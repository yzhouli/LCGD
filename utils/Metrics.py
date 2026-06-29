import torch
import numpy as np


class Metrics(object):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.PAD = 0
        self.device = torch.device(device)
        # 预先计算常量，加载到设备上
        self.log2_val = torch.log2(torch.tensor(2.0, device=self.device, dtype=torch.float32))

    def apk(self, actual, predicted, k=10):
        """单标签场景下的 Average Precision 计算 (用于验证GPU版本的正确性)"""
        score = 0.0
        num_hits = 0.0
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        # 对于单标签场景，min(len(actual), k) 当 k>=1 时总是 1
        # 所以 AP@K = score / 1.0 = score
        # 而对于单个正例，只有当它在 top-k 中时，score 才是 1/(rank)。
        # 因此，AP@K = 1 / (rank of true label in top-k) if in top-k, else 0
        return score / min(len(actual), k)

    def ndcg_k(self, actual, predicted, k=10):
        """单标签场景下的 NDCG 计算 (用于验证GPU版本的正确性)"""
        # 理想排序：如果有相关项，则排在第一位，其余为0
        ideal_ranking = [1.0] + [0.0] * (k - 1)

        # 实际排序：将预测列表与真实标签比较
        actual_ranking = []
        for item in predicted[:k]:
            if item == actual:
                actual_ranking.append(1.0)
            else:
                actual_ranking.append(0.0)

        # 计算 DCG
        dcg = 0.0
        for i, rel in enumerate(actual_ranking):
            dcg += rel / np.log2(i + 2)

        # 计算 IDCG (对于单个正例，IDCG 总是 1 / log2(2))
        idcg = 0.0
        for i, rel in enumerate(ideal_ranking):
            idcg += rel / np.log2(i + 2)

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def compute_metric(self, y_prob, y_true, k_list=[10, 20, 50, 100]):
        """原始的 NumPy 版本"""
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        scores.update({'ndcg@' + str(k): [] for k in k_list})

        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                scores_len += 1.0
                p_sort = p_.argsort()

                for k in k_list:
                    topk = p_sort[-k:][::-1]  # Get top-k indices

                    scores['hits@' + str(k)].append(1. if y_ in topk else 0.)
                    scores['map@' + str(k)].append(self.apk([y_], topk, k))
                    scores['ndcg@' + str(k)].append(self.ndcg_k(y_, topk, k))

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len

    def compute_metric_gpu(self, y_prob, y_true, k_list=[10, 20, 50, 100]):
        """
        使用 PyTorch 和 GPU 加速的评估指标计算函数。
        针对单标签场景进行了优化。
        """
        # 将输入数据转换为 PyTorch 张量并移动到指定设备
        y_prob_torch = torch.as_tensor(y_prob, dtype=torch.float32, device=self.device)
        y_true_torch = torch.as_tensor(y_true, dtype=torch.long, device=self.device)

        # 1. 筛选出非PAD的有效样本
        valid_mask = (y_true_torch != self.PAD)
        if not valid_mask.any():
            empty_scores = {f'hits@{k}': 0.0 for k in k_list}
            empty_scores.update({f'map@{k}': 0.0 for k in k_list})
            empty_scores.update({f'ndcg@{k}': 0.0 for k in k_list})
            return empty_scores, 0.0

        y_prob_valid = y_prob_torch[valid_mask]
        y_true_valid = y_true_torch[valid_mask]
        scores_len = y_true_valid.size(0)

        scores = {}

        for k in k_list:
            # 2. 获取 Top-K 的预测索引
            # 使用 torch.topk 是最高效的方式
            top_k_scores, top_k_indices_unsorted = torch.topk(y_prob_valid, k, dim=1, largest=True, sorted=False)
            # 对 Top-K 进行排序以得到降序
            _, top_k_indices_sorted_desc = torch.sort(top_k_scores, dim=1, descending=True)
            # 使用 gather 获取最终排序后的索引
            top_k_indices = torch.gather(top_k_indices_unsorted, 1, top_k_indices_sorted_desc)

            # 3. 向量化计算指标
            # --- Hits@K ---
            hits_mask = (top_k_indices == y_true_valid.unsqueeze(1))
            hits_at_k = hits_mask.any(dim=1).float()
            scores[f'hits@{k}'] = hits_at_k

            # --- NDCG@K (针对单个正例) ---
            # 找到真实标签在 Top-K 中的位置
            true_pos_mask = (top_k_indices == y_true_valid.unsqueeze(1))
            # 为每个位置分配一个排名 (1-based)
            ranks_tensor = torch.arange(1, k + 1, device=self.device, dtype=torch.float32).unsqueeze(0).expand_as(
                top_k_indices)
            # 如果不在 Top-K 中，则将其排名设为一个很大的数
            true_pos_ranks = torch.where(true_pos_mask, ranks_tensor, k + 1.0)
            # 取最小排名 (即第一次出现的位置)
            true_pos_ranks = torch.min(true_pos_ranks, dim=1)[0]

            # 计算 DCG: 如果在 top-k 中，DCG = 1 / log2(rank + 1)
            dcg_at_k = torch.where(
                true_pos_ranks <= k,
                1.0 / torch.log2(true_pos_ranks + 1),
                torch.tensor(0.0, device=self.device, dtype=torch.float32)
            )
            # 计算 IDCG: 对于单个正例，IDCG = 1 / log2(2)
            idcg_at_k = 1.0 / self.log2_val
            # 计算 NDCG
            ndcg_at_k = dcg_at_k / idcg_at_k
            # 对于不在 top-k 中的样本，NDCG 应为 0
            ndcg_at_k = torch.where(true_pos_ranks <= k, ndcg_at_k,
                                    torch.tensor(0.0, device=self.device, dtype=torch.float32))
            scores[f'ndcg@{k}'] = ndcg_at_k

            # --- MAP@K (针对单个正例) ---
            # AP@K = 1 / rank if true label is in top-k, else 0
            map_at_k_per_sample = torch.where(
                true_pos_ranks <= k,
                1.0 / true_pos_ranks,
                torch.tensor(0.0, device=self.device, dtype=torch.float32)
            )
            scores[f'map@{k}'] = map_at_k_per_sample

        # 4. 计算平均值并转移到 CPU
        final_scores = {metric_name: metric_values.mean().cpu().item() for metric_name, metric_values in scores.items()}

        return final_scores, float(scores_len)