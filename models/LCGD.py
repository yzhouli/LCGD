"""LCGD: Langevin-Inspired Controlled Graph Dynamics for Robust Cascades Prediction.

A cascade prediction model that replaces the linear GNN graph encoder with the
Controlled Graph Dynamics (CGD) — a Langevin-inspired, energy-based
dynamical system with excitation-inhibition modulation and a
low-rank global attractor (see layers/CGD.py).

Run with:
    python run.py --model_name LCGD -data_name douban
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utils import Constants
from layers.TransformerBlock import TransformerBlock
from layers.CGD import CGDEncoder

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LCGD(nn.Module):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--pos_dim', type=int, default=64)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--time_step_split', type=int, default=8)
        parser.add_argument('--ssl_reg', type=float, default=1e-5)
        parser.add_argument('--alpha', type=float, default=1.0)
        parser.add_argument('--ssl_temp', type=float, default=0.8)
        parser.add_argument('--K', type=int, default=3)
        parser.add_argument('--nc', type=int, default=1)
        # CGD (controlled graph dynamics) options
        parser.add_argument('--cgd_layers', type=int, default=2,
                            help='CGD layers: cgd_layers-1 dynamics iterations + 1 output layer')
        parser.add_argument('--cgd_heads', type=int, default=4)
        parser.add_argument('--cgd_alpha', type=float, default=0.2,
                            help='CGD dynamics iteration step size')
        parser.add_argument('--cgd_noise_std', type=float, default=0.02,
                            help='CGD noise std, 0 disables noise')
        parser.add_argument('--cgd_rank', type=int, default=8,
                            help='rank of the CGD low-rank interaction term')
        parser.add_argument('--cgd_layer_norm', type=int, default=1)
        parser.add_argument('--cgd_residual', type=int, default=1)
        parser.add_argument('--cgd_use_lowrank', type=int, default=1,
                            help='1=use low-rank P^T Q P term, 0=ablate (Part 2)')
        parser.add_argument('--cgd_use_dynamics', type=int, default=1,
                            help='1=controlled dynamics, 0=plain transformer (Part 2)')
        return parser

    def __init__(self, args, data_loader):
        super(LCGD, self).__init__()
        self.args = args
        self.device = args.device
        self.user_num = data_loader.user_num
        self.cas_num = data_loader.cas_num
        self.embedding_size = args.d_model
        self.pos_dim = args.d_model  # Must match d_model to avoid dim mismatch
        self.n_heads = args.n_heads
        self.drop_timestamp = nn.Dropout(args.dropout)
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.cas_embedding = nn.Embedding(self.cas_num, self.embedding_size)
        self.user_bias = nn.Embedding(self.user_num, 1)
        self.pos_embedding = nn.Embedding(data_loader.cas_num, self.pos_dim)
        self.align_attention = TransformerBlock(input_size=self.embedding_size, n_heads=self.n_heads, device=self.device)
        self.linear = nn.Linear(self.embedding_size, self.user_num)

        # CGD graph encoder over the cascade-user bipartite graph
        # (replaces PMRCA's normalized-adjacency LightGCN propagation)
        train_cas_user_dict = data_loader.train_cas_user_dict
        edge_index = self.build_edge_index(self.cas_num, self.user_num, train_cas_user_dict)
        self.register_buffer('edge_index', edge_index)
        self.cgd_encoder = CGDEncoder(
            hidden_dim=self.embedding_size,
            num_heads=args.cgd_heads,
            n_layers=args.cgd_layers,
            num_nodes=self.cas_num + self.user_num,
            alpha=args.cgd_alpha,
            noise_std=args.cgd_noise_std,
            rank=args.cgd_rank,
            dropout=args.dropout,
            layer_norm=bool(args.cgd_layer_norm),
            residual=bool(args.cgd_residual),
            use_lowrank=bool(args.cgd_use_lowrank),
            use_dynamics=bool(args.cgd_use_dynamics),
        )

        self.ssl_reg = args.ssl_reg
        self.alpha = args.alpha
        self.ssl_temp = args.ssl_temp
        self.proto_reg = 8e-8
        self.num_clusters = 10
        self.attn_size = 8
        self.K = args.K
        self.W1 = nn.Linear(self.embedding_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)
        self.linear2 = nn.Linear(self.embedding_size, self.embedding_size)
        self.k = self.num_clusters
        self.user_centroids = None
        self.user_2cluster = None
        self.cas_centroids = None
        self.cas_2cluster = None
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.pos_embedding.weight)
        init.xavier_normal_(self.linear.weight)

    @staticmethod
    def build_edge_index(cas_count, user_count, train_mat):
        """Bidirectional cascade<->user edges plus self-loops.

        Node ids: 0..cas_count-1 are cascades, cas_count.. are users,
        matching the ordering of get_ego_embeddings(). Self-loops keep the
        CGD attention normalizer z strictly positive for isolated nodes.
        """
        src, dst = [], []
        for cas in train_mat:
            for user in train_mat[cas]:
                u = cas_count + user
                src.extend([cas, u])
                dst.extend([u, cas])
        n = cas_count + user_count
        loops = list(range(n))
        src.extend(loops)
        dst.extend(loops)
        return torch.tensor([src, dst], dtype=torch.long)

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        cas_embeddings = self.cas_embedding.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.cas_centroids, self.cas_2cluster = self.run_kmeans(cas_embeddings)

    def run_kmeans(self, x, n_iters=20):
        """Pure-torch Lloyd's k-means.

        Replaces faiss (used by PMRCA) to avoid the duplicate-OpenMP
        crash when pip-installed faiss and torch coexist on macOS.
        """
        x = torch.from_numpy(x).to(self.device)
        n = x.shape[0]
        centroids = x[torch.randperm(n, device=self.device)[:self.k]].clone()

        assign = None
        for _ in range(n_iters):
            assign = torch.cdist(x, centroids).argmin(dim=1)
            counts = torch.bincount(assign, minlength=self.k)
            new_centroids = torch.zeros_like(centroids).index_add_(0, assign, x)
            new_centroids = new_centroids / counts.clamp(min=1).unsqueeze(1)
            empty = counts == 0
            new_centroids[empty] = centroids[empty]  # keep empty clusters in place
            if torch.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        centroids = F.normalize(centroids, p=2, dim=1)
        node2cluster = assign.long().to(self.device)
        return centroids, node2cluster

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        cas_embeddings = self.cas_embedding.weight
        ego_embeddings = torch.cat([cas_embeddings, user_embeddings], dim=0)
        return ego_embeddings

    def ssl_layer_loss(self, current_embedding, previous_embedding, cas_idx, user_idx):
        current_cas_embeddings, current_user_embeddings = torch.split(
            current_embedding, [self.cas_num, self.user_num]
        )
        previous_cas_embeddings_all, previous_user_embeddings_all = torch.split(
            previous_embedding, [self.cas_num, self.user_num]
        )

        current_user_embeddings = current_user_embeddings[user_idx]
        previous_user_embeddings = previous_user_embeddings_all[user_idx]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_cas_embeddings[cas_idx]
        previous_item_embeddings = previous_cas_embeddings_all[cas_idx]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_cas_embeddings_all)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def aware_loss(self, node_embedding, cas, tg_user):
        cas_embeddings_all, tuser_embeddings_all = torch.split(
            node_embedding, [self.cas_num, self.user_num]
        )

        user_embeddings = tuser_embeddings_all[tg_user]
        norm_user_embeddings = F.normalize(user_embeddings)

        user2cluster = self.user_2cluster[tg_user]
        user2centroids = self.user_centroids[user2cluster]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(
            norm_user_embeddings, self.user_centroids.transpose(0, 1)
        )
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        item_embeddings = cas_embeddings_all[cas]
        norm_item_embeddings = F.normalize(item_embeddings)

        item2cluster = self.cas_2cluster[cas]
        item2centroids = self.cas_centroids[item2cluster]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(
            norm_item_embeddings, self.cas_centroids.transpose(0, 1)
        )
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def forward(self, input_seq, input_timestamp, tgt_idx):
        ego_embeddings = self.get_ego_embeddings().to(self.device)

        # CGD controlled-dynamics propagation over the bipartite graph;
        # embedding_list keeps the per-iteration embeddings so the SSL
        # losses can contrast layer 0 (ego) with layer 1, as in PMRCA
        embedding_list = [ego_embeddings] + self.cgd_encoder(ego_embeddings, self.edge_index)

        gcn_all_embeddings = torch.stack(embedding_list, dim=1)
        gcn_all_embeddings = torch.mean(gcn_all_embeddings, dim=1)

        cas_all_embeddings, user_all_embeddings = torch.split(
            gcn_all_embeddings, [self.cas_num, self.user_num]
        )

        input_seq = input_seq[:, :-1]
        mask = (input_seq == Constants.PAD)

        batch_t = torch.arange(input_seq.size(1)).expand(input_seq.size()).to(self.device)
        position_embed = self.pos_embedding(batch_t)

        batch_size, max_len = input_seq.size()

        original_seq_emb = self.user_embedding(input_seq.to(self.device))
        dyemb = user_all_embeddings[input_seq.to(self.device)]
        dyemb += position_embed

        valid_his = (input_seq > 0).long().to(self.device)

        attn_score = self.W2(self.W1(original_seq_emb).tanh()).to(self.device)
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf).to(self.device)
        attn_score = attn_score.transpose(-1, -2).to(self.device)
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1).to(self.device)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0).to(self.device)
        intention_vectors = (original_seq_emb[:, None, :, :] * attn_score[:, :, :, None]).sum(-2).sum(
            -2).to(self.device)

        intention_vectors = intention_vectors.unsqueeze(1)
        intention_vectors = intention_vectors.expand(-1, original_seq_emb.size(1), -1)

        att_out = self.align_attention(dyemb.to(self.device),
                                       intention_vectors.to(self.device),
                                       intention_vectors.to(self.device),
                                       mask=mask.to(self.device))

        output = self.linear(att_out.to(self.device))

        mask = self.get_previous_user_mask(input_seq.to(self.device), self.user_num)
        output = output.to(self.device) + mask.to(self.device)

        if self.training:
            return output.view(-1, output.size(-1)), user_all_embeddings, cas_all_embeddings, embedding_list
        else:
            return output.view(-1, output.size(-1))

    def get_previous_user_mask(self, seq, user_size):
        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask)
        if seq.is_cuda:
            previous_mask = previous_mask.to(self.device)
        masked_seq = previous_mask * seqs.data.float()

        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
        if seq.is_cuda:
            PAD_tmp = PAD_tmp.to(self.device)
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
        if seq.is_cuda:
            ans_tmp = ans_tmp.to(self.device)
        masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
        masked_seq = masked_seq.detach()
        return masked_seq.to(self.device)

    def get_performance(self, input_seq, input_seq_timestamp, history_seq_idx, loss_func, gold):
        pred, user_all_embeddings, cas_all_embeddings, embedding_list = self.forward(input_seq,
                                                                                     input_seq_timestamp,
                                                                                     history_seq_idx)

        center_embedding = embedding_list[0].to(self.device)
        context_embedding = embedding_list[1].to(self.device)
        ssl_loss = torch.tensor(0.0).to(self.device)
        proto_loss = torch.tensor(0.0).to(self.device)

        ssl_loss = self.ssl_layer_loss(
            context_embedding, center_embedding, history_seq_idx.to(self.device), gold[:, -1].to(self.device)
        )

        # proto_loss = self.aware_loss(center_embedding, history_seq_idx.to(self.device), gold[:, -1].to(self.device))

        loss = loss_func(pred, gold.contiguous().view(-1).to(self.device))

        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        n_correct = pred.data.eq(gold.data)
        n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
        return loss + ssl_loss + proto_loss, n_correct

    def before_epoch(self):
        # self.e_step()
        pass
