"""CGD (Controlled Graph Dynamics) encoder.

Pure PyTorch: message passing is expressed with edge-index gather + index_add_
scatter, so no DGL / torch_geometric extension is required.

Components:
- MultiHeadAttentionLayer / GraphTransformerLayer: the CGD graph transformer
  attention layer, operating on (h, edge_index).
- CGDEncoder: stacks the transformer layer with controlled dynamics iteration
  h <- h - a*grad - a*(h + c*h) + a*(P^T Q P) h + sqrt(a)*noise
  and returns the embedding of every dynamics step, so callers (e.g.
  LCGD) can reuse the per-layer embedding list for SSL losses.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.beta = nn.Parameter(nn.init.constant_(torch.empty(1), np.sqrt(self.out_dim)))

        self.Q = nn.Parameter(nn.init.normal_(torch.empty(in_dim, num_heads, out_dim), std=0.01))
        self.K = nn.Parameter(nn.init.normal_(torch.empty(in_dim, num_heads, out_dim), std=0.01))

    def propagate_attention(self, edge_index, Q_h, K_h, F1, F2):
        src, dst = edge_index
        n = Q_h.shape[0]

        # score(e) = <K_h[src], Q_h[dst]>, clamp for softmax numerical stability
        score = (K_h[src] * Q_h[dst]).sum(-1, keepdim=True)
        score = torch.exp((score / self.beta).clamp(-5, 5))

        # T1[v] = sum_{e:(u->v)} F1[u] * score(e);  z[v] = sum_{e:(u->v)} score(e)
        T1 = torch.zeros_like(F1).index_add_(0, dst, F1[src] * score)
        z = torch.zeros(n, self.num_heads, 1, dtype=score.dtype, device=score.device)
        z = z.index_add_(0, dst, score)

        # second attention pass: score_2(e) = <Q_h[src], K_h[dst]>
        score_2 = (Q_h[src] * K_h[dst]).sum(-1, keepdim=True)
        score_2 = torch.exp((score_2 / self.beta).clamp(-5, 5))

        # T2[v] = sum_{e:(u->v)} F2[u] * score_2(e) / z[u]
        # nodes need at least one in-edge (e.g. a self-loop) so that z > 0
        T2 = torch.zeros_like(F2).index_add_(0, dst, F2[src] * score_2 / z[src])

        return T1, T2, z

    def forward(self, h, edge_index):
        Q_h = torch.einsum("bd, dhz-> bhz", h, self.Q)
        K_h = torch.einsum("bd, dhz-> bhz", h, self.K)

        F1 = torch.einsum("dzh,bhz->bhd", self.Q.permute(0, 2, 1), K_h)
        F2 = torch.einsum("dzh,bhz->bhd", self.K.permute(0, 2, 1), Q_h)

        T1, T2, z = self.propagate_attention(edge_index, Q_h, K_h, F1, F2)

        head_out_1 = T1 / z
        head_out_2 = T2

        final = torch.sum(head_out_1 + head_out_2, dim=1)  # B * in_dim
        return final


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, ffn=2, layer_norm=False, batch_norm=False, residual=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim, num_heads)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # tied-weight FFN
        self.W = nn.Parameter(nn.init.uniform_(torch.empty(out_dim, out_dim * ffn),
                                               a=-1.0 / np.sqrt(out_dim), b=1.0 / np.sqrt(out_dim)))

    def forward(self, h, edge_index):
        if self.layer_norm:
            h = self.layer_norm1(h)
        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in1 = h  # for residual connection
        h2 = h

        h = self.attention(h, edge_index)

        h2 = torch.matmul(h2, self.W)
        h2 = F.relu(h2)
        h2 = torch.matmul(h2, self.W.t())
        h = h + h2
        h = F.dropout(h, self.dropout, training=self.training)

        if self.residual:
            h = h_in1 + h

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_heads, self.residual)


class CGDEncoder(nn.Module):
    """CGD controlled-dynamics graph encoder over a fixed node set.

    forward(h, edge_index) runs (n_layers - 1) controlled dynamics
    iterations followed by an output transformer layer, and returns the
    list of intermediate embeddings (one per iteration, plus the final
    output), mirroring the layer-wise embedding list of a GCN stack.
    """

    def __init__(self, hidden_dim, num_heads, n_layers, num_nodes,
                 alpha=0.1, noise_std=0.02, rank=4, dropout=0.1, ffn=4,
                 layer_norm=True, batch_norm=False, residual=True,
                 use_lowrank=True, use_dynamics=True):
        super().__init__()

        self.alpha = alpha
        self.noise_std = noise_std
        self.n_layers = max(1, n_layers - 1)
        self.use_lowrank = use_lowrank
        self.use_dynamics = use_dynamics

        self.layers = GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout, ffn,
                                            layer_norm, batch_norm, residual)
        self.layers_out = GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout, ffn,
                                                layer_norm, batch_norm, residual)

        # suppression coefficient
        self.c = nn.Parameter(torch.rand(1))
        # low-rank interaction term P^T Q P, avoids the NxN matrix
        self.P = nn.Parameter(torch.rand(rank, num_nodes))
        self.Q = nn.Parameter(torch.randn(rank, rank) * (1.0 / rank) ** 0.5)

    def _low_rank_mul(self, X):
        P = self.P.to(X.device, dtype=X.dtype)
        Q = self.Q.to(X.device, dtype=X.dtype)
        return P.t() @ (Q @ (P @ X))

    def forward(self, h, edge_index):
        embedding_list = []
        h_cur = h
        for _ in range(self.n_layers):
            if not self.use_dynamics:
                # Ablation: plain graph-transformer layer, no controlled dynamics
                h_cur = self.layers(h_cur, edge_index)
                embedding_list.append(h_cur)
                continue
            grad = self.layers(h_cur, edge_index)
            if self.noise_std > 0:
                noise = torch.normal(mean=0.0, std=self.noise_std, size=(1,), device=h_cur.device)
            else:
                noise = 0
            lowrank_h = self._low_rank_mul(h_cur) if self.use_lowrank else 0
            h_cur = h_cur - self.alpha * grad \
                    - self.alpha * (h_cur + self.c * h_cur) \
                    + self.alpha * lowrank_h \
                    + (self.alpha ** 0.5) * noise
            embedding_list.append(h_cur)
        h_out = self.layers_out(h_cur, edge_index)
        embedding_list.append(h_out)
        return embedding_list
