import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

from fedgb.algorithms.graph_fgl.fedssp.fedssp_config import config
from fedgb.algorithms.graph_fgl.fedssp.utils import compute_laplacian_eigh_batch


class SineEncoding(nn.Module):
    def __init__(self, hid_dim):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hid_dim = hid_dim
        self.eig_w = nn.Linear(hid_dim + 1, hid_dim)

    def forward(self, eigenvalues):
        scaled = eigenvalues * self.constant
        div = torch.exp(
            torch.arange(0, self.hid_dim, 2, device=eigenvalues.device)
            * (-math.log(10000.0) / self.hid_dim)
        )
        encoded = scaled.unsqueeze(2) * div
        encoded = torch.cat((eigenvalues.unsqueeze(2), torch.sin(encoded), torch.cos(encoded)), dim=2)
        return self.eig_w(encoded[:, :, : self.hid_dim + 1])


class FeedForwardNetwork(nn.Module):
    def __init__(self, hid_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hid_dim, hid_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        return self.layer2(self.gelu(self.layer1(x)))


class SpectralConv(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(SpectralConv, self).__init__()
        self.pre_ffn = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.GELU())
        self.preffn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, bases):
        src, dst = edge_index
        edge_feat = self.pre_ffn(x[src]) * bases
        aggr = x.new_zeros(x.shape)
        aggr.index_add_(0, dst, edge_feat)
        x = x + self.preffn_dropout(aggr)
        return x + self.ffn_dropout(self.ffn(x))


_POOLING_FNS = {
    "sum": global_add_pool,
    "add": global_add_pool,
    "mean": global_mean_pool,
    "avg": global_mean_pool,
    "average": global_mean_pool,
    "max": global_max_pool,
}


class FedSSPGraphModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        output_dim,
        num_layers=3,
        num_heads=4,
        dropout=0.5,
        graph_pooling_type="mean",
    ):
        super(FedSSPGraphModel, self).__init__()
        self.hid_dim = hid_dim
        self.num_heads = int(num_heads)
        if graph_pooling_type not in _POOLING_FNS:
            raise ValueError(
                "Unsupported fedssp_graph_pooling_type '{}'. Use one of: {}.".format(
                    graph_pooling_type, ", ".join(sorted(_POOLING_FNS))
                )
            )
        self.graph_pooling_type = graph_pooling_type
        self.graph_pooling = _POOLING_FNS[graph_pooling_type]
        self.atom_encoder = nn.Sequential(nn.Linear(input_dim, hid_dim), nn.ReLU())
        self.eig_encoder = SineEncoding(hid_dim)
        self.mha_norm = nn.LayerNorm(hid_dim)
        self.ffn_norm = nn.LayerNorm(hid_dim)
        self.mha_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(hid_dim, self.num_heads, dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(hid_dim)
        self.filter_encoder = nn.Sequential(
            nn.Linear(self.num_heads + 1, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.GELU(),
        )
        self.decoder = nn.Linear(hid_dim, self.num_heads)
        self.adj_dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList([SpectralConv(hid_dim, dropout) for _ in range(num_layers)])
        self.head = nn.Linear(hid_dim, output_dim)
        self.preference = nn.Parameter(torch.zeros(hid_dim))

    def forward(self, data, use_preference=False):
        embedding = self.encode(data)
        if use_preference:
            logits = self.head(embedding + self.preference)
        else:
            logits = self.head(embedding)
        return embedding, logits

    def encode(self, data):
        eigenvalues, eigenvectors, lengths = compute_laplacian_eigh_batch(data)
        if eigenvalues.shape[0] == 0:
            return data.x.new_zeros((0, self.hid_dim))

        x = self.atom_encoder(data.x.float())
        spectral_filters = self._spectral_edge_filters(eigenvalues, eigenvectors, lengths, data)
        for conv in self.convs:
            x = conv(x, data.edge_index, spectral_filters)
        return self.graph_pooling(x, data.batch)

    def _spectral_edge_filters(self, eigenvalues, eigenvectors, lengths, data):
        eig = self.eig_encoder(eigenvalues)
        mask = torch.arange(eigenvalues.shape[1], device=eigenvalues.device).expand(
            eigenvalues.shape[0], eigenvalues.shape[1]
        ) >= lengths.unsqueeze(1)

        attn_input = self.mha_norm(eig)
        attn_output, _ = self.mha(attn_input, attn_input, attn_input, key_padding_mask=mask)
        eig = eig + self.mha_dropout(attn_output)
        eig = eig + self.ffn_dropout(self.ffn(self.ffn_norm(eig)))

        new_e = self.decoder(eig).transpose(2, 1)
        edge_basis = self._edge_basis(new_e, eigenvectors, data)
        edge_basis = self.adj_dropout(self.filter_encoder(edge_basis))
        return softmax(edge_basis, data.edge_index[1], num_nodes=data.num_nodes)

    def _edge_basis(self, new_e, eigenvectors, data):
        src, dst = data.edge_index
        graph_id = data.batch[src]
        node_start = torch.zeros(new_e.shape[0], dtype=torch.long, device=src.device)
        node_start[1:] = torch.cumsum(torch.bincount(data.batch, minlength=new_e.shape[0]), dim=0)[:-1]
        local_src = src - node_start[graph_id]
        local_dst = dst - node_start[graph_id]

        bases = [torch.ones(src.shape[0], device=src.device, dtype=new_e.dtype)]
        for head_id in range(self.num_heads):
            values = (
                eigenvectors[graph_id, local_src, :]
                * new_e[graph_id, head_id, :]
                * eigenvectors[graph_id, local_dst, :]
            ).sum(dim=-1)
            bases.append(values)
        return torch.stack(bases, dim=-1)

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)


def build_fedssp_model(args, task):
    return FedSSPGraphModel(
        input_dim=task.num_feats,
        hid_dim=getattr(args, "fedssp_hidden_dim", config["fedssp_hidden_dim"]),
        output_dim=task.num_targets if getattr(args, "task", None) == "graph_reg" else task.num_global_classes,
        num_layers=getattr(args, "fedssp_num_layers", config["fedssp_num_layers"]),
        num_heads=getattr(args, "fedssp_num_heads", config["fedssp_num_heads"]),
        dropout=getattr(args, "fedssp_dropout", config["fedssp_dropout"]),
        graph_pooling_type=getattr(args, "fedssp_graph_pooling_type", config.get("fedssp_graph_pooling_type", "mean")),
    )
