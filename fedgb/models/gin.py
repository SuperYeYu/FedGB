import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool


_POOLING_FNS = {
    "sum": global_add_pool,
    "add": global_add_pool,
    "mean": global_mean_pool,
    "avg": global_mean_pool,
    "average": global_mean_pool,
    "max": global_max_pool,
}

class GINEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=2, dropout=0.5, graph_pooling_type="sum"):
        super(GINEncoder, self).__init__()
        self.dropout = dropout
        self.graph_pooling_type = graph_pooling_type
        if graph_pooling_type not in _POOLING_FNS:
            raise ValueError(
                "Unsupported graph_pooling_type '{}'. Use one of: {}.".format(
                    graph_pooling_type, ", ".join(sorted(_POOLING_FNS))
                )
            )
        self.graph_pooling = _POOLING_FNS[graph_pooling_type]

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        dim = input_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dim, 2 * hid_dim),
                nn.BatchNorm1d(2 * hid_dim),
                nn.ReLU(),
                nn.Linear(2 * hid_dim, hid_dim),
            )
            conv = GINConv(mlp, train_eps=True)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))
            dim = hid_dim

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = self.graph_pooling(x, batch)
        return embedding


class GIN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5, graph_pooling_type="sum"):
        super(GIN, self).__init__()
        self.encoder = GINEncoder(input_dim, hid_dim, num_layers, dropout, graph_pooling_type)
        self.head = nn.Linear(hid_dim, output_dim)

    def forward(self, data):
        embedding = self.encoder(data)
        logits = self.head(embedding)
        return embedding, logits
