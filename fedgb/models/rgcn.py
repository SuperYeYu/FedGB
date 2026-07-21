import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_relations=8, num_layers=2, dropout=0.5):
        super(RGCNEncoder, self).__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.dropout = dropout
        self.num_relations = int(num_relations)
        self.input_proj = nn.Linear(input_dim, hid_dim)
        self.convs = nn.ModuleList(
            [RGCNConv(hid_dim, hid_dim, self.num_relations) for _ in range(int(num_layers))]
        )

    def forward(self, data):
        x = F.relu(self.input_proj(data.x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        edge_index = data.edge_index
        edge_type = getattr(data, "edge_type", None)
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
        else:
            edge_type = edge_type.to(device=edge_index.device, dtype=torch.long)

        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class RGCN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_relations=8, num_layers=2, dropout=0.5):
        super(RGCN, self).__init__()
        self.encoder = RGCNEncoder(input_dim, hid_dim, num_relations, num_layers, dropout)
        self.head = nn.Linear(hid_dim, output_dim)

    def forward(self, data):
        x = self.encoder(data)
        logits = self.head(x)
        return x, logits
