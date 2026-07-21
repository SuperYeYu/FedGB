import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from fedgb.models.pregc_common import scalar_edge_weight


class ChebNetEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=2, dropout=0.5, k=2):
        super(ChebNetEncoder, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for layer_id in range(max(1, num_layers)):
            in_dim = input_dim if layer_id == 0 else hid_dim
            self.convs.append(ChebConv(in_dim, hid_dim, K=k))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = scalar_edge_weight(data)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ChebNet(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5, k=2):
        super(ChebNet, self).__init__()
        self.encoder = ChebNetEncoder(input_dim, hid_dim, num_layers, dropout, k)
        self.head = nn.Linear(hid_dim, output_dim)

    def forward(self, data):
        x = self.encoder(data)
        logits = self.head(x)
        return x, logits
