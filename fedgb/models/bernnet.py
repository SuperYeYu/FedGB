import torch.nn as nn
import torch.nn.functional as F

from fedgb.models.pregc_common import BernPropagation, scalar_edge_weight


class BernNetEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=2, dropout=0.5, k=10, dprate=0.0):
        super(BernNetEncoder, self).__init__()
        self.dropout = dropout
        self.dprate = dprate
        self.layers = nn.ModuleList()
        self.propagations = nn.ModuleList()
        for layer_id in range(max(1, num_layers)):
            in_dim = input_dim if layer_id == 0 else hid_dim
            self.layers.append(nn.Linear(in_dim, hid_dim))
            self.propagations.append(BernPropagation(k=k))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = scalar_edge_weight(data)
        for layer, propagation in zip(self.layers, self.propagations):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x))
            if self.dprate > 0:
                x = F.dropout(x, p=self.dprate, training=self.training)
            x = propagation(x, edge_index, edge_weight)
        return x


class BernNet(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5, k=10, dprate=0.0):
        super(BernNet, self).__init__()
        self.encoder = BernNetEncoder(input_dim, hid_dim, num_layers, dropout, k, dprate)
        self.head = nn.Linear(hid_dim, output_dim)

    def forward(self, data):
        x = self.encoder(data)
        logits = self.head(x)
        return x, logits
