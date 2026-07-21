import torch.nn as nn
from torch_geometric.nn import SGConv


class SGCEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=2, dropout=0.5):
        super(SGCEncoder, self).__init__()
        self.dropout = dropout
        self.conv = SGConv(input_dim, hid_dim, K=num_layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index)
        return x


class SGC(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(SGC, self).__init__()
        self.encoder = SGCEncoder(input_dim, hid_dim, num_layers, dropout)
        self.head = nn.Linear(hid_dim, output_dim)

    def forward(self, data):
        x = self.encoder(data)
        logits = self.head(x)
        return x, logits
