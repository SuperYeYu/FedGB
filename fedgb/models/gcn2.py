import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv

class GCN2Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, alpha=0.1, num_layers=2, dropout=0.5):
        super(GCN2Encoder, self).__init__()
        self.alpha = alpha
        self.dropout = dropout

        self.linear_embed = nn.Linear(input_dim, hid_dim)
        self.convs = nn.ModuleList([GCN2Conv(hid_dim, alpha=alpha) for _ in range(num_layers)])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.linear_embed(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x0 = x
        for conv in self.convs:
            x = conv(x, x0, edge_index)
        return x


class GCN2(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, alpha=0.1, num_layers=2, dropout=0.5):
        super(GCN2, self).__init__()
        self.encoder = GCN2Encoder(input_dim, hid_dim, alpha, num_layers, dropout)
        self.head = nn.Linear(hid_dim, output_dim)

    def forward(self, data):
        x = self.encoder(data)
        logits = self.head(x)
        return x, logits
