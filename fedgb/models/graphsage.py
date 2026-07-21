import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=2, dropout=0.5):
        super(GraphSAGEEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hid_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hid_dim, hid_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.encoder = GraphSAGEEncoder(input_dim, hid_dim, num_layers, dropout)
        self.head = nn.Linear(hid_dim, output_dim)

    def forward(self, data):
        x = self.encoder(data)
        logits = self.head(x)
        return x, logits
