import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PANConv
from torch_geometric.nn.pool import global_add_pool, PANPooling

class GlobalPANEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=2, dropout=0.5):
        super(GlobalPANEncoder, self).__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(PANConv(input_dim, hid_dim, filter_size=0))
        self.batch_norms.append(nn.BatchNorm1d(hid_dim))
        for _ in range(num_layers - 1):
            self.convs.append(PANConv(hid_dim, hid_dim, filter_size=0))
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))

        self.pan_pooling = PANPooling(hid_dim, ratio=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x, M = conv(x, edge_index)
            x = F.relu(batch_norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x, connect_out_edge_index, connect_out_edge_attr, connect_out_batch, perm, score = self.pan_pooling(x, M, batch)
        embedding = global_add_pool(x, connect_out_batch)
        return embedding


class GlobalPAN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(GlobalPAN, self).__init__()
        self.encoder = GlobalPANEncoder(input_dim, hid_dim, num_layers, dropout)
        self.head = nn.Linear(hid_dim, output_dim)

    def forward(self, data):
        embedding = self.encoder(data)
        logits = self.head(embedding)
        return embedding, logits
