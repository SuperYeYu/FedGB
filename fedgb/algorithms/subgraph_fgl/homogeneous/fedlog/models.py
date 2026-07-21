import torch
import torch.nn as nn
import torch.nn.functional as F


def diff_mean(data, segment_ids, num_nodes):
    result = data.new_zeros((num_nodes, data.size(1)))
    count = data.new_zeros((num_nodes, data.size(1)))
    expanded = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, expanded, data)
    count.scatter_add_(0, expanded, torch.ones_like(data))
    return result / count.clamp_min(1.0)


class FedLoGTaskAdapterLayer(nn.Module):
    def __init__(self, raw_dim, hid_dim):
        super(FedLoGTaskAdapterLayer, self).__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(raw_dim + 1, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
        )
        self.trans_mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, 1, bias=False),
        )
        nn.init.xavier_uniform_(self.trans_mlp[-1].weight, gain=0.001)

    def forward(self, edge_index, neighbor, query_embedding, proto_embedding):
        x = torch.cat([proto_embedding, query_embedding], dim=0)
        x_neighbor = torch.cat([proto_embedding, neighbor], dim=0)
        row, col = edge_index
        coord_diff = x[row] - x[col]
        sqr_dist = torch.sum(coord_diff**2, dim=1, keepdim=True)
        msg = self.msg_mlp(torch.cat([x_neighbor[col], sqr_dist], dim=1))
        trans = coord_diff * self.trans_mlp(msg)
        trans = diff_mean(trans, row, x.size(0))
        query_embedding = query_embedding + trans[proto_embedding.size(0) :]
        return neighbor, query_embedding


class FedLoGTaskAdapter(nn.Module):
    def __init__(self, str_dim, hid_dim, num_layers=2):
        super(FedLoGTaskAdapter, self).__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.layers = nn.ModuleList(
            [FedLoGTaskAdapterLayer(str_dim, hid_dim) for _ in range(num_layers)]
        )

    def forward(self, query_embedding, neighbor, proto_embedding, edge_index):
        query_embedding = self.layer_norm(query_embedding)
        for layer in self.layers:
            neighbor, query_embedding = layer(edge_index, neighbor, query_embedding, proto_embedding)
        return neighbor, query_embedding


class FedLoGNeighborGenerator(nn.Module):
    def __init__(self, feat_dim, hid_dim=256, dropout=0.5):
        super(FedLoGNeighborGenerator, self).__init__()
        self.fc1 = nn.Linear(feat_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim * 2)
        self.fc_out = nn.Linear(hid_dim * 2, feat_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.tanh(self.fc_out(x))
