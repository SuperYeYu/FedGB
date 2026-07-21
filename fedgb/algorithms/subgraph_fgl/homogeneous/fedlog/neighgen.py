import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph


class NeighGen(nn.Module):
    def __init__(self, feat_dim, hid_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, feat_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.tanh(self.fc_out(x))


def pretrain_neighgen(local_data, neigh_gen_model, pre_gen_epochs, device):
    x = local_data.x.to(device)
    edge_index = local_data.edge_index.to(device)
    N = x.size(0)
    gen_opt = torch.optim.Adam(neigh_gen_model.parameters(), lr=0.01)
    mse = nn.MSELoss()

    true_neigh_feats = torch.zeros(N, x.size(1)).to(device)
    for i in range(N):
        neighbors = k_hop_subgraph(i, 1, edge_index)[0]
        neighbors = neighbors[neighbors != i]
        if len(neighbors) > 0:
            true_neigh_feats[i] = x[neighbors].mean(dim=0)

    for _ in range(pre_gen_epochs):
        neigh_gen_model.train()
        gen_opt.zero_grad()
        pred = neigh_gen_model(x)
        loss = mse(pred, true_neigh_feats)
        loss.backward()
        gen_opt.step()
