import torch
import torch.nn as nn
import torch.nn.functional as F


class OptGDBAGenerator(nn.Module):
    def __init__(self, max_nodes, feat_dim, layernum, trigger_size, dropout=0.05):
        super(OptGDBAGenerator, self).__init__()
        self.max_nodes = int(max_nodes)
        self.feat_dim = int(feat_dim)
        self.trigger_size = int(trigger_size)
        self.topology_net = _make_mlp(self.max_nodes, self.max_nodes, layernum, dropout)
        self.feature_net = _make_mlp(self.feat_dim, self.feat_dim, layernum, dropout)
        self.view_topology_net = _make_mlp(self.max_nodes, self.max_nodes, layernum, dropout)
        self.view_feature_net = _make_mlp(self.feat_dim, self.feat_dim, layernum, dropout)
        self.client_topology = nn.Linear(1, self.max_nodes * self.max_nodes)
        self.client_feature = nn.Linear(1, self.max_nodes * self.feat_dim)

    def forward(self, a_input, x_input, client_id, num_nodes):
        device = a_input.device
        num_nodes = min(int(num_nodes), self.max_nodes)

        view_a = torch.sigmoid(self.view_topology_net(a_input))
        view_x = F.relu(self.view_feature_net(x_input))
        node_scores = (view_a.mean(dim=1) * view_x.mean(dim=1))[:num_nodes]

        client_tensor = torch.as_tensor([[float(client_id)]], dtype=a_input.dtype, device=device)
        topo_scale = self.client_topology(client_tensor).view(self.max_nodes, self.max_nodes)
        feat_scale = self.client_feature(client_tensor).view(self.max_nodes, self.feat_dim)

        topo_logits = torch.sigmoid(self.topology_net(a_input * topo_scale))
        topo_logits = 0.5 * (topo_logits + topo_logits.t())
        feat_logits = F.relu(self.feature_net(x_input * feat_scale))
        return topo_logits, feat_logits, node_scores


def _make_mlp(input_dim, output_dim, layernum, dropout):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    for _ in range(max(int(layernum) - 1, 0)):
        layers.append(nn.Linear(input_dim, input_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)
