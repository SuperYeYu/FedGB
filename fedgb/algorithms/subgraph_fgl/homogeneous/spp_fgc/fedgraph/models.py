import torch
import torch.nn as nn

from fedgb.algorithms.subgraph_fgl.homogeneous.spp_fgc.utils import (
    cal_weights_via_can,
    graph_diff_loss,
)


class DecoderLinear(nn.Module):
    def __init__(self, input_dim):
        super(DecoderLinear, self).__init__()
        self.decoder = nn.Linear(input_dim, input_dim)
        nn.init.eye_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        return self.decoder(x)


class SPPFGCLocalModel(nn.Module):
    """Official SPP-FGC local structure model adapted to OpenFGL tensors."""

    def __init__(self, input_dim, num_neighbors, device):
        super(SPPFGCLocalModel, self).__init__()
        self.num_neighbors = num_neighbors
        self.device = device
        self.encoder = DecoderLinear(input_dim).to(device)

    def forward(self, features):
        features = features.to(self.device)
        embedding = self.encoder(features)
        graph_a = cal_weights_via_can(embedding, self.num_neighbors)
        return embedding, graph_a

    def train_to_global_s(self, features, global_s, num_epochs, lr, weight_decay):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        features = features.to(self.device)
        global_s = global_s.to(self.device)
        final_loss = None
        for _ in range(int(num_epochs)):
            self.train()
            optimizer.zero_grad()
            _, graph_a = self.forward(features)
            loss = graph_diff_loss(graph_a, global_s)
            loss.backward()
            optimizer.step()
            final_loss = loss
        return float(final_loss.detach().cpu()) if final_loss is not None else 0.0
