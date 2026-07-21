import torch
import torch.nn as nn


class GraphTrojanNet(nn.Module):
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.0):
        super(GraphTrojanNet, self).__init__()
        self.nfeat = int(nfeat)
        self.nout = max(int(nout), 3)

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for _ in range(max(int(layernum) - 1, 0)):
            layers.append(nn.Linear(self.nfeat, self.nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers).to(device)
        self.feat = nn.Linear(self.nfeat, self.nfeat)
        self.edge_weights = nn.Linear(self.nfeat, self.nout)
        self.device = device

    def forward(self, input):
        h = self.layers(input)
        feat = self.feat(h)
        edge_weight = self.edge_weights(h)
        return feat, edge_weight
