import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=2, dropout=0.5):
        super(MLPEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hid_dim))
        for _ in range(num_layers - 1):
            self.linears.append(nn.Linear(hid_dim, hid_dim))

    def forward(self, data):
        x = data.x
        for linear in self.linears:
            x = linear(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(MLP, self).__init__()
        self.encoder = MLPEncoder(input_dim, hid_dim, num_layers, dropout)
        self.head = nn.Linear(hid_dim, output_dim)

    def forward(self, data):
        x = self.encoder(data)
        logits = self.head(x)
        return x, logits
