import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PyGFedLITModel(nn.Module):
    def __init__(self, nlinktype, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super(PyGFedLITModel, self).__init__()
        self.nlinktype = int(nlinktype)
        self.hid_dim = int(hid_dim)
        self.dropout = dropout
        self.feature1 = nn.Linear(input_dim, hid_dim)
        self.split_gnns = nn.ModuleList()
        for _ in range(self.nlinktype):
            feature2 = nn.Linear(hid_dim, hid_dim)
            convs = nn.ModuleList()
            for _ in range(num_layers):
                convs.append(GCNConv(hid_dim, hid_dim))
            self.split_gnns.append(nn.ModuleList([feature2, convs]))
        self.classifier = nn.Linear(hid_dim, output_dim)

    def feature_projection(self, x):
        return self.feature1(x)

    def split_forward_subgraph(self, idx_subgraph, subgraph):
        feature2, convs = self.split_gnns[idx_subgraph]
        x = feature2(subgraph.x)
        for conv in convs:
            x = conv(x, subgraph.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def split_forward(self, subgraphs, num_nodes, device):
        out = torch.zeros((num_nodes, self.hid_dim), device=device)
        for idx_subgraph, subgraph in enumerate(subgraphs):
            if subgraph.x.size(0) == 0:
                continue
            x = self.split_forward_subgraph(idx_subgraph, subgraph)
            out[subgraph.orig_node_ids.to(device)] += x
        return out

    def classify(self, x):
        return self.classifier(x)

    def forward(self, data):
        projected = self.feature_projection(data.x)
        if not hasattr(data, "fedlit_clusters"):
            logits = self.classifier(projected)
            return projected, logits
        subgraphs = getattr(data, "fedlit_subgraphs", None)
        if subgraphs is None:
            raise RuntimeError("FedLIT data has clusters but no fedlit_subgraphs.")
        branch_embedding = self.split_forward(subgraphs, data.x.size(0), data.x.device)
        logits = self.classify(branch_embedding)
        return branch_embedding, logits
