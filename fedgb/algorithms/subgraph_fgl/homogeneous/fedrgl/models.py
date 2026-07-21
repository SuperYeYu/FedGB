import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from fedgb.algorithms.subgraph_fgl.homogeneous.fedrgl.fedrgl_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedrgl.utils import drop_edge, drop_feature


class FedRGLGCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        output_dim,
        dropout,
        drop_feature_rate_1,
        drop_feature_rate_2,
        drop_edge_rate_1,
        drop_edge_rate_2,
    ):
        super(FedRGLGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, output_dim)
        self.projection = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )
        self.dropout = dropout
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2

    def _encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return F.dropout(x, p=self.dropout, training=self.training)

    def forward(self, data):
        embedding = self._encode(data.x, data.edge_index)
        logits = self.conv2(embedding, data.edge_index)
        return embedding, logits

    def forward_logits(self, data):
        _, logits = self.forward(data)
        return logits

    def forward_full(self, x, edge_index):
        embedding = self._encode(x, edge_index)
        return self.conv2(embedding, edge_index)

    def rep_forward(self, data):
        embedding = self._encode(data.x, data.edge_index)
        return self.projection(embedding)

    def forward_two_views(self, data):
        edge_index_1 = drop_edge(data.edge_index, self.drop_edge_rate_1)
        edge_index_2 = drop_edge(data.edge_index, self.drop_edge_rate_2)
        view1 = Data(
            x=drop_feature(data.x, self.drop_feature_rate_1),
            edge_index=edge_index_1,
        ).to(data.x.device)
        view2 = Data(
            x=drop_feature(data.x, self.drop_feature_rate_2),
            edge_index=edge_index_2,
        ).to(data.x.device)

        emb1 = self._encode(view1.x, view1.edge_index)
        emb2 = self._encode(view2.x, view2.edge_index)
        z1 = self.projection(emb1)
        z2 = self.projection(emb2)
        logits1 = self.conv2(emb1, view1.edge_index)
        logits2 = self.conv2(emb2, view2.edge_index)
        return z1, z2, logits1, logits2


def build_fedrgl_model(args, task):
    return FedRGLGCN(
        input_dim=task.num_feats,
        hid_dim=getattr(args, "fedrgl_hid_dim", args.hid_dim),
        output_dim=task.num_global_classes,
        dropout=getattr(args, "fedrgl_dropout", args.dropout),
        drop_feature_rate_1=getattr(args, "fedrgl_drop_feature_rate_1", config["fedrgl_drop_feature_rate_1"]),
        drop_feature_rate_2=getattr(args, "fedrgl_drop_feature_rate_2", config["fedrgl_drop_feature_rate_2"]),
        drop_edge_rate_1=getattr(args, "fedrgl_drop_edge_rate_1", config["fedrgl_drop_edge_rate_1"]),
        drop_edge_rate_2=getattr(args, "fedrgl_drop_edge_rate_2", config["fedrgl_drop_edge_rate_2"]),
    )
