import torch
import torch.nn as nn
import torch.nn.functional as F

from fedgb.algorithms.subgraph_fgl.homogeneous.s2fgl.utils import build_s2fgl_adjs


class S2FGLGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(S2FGLGraphConvolution, self).__init__()
        self.weight_low = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_high = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_mlp = nn.Parameter(torch.empty(in_features, out_features))
        self.att_vec_low = nn.Parameter(torch.empty(out_features, 1))
        self.att_vec_high = nn.Parameter(torch.empty(out_features, 1))
        self.att_vec_mlp = nn.Parameter(torch.empty(out_features, 1))
        self.att_vec_3 = nn.Parameter(torch.empty(3, 3))
        self.layer_norm_low = nn.LayerNorm(out_features)
        self.layer_norm_high = nn.LayerNorm(out_features)
        self.layer_norm_mlp = nn.LayerNorm(out_features)
        self.att_low = None
        self.att_high = None
        self.att_mlp = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_low)
        nn.init.xavier_uniform_(self.weight_high)
        nn.init.xavier_uniform_(self.weight_mlp)
        nn.init.xavier_uniform_(self.att_vec_low)
        nn.init.xavier_uniform_(self.att_vec_high)
        nn.init.xavier_uniform_(self.att_vec_mlp)
        nn.init.xavier_uniform_(self.att_vec_3)

    def _attention(self, output_low, output_high, output_mlp):
        low = self.layer_norm_low(output_low) @ self.att_vec_low
        high = self.layer_norm_high(output_high) @ self.att_vec_high
        mlp = self.layer_norm_mlp(output_mlp) @ self.att_vec_mlp
        logits = torch.sigmoid(torch.cat([low, high, mlp], dim=1)) @ self.att_vec_3 / 3.0
        attention = torch.softmax(logits, dim=1)
        return attention[:, 0:1], attention[:, 1:2], attention[:, 2:3]

    def _matmul_adj(self, adj, features):
        if getattr(adj, "is_sparse", False):
            return torch.sparse.mm(adj, features)
        return adj @ features

    def forward(self, x, adj_low, adj_high):
        low_input = x @ self.weight_low
        output_low = F.relu(self._matmul_adj(adj_low, low_input))

        high_input = x @ self.weight_high
        if adj_high is None:
            output_high = F.relu(high_input - self._matmul_adj(adj_low, high_input))
        else:
            output_high = F.relu(self._matmul_adj(adj_high, high_input))

        output_mlp = F.relu(x @ self.weight_mlp)
        self.att_low, self.att_high, self.att_mlp = self._attention(
            output_low,
            output_high,
            output_mlp,
        )
        return 3.0 * (
            self.att_low * output_low
            + self.att_high * output_high
            + self.att_mlp * output_mlp
        )


class S2FGLACM(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.1):
        super(S2FGLACM, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(S2FGLGraphConvolution(input_dim, hid_dim))
        for _ in range(max(1, num_layers) - 1):
            self.layers.append(S2FGLGraphConvolution(hid_dim, hid_dim))
        self.head = nn.Linear(hid_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, data):
        if not hasattr(data, "adj_low"):
            build_s2fgl_adjs(data)

        x = F.dropout(data.x, p=self.dropout, training=self.training)
        adj_high = getattr(data, "adj_high", None)
        for layer_id, layer in enumerate(self.layers):
            x = layer(x, data.adj_low, adj_high)
            if layer_id < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.head(x)
        return x, logits


def build_s2fgl_model(args, task):
    return S2FGLACM(
        input_dim=task.num_feats,
        hid_dim=args.hid_dim,
        output_dim=task.num_global_classes,
        num_layers=getattr(args, "num_layers", 2),
        dropout=getattr(args, "dropout", 0.1),
    )
