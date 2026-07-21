import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import add_self_loops

from fedgb.algorithms.graph_fgl.fedgmark.fedgmark_config import config


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.linear_or_not = num_layers == 1
        self.num_layers = num_layers
        if self.linear_or_not:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        h = x
        for layer in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[-1](h)


class FedGMarkBranchModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        output_dim,
        branch_id,
        num_layers=5,
        num_mlp_layers=2,
        num_branches=4,
        dropout=0.5,
        graph_pooling_type="sum",
        learn_eps=False,
        average_layer_logits=False,
        use_final_layer_prediction=False,
    ):
        super(FedGMarkBranchModel, self).__init__()
        self.branch_id = int(branch_id)
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.dropout = dropout
        self.graph_pooling_type = graph_pooling_type
        self.learn_eps = learn_eps
        self.average_layer_logits = average_layer_logits
        self.use_final_layer_prediction = use_final_layer_prediction
        self.eps = nn.Parameter(torch.zeros(num_layers))
        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.linears_prediction = nn.ModuleList()

        final_dim = max(1, int(hid_dim / (2 ** self.branch_id)))
        for layer in range(num_layers - 1):
            if layer == num_layers - 2:
                in_dim = input_dim if layer == 0 else hid_dim
                self.mlps.append(MLP(num_mlp_layers, in_dim, final_dim, final_dim))
                self.batch_norms.append(nn.BatchNorm1d(final_dim))
            elif layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hid_dim, hid_dim))
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hid_dim, hid_dim, hid_dim))
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            elif layer != num_layers - 1:
                self.linears_prediction.append(nn.Linear(hid_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(final_dim, output_dim))

    def forward(self, data):
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
        h = data.x.float()
        hidden_rep = [h]
        for layer in range(self.num_layers - 1):
            pooled = self._sum_neighbor_features(h, edge_index, data.num_nodes)
            if self.learn_eps:
                pooled = pooled + (1.0 + self.eps[layer]) * h
            h = F.relu(self.batch_norms[layer](self.mlps[layer](pooled)))
            hidden_rep.append(h)

        embedding = self._graph_pool(hidden_rep[-1], data.batch)
        if self.use_final_layer_prediction:
            logits = F.dropout(
                self.linears_prediction[-1](embedding),
                self.dropout,
                training=self.training,
            )
            return embedding, logits

        logits = 0
        for layer, h_layer in enumerate(hidden_rep):
            pooled_h = self._graph_pool(h_layer, data.batch)
            logits = logits + F.dropout(
                self.linears_prediction[layer](pooled_h),
                self.dropout,
                training=self.training,
            )
        if self.average_layer_logits:
            logits = logits / float(len(hidden_rep))
        return embedding, logits

    def _sum_neighbor_features(self, h, edge_index, num_nodes):
        row, col = edge_index
        pooled = h.new_zeros((num_nodes, h.shape[1]))
        pooled.index_add_(0, row, h[col])
        return pooled

    def _graph_pool(self, h, batch):
        if self.graph_pooling_type == "average":
            return global_mean_pool(h, batch)
        return global_add_pool(h, batch)


class FedGMarkGraphModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        output_dim,
        num_layers=5,
        num_mlp_layers=2,
        num_branches=4,
        dropout=0.5,
        graph_pooling_type="sum",
        learn_eps=False,
        average_branch_logits=False,
        use_final_layer_prediction=False,
    ):
        super(FedGMarkGraphModel, self).__init__()
        self.hid_dim = hid_dim
        self.num_branches = int(num_branches)
        self.average_branch_logits = average_branch_logits
        self.use_final_layer_prediction = use_final_layer_prediction
        self.branches = nn.ModuleList(
            [
                FedGMarkBranchModel(
                    input_dim,
                    hid_dim,
                    output_dim,
                    branch_id=branch_id,
                    num_layers=num_layers,
                    num_mlp_layers=num_mlp_layers,
                    num_branches=num_branches,
                    dropout=dropout,
                    graph_pooling_type=graph_pooling_type,
                    learn_eps=learn_eps,
                    average_layer_logits=average_branch_logits,
                    use_final_layer_prediction=use_final_layer_prediction,
                )
                for branch_id in range(self.num_branches)
            ]
        )

    def forward(self, data):
        logits = None
        embedding = None
        for branch in self.branches:
            branch_embedding, branch_logits = branch(data)
            if logits is None:
                logits = branch_logits
            else:
                logits = logits + branch_logits
            if embedding is None:
                embedding = branch_embedding.new_zeros((branch_embedding.shape[0], self.hid_dim))
            keep_dim = min(embedding.shape[1], branch_embedding.shape[1])
            embedding[:, :keep_dim] = embedding[:, :keep_dim] + branch_embedding[:, :keep_dim]
        if self.average_branch_logits and self.num_branches > 0:
            logits = logits / float(self.num_branches)
        return embedding, logits

    def forward_branch(self, data, branch_id):
        _, logits = self.branches[int(branch_id)](data)
        return logits


def global_to_sub(global_model, sub_models):
    for branch_id, sub_model in enumerate(sub_models):
        sub_model.load_state_dict(global_model.branches[branch_id].state_dict())


def sub_to_global(global_model, sub_models):
    for branch_id, sub_model in enumerate(sub_models):
        global_model.branches[branch_id].load_state_dict(sub_model.state_dict())


def build_fedgmark_model(args, task):
    return FedGMarkGraphModel(
        input_dim=task.num_feats,
        hid_dim=getattr(args, "fedgmark_hidden_dim", getattr(args, "hid_dim", config["hidden_dim"])),
        output_dim=task.num_targets if getattr(args, "task", None) == "graph_reg" else task.num_global_classes,
        num_layers=getattr(args, "fedgmark_num_layers", config["num_layers"]),
        num_mlp_layers=getattr(args, "fedgmark_num_mlp_layers", config["num_mlp_layers"]),
        num_branches=getattr(args, "fedgmark_num_branches", config["num_branches"]),
        dropout=getattr(args, "fedgmark_dropout", config["dropout"]),
        graph_pooling_type=getattr(args, "fedgmark_graph_pooling_type", config["graph_pooling_type"]),
        learn_eps=getattr(args, "fedgmark_learn_eps", config["learn_eps"]),
        average_branch_logits=getattr(args, "task", None) == "graph_reg",
        use_final_layer_prediction=getattr(args, "task", None) == "graph_reg",
    )
