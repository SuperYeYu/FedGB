import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_add_pool, global_mean_pool

from fedgb.algorithms.graph_fgl.fedvn.fedvn_config import config


class MLPGIN(nn.Module):
    def __init__(self, emb_dim):
        super(MLPGIN, self).__init__()
        self.in_features = emb_dim
        self.l1 = nn.Linear(emb_dim, 2 * emb_dim)
        self.bn = nn.BatchNorm1d(2 * emb_dim, track_running_stats=False)
        self.l2 = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, x):
        return self.l2(F.relu(self.bn(self.l1(x))))


class MLPVirtual(nn.Module):
    def __init__(self, emb_dim, num_vn):
        super(MLPVirtual, self).__init__()
        self.l1 = nn.Linear(emb_dim, 2 * emb_dim)
        self.l2 = nn.Linear(2 * emb_dim, emb_dim)
        self.bn1 = nn.BatchNorm1d(2 * num_vn * emb_dim, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(num_vn * emb_dim, track_running_stats=False)

    def forward(self, x):
        batch_size, num_vn, emb_dim = x.shape
        x = self.l1(x)
        x_flat = x.reshape(batch_size, -1)
        if batch_size == 1:
            if self.bn1.affine:
                x_flat = x_flat * self.bn1.weight + self.bn1.bias
        else:
            x_flat = self.bn1(x_flat)
        x = x_flat.reshape(batch_size, num_vn, 2 * emb_dim)
        x = F.relu(x)
        x = self.l2(x)
        x_flat = x.reshape(batch_size, -1)
        if batch_size == 1:
            if self.bn2.affine:
                x_flat = x_flat * self.bn2.weight + self.bn2.bias
        else:
            x_flat = self.bn2(x_flat)
        x = x_flat.reshape(batch_size, num_vn, emb_dim)
        return F.relu(x)


def _is_gine(model_name):
    if isinstance(model_name, (list, tuple)):
        model_name = model_name[0] if model_name else "gin"
    return str(model_name).lower() == "gine"


def _edge_attr(data, enabled):
    if not enabled:
        return None
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        raise ValueError("FedVN GINE mode requires data.edge_attr, but the graph batch has no edge_attr.")
    return edge_attr.to(dtype=data.x.dtype, device=data.x.device)


def _make_gin_conv(emb_dim, model_name="gin", edge_dim=None):
    if _is_gine(model_name):
        return GINEConv(MLPGIN(emb_dim), train_eps=False, edge_dim=edge_dim)
    return GINConv(MLPGIN(emb_dim), train_eps=False)


def _run_conv(conv, x, edge_index, edge_attr):
    if isinstance(conv, GINEConv):
        return conv(x, edge_index, edge_attr)
    return conv(x, edge_index)


class FedVNScoringGIN(nn.Module):
    def __init__(self, input_dim, hid_dim, num_vn, num_layers=3, dropout=0.5, model_name="gin", edge_dim=None):
        super(FedVNScoringGIN, self).__init__()
        self.dropout = dropout
        self.use_edge_attr = _is_gine(model_name)
        self.input_proj = nn.Linear(input_dim, hid_dim)
        self.convs = nn.ModuleList([_make_gin_conv(hid_dim, model_name, edge_dim) for _ in range(max(int(num_layers), 1))])
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hid_dim, track_running_stats=False) for _ in range(max(int(num_layers), 1))]
        )
        self.l1 = nn.Linear(hid_dim, hid_dim)
        self.l2 = nn.Linear(hid_dim, num_vn)

    def forward(self, data):
        edge_attr = _edge_attr(data, self.use_edge_attr)
        h = self.input_proj(data.x.float())
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            h = _run_conv(conv, h, data.edge_index, edge_attr)
            h = F.relu(batch_norm(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        score = self.l2(F.dropout(F.relu(self.l1(h)), p=self.dropout, training=self.training))
        return score, h


class FedVNGINEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_vn, num_layers=3, dropout=0.5, model_name="gin", edge_dim=None):
        super(FedVNGINEncoder, self).__init__()
        self.dropout = dropout
        self.use_edge_attr = _is_gine(model_name)
        self.input_proj = nn.Linear(input_dim, hid_dim)
        self.convs = nn.ModuleList([_make_gin_conv(hid_dim, model_name, edge_dim) for _ in range(max(int(num_layers), 1))])
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hid_dim, track_running_stats=False) for _ in range(max(int(num_layers), 1))]
        )
        self.virtual_mlps = nn.ModuleList([MLPVirtual(hid_dim, num_vn) for _ in range(max(int(num_layers) - 1, 0))])

    def forward(self, data, vn_embedding, score):
        edge_attr = _edge_attr(data, self.use_edge_attr)
        batch = data.batch
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        vn = vn_embedding.expand(batch_size, -1, -1)
        h = self.input_proj(data.x.float())

        h, vn = self._inject_virtual_nodes(h, batch, vn, score)
        for layer, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = _run_conv(conv, h, data.edge_index, edge_attr)
            h = batch_norm(h)
            if layer < len(self.convs) - 1:
                h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if layer < len(self.virtual_mlps):
                vn = F.dropout(self.virtual_mlps[layer](vn), p=self.dropout, training=self.training)
                h, vn = self._inject_virtual_nodes(h, batch, vn, score)
        return h

    def _inject_virtual_nodes(self, h, batch, vn, score):
        vn_neigh = score.unsqueeze(2) * h.unsqueeze(1)
        vn_temp = global_add_pool(vn_neigh.transpose(0, 1), batch).transpose(0, 1) + vn
        h = h + (score[:, None] @ vn[batch]).squeeze(1)
        return h, vn_temp


class Classifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(Classifier, self).__init__()
        self.dropout = dropout
        self.l1 = nn.Linear(in_channels, hidden_channels)
        self.l2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, h):
        h = F.relu(self.l1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.l2(h)


class FedVNGIN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_vn, num_layers=3, dropout=0.5, model_name="gin", edge_dim=None):
        super(FedVNGIN, self).__init__()
        self.hid_dim = hid_dim
        self.num_vn = num_vn
        self.encoder = FedVNGINEncoder(input_dim, hid_dim, num_vn, num_layers, dropout, model_name, edge_dim)
        self.classifier = Classifier(hid_dim, hid_dim, output_dim, dropout)
        self._context_vn_embedding = None
        self._context_scoring_model = None

    def set_virtual_node_context(self, vn_embedding, scoring_model=None):
        self._parameters.pop("_context_vn_embedding", None)
        self._modules.pop("_context_scoring_model", None)
        object.__setattr__(self, "_context_vn_embedding", vn_embedding)
        object.__setattr__(self, "_context_scoring_model", scoring_model)

    def forward(self, data, vn_embedding=None, score=None):
        if vn_embedding is None:
            vn_embedding = self._context_vn_embedding
        if score is None and self._context_scoring_model is not None:
            score, _ = self._context_scoring_model(data)
            score = torch.sigmoid(score)
        if vn_embedding is None:
            vn_embedding = data.x.new_zeros(self.num_vn, self.hid_dim)
        if score is None:
            score = data.x.new_full((data.num_nodes, self.num_vn), 1.0 / self.num_vn)
        h = self.encoder(data, vn_embedding, score)
        embedding = global_mean_pool(h, data.batch)
        logits = self.classifier(embedding)
        return embedding, logits


def build_fedvn_model(args, task):
    return FedVNGIN(
        input_dim=task.num_feats,
        hid_dim=getattr(args, "fedvn_hidden_dim", getattr(args, "hid_dim", config["fedvn_hidden_dim"])),
        output_dim=task.num_targets if getattr(args, "task", None) == "graph_reg" else task.num_global_classes,
        num_vn=getattr(args, "fedvn_num_vn", config["fedvn_num_vn"]),
        num_layers=getattr(args, "fedvn_num_layers", getattr(args, "num_layers", config["fedvn_num_layers"])),
        dropout=getattr(args, "fedvn_dropout", getattr(args, "dropout", config["fedvn_dropout"])),
        model_name=getattr(args, "model", ["gin"]),
        edge_dim=getattr(args, "edge_dim", None),
    )


def build_fedvn_scoring_model(args, task):
    return FedVNScoringGIN(
        input_dim=task.num_feats,
        hid_dim=getattr(args, "fedvn_hidden_eg", config["fedvn_hidden_eg"]),
        num_vn=getattr(args, "fedvn_num_vn", config["fedvn_num_vn"]),
        num_layers=getattr(args, "fedvn_num_layers", getattr(args, "num_layers", config["fedvn_num_layers"])),
        dropout=getattr(args, "fedvn_dropout", getattr(args, "dropout", config["fedvn_dropout"])),
        model_name=getattr(args, "model", ["gin"]),
        edge_dim=getattr(args, "edge_dim", None),
    )
