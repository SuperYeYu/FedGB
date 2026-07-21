import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def _arg(args, name, default):
    return getattr(args, name, default)


class MaskedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, l1=0.001, args=None, one_init=True):
        super(MaskedLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.mask = nn.Parameter(torch.ones(out_dim, in_dim))
        self.l1 = l1
        self.args = args
        nn.init.xavier_uniform_(self.weight)
        if not one_init:
            nn.init.xavier_uniform_(self.mask)

    def _current_mask(self):
        mask = self.mask
        if _arg(self.args, "cufl_mask_noise", False) and self.training:
            mask = mask + torch.empty_like(mask).normal_(mean=0.0, std=self.l1)
        if _arg(self.args, "cufl_mask_drop", False):
            mask = F.dropout(mask, p=_arg(self.args, "cufl_mask_drop_ratio", 0.5), training=self.training)
        if not self.training:
            mask = mask.masked_fill(mask.abs() < self.l1, 0.0)
        return mask

    def forward(self, x):
        return F.linear(x, self.weight * self._current_mask(), self.bias)


class MaskedGCNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim, l1=0.001, args=None):
        super(MaskedGCNLayer, self).__init__(aggr="add")
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.mask = nn.Parameter(torch.ones(out_dim, in_dim))
        self.l1 = l1
        self.args = args
        nn.init.xavier_uniform_(self.weight)
        if not _arg(args, "cufl_mask_layer_one", True):
            nn.init.xavier_uniform_(self.mask)

    def _current_mask(self):
        mask = self.mask
        if _arg(self.args, "cufl_mask_noise", False) and self.training:
            mask = mask + torch.empty_like(mask).normal_(mean=0.0, std=self.l1)
        if _arg(self.args, "cufl_mask_drop", False):
            mask = F.dropout(mask, p=_arg(self.args, "cufl_mask_drop_ratio", 0.5), training=self.training)
        if not self.training:
            mask = mask.masked_fill(mask.abs() < self.l1, 0.0)
        return mask

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            x.size(0),
            improved=False,
            add_self_loops=True,
            dtype=x.dtype,
        )
        x = F.linear(x, self.weight * self._current_mask())
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return out + self.bias

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class CUFLMaskedGCN(nn.Module):
    """OpenFGL-compatible CUFL MaskedGCN following the official mask-gated GCN idea."""

    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.0, l1=0.001, args=None):
        super(CUFLMaskedGCN, self).__init__()
        self.args = args
        self.dropout = dropout
        self.use_dropout = _arg(args, "cufl_use_dropout", False)
        self.debug = _arg(args, "debug", False)
        self.l1 = l1

        self.convs = nn.ModuleDict()
        for layer_id in range(max(1, num_layers)):
            in_dim = input_dim if layer_id == 0 else hid_dim
            self.convs[str(layer_id)] = MaskedGCNLayer(in_dim, hid_dim, l1=l1, args=args)

        if _arg(args, "cufl_use_classifier_mask", False):
            self.classifier = MaskedLinear(
                hid_dim,
                output_dim,
                l1=l1,
                args=args,
                one_init=_arg(args, "cufl_mask_classifier_one", True),
            )
        else:
            self.classifier = nn.Linear(hid_dim, output_dim)

    def forward(self, data, get_feature=False):
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_attr", None)
        for layer in self.convs.values():
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            if self.use_dropout and not self.debug:
                x = F.dropout(x, p=self.dropout, training=self.training)
        embedding = x
        if get_feature:
            return embedding
        logits = self.classifier(embedding)
        return embedding, logits


def build_cufl_model(args, task):
    return CUFLMaskedGCN(
        input_dim=task.num_feats,
        hid_dim=args.hid_dim,
        output_dim=task.num_global_classes,
        num_layers=args.num_layers,
        dropout=getattr(args, "dropout", 0.0),
        l1=getattr(args, "cufl_l1", 0.001),
        args=args,
    )
