import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_fedhgn_metadata(data):
    num_node_types = _infer_num_types(data, "node_type", "hetero_node_types")
    num_edge_types = _infer_num_types(data, "edge_type", "hetero_edge_types")
    return max(num_node_types, 1), max(num_edge_types, 1)


def _infer_num_types(data, tensor_attr, names_attr):
    names = getattr(data, names_attr, None)
    if names is not None:
        try:
            return len(names)
        except TypeError:
            pass
    values = getattr(data, tensor_attr, None)
    if torch.is_tensor(values) and values.numel() > 0:
        return int(values.max().item()) + 1
    return 1


def _safe_type_tensor(data, attr, length, max_types, device):
    values = getattr(data, attr, None)
    if torch.is_tensor(values) and values.numel() == length:
        values = values.to(device=device, dtype=torch.long).view(-1)
        return values.clamp_(0, max_types - 1)
    if torch.is_tensor(values) and values.numel() > 0:
        values = values.to(device=device, dtype=torch.long).view(-1)
        index = torch.arange(length, device=device) % values.numel()
        return values[index].clamp_(0, max_types - 1)
    return torch.zeros(length, dtype=torch.long, device=device)


class PyGRGCNLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_edge_types,
        num_bases,
        use_weight=True,
        use_bias=True,
        activation=None,
        dropout=0.0,
        use_self_loop=False,
    ):
        super(PyGRGCNLayer, self).__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_edge_types = int(num_edge_types)
        self.num_bases = int(num_bases)
        self.use_weight = bool(use_weight)
        self.use_bias = bool(use_bias)
        self.activation = activation
        self.use_self_loop = bool(use_self_loop)

        if self.use_weight:
            self.bases = nn.Parameter(torch.empty(self.num_bases, self.in_dim, self.out_dim))
        if self.use_bias:
            self.h_bias = nn.Parameter(torch.zeros(self.out_dim))
        if self.use_self_loop:
            self.loop_weight = nn.Parameter(torch.empty(self.in_dim, self.out_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.use_weight:
            nn.init.xavier_uniform_(self.bases, gain=gain)
        if self.use_self_loop:
            nn.init.xavier_uniform_(self.loop_weight, gain=gain)
        if self.use_bias:
            nn.init.zeros_(self.h_bias)

    def forward(self, x, edge_index, edge_type, basis_coeffs=None):
        if edge_index.numel() == 0:
            out = x.new_zeros(x.size(0), self.out_dim)
        else:
            src, dst = edge_index
            out = x.new_zeros(x.size(0), self.out_dim)
            for rel_id in range(self.num_edge_types):
                mask = edge_type == rel_id
                if not mask.any():
                    continue
                rel_src = src[mask]
                rel_dst = dst[mask]
                if self.use_weight:
                    coeff = basis_coeffs[f"rel_{rel_id}"]
                    weight = torch.matmul(coeff, self.bases.view(self.num_bases, -1)).view(
                        self.in_dim,
                        self.out_dim,
                    )
                    msg = x[rel_src].matmul(weight)
                else:
                    msg = x[rel_src]
                deg = torch.bincount(rel_dst, minlength=x.size(0)).to(dtype=msg.dtype, device=msg.device)
                msg = msg / deg[rel_dst].clamp(min=1.0).unsqueeze(-1)
                out.index_add_(0, rel_dst, msg)

        if self.use_self_loop:
            out = out + x.matmul(self.loop_weight)
        if self.use_bias:
            out = out + self.h_bias
        if self.activation is not None:
            out = self.activation(out)
        return self.dropout(out)


class PyGFedHGNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_node_types,
        num_edge_types,
        num_bases=20,
        num_layers=2,
        dropout=0.0,
        max_nodes=1,
        use_self_loop=False,
    ):
        super(PyGFedHGNModel, self).__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_node_types = max(int(num_node_types), 1)
        self.num_edge_types = max(int(num_edge_types), 1)
        self.num_bases = max(int(num_bases), 1)
        self.num_layers = max(int(num_layers), 2)
        self.max_nodes = max(int(max_nodes), 1)

        self.feature_proj = nn.ModuleList(
            [nn.Linear(self.input_dim, self.hidden_dim) for _ in range(self.num_node_types)]
        )
        self.layers = nn.ModuleList()
        self.layers.append(
            PyGRGCNLayer(
                self.hidden_dim,
                self.hidden_dim,
                self.num_edge_types,
                self.num_bases,
                use_weight=False,
                activation=F.relu,
                dropout=dropout,
                use_self_loop=use_self_loop,
            )
        )
        self.basis_coeffs_encoder = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.layers.append(
                PyGRGCNLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.num_edge_types,
                    self.num_bases,
                    use_weight=True,
                    activation=F.relu,
                    dropout=dropout,
                    use_self_loop=use_self_loop,
                )
            )
            coeffs = nn.ParameterDict()
            for rel_id in range(self.num_edge_types):
                coeffs[f"rel_{rel_id}"] = nn.Parameter(torch.empty(self.num_bases))
            self.basis_coeffs_encoder.append(coeffs)
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for proj in self.feature_proj:
            nn.init.xavier_uniform_(proj.weight, gain=gain)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        for coeffs in self.basis_coeffs_encoder:
            for coeff in coeffs.values():
                nn.init.xavier_uniform_(coeff.view(1, -1), gain=gain)
        nn.init.xavier_uniform_(self.decoder.weight, gain=gain)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def forward(self, data):
        device = data.x.device
        num_nodes = data.x.size(0)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device=device, dtype=torch.long)
        node_type = _safe_type_tensor(data, "node_type", num_nodes, self.num_node_types, device)
        edge_type = _safe_type_tensor(data, "edge_type", edge_index.size(1), self.num_edge_types, device)

        h = self._typed_feature_projection(data.x, node_type)
        h = self.layers[0](h, edge_index, edge_type, None)
        for layer, coeffs in zip(self.layers[1:], self.basis_coeffs_encoder):
            h = layer(h, edge_index, edge_type, coeffs)
        logits = self.decoder(F.relu(h))
        return h, logits

    def _typed_feature_projection(self, x, node_type):
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        for type_id, proj in enumerate(self.feature_proj):
            mask = node_type == type_id
            if not mask.any():
                continue
            h[mask] = proj(x[mask])
        return h
