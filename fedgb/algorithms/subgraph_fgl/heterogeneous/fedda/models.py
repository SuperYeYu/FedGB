import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, softmax


def infer_fedda_metadata(data):
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
    return torch.zeros(length, dtype=torch.long, device=device)


class FedDAEdgeGATLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        edge_dim,
        num_node_types,
        num_edge_types,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        activation=None,
        residual=False,
        residual_attention_alpha=0.0,
    ):
        super(FedDAEdgeGATLayer, self).__init__()
        self.out_dim = int(out_dim)
        self.edge_dim = int(edge_dim)
        self.num_heads = int(num_heads)
        self.num_node_types = int(num_node_types)
        self.num_edge_types = int(num_edge_types)
        self.activation = activation
        self.residual_attention_alpha = float(residual_attention_alpha)

        self.lin = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.edge_lin = nn.Linear(edge_dim, out_dim * num_heads, bias=False)
        self.edge_emb = nn.ParameterList(
            [nn.Parameter(torch.empty(1, edge_dim)) for _ in range(self.num_edge_types)]
        )
        self.attn_l = nn.ParameterList(
            [nn.Parameter(torch.empty(1, num_heads, out_dim)) for _ in range(self.num_node_types)]
        )
        self.attn_r = nn.ParameterList(
            [nn.Parameter(torch.empty(1, num_heads, out_dim)) for _ in range(self.num_node_types)]
        )
        self.attn_e = nn.ParameterList(
            [nn.Parameter(torch.empty(1, num_heads, out_dim)) for _ in range(self.num_edge_types)]
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        else:
            self.res_fc = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.lin.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_lin.weight, gain=gain)
        if self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        for param in self.edge_emb:
            nn.init.xavier_normal_(param, gain=gain)
        for param in self.attn_l:
            nn.init.xavier_normal_(param, gain=gain)
        for param in self.attn_r:
            nn.init.xavier_normal_(param, gain=gain)
        for param in self.attn_e:
            nn.init.xavier_normal_(param, gain=gain)

    def forward(self, x, edge_index, edge_type, node_type, res_attn=None):
        num_nodes = x.size(0)
        x_src = self.feat_drop(x)
        h = self.lin(x_src).view(num_nodes, self.num_heads, self.out_dim)

        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            edge_type = torch.empty(0, dtype=torch.long, device=x.device)
        edge_index, edge_type = add_remaining_self_loops(
            edge_index,
            edge_type,
            fill_value=0,
            num_nodes=num_nodes,
        )
        edge_type = edge_type.to(device=x.device, dtype=torch.long).view(-1).clamp_(0, self.num_edge_types - 1)
        src, dst = edge_index

        edge_emb_table = torch.cat([param for param in self.edge_emb], dim=0)
        edge_hidden = self.edge_lin(edge_emb_table[edge_type]).view(-1, self.num_heads, self.out_dim)

        node_type = node_type.to(device=x.device, dtype=torch.long).view(-1).clamp_(0, self.num_node_types - 1)
        attn_l_table = torch.cat([param for param in self.attn_l], dim=0)
        attn_r_table = torch.cat([param for param in self.attn_r], dim=0)
        attn_e_table = torch.cat([param for param in self.attn_e], dim=0)
        scores = (
            h[src] * attn_l_table[node_type[src]]
            + h[dst] * attn_r_table[node_type[dst]]
            + edge_hidden * attn_e_table[edge_type]
        ).sum(dim=-1)
        scores = self.leaky_relu(scores)
        alpha = softmax(scores, dst, num_nodes=num_nodes)
        alpha = self.attn_drop(alpha)
        if res_attn is not None and res_attn.shape == alpha.shape:
            mix = self.residual_attention_alpha
            alpha = alpha * (1.0 - mix) + res_attn * mix

        message = (h[src] + edge_hidden) * alpha.unsqueeze(-1)
        out = h.new_zeros(num_nodes, self.num_heads, self.out_dim)
        out.index_add_(0, dst, message)
        if self.res_fc is not None:
            out = out + self.res_fc(x).view(num_nodes, self.num_heads, self.out_dim)
        if self.activation is not None:
            out = self.activation(out)
        return out, alpha.detach()


class PyGFedDAModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        output_dim,
        num_layers=2,
        num_heads=3,
        num_node_types=1,
        num_edge_types=1,
        edge_dim=32,
        dropout=0.5,
        attn_dropout=None,
        negative_slope=0.2,
        residual=False,
        residual_attention_alpha=0.0,
    ):
        super(PyGFedDAModel, self).__init__()
        self.hid_dim = int(hid_dim)
        self.num_layers = max(int(num_layers), 1)
        self.num_heads = int(num_heads)
        self.num_node_types = max(int(num_node_types), 1)
        self.num_edge_types = max(int(num_edge_types), 1)
        self.dropout = float(dropout)
        self.embedding_dim = self.hid_dim * (self.num_layers + 2)

        self.fc_list = nn.ModuleList(
            [nn.Linear(input_dim, self.hid_dim, bias=True) for _ in range(self.num_node_types)]
        )
        self.gat_layers = nn.ModuleList()
        attn_dropout = self.dropout if attn_dropout is None else attn_dropout
        for layer_idx in range(self.num_layers):
            in_dim = self.hid_dim if layer_idx == 0 else self.hid_dim * self.num_heads
            self.gat_layers.append(
                FedDAEdgeGATLayer(
                    in_dim=in_dim,
                    out_dim=self.hid_dim,
                    edge_dim=edge_dim,
                    num_node_types=self.num_node_types,
                    num_edge_types=self.num_edge_types,
                    num_heads=self.num_heads,
                    feat_drop=self.dropout,
                    attn_drop=attn_dropout,
                    negative_slope=negative_slope,
                    activation=F.elu,
                    residual=residual and layer_idx > 0,
                    residual_attention_alpha=residual_attention_alpha,
                )
            )
        self.gat_layers.append(
            FedDAEdgeGATLayer(
                in_dim=self.hid_dim * self.num_heads,
                out_dim=self.hid_dim,
                edge_dim=edge_dim,
                num_node_types=self.num_node_types,
                num_edge_types=self.num_edge_types,
                num_heads=self.num_heads,
                feat_drop=self.dropout,
                attn_drop=attn_dropout,
                negative_slope=negative_slope,
                activation=None,
                residual=residual,
                residual_attention_alpha=residual_attention_alpha,
            )
        )
        self.decoder = nn.Linear(self.embedding_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=gain)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)
        nn.init.xavier_normal_(self.decoder.weight, gain=gain)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def forward(self, data):
        x = data.x
        num_nodes = x.size(0)
        node_type = _safe_type_tensor(data, "node_type", num_nodes, self.num_node_types, x.device)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        else:
            edge_index = edge_index.to(x.device)
        edge_type = _safe_type_tensor(data, "edge_type", edge_index.size(1), self.num_edge_types, x.device)

        h = self._typed_input_projection(x, node_type)
        embeddings = [self._l2_norm(h)]
        res_attn = None
        for layer in self.gat_layers[:-1]:
            layer_out, res_attn = layer(h, edge_index, edge_type, node_type, res_attn=res_attn)
            h_mean = layer_out.mean(dim=1)
            embeddings.append(self._l2_norm(h_mean))
            h = layer_out.flatten(1)

        out, _ = self.gat_layers[-1](h, edge_index, edge_type, node_type, res_attn=res_attn)
        final_embedding = self._l2_norm(out.mean(dim=1))
        embeddings.append(final_embedding)
        embedding = torch.cat(embeddings, dim=1)
        logits = self.decoder(F.dropout(embedding, p=self.dropout, training=self.training))
        return embedding, logits

    def _typed_input_projection(self, x, node_type):
        h = x.new_zeros(x.size(0), self.hid_dim)
        for type_id, fc in enumerate(self.fc_list):
            mask = node_type == type_id
            if mask.any():
                h[mask] = fc(x[mask])
        return h

    def _l2_norm(self, x):
        return x / torch.clamp(torch.norm(x, dim=1, keepdim=True), min=1e-12)
