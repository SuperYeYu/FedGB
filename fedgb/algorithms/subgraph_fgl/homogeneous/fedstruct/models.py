import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from pathlib import Path
from torch_geometric.nn import GCNConv, SAGEConv

from fedgb.algorithms.subgraph_fgl.homogeneous.fedstruct.utils import (
    build_laplacian_eigenbasis,
    spectral_sfv_regularizer,
)


class _GraphEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers, dropout, conv_type="sage"):
        super(_GraphEncoder, self).__init__()
        self.dropout = dropout
        conv_cls = SAGEConv if conv_type == "sage" else GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(conv_cls(input_dim, hid_dim))
        for _ in range(max(1, num_layers) - 1):
            self.convs.append(conv_cls(hid_dim, hid_dim))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class _MLPEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(_MLPEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class FedStructModel(nn.Module):
    """Feature+structure classifier following FedStruct's SFV training path."""

    def __init__(
        self,
        input_dim,
        hid_dim,
        output_dim,
        num_layers,
        dropout,
        data,
        spectral_len,
        structure_dim,
        conv_type="sage",
        sfv_basis=None,
        sfv_eigvals=None,
        sfv_coefficients=None,
    ):
        super(FedStructModel, self).__init__()
        device = data.x.device
        self.feature_encoder = _GraphEncoder(input_dim, hid_dim, num_layers, dropout, conv_type=conv_type)
        self.structure_encoder = _MLPEncoder(structure_dim, hid_dim, dropout)
        self.head = nn.Linear(hid_dim, output_dim)
        self.structure_head = nn.Linear(hid_dim, output_dim)

        if sfv_basis is None or sfv_eigvals is None:
            basis, eigvals = build_laplacian_eigenbasis(
                data.edge_index,
                data.num_nodes,
                k=spectral_len,
                device=device,
            )
        else:
            basis = sfv_basis.to(device).float()
            eigvals = sfv_eigvals.to(device).float()
        self.register_buffer("sfv_basis", basis)
        self.register_buffer("sfv_eigvals", eigvals)
        if sfv_coefficients is None:
            self.sfv_coefficients = nn.Parameter(torch.empty(basis.size(1), structure_dim, device=device))
            nn.init.normal_(self.sfv_coefficients, mean=0.0, std=0.05)
        else:
            self.sfv_coefficients = nn.Parameter(sfv_coefficients.to(device).float().clone())

    def get_sfv(self):
        return self.sfv_basis @ self.sfv_coefficients

    def spectral_regularizer(self):
        return spectral_sfv_regularizer(self.sfv_coefficients, self.sfv_eigvals)

    def federated_parameters(self):
        for module in [self.feature_encoder, self.structure_encoder, self.head, self.structure_head]:
            yield from module.parameters()

    def forward(self, data):
        feature_embedding = self.feature_encoder(data.x, data.edge_index)
        structure_embedding = self.structure_encoder(self.get_sfv())
        embedding = feature_embedding + structure_embedding
        logits = self.head(feature_embedding) + self.structure_head(structure_embedding)
        return embedding, logits


def _normalize_global_map(data, device):
    if not hasattr(data, "global_map"):
        return None
    global_map = data.global_map
    if torch.is_tensor(global_map):
        return global_map.long().to(device)
    if isinstance(global_map, dict):
        if all(isinstance(key, int) and 0 <= key < data.num_nodes for key in global_map.keys()):
            return torch.tensor(
                [global_map[local_id] for local_id in range(data.num_nodes)],
                dtype=torch.long,
                device=device,
            )
        ordered = [None] * data.num_nodes
        for global_id, local_id in global_map.items():
            ordered[int(local_id)] = int(global_id)
        if any(value is None for value in ordered):
            return None
        return torch.tensor(ordered, dtype=torch.long, device=device)
    return torch.tensor(global_map, dtype=torch.long, device=device)


def _graph_cache_key(edge_index, num_nodes, spectral_len, normalization="sym"):
    edge_cpu = edge_index.detach().cpu().contiguous()
    digest = hashlib.sha1()
    digest.update(str(num_nodes).encode("utf-8"))
    digest.update(str(spectral_len).encode("utf-8"))
    digest.update(normalization.encode("utf-8"))
    digest.update(str(tuple(edge_cpu.shape)).encode("utf-8"))
    digest.update(edge_cpu.numpy().tobytes())
    return digest.hexdigest()


def _load_or_build_sfv_basis(data, spectral_len, device):
    cache_root = Path(getattr(data, "fedstruct_cache_dir", ".cache/fedstruct_spectral"))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = _graph_cache_key(data.edge_index, data.num_nodes, spectral_len)
    cache_path = cache_root / f"sfv_n{data.num_nodes}_k{spectral_len}_{cache_key[:16]}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        return cached["basis"].to(device).float(), cached["eigvals"].to(device).float()

    basis, eigvals = build_laplacian_eigenbasis(
        data.edge_index,
        data.num_nodes,
        k=spectral_len,
        device=device,
    )
    torch.save(
        {"basis": basis.detach().cpu(), "eigvals": eigvals.detach().cpu()},
        cache_path,
    )
    return basis, eigvals


def build_fedstruct_model(args, task, data=None, sfv_basis=None, sfv_eigvals=None, sfv_coefficients=None):
    if data is None:
        data = task.data
    model_name = args.model[0] if getattr(args, "model", None) else "graphsage"
    conv_type = "gcn" if model_name == "gcn" else "sage"
    spectral_len = getattr(args, "fedstruct_spectral_len", None)
    if spectral_len is None:
        spectral_len = min(200, data.num_nodes)
    if sfv_basis is None and sfv_eigvals is None:
        sfv_basis, sfv_eigvals = _load_or_build_sfv_basis(
            data,
            spectral_len,
            data.x.device,
        )
    if sfv_basis is not None:
        global_map = _normalize_global_map(data, data.x.device)
        if global_map is not None and sfv_basis.size(0) != data.num_nodes:
            sfv_basis = sfv_basis.to(data.x.device)[global_map]
    return FedStructModel(
        input_dim=task.num_feats,
        hid_dim=args.hid_dim,
        output_dim=task.num_global_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        data=data,
        spectral_len=spectral_len,
        structure_dim=getattr(args, "fedstruct_structure_dim", 128),
        conv_type=conv_type,
        sfv_basis=sfv_basis,
        sfv_eigvals=sfv_eigvals,
        sfv_coefficients=sfv_coefficients,
    )
