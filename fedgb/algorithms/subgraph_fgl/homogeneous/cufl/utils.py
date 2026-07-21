import copy
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def _cosine_similarity(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    size = min(left.size, right.size)
    if size == 0:
        return 0.0

    left = left[:size]
    right = right[:size]
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def build_similarity_matrix(recon_edges, scales=1.0, metric="exp", filter_below_mean=False):
    n = len(recon_edges)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    if isinstance(scales, (int, float)):
        scales = np.ones(n, dtype=np.float64) * float(scales)
    else:
        scales = np.asarray(scales, dtype=np.float64)
        if scales.size != n:
            scales = np.ones(n, dtype=np.float64)

    sim_matrix = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = _cosine_similarity(recon_edges[i], recon_edges[j])

    if metric == "exp":
        sim_matrix = np.exp(scales[:, np.newaxis] * sim_matrix)

    if filter_below_mean:
        for i in range(n):
            row_mean = sim_matrix[i].mean()
            sim_matrix[i, sim_matrix[i] < row_mean] = 0.0

    row_sum = sim_matrix.sum(axis=1, keepdims=True)
    degenerate = (~np.isfinite(row_sum[:, 0])) | (row_sum[:, 0] <= 0)
    sim_matrix = np.divide(
        sim_matrix,
        row_sum,
        out=np.zeros_like(sim_matrix),
        where=row_sum > 0,
    )
    if np.any(degenerate):
        sim_matrix[degenerate] = 1.0 / n
    return sim_matrix


class VectorGSS:
    """Official CUFL vector greedy scale scheduler."""

    def __init__(self, init_scale, window_size, patience, varying_factor, max_scale, min_scale, prefer_larger=False):
        self.scale = init_scale
        self.window_size = window_size
        self.patience = patience
        self.varying_factor = varying_factor
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.prefer_larger = prefer_larger
        self.accum_metric = deque(maxlen=window_size)
        self.best_metric = -np.inf if prefer_larger else np.inf
        self.num_good = 0
        self.num_bad = 0
        self.is_increase = True

    def compare(self, challenger, best_metric):
        return challenger > best_metric if self.prefer_larger else challenger < best_metric

    def improve(self, flag):
        if flag:
            self.scale *= self.varying_factor
        else:
            self.scale /= self.varying_factor

    def clip(self):
        self.scale = min(max(self.scale, self.min_scale), self.max_scale)

    def evaluate(self, val_metric):
        self.accum_metric.append(float(val_metric))
        mean_metric = float(np.mean(self.accum_metric))
        if self.compare(mean_metric, self.best_metric):
            self.best_metric = mean_metric
            self.num_good += 1
            self.num_bad = 0
        else:
            self.num_good = 0
            self.num_bad += 1

        if self.num_good >= self.patience:
            self.improve(self.is_increase)
            self.num_good = 0
        elif self.num_bad >= self.patience:
            self.is_increase = not self.is_increase
            self.improve(self.is_increase)
            self.num_bad = 0
        self.clip()


def build_proxy_data(num_features, num_proxy=5, num_nodes=100, p_in=0.1, p_out=0.0, seed=0, device=None):
    try:
        import networkx as nx
        from torch_geometric.utils import from_networkx

        graph = nx.random_partition_graph(
            [num_nodes] * num_proxy,
            p_in=p_in,
            p_out=p_out,
            seed=seed,
        )
        data = from_networkx(graph)
        edge_index = data.edge_index.long()
    except Exception:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        edges = []
        for part in range(num_proxy):
            offset = part * num_nodes
            for src in range(num_nodes):
                for dst in range(src + 1, num_nodes):
                    if torch.rand((), generator=generator).item() < p_in:
                        edges.append([offset + src, offset + dst])
                        edges.append([offset + dst, offset + src])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2, 0, dtype=torch.long)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    x = torch.normal(
        mean=0.0,
        std=1.0,
        size=(num_proxy * num_nodes, num_features),
        generator=generator,
    )
    proxy = Data(x=x, edge_index=edge_index)
    proxy.edge_attr = torch.ones(proxy.edge_index.size(1), dtype=torch.float)
    return proxy.to(device) if device is not None else proxy


def _clone_detached_state(state_dict, device=None):
    cloned = {}
    for name, tensor in state_dict.items():
        cloned[name] = tensor.detach().clone().to(device or tensor.device)
    return cloned


def parameters_to_state_dict(model, parameters):
    names = list(model.state_dict().keys())
    return {name: param.detach().clone() for name, param in zip(names, parameters)}


def state_dict_to_parameter_list(model, state_dict, device=None):
    params = []
    for name in model.state_dict().keys():
        tensor = state_dict[name].detach().clone()
        params.append(tensor.to(device or tensor.device))
    return params


def _ratio_index(name, num_layers):
    if name.startswith("convs"):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            idx = int(parts[1])
            return idx if idx < num_layers else -1
    return -1


def aggregate_state_dicts(
    local_weights,
    ratios=None,
    client_id=-1,
    aggregate_classifier=True,
    mask_aggr=False,
    l1=0.001,
    device=None,
):
    if not local_weights:
        return {}

    if ratios is None:
        ratio_matrix = np.ones((1, len(local_weights)), dtype=np.float64) / len(local_weights)
    else:
        ratio_matrix = np.asarray(ratios, dtype=np.float64)
        if ratio_matrix.ndim == 1:
            ratio_matrix = ratio_matrix[np.newaxis, :]
    num_layers = ratio_matrix.shape[0]

    aggregated = {}
    for name in local_weights[0].keys():
        idx = _ratio_index(name, num_layers)
        if name.startswith("classifier") and not aggregate_classifier and client_id != -1:
            source_index = int(client_id)
            if source_index >= len(local_weights):
                source_index = len(local_weights) - 1
            aggregated[name] = local_weights[source_index][name].detach().clone().to(
                device or local_weights[source_index][name].device
            )
            continue

        base_device = device or local_weights[0][name].device
        row = ratio_matrix[idx]
        if mask_aggr and "mask" in name:
            active = []
            for ratio, weights in zip(row, local_weights):
                mask = (weights[name].detach().abs() >= l1).float().to(base_device)
                active.append(mask * float(ratio) + 1e-8)
            denom = torch.stack(active, dim=0).sum(dim=0).clamp_min(1e-12)
            value = torch.zeros_like(local_weights[0][name], device=base_device)
            for weights, active_ratio in zip(local_weights, active):
                value += weights[name].detach().to(base_device) * (active_ratio / denom)
        else:
            value = torch.zeros_like(local_weights[0][name], device=base_device)
            for ratio, weights in zip(row, local_weights):
                value += weights[name].detach().to(base_device) * float(ratio)
        aggregated[name] = value
    return aggregated


def aggregate_parameters(local_weights, ratios, device=None):
    ratios = list(ratios)
    aggregated = [torch.zeros_like(param, device=device or param.device) for param in local_weights[0]]
    for ratio, weights in zip(ratios, local_weights):
        for idx, param in enumerate(weights):
            aggregated[idx] += param.detach().to(aggregated[idx].device) * float(ratio)
    return aggregated


def clone_parameter_list(params):
    return [param.detach().clone() for param in params]


def copy_state_dict_skip_mask(model, incoming_state, skip_mask=True):
    current = model.state_dict()
    for name, tensor in incoming_state.items():
        if name not in current:
            continue
        if skip_mask and "mask" in name:
            continue
        current[name].copy_(tensor.to(current[name].device))


def make_masked_split(splitted_data, edge_index, device, edge_weight=None):
    masked = dict(splitted_data)
    data = copy.copy(splitted_data["data"])
    data.edge_index = edge_index.to(device)
    data.edge_attr = edge_weight.to(device) if edge_weight is not None else torch.ones(edge_index.size(1), device=device)
    masked["data"] = data
    return masked


def ensure_edge_attr(data, device=None):
    if not hasattr(data, "edge_attr") or data.edge_attr is None:
        data.edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float, device=data.edge_index.device)
    if device is not None:
        data.edge_attr = data.edge_attr.to(device)
    return data


def compute_edge_confidence(logits, labels, edge_index, train_mask, norm_scale=2.0, norm_method="min"):
    labels = labels.to(logits.device).long()
    train_mask = train_mask.to(logits.device).bool()
    edge_index = edge_index.to(logits.device)
    num_classes = logits.size(-1)
    valid_mask = train_mask & (labels >= 0) & (labels < num_classes)

    norm_difficulty = torch.ones(labels.size(0), device=logits.device, dtype=logits.dtype)
    if valid_mask.any():
        difficulty = F.cross_entropy(
            logits[valid_mask],
            labels[valid_mask],
            reduction="none",
        ).detach()
        valid_difficulty = torch.exp(float(norm_scale) * difficulty)
        if norm_method == "min":
            valid_difficulty = valid_difficulty / valid_difficulty.min().clamp_min(1e-12)
        elif norm_method == "sum":
            valid_difficulty = valid_difficulty / valid_difficulty.sum().clamp_min(1e-12)
        norm_difficulty[valid_mask] = valid_difficulty

    src, dst = edge_index
    n1 = norm_difficulty[src].clone()
    n2 = norm_difficulty[dst].clone()
    n1[~valid_mask[src]] = 1.0
    n2[~valid_mask[dst]] = 1.0
    return (n1 * n2).clamp_min(1e-12)


def update_personalization_degree(curr_round, num_rounds, base_pd, warmup_pd, rule):
    if rule == "1":
        boundary = num_rounds * 2 // 3
        return base_pd / (boundary + 1 - curr_round) if curr_round < boundary else base_pd
    if rule == "descend_1":
        boundary = num_rounds * 2 // 3
        return base_pd - base_pd / (boundary + 1 - curr_round) if curr_round < boundary else base_pd
    if rule == "2":
        return base_pd / (num_rounds + 1 - curr_round) if curr_round < num_rounds else base_pd
    if rule == "3":
        return base_pd * curr_round / max(1, num_rounds) + warmup_pd
    return base_pd


def transfer_proxy_reconstruction(spcl_model, method="ratio", ratio=0.3, threshold=0.4):
    proxy_recon = (
        spcl_model.graph_recon_degree
        / max(float(spcl_model.num), spcl_model.epsilon)
    ).detach().cpu().numpy()
    if proxy_recon.size == 0:
        return proxy_recon
    if method == "ratio":
        proxy_recon[proxy_recon < np.quantile(proxy_recon, ratio)] = 0.0
    elif method == "threshold":
        proxy_recon[proxy_recon < threshold] = 0.0
    return proxy_recon
