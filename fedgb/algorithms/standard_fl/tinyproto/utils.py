import itertools
from collections import defaultdict

import numpy as np
import torch


def _block_vectors(num_classes, ones_count, seed=0):
    ones_count = max(1, min(int(ones_count), int(num_classes)))
    if ones_count == num_classes:
        return np.ones((num_classes, num_classes), dtype=np.int64)

    candidates = list(itertools.combinations(range(num_classes), ones_count))
    selected = [set(candidates[-1])]
    rng = np.random.default_rng(seed)
    remaining = [set(item) for item in candidates[:-1]]

    while len(selected) < num_classes:
        best_idx = None
        best_score = -1
        order = rng.permutation(len(remaining))
        for idx in order:
            candidate = remaining[int(idx)]
            score = sum(len(candidate.symmetric_difference(prev)) for prev in selected)
            if score > best_score:
                best_score = score
                best_idx = int(idx)
        selected.append(remaining.pop(best_idx))

    vectors = np.zeros((num_classes, num_classes), dtype=np.int64)
    for row, active_blocks in enumerate(selected):
        for block in active_blocks:
            vectors[row, block] = 1
    return vectors


def build_classwise_masks(feature_dim, num_classes, csr_ratio=10, seed=0, device=None):
    block = max(1, feature_dim // num_classes)
    per_mask = max(1, int(num_classes * (float(csr_ratio) / 100.0)))
    vectors = _block_vectors(num_classes, per_mask, seed=seed)
    masks = {}
    for class_id in range(num_classes):
        indices = []
        for block_id, active in enumerate(vectors[class_id]):
            if not active:
                continue
            start = block_id * block
            end = min(start + block, feature_dim)
            indices.extend(range(start, end))
        if not indices:
            indices = [class_id % feature_dim]
        masks[class_id] = torch.tensor(indices, dtype=torch.long, device=device)
    return masks


def compute_class_prototypes(embedding, labels, mask, num_classes, simple_scale=True):
    device = embedding.device
    prototypes = {}
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    labels = labels.to(device)
    mask = mask.to(device).bool()

    for class_id in range(num_classes):
        selected = mask & (labels == class_id)
        counts[class_id] = selected.sum().float()
        if selected.any():
            proto = embedding[selected].mean(dim=0).detach()
            if simple_scale:
                proto = proto * counts[class_id]
            prototypes[class_id] = proto
    return prototypes, counts


def sparsify_prototypes(prototypes, masks):
    sparse = {}
    for class_id, proto in prototypes.items():
        if class_id not in masks:
            sparse[class_id] = proto.detach().clone()
        else:
            sparse[class_id] = proto.detach()[masks[class_id].to(proto.device)].clone()
    return sparse


def expand_sparse_prototypes(sparse_prototypes, masks, feature_dim, device=None):
    expanded = {}
    for class_id, proto in sparse_prototypes.items():
        value = proto.detach()
        if device is not None:
            value = value.to(device)
        if class_id not in masks:
            expanded[class_id] = value.clone()
            continue
        full = value.new_zeros(feature_dim)
        full[masks[class_id].to(value.device)] = value
        expanded[class_id] = full
    return expanded


def aggregate_sparse_prototypes(
    local_prototypes,
    simple_scale=True,
    constant_scale_factor=1.0,
):
    by_class = defaultdict(list)
    for client_protos in local_prototypes:
        for class_id, proto in client_protos.items():
            by_class[class_id].append(proto.detach())

    aggregated = {}
    for class_id, protos in by_class.items():
        stacked = torch.stack(protos, dim=0)
        value = torch.mean(stacked, dim=0).detach()
        if simple_scale:
            value = len(protos) * float(constant_scale_factor) * value
        aggregated[class_id] = value
    return aggregated


def sparse_proto_targets(labels, sparse_global_protos, masks, feature_dim, device):
    full = expand_sparse_prototypes(sparse_global_protos, masks, feature_dim, device=device)
    targets = torch.zeros(labels.shape[0], feature_dim, device=device)
    for row, label in enumerate(labels.detach().cpu().tolist()):
        if label in full:
            targets[row] = full[label].to(device)
    return targets


def prototype_distance_logits(embedding, sparse_global_protos, masks, num_classes, feature_dim):
    if not sparse_global_protos:
        return None
    full = expand_sparse_prototypes(
        sparse_global_protos,
        masks,
        feature_dim,
        device=embedding.device,
    )
    rows = []
    for class_id in range(num_classes):
        proto = full.get(class_id)
        if proto is None:
            rows.append(embedding.new_full((feature_dim,), float("nan")))
        else:
            rows.append(proto.to(embedding.device))
    proto_matrix = torch.stack(rows, dim=0)
    valid = ~torch.isnan(proto_matrix).any(dim=1)
    safe_proto = torch.nan_to_num(proto_matrix, nan=0.0)
    distances = torch.mean((embedding.unsqueeze(1) - safe_proto.unsqueeze(0)) ** 2, dim=2)
    distances[:, ~valid] = float("inf")
    return -distances
