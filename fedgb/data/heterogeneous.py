"""Lightweight heterogeneous-to-relation graph conversion."""

import torch
from torch_geometric.data import Data


def _ensure_node_features(data):
    fallback_dtype = next(
        (data[node_type].x.dtype for node_type in data.node_types if hasattr(data[node_type], "x")),
        torch.float32,
    )
    for node_type in data.node_types:
        if not hasattr(data[node_type], "x"):
            data[node_type].x = torch.eye(data[node_type].num_nodes, dtype=fallback_dtype)


def hetero_to_relation_data(data, target_node=None, global_node_offsets=None):
    target_node = target_node or data.node_types[0]
    _ensure_node_features(data)
    if global_node_offsets is None:
        global_node_offsets = {}
        cursor = 0
        for node_type in data.node_types:
            global_node_offsets[node_type] = cursor
            cursor += data[node_type].num_nodes

    max_dim = max(data[node_type].x.shape[1] for node_type in data.node_types)
    local_offsets = {}
    xs, ys, node_types = [], [], []
    global_map = {}
    cursor = 0
    source_global_map = getattr(data, "global_map", {})
    for type_id, node_type in enumerate(data.node_types):
        x = data[node_type].x.to(torch.float32)
        if x.shape[1] < max_dim:
            x = torch.cat(
                [x, torch.zeros(x.shape[0], max_dim - x.shape[1], dtype=x.dtype, device=x.device)],
                dim=1,
            )
        xs.append(x)
        y = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        if node_type == target_node and hasattr(data[node_type], "y"):
            y = data[node_type].y.long().view(-1)
        ys.append(y)
        node_types.append(torch.full((x.shape[0],), type_id, dtype=torch.long, device=x.device))
        local_offsets[node_type] = cursor
        per_type = source_global_map.get(node_type, {}) if isinstance(source_global_map, dict) else {}
        for local_id in range(x.shape[0]):
            global_map[cursor + local_id] = global_node_offsets[node_type] + int(per_type.get(local_id, local_id))
        cursor += x.shape[0]

    edge_parts, edge_types = [], []
    for edge_type_id, edge_type in enumerate(data.edge_types):
        if not hasattr(data[edge_type], "edge_index"):
            continue
        edge_index = data[edge_type].edge_index.long()
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        shifted = edge_index.clone()
        shifted[0] += local_offsets[src_type]
        shifted[1] += local_offsets[dst_type]
        edge_parts.append(shifted)
        edge_types.append(torch.full((shifted.shape[1],), edge_type_id, dtype=torch.long, device=shifted.device))

    edge_index = torch.cat(edge_parts, dim=1) if edge_parts else torch.empty((2, 0), dtype=torch.long)
    relation = Data(x=torch.cat(xs), y=torch.cat(ys), edge_index=edge_index.contiguous())
    relation.node_type = torch.cat(node_types)
    relation.edge_type = (
        torch.cat(edge_types) if edge_types else torch.empty((0,), dtype=torch.long)
    )
    relation.target_node_type = target_node
    relation.hetero_node_types = list(data.node_types)
    relation.hetero_edge_types = list(data.edge_types)
    relation.global_map = global_map
    if hasattr(data, "num_global_classes"):
        relation.num_global_classes = int(data.num_global_classes)
    elif hasattr(data[target_node], "y") and data[target_node].y.numel() > 0:
        relation.num_global_classes = int(data[target_node].y.max().item()) + 1
    return relation

