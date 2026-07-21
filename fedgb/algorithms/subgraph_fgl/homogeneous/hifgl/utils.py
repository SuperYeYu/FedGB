import torch
import torch.nn.functional as F


def _as_long_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    return torch.tensor(value, device=device, dtype=torch.long)


def normalize_global_map(global_map, num_nodes, device):
    if torch.is_tensor(global_map):
        return global_map.to(device=device, dtype=torch.long)

    if isinstance(global_map, dict):
        keys = list(global_map.keys())
        values = list(global_map.values())
        local_keys = all(isinstance(key, int) and 0 <= key < num_nodes for key in keys)
        local_values = all(isinstance(value, int) and 0 <= value < num_nodes for value in values)

        if local_keys and not local_values:
            ordered = [global_map[local_id] for local_id in range(num_nodes)]
        elif local_values:
            ordered = [None] * num_nodes
            for global_id, local_id in global_map.items():
                ordered[int(local_id)] = int(global_id)
            if any(item is None for item in ordered):
                raise RuntimeError("Cannot normalize global_map dict with missing local ids.")
        else:
            ordered = [global_map[local_id] for local_id in range(num_nodes)]
        return torch.tensor(ordered, device=device, dtype=torch.long)

    return torch.tensor(global_map, device=device, dtype=torch.long)


def build_cross_client_messages(global_edge_index, client_payloads, feature_dim, device):
    global_edge_index = global_edge_index.to(device)
    owner = {}
    local_pos = {}
    embeddings = {}
    sizes = {}

    for client_id, payload in client_payloads.items():
        global_ids = _as_long_tensor(payload["global_ids"], device)
        node_embeddings = payload["node_embeddings"].to(device)
        embeddings[client_id] = node_embeddings
        sizes[client_id] = node_embeddings.shape[0]
        for idx, global_id in enumerate(global_ids.tolist()):
            owner[global_id] = client_id
            local_pos[global_id] = idx

    messages = {
        client_id: torch.zeros(size, feature_dim, device=device)
        for client_id, size in sizes.items()
    }
    counts = {
        client_id: torch.zeros(size, device=device)
        for client_id, size in sizes.items()
    }

    src_nodes, dst_nodes = global_edge_index
    for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
        if src not in owner or dst not in owner:
            continue
        src_client = owner[src]
        dst_client = owner[dst]
        if src_client == dst_client:
            continue
        dst_idx = local_pos[dst]
        src_idx = local_pos[src]
        messages[dst_client][dst_idx] += embeddings[src_client][src_idx]
        counts[dst_client][dst_idx] += 1.0

    masks = {}
    for client_id in messages:
        masks[client_id] = counts[client_id] > 0
        messages[client_id][masks[client_id]] = (
            messages[client_id][masks[client_id]]
            / counts[client_id][masks[client_id]].unsqueeze(-1)
        )
    return messages, masks


def build_hifgl_cross_edge_messages(global_edge_index, client_payloads, feature_dim, device):
    global_edge_index = global_edge_index.to(device)
    owner = {}
    local_pos = {}
    embeddings = {}
    sizes = {}

    for client_id, payload in client_payloads.items():
        global_ids = _as_long_tensor(payload["global_ids"], device)
        node_embeddings = payload["node_embeddings"].to(device)
        embeddings[client_id] = node_embeddings
        sizes[client_id] = node_embeddings.shape[0]
        for idx, global_id in enumerate(global_ids.tolist()):
            owner[int(global_id)] = client_id
            local_pos[int(global_id)] = idx

    global_degree = {}
    for src, dst in zip(global_edge_index[0].tolist(), global_edge_index[1].tolist()):
        global_degree[int(src)] = global_degree.get(int(src), 0) + 1
        global_degree[int(dst)] = global_degree.get(int(dst), 0)

    messages = {
        client_id: torch.zeros(size, feature_dim, device=device)
        for client_id, size in sizes.items()
    }
    counts = {
        client_id: torch.zeros(size, device=device)
        for client_id, size in sizes.items()
    }

    src_nodes, dst_nodes = global_edge_index
    for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
        src = int(src)
        dst = int(dst)
        if src not in owner or dst not in owner:
            continue
        src_client = owner[src]
        dst_client = owner[dst]
        if src_client == dst_client:
            continue

        src_idx = local_pos[src]
        dst_idx = local_pos[dst]
        src_degree = max(global_degree.get(src, 1), 1)
        messages[dst_client][dst_idx] += embeddings[src_client][src_idx] / (src_degree ** 0.5)
        counts[dst_client][dst_idx] += 1.0

    output = {}
    for client_id in messages:
        mask = counts[client_id] > 0
        output[client_id] = {
            "message": messages[client_id],
            "count": counts[client_id],
            "mask": mask,
        }
    return output


def apply_hifgl_messages_to_embedding(local_embedding, cross_payload, weight=1.0):
    if cross_payload is None:
        return local_embedding

    message = cross_payload["message"].to(local_embedding.device)
    count = cross_payload["count"].to(local_embedding.device)
    mask = cross_payload["mask"].to(local_embedding.device).bool()
    updated = local_embedding.clone()
    if mask.any():
        averaged = message[mask] / count[mask].clamp_min(1.0).unsqueeze(-1)
        updated[mask] = (1.0 - weight) * updated[mask] + weight * averaged
    return updated


def masked_cross_message_loss(local_embedding, cross_embedding, mask):
    mask = mask.to(local_embedding.device).bool()
    if not mask.any():
        return local_embedding.sum() * 0.0
    return 1.0 - F.cosine_similarity(local_embedding[mask], cross_embedding[mask], dim=-1).mean()


def extract_global_ids(data, device):
    if hasattr(data, "global_map"):
        return normalize_global_map(data.global_map, data.x.shape[0], device)
    return torch.arange(data.x.shape[0], device=device)
