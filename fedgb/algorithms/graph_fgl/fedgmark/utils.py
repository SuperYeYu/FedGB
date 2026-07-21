import hashlib

import torch
from torch_geometric.data import Batch


def threshold_mask(values, threshold):
    return (values > threshold).to(dtype=values.dtype)


def direct_average_state_dicts(state_dicts, device):
    if not state_dicts:
        return {}
    averaged = {}
    for name in state_dicts[0]:
        total = torch.zeros_like(state_dicts[0][name], device=device)
        for state in state_dicts:
            total = total + state[name].to(device)
        averaged[name] = total / len(state_dicts)
    return averaged


def weighted_average_state_dicts(state_dicts, weights, device):
    if not state_dicts:
        return {}
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return direct_average_state_dicts(state_dicts, device)

    averaged = {}
    for name in state_dicts[0]:
        total = torch.zeros_like(state_dicts[0][name], device=device)
        for state, weight in zip(state_dicts, weights):
            total = total + state[name].to(device) * (float(weight) / total_weight)
        averaged[name] = total
    return averaged


def shared_parameter_names(model, task_name):
    names = []
    for name, _ in model.named_parameters():
        if task_name == "graph_reg" and ".linears_prediction." in name:
            continue
        names.append(name)
    return names


def shared_parameter_payload(model, task_name):
    named_params = dict(model.named_parameters())
    names = shared_parameter_names(model, task_name)
    return names, [named_params[name] for name in names]


def load_shared_parameters(model, names, weights, device):
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, weight in zip(names, weights):
            named_params[name].data.copy_(weight.to(device))


def watermark_seed(client_id, prefix="my_seed"):
    digest = hashlib.sha256(f"{prefix}_{client_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def generate_trigger_masks(feature_dim, max_nodes, selected_graph_ids, node_groups):
    selected_graph_ids = list(selected_graph_ids)
    topomask = torch.zeros(len(selected_graph_ids), max_nodes, max_nodes)
    featmask = torch.zeros(len(selected_graph_ids), max_nodes, feature_dim)

    for row, graph_id in enumerate(selected_graph_ids):
        trigger_nodes = [int(node) for node in node_groups.get(graph_id, [])]
        for node_id in trigger_nodes:
            if node_id < 0 or node_id >= max_nodes:
                continue
            featmask[row, node_id, :] = 1
            for other_id in trigger_nodes:
                if other_id < 0 or other_id >= max_nodes or other_id == node_id:
                    continue
                topomask[row, node_id, other_id] = 1
    return topomask, featmask


def graph_topology_input(data, max_nodes):
    adj = torch.zeros(max_nodes, max_nodes, dtype=data.x.dtype, device=data.x.device)
    if data.edge_index.numel() > 0:
        src, dst = data.edge_index
        keep = (src < max_nodes) & (dst < max_nodes)
        adj[src[keep], dst[keep]] = 1.0
    return adj


def random_trigger_nodes(data, trigger_size, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    num_nodes = int(data.num_nodes)
    if num_nodes <= 0:
        return []
    if num_nodes >= trigger_size:
        return torch.randperm(num_nodes, generator=generator)[: int(trigger_size)].tolist()
    base = torch.randint(0, num_nodes, (int(trigger_size),), generator=generator)
    return base.tolist()


def build_topomask(num_nodes, max_nodes, trigger_nodes, device):
    mask = torch.zeros(max_nodes, max_nodes, device=device)
    trigger_nodes = [int(node) for node in trigger_nodes if 0 <= int(node) < num_nodes]
    for src in trigger_nodes:
        for dst in trigger_nodes:
            if src != dst:
                mask[src, dst] = 1.0
    return mask


def apply_watermark_to_graph(data, trigger_nodes, target_label, topo_logits=None, topo_threshold=0.5, init_feat=0.0):
    watermarked = data.clone()
    trigger_nodes = [int(node) for node in trigger_nodes if 0 <= int(node) < watermarked.num_nodes]
    if not trigger_nodes:
        watermarked.y = torch.as_tensor([target_label], dtype=torch.long, device=watermarked.y.device)
        return watermarked

    edges = set(tuple(edge) for edge in watermarked.edge_index.t().detach().cpu().tolist())
    for src in trigger_nodes:
        for dst in trigger_nodes:
            if src == dst:
                continue
            add_edge = True
            if topo_logits is not None:
                add_edge = bool(topo_logits[src, dst].detach().item() > topo_threshold)
            if add_edge:
                edges.add((src, dst))

    if edges:
        watermarked.edge_index = torch.tensor(
            sorted(edges),
            dtype=torch.long,
            device=watermarked.edge_index.device,
        ).t().contiguous()
    x = watermarked.x.clone()
    x[trigger_nodes] = init_feat
    watermarked.x = x
    watermarked.y = torch.as_tensor([target_label], dtype=torch.long, device=watermarked.y.device)
    return watermarked


def watermark_batch(batch, generator, client_id, target_label, trigger_size, max_nodes, topo_threshold, seed_prefix):
    watermarked = []
    selected_indices = []
    for graph_idx, data in enumerate(batch.to_data_list()):
        if int(data.y.item()) == int(target_label):
            watermarked.append(data)
            continue
        seed = watermark_seed(f"{client_id}_{graph_idx}", seed_prefix)
        trigger_nodes = random_trigger_nodes(data, trigger_size, seed)
        topomask = build_topomask(data.num_nodes, max_nodes, trigger_nodes, data.x.device)
        a_input = graph_topology_input(data, max_nodes)
        topo_logits = generator(a_input, client_id=client_id, topomask=topomask, threshold=topo_threshold)
        watermarked.append(
            apply_watermark_to_graph(
                data,
                trigger_nodes,
                target_label,
                topo_logits=topo_logits,
                topo_threshold=topo_threshold,
            )
        )
        selected_indices.append(graph_idx)
    return Batch.from_data_list(watermarked).to(batch.x.device), selected_indices


def recover_mask(num_nodes, mask, target):
    if target == "topo":
        return mask[:num_nodes, :num_nodes]
    if target == "feat":
        return mask[:num_nodes, :]
    raise ValueError("target must be 'topo' or 'feat'")
