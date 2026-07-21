import torch
from torch_geometric.data import Batch


def threshold_mask(values, threshold):
    return (values >= threshold).to(dtype=values.dtype)


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
    return [name for name, _ in model.named_parameters()]


def shared_parameter_payload(model, task_name):
    named_params = dict(model.named_parameters())
    names = shared_parameter_names(model, task_name)
    return names, [named_params[name] for name in names]


def load_shared_parameters(model, names, weights, device):
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, weight in zip(names, weights):
            named_params[name].data.copy_(weight.to(device))


def select_top_trigger_nodes(scores, trigger_size, num_nodes):
    valid_scores = scores[:num_nodes]
    k = min(int(trigger_size), int(num_nodes))
    if k <= 0:
        return []
    _, indices = torch.topk(valid_scores, k=k)
    return indices.detach().cpu().tolist()


def graph_trigger_inputs(data, max_nodes):
    num_nodes = int(data.num_nodes)
    device = data.x.device
    adj = torch.zeros(max_nodes, max_nodes, dtype=data.x.dtype, device=device)
    if data.edge_index.numel() > 0:
        src, dst = data.edge_index
        keep = (src < max_nodes) & (dst < max_nodes)
        adj[src[keep], dst[keep]] = 1.0
    x_input = torch.zeros(max_nodes, data.x.shape[1], dtype=data.x.dtype, device=device)
    x_input[: min(num_nodes, max_nodes)] = torch.matmul(adj[:num_nodes, :num_nodes], data.x.float())[
        : min(num_nodes, max_nodes)
    ]
    return adj, x_input


def apply_trigger_to_graph(
    data,
    trigger_nodes,
    target_label,
    feature_value=1.0,
    topo_logits=None,
    feat_logits=None,
    topo_threshold=0.5,
    feat_threshold=0.0,
):
    poisoned = data.clone()
    trigger_nodes = [int(node) for node in trigger_nodes if 0 <= int(node) < poisoned.num_nodes]
    if not trigger_nodes:
        poisoned.y = torch.as_tensor([target_label], dtype=torch.long, device=poisoned.y.device)
        return poisoned

    edges = set(tuple(edge) for edge in poisoned.edge_index.t().detach().cpu().tolist())
    for src in trigger_nodes:
        for dst in trigger_nodes:
            if src == dst:
                continue
            add_edge = True
            if topo_logits is not None:
                add_edge = bool(topo_logits[src, dst].detach().item() >= topo_threshold)
            if add_edge:
                edges.add((src, dst))

    if edges:
        poisoned.edge_index = torch.tensor(
            sorted(edges),
            dtype=torch.long,
            device=poisoned.edge_index.device,
        ).t().contiguous()

    x = poisoned.x.clone()
    if feat_logits is None:
        x[trigger_nodes] = feature_value
    else:
        feat_delta = feat_logits[trigger_nodes]
        feat_mask = (feat_delta >= feat_threshold).to(dtype=x.dtype)
        x[trigger_nodes] = x[trigger_nodes] + feat_delta.detach() * feat_mask
    poisoned.x = x
    poisoned.y = torch.as_tensor([target_label], dtype=torch.long, device=poisoned.y.device)
    return poisoned


def poison_batch_with_generator(
    batch,
    generator,
    client_id,
    target_label,
    trigger_size,
    max_nodes,
    topo_threshold,
    feat_threshold,
):
    data_list = batch.to_data_list()
    poisoned_graphs = []
    poisoned_indices = []
    for graph_idx, data in enumerate(data_list):
        if int(data.y.item()) == int(target_label):
            poisoned_graphs.append(data)
            continue
        a_input, x_input = graph_trigger_inputs(data, max_nodes)
        topo_logits, feat_logits, node_scores = generator(a_input, x_input, client_id, data.num_nodes)
        trigger_nodes = select_top_trigger_nodes(node_scores, trigger_size, data.num_nodes)
        poisoned_graphs.append(
            apply_trigger_to_graph(
                data,
                trigger_nodes,
                target_label,
                topo_logits=topo_logits,
                feat_logits=feat_logits,
                topo_threshold=topo_threshold,
                feat_threshold=feat_threshold,
            )
        )
        poisoned_indices.append(graph_idx)
    return Batch.from_data_list(poisoned_graphs).to(batch.x.device), poisoned_indices


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


def recover_mask(num_nodes, mask, target):
    if target == "topo":
        return mask[:num_nodes, :num_nodes]
    if target == "feat":
        return mask[:num_nodes, :]
    raise ValueError("target must be 'topo' or 'feat'")
