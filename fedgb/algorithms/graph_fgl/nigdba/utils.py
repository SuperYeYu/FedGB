import copy

import numpy as np
import torch
import torch.nn.functional as F
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
            total = total + (float(weight) / total_weight) * state[name].to(device)
        averaged[name] = total
    return averaged


def is_private_prediction_head(name, task_name, private_head=False):
    graph_task_with_private_head = task_name == "graph_reg" or (task_name == "graph_cls" and private_head)
    return graph_task_with_private_head and name.startswith("head.")


def shared_parameter_names(model, task_name=None, private_head=False):
    return [
        name
        for name, _ in model.named_parameters()
        if not is_private_prediction_head(name, task_name, private_head)
    ]


def shared_parameter_payload(model, task_name=None, private_head=False):
    names = shared_parameter_names(model, task_name, private_head)
    named_params = dict(model.named_parameters())
    return names, [named_params[name] for name in names]


def load_shared_parameters(model, names, weights, device):
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, weight in zip(names, weights):
            if name in named_params:
                named_params[name].data.copy_(weight.to(device))


def compute_trigger_size(avg_nodes, frac_of_avg, min_trigger_nodes=3):
    return max(int(round(float(avg_nodes) * float(frac_of_avg))), int(min_trigger_nodes))


def average_graph_nodes(graphs):
    sizes = sorted(int(data.num_nodes) for data in graphs)
    if not sizes:
        return 1
    trim = int(0.3 * len(sizes))
    middle = sizes[trim : len(sizes) - trim] if len(sizes) - trim > trim else sizes
    return max(1, int(round(sum(middle) / len(middle))))


def select_trigger_nodes(data, trigger_size, trigger_position="random", seed=None):
    num_nodes = int(data.num_nodes)
    k = min(int(trigger_size), num_nodes)
    if k <= 0:
        return []
    if trigger_position == "degree":
        degrees = torch.bincount(data.edge_index[0].detach().cpu(), minlength=num_nodes)
        _, order = torch.sort(degrees, descending=True, stable=True)
        return order[:k].tolist()
    if trigger_position == "cluster":
        return _select_cluster_nodes(data, k)
    if trigger_position != "random":
        raise ValueError("trigger_position must be 'random', 'degree', or 'cluster'")
    rng = np.random.RandomState(seed)
    return rng.choice(np.arange(num_nodes), k, replace=False).astype(int).tolist()


def _select_cluster_nodes(data, k):
    num_nodes = int(data.num_nodes)
    neighbors = [set() for _ in range(num_nodes)]
    for src, dst in data.edge_index.detach().cpu().t().tolist():
        if src != dst:
            neighbors[int(src)].add(int(dst))
            neighbors[int(dst)].add(int(src))
    scores = []
    for node_id, neigh in enumerate(neighbors):
        degree = len(neigh)
        if degree < 2:
            scores.append((0.0, degree, -node_id))
            continue
        links = 0
        neigh_list = list(neigh)
        for i, src in enumerate(neigh_list):
            for dst in neigh_list[i + 1 :]:
                if dst in neighbors[src]:
                    links += 1
        denom = degree * (degree - 1) / 2
        scores.append((links / denom, degree, -node_id))
    scores.sort(reverse=True)
    return [-entry[2] for entry in scores[:k]]


def apply_trojan_trigger(data, trigger_nodes, trojan, target_label, weight_threshold):
    poisoned = data.clone()
    trigger_nodes = [int(node) for node in trigger_nodes if 0 <= int(node) < int(poisoned.num_nodes)]
    if not trigger_nodes:
        poisoned.y = torch.as_tensor([target_label], dtype=torch.long, device=poisoned.y.device)
        return poisoned

    trigger_feat, edge_weights = trojan(poisoned.x[trigger_nodes].float())
    trigger_feat = trigger_feat.view(len(trigger_nodes), poisoned.x.shape[1])
    x = poisoned.x.clone()
    x[trigger_nodes] = trigger_feat.to(dtype=x.dtype, device=x.device)
    poisoned.x = x

    edges = set(tuple(edge) for edge in poisoned.edge_index.t().detach().cpu().tolist())
    for src_idx, src in enumerate(trigger_nodes):
        for dst_idx in range(src_idx, len(trigger_nodes)):
            dst = trigger_nodes[dst_idx]
            if src == dst:
                continue
            if edge_weights[src_idx, dst_idx].detach().item() > float(weight_threshold):
                edges.add((src, dst))
                edges.add((dst, src))
    if edges:
        poisoned.edge_index = torch.tensor(sorted(edges), dtype=torch.long, device=poisoned.edge_index.device).t()
    poisoned.y = torch.as_tensor([target_label], dtype=torch.long, device=poisoned.y.device)
    return poisoned


def poison_batch_with_trojan(
    batch,
    trojan,
    target_label,
    trigger_size,
    trigger_position,
    weight_threshold,
    seed=None,
):
    poisoned_graphs = []
    poisoned_indices = []
    for graph_idx, data in enumerate(batch.to_data_list()):
        if int(data.y.item()) == int(target_label) or int(data.num_nodes) < int(trigger_size):
            poisoned_graphs.append(data)
            continue
        trigger_nodes = select_trigger_nodes(data, trigger_size, trigger_position, None if seed is None else seed + graph_idx)
        poisoned_graphs.append(apply_trojan_trigger(data, trigger_nodes, trojan, target_label, weight_threshold))
        poisoned_indices.append(graph_idx)
    return Batch.from_data_list(poisoned_graphs).to(batch.x.device), poisoned_indices


def train_trojan_generator(
    model,
    trojan,
    optimizer,
    train_dataloader,
    device,
    target_label,
    avg_nodes,
    frac_of_avg,
    trigger_position,
    weight_threshold,
    trojan_epochs,
    target_loss_weight,
    seed,
):
    trigger_size = compute_trigger_size(avg_nodes, frac_of_avg)
    trojan.train()
    model.eval()
    trained_steps = 0
    for epoch in range(int(trojan_epochs)):
        loss_total = None
        poisoned_seen = 0
        for batch_idx, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            poisoned, poisoned_indices = poison_batch_with_trojan(
                batch,
                trojan,
                target_label,
                trigger_size,
                trigger_position,
                weight_threshold,
                seed + epoch * 10007 + batch_idx,
            )
            if not poisoned_indices:
                continue
            _, logits = model(poisoned)
            poison_logits = logits[poisoned_indices]
            poison_labels = torch.full(
                (len(poisoned_indices),),
                int(target_label),
                dtype=torch.long,
                device=device,
            )
            loss = float(target_loss_weight) * F.cross_entropy(poison_logits, poison_labels)
            loss_total = loss if loss_total is None else loss_total + loss
            poisoned_seen += len(poisoned_indices)
        if loss_total is None or poisoned_seen == 0:
            break
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        trained_steps += 1
    return trained_steps


def _graph_size(adj):
    return int(adj.shape[0]) if hasattr(adj, "shape") else len(adj)


def select_candidate_graphs(args, data_ids, adj_list, subset):
    rng = np.random.RandomState(args.seed)
    ratio = args.bkd_gratio_train if subset == "train" else args.bkd_gratio_test
    target_count = int(np.ceil(ratio * len(data_ids)))
    if target_count <= 0:
        return []
    if len(data_ids) <= target_count:
        raise ValueError("Graph instances are not enough for the requested backdoor ratio")

    picked_ids = []
    remained_set = copy.deepcopy(list(data_ids))
    loop_count = 0
    thresholds = [3.0, 1.5, 1.0]
    min_required = args.bkd_size * args.bkd_num_pergraph

    while target_count - len(picked_ids) > 0 and remained_set and loop_count <= 50:
        loop_count += 1
        draw_count = min(target_count - len(picked_ids), len(remained_set))
        candidate_ids = rng.choice(remained_set, draw_count, replace=False).tolist()

        for multiplier in thresholds:
            for graph_id in candidate_ids:
                if target_count - len(picked_ids) <= 0:
                    break
                if graph_id in picked_ids:
                    continue
                if _graph_size(adj_list[graph_id]) >= multiplier * min_required:
                    picked_ids.append(int(graph_id))

        picked_ids = sorted(set(picked_ids))
        remained_set = sorted(set(remained_set) - set(picked_ids))

    return picked_ids


def select_candidate_nodes(args, graph_candidate_ids, adj_list):
    rng = np.random.RandomState(args.seed)
    picked_nodes = []
    node_groups = []
    node_count = int(args.bkd_num_pergraph * args.bkd_size)

    for graph_id in graph_candidate_ids:
        num_nodes = _graph_size(adj_list[graph_id])
        if node_count > num_nodes:
            raise ValueError("Candidate graph is too small for the requested NI-GDBA trigger")
        nodes = rng.choice(list(range(num_nodes)), node_count, replace=False).tolist()
        picked_nodes.append(nodes)
        groups = np.array_split(np.array(nodes), len(nodes) // args.bkd_size)
        node_groups.append([group.astype(int).tolist() for group in groups])

    return picked_nodes, node_groups


def generate_trigger_masks(features, max_nodes, selected_graph_ids, node_groups):
    feature_dim = int(torch.as_tensor(features[0]).shape[1])
    topomask = {}
    featmask = {}

    for idx, graph_id in enumerate(selected_graph_ids):
        topomask[graph_id] = torch.zeros(max_nodes, max_nodes)
        featmask[graph_id] = torch.zeros(max_nodes, feature_dim)
        for group in node_groups[idx]:
            for node_id in group:
                if node_id < 0 or node_id >= max_nodes:
                    continue
                for other_id in group:
                    if other_id < 0 or other_id >= max_nodes or other_id == node_id:
                        continue
                    topomask[graph_id][node_id, other_id] = 1
                featmask[graph_id][node_id, :] = 1

    return topomask, featmask


def recover_mask(num_nodes, mask, target):
    recovered = copy.deepcopy(mask)
    if target == "topo":
        return recovered[:num_nodes, :num_nodes]
    if target == "feat":
        return recovered[:num_nodes]
    raise ValueError("target must be 'topo' or 'feat'")
