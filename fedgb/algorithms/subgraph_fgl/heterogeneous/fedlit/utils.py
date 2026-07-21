from collections import defaultdict, OrderedDict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def is_fedlit_state(state_dict):
    keys = set(state_dict.keys())
    has_shared = any(key.startswith("feature1.") for key in keys)
    has_classifier = any(key.startswith("classifier.") for key in keys)
    has_split_gnn = any(key.startswith("split_gnns.") for key in keys)
    return has_shared and has_classifier and has_split_gnn


def canonical_edge_index(edge_index):
    src = torch.minimum(edge_index[0], edge_index[1])
    dst = torch.maximum(edge_index[0], edge_index[1])
    return torch.stack([src, dst], dim=0)


def edge_embeddings_from_node_embeddings(node_embeddings, edge_index):
    canonical = canonical_edge_index(edge_index)
    return torch.cat([node_embeddings[canonical[0]], node_embeddings[canonical[1]]], dim=1)


def _init_centroids_from_edge_embeddings(edge_embeddings, nlinktype):
    if edge_embeddings.size(0) == 0:
        return torch.zeros(nlinktype, edge_embeddings.size(1), device=edge_embeddings.device)
    if edge_embeddings.size(0) >= nlinktype:
        positions = torch.linspace(
            0,
            edge_embeddings.size(0) - 1,
            steps=nlinktype,
            device=edge_embeddings.device,
        ).long()
        return edge_embeddings[positions].detach().clone()
    repeats = []
    for idx in range(nlinktype):
        repeats.append(edge_embeddings[idx % edge_embeddings.size(0)])
    return torch.stack(repeats, dim=0).detach().clone()


def cluster_edges_by_centroids(edge_embeddings, centroids, num_iter=1):
    if edge_embeddings.size(0) == 0:
        clusters = torch.empty(0, dtype=torch.long, device=edge_embeddings.device)
        return centroids.detach().clone(), clusters

    next_centroids = centroids.detach().clone().to(edge_embeddings.device)
    clusters = torch.zeros(edge_embeddings.size(0), dtype=torch.long, device=edge_embeddings.device)
    for _ in range(max(1, int(num_iter))):
        similarity = torch.stack(
            [F.cosine_similarity(edge_embeddings, center.unsqueeze(0), dim=1) for center in next_centroids],
            dim=0,
        )
        clusters = torch.argmax(similarity, dim=0)
        updated = next_centroids.clone()
        for idx_cluster in range(next_centroids.size(0)):
            mask = clusters == idx_cluster
            if bool(mask.any()):
                updated[idx_cluster] = edge_embeddings[mask].mean(dim=0)
        next_centroids = updated
    return next_centroids, clusters


def assign_or_update_edge_clusters(node_embeddings, edge_index, nlinktype, centroids=None, num_iter=1):
    edge_embeddings = edge_embeddings_from_node_embeddings(node_embeddings, edge_index)
    if centroids is None:
        centroids = _init_centroids_from_edge_embeddings(edge_embeddings, nlinktype)
    return cluster_edges_by_centroids(edge_embeddings, centroids, num_iter=num_iter)


def pyg_subgraphs_by_linktype(data, clusters, nlinktype):
    subgraphs = []
    for idx_cluster in range(nlinktype):
        edge_mask = clusters == idx_cluster
        if bool(edge_mask.any()):
            edge_index = data.edge_index[:, edge_mask]
            node_ids = torch.unique(edge_index.reshape(-1)).sort()[0]
            local_edge_index, _ = subgraph(
                node_ids,
                edge_index,
                relabel_nodes=True,
                num_nodes=data.x.size(0),
            )
        else:
            node_ids = torch.empty(0, dtype=torch.long, device=data.x.device)
            local_edge_index = torch.empty((2, 0), dtype=torch.long, device=data.x.device)

        subg = Data(
            x=data.x[node_ids],
            edge_index=local_edge_index,
            y=data.y[node_ids] if hasattr(data, "y") else None,
        )
        subg.orig_node_ids = node_ids
        for mask_name in ["train_mask", "val_mask", "test_mask"]:
            if hasattr(data, mask_name):
                setattr(subg, mask_name, getattr(data, mask_name)[node_ids])
        subgraphs.append(subg)
    return subgraphs


def fedlit_forward_with_centroids(model, data, centroids, nlinktype, num_iter, device):
    projected = model.feature_projection(data.x)
    next_centroids, clusters = assign_or_update_edge_clusters(
        node_embeddings=projected.detach(),
        edge_index=data.edge_index,
        nlinktype=nlinktype,
        centroids=centroids,
        num_iter=num_iter,
    )
    subgraphs = pyg_subgraphs_by_linktype(data, clusters, nlinktype)
    for subg in subgraphs:
        subg.x = projected[subg.orig_node_ids].clone()
        subg = subg.to(device)
    embedding = model.split_forward(subgraphs, data.x.size(0), device)
    logits = model.classify(embedding)
    return embedding, logits, next_centroids, clusters, subgraphs


def build_groups_from_similarity(nlinktype, client_id, similarity):
    """Greedily align one client's local clusters to global latent link-type groups."""
    sim = similarity.detach().clone()
    groups = {}
    for _ in range(nlinktype):
        flat_idx = torch.argmax(sim).item()
        idx_cluster = flat_idx // sim.shape[1]
        idx_group = flat_idx % sim.shape[1]
        groups[(client_id, idx_cluster)] = idx_group
        sim[idx_cluster, :] = -float("inf")
        sim[:, idx_group] = -float("inf")
    return groups


def group_centroids(client_centroids, nlinktype, previous_centers=None):
    """Align all client centroids and return updated groups and global centers."""
    if not client_centroids:
        return {}, previous_centers

    groups = {}
    if previous_centers is None:
        init_client_id = next(iter(client_centroids))
        centers = client_centroids[init_client_id].detach().clone()
        groups.update({(init_client_id, idx): idx for idx in range(nlinktype)})
        remaining = [(cid, c) for cid, c in client_centroids.items() if cid != init_client_id]
    else:
        centers = previous_centers.detach().clone()
        remaining = list(client_centroids.items())

    for client_id, centroids in remaining:
        similarity = torch.stack([F.cosine_similarity(centroids, center.unsqueeze(0), dim=1) for center in centers])
        groups.update(build_groups_from_similarity(nlinktype, client_id, similarity))

    centers = update_centers_from_groups(groups, centers, client_centroids)
    return groups, centers


def update_centers_from_groups(groups, centers, client_centroids):
    next_centers = centers.detach().clone()
    grouped = defaultdict(list)
    for (client_id, idx_cluster), idx_group in groups.items():
        grouped[idx_group].append(client_centroids[client_id][idx_cluster].detach())

    for idx_group, values in grouped.items():
        next_centers[idx_group] = torch.mean(torch.stack(values, dim=0), dim=0)
    return next_centers


def aggregate_shared_state(client_states, train_sizes, layer_names):
    """FedLIT shared feature/classifier aggregation weighted by client train size."""
    if len(client_states) != len(train_sizes):
        raise ValueError("client_states and train_sizes must have the same length")

    total_size = float(sum(train_sizes))
    if total_size <= 0:
        return OrderedDict()

    aggregated = OrderedDict()
    prefixes = tuple(f"{name}." for name in layer_names)
    keys = [
        key
        for key in client_states[0].keys()
        if key.startswith(prefixes)
    ]
    for key in keys:
        value = None
        for state, size in zip(client_states, train_sizes):
            if key not in state:
                continue
            cur = state[key].detach() * (float(size) / total_size)
            value = cur.clone() if value is None else value + cur
        if value is not None:
            aggregated[key] = value
    return aggregated


def aggregate_branch_state(server_state, client_states, groups, cluster_train_sizes, idx_group):
    """Aggregate one aligned split-GCN branch and skip empty local clusters."""
    aggregated = OrderedDict((key, value.detach().clone()) for key, value in server_state.items())
    group_members = [
        (client_id, idx_cluster)
        for (client_id, idx_cluster), group_id in groups.items()
        if group_id == idx_group
    ]

    for name in server_state:
        prefix = f"split_gnns.{idx_group}"
        if not name.startswith(prefix):
            continue
        suffix = name.split(prefix, 1)[1]
        values = []
        for client_id, idx_cluster in group_members:
            if cluster_train_sizes.get(client_id, [])[idx_cluster] == 0:
                continue
            client_key = f"split_gnns.{idx_cluster}{suffix}"
            if client_key in client_states[client_id]:
                values.append(client_states[client_id][client_key].detach())
        if values:
            aggregated[name] = torch.mean(torch.stack(values, dim=0), dim=0)
    return aggregated


def aggregate_fedlit_state(server_state, client_payloads, groups, nlinktype, shared_layers=None):
    """Aggregate shared layers and aligned split branches from FedLIT client payloads."""
    shared_layers = ["feature1", "classifier"] if shared_layers is None else shared_layers
    next_state = OrderedDict((key, value.detach().clone()) for key, value in server_state.items())
    client_ids = list(client_payloads.keys())
    client_states = [client_payloads[cid]["state_dict"] for cid in client_ids]
    train_sizes = [client_payloads[cid]["total_train_size"] for cid in client_ids]

    next_state.update(aggregate_shared_state(client_states, train_sizes, shared_layers))
    state_by_client = {cid: client_payloads[cid]["state_dict"] for cid in client_ids}
    cluster_sizes = {cid: client_payloads[cid].get("cluster_train_size", [1] * nlinktype) for cid in client_ids}
    for idx_group in range(nlinktype):
        next_state.update(aggregate_branch_state(next_state, state_by_client, groups, cluster_sizes, idx_group))
    return next_state
