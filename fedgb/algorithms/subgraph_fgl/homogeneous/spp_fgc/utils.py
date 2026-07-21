import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans


EPS = 2.2204e-16


def cal_weights_via_can(x, num_neighbors, links=None):
    if x.dim() != 2:
        raise ValueError("x must be a 2-D tensor")

    num_nodes = x.size(0)
    if num_nodes == 0:
        return torch.empty((0, 0), dtype=x.dtype, device=x.device)
    if num_nodes == 1:
        return torch.ones((1, 1), dtype=x.dtype, device=x.device)

    k = max(1, min(int(num_neighbors), num_nodes - 1))
    distances = torch.cdist(x, x, p=2).pow(2)
    distances = torch.maximum(distances, distances.t())
    sorted_distances, _ = distances.sort(dim=1)

    top_k = sorted_distances[:, k].unsqueeze(1) + 1e-10
    sum_top_k = sorted_distances[:, :k].sum(dim=1, keepdim=True)
    weights = (top_k - distances) / (k * top_k - sum_top_k + 1e-12)
    weights = weights.relu()
    if links is not None:
        link_tensor = torch.as_tensor(links, dtype=weights.dtype, device=weights.device)
        weights = weights + torch.eye(num_nodes, dtype=weights.dtype, device=weights.device) + link_tensor
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    weights = (weights + weights.t()) / 2
    return torch.clamp(weights, min=0.0, max=1.0)


def graph_diff_loss(graph_a, target_s):
    return torch.norm(graph_a - target_s.to(graph_a.device), p="fro").pow(2)


def _row_l2_normalize(matrix):
    norm = torch.norm(matrix, p=2, dim=1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    return matrix / norm


def clr_graph_learning(graph, num_clusters, device=None, max_iter=30):
    graph = torch.as_tensor(graph, dtype=torch.float32, device=device)
    graph = graph.clone()
    graph.fill_diagonal_(0)
    graph = (graph + graph.t()) / 2
    if graph.numel() == 0:
        return np.empty((0,), dtype=int), graph

    num_nodes = graph.size(0)
    num_clusters = max(1, min(int(num_clusters), num_nodes))
    affinity = graph.detach().cpu().numpy()
    if np.allclose(affinity, 0):
        labels = np.arange(num_nodes) % num_clusters
        return labels.astype(int), torch.eye(num_nodes, dtype=torch.float32, device=graph.device)

    degree = torch.diag(graph.sum(dim=1))
    laplacian = degree - graph
    try:
        eigvals, eigvecs = torch.linalg.eigh(laplacian)
        features = eigvecs[:, :num_clusters]
        features = _row_l2_normalize(features).detach().cpu().numpy()
        labels = KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit_predict(features)
    except Exception:
        labels = np.arange(num_nodes) % num_clusters

    learned = torch.zeros_like(graph)
    for cluster_id in range(num_clusters):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        block = graph[idx][:, idx]
        if block.numel() == 0 or float(block.sum()) <= 0:
            block = torch.ones((len(idx), len(idx)), dtype=graph.dtype, device=graph.device)
        row_sum = block.sum(dim=1, keepdim=True).clamp_min(1e-12)
        block = block / row_sum
        learned[idx[:, None], idx] = block

    learned = (learned + learned.t()) / 2
    learned.fill_diagonal_(1.0)
    return labels.astype(int), learned


def compute_cluster_prototypes(embedding, labels, num_clusters):
    embedding_np = embedding.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int) if torch.is_tensor(labels) else np.asarray(labels, dtype=int)
    prototypes = np.zeros((num_clusters, embedding_np.shape[1]), dtype=np.float32)
    for class_id in range(num_clusters):
        mask = labels_np == class_id
        if mask.any():
            prototypes[class_id] = embedding_np[mask].mean(axis=0)
    return prototypes


def cluster_embeddings(embedding, num_clusters, random_state=0):
    embedding_np = embedding.detach().cpu().numpy()
    n_clusters = max(1, min(int(num_clusters), embedding_np.shape[0]))
    if embedding_np.shape[0] > 10000:
        labels = MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=3,
            batch_size=4096,
            random_state=random_state,
        ).fit_predict(embedding_np)
    else:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(embedding_np)
    prototypes = compute_cluster_prototypes(
        embedding,
        torch.tensor(labels, device=embedding.device),
        n_clusters,
    )
    if n_clusters < num_clusters:
        padded = np.zeros((num_clusters, embedding_np.shape[1]), dtype=np.float32)
        padded[:n_clusters] = prototypes
        prototypes = padded
    return labels.astype(int), prototypes


def calculate_prototype_sensitivity(
    data_samples_list,
    n_clusters=10,
    num_removals=1,
    noise_scale=0.1,
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    sensitivities = []
    for data_samples in data_samples_list:
        data_samples = np.asarray(data_samples, dtype=np.float32)
        samples = data_samples.T if data_samples.shape[0] < data_samples.shape[1] else data_samples
        if samples.shape[0] <= n_clusters:
            sensitivities.append(0.0)
            continue
        k = max(1, min(int(n_clusters), samples.shape[0] - 1))
        prototypes = KMeans(n_clusters=k, n_init=10, random_state=0).fit(samples).cluster_centers_
        perturbed = samples.copy()
        idx = rng.choice(samples.shape[0], size=min(num_removals, samples.shape[0]), replace=False)
        perturbed[idx] += rng.normal(0.0, noise_scale, size=perturbed[idx].shape)
        perturbed_prototypes = KMeans(n_clusters=k, n_init=10, random_state=0).fit(perturbed).cluster_centers_
        sensitivities.append(float(np.max(np.linalg.norm(prototypes - perturbed_prototypes, axis=1))))
    return float(np.max(sensitivities)) if sensitivities else 0.0


def calculate_graph_sensitivity(
    data_samples_list,
    device,
    num_neighbors,
    num_removals=1,
    noise_scale=0.1,
    random_state=None,
):
    rng = np.random.default_rng(random_state)
    sensitivities = []
    for data_samples in data_samples_list:
        data_samples = np.asarray(data_samples, dtype=np.float32)
        samples = data_samples.T if data_samples.shape[0] < data_samples.shape[1] else data_samples
        if samples.shape[0] <= 1:
            sensitivities.append(0.0)
            continue
        graph = cal_weights_via_can(torch.tensor(samples, dtype=torch.float32, device=device), num_neighbors)
        perturbed = samples.copy()
        idx = rng.choice(samples.shape[0], size=min(num_removals, samples.shape[0]), replace=False)
        perturbed[idx] += rng.normal(0.0, noise_scale, size=perturbed[idx].shape)
        perturbed_graph = cal_weights_via_can(torch.tensor(perturbed, dtype=torch.float32, device=device), num_neighbors)
        sensitivities.append(float(torch.max(torch.abs(graph - perturbed_graph)).detach().cpu()))
    return float(np.max(sensitivities)) if sensitivities else 0.0


def allocate_privacy_budget(class_counter, total_epsilon, min_ratio=0.2):
    class_counter = np.asarray(class_counter, dtype=np.float64)
    if class_counter.size == 0:
        return np.empty((0,), dtype=np.float64)

    if total_epsilon <= 0 or class_counter.sum() <= 0:
        return np.ones_like(class_counter) * (1.0 / class_counter.size)

    weights = class_counter / class_counter.sum()
    epsilon = weights * total_epsilon
    min_epsilon = total_epsilon * float(min_ratio) / class_counter.size
    low = epsilon < min_epsilon
    if low.any():
        epsilon[low] = min_epsilon
        remaining = max(total_epsilon - epsilon.sum(), 0.0)
        high = ~low
        if high.any():
            high_weights = weights[high] / weights[high].sum()
            epsilon[high] += remaining * high_weights
        else:
            epsilon[:] = total_epsilon / class_counter.size
    return epsilon


def laplace_noise_addition(true_values, sensitivity, epsilon, random_state=None):
    values = np.asarray(true_values, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    if values.size == 0:
        return values
    if epsilon.ndim == 0:
        scale = sensitivity / max(float(epsilon) * max(values.shape[-1], 1), 1e-12)
        return np.clip(values + rng.laplace(0, scale, size=values.shape), 0, 1)

    noisy = values.copy()
    for row in range(values.shape[0]):
        eps = max(float(epsilon[min(row, epsilon.size - 1)]) * values.shape[1], 1e-12)
        scale = sensitivity / eps
        noisy[row] += rng.laplace(0, scale, size=values.shape[1])
    return np.clip(noisy, 0, 1)


def add_noise_to_graph(graph, graph_sensitivity, epsilon_graph, random_state=None):
    graph = np.asarray(graph, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    scale = graph_sensitivity / max(float(epsilon_graph) * max(graph.shape[0], 1), 1e-12)
    return np.clip(graph + rng.laplace(0, scale, size=graph.shape), 0, 1)


def match_clusters_greedy(means):
    if not means:
        return {}

    reference = np.asarray(means[0])
    matches = {0: {idx: idx for idx in range(len(reference))}}
    for client_id, client_means in enumerate(means[1:], start=1):
        client_means = np.asarray(client_means)
        if len(client_means) == 0 or len(reference) == 0:
            matches[client_id] = {}
            continue

        distance_matrix = cdist(client_means, reference, metric="euclidean")
        matched = [-1] * len(client_means)
        used = set()
        while True:
            best = (float("inf"), -1, -1)
            for row in range(len(client_means)):
                if matched[row] != -1:
                    continue
                for col in range(len(reference)):
                    if col in used:
                        continue
                    if distance_matrix[row, col] < best[0]:
                        best = (distance_matrix[row, col], row, col)
            if best[1] == -1:
                break
            matched[best[1]] = best[2]
            used.add(best[2])
        matches[client_id] = {idx: value for idx, value in enumerate(matched) if value != -1}
    return matches


def build_global_similarity_graph(local_graphs, pseudo_labels, matches, cross_cluster_weight=0.03):
    sizes = [len(labels) for labels in pseudo_labels]
    total = int(sum(sizes))
    global_graph = np.zeros((total, total), dtype=np.float32)

    offsets = np.cumsum([0] + sizes)
    for client_id, local_graph in enumerate(local_graphs):
        start, end = offsets[client_id], offsets[client_id + 1]
        global_graph[start:end, start:end] = np.asarray(local_graph, dtype=np.float32)

    for src_client, src_labels in enumerate(pseudo_labels):
        src_start, src_end = offsets[src_client], offsets[src_client + 1]
        src_match = matches.get(src_client, {})
        for dst_client, dst_labels in enumerate(pseudo_labels):
            if src_client == dst_client:
                continue
            dst_start, dst_end = offsets[dst_client], offsets[dst_client + 1]
            dst_match = matches.get(dst_client, {})
            block = np.zeros((len(src_labels), len(dst_labels)), dtype=np.float32)
            for i, src_label in enumerate(src_labels):
                src_global = src_match.get(int(src_label))
                for j, dst_label in enumerate(dst_labels):
                    if src_global is not None and src_global == dst_match.get(int(dst_label)):
                        block[i, j] = cross_cluster_weight
            global_graph[src_start:src_end, dst_start:dst_end] = block

    return np.maximum(global_graph, global_graph.T)


def global_s_slices(global_s, sizes, device):
    offsets = np.cumsum([0] + list(map(int, sizes)))
    output = []
    for idx in range(len(sizes)):
        start, end = offsets[idx], offsets[idx + 1]
        output.append(global_s[start:end, start:end].detach().clone().to(device))
    return output


def dense_graph_to_edge_split(
    data,
    graph,
    device,
    train_mask=None,
    val_mask=None,
    test_mask=None,
    top_k=None,
    threshold=0.0,
    keep_original_edges=True,
):
    graph = torch.as_tensor(graph, dtype=torch.float32, device=device)
    graph = graph.clone()
    graph.fill_diagonal_(0)
    if top_k is not None and top_k > 0 and graph.size(0) > 0:
        k = min(int(top_k), max(graph.size(1) - 1, 1))
        sparse_mask = torch.zeros_like(graph, dtype=torch.bool)
        values, indices = torch.topk(graph, k=k, dim=1)
        sparse_mask.scatter_(1, indices, values > threshold)
        sparse_mask = sparse_mask | sparse_mask.t()
    else:
        sparse_mask = graph > threshold

    edge_index = sparse_mask.nonzero(as_tuple=False).t().contiguous().long()
    if keep_original_edges and hasattr(data, "edge_index") and data.edge_index is not None:
        original_edges = data.edge_index.to(device).long()
        if edge_index.numel() == 0:
            edge_index = original_edges
        else:
            edge_index = torch.cat([original_edges, edge_index], dim=1)

    if edge_index.numel() > 0:
        edge_index = torch.unique(edge_index.t(), dim=0).t().contiguous()
    else:
        edge_index = data.edge_index.to(device)

    target_data = copy.copy(data)
    target_data.edge_index = edge_index.to(device)
    train_mask = train_mask if train_mask is not None else data.train_mask
    val_mask = val_mask if val_mask is not None else data.val_mask
    test_mask = test_mask if test_mask is not None else data.test_mask
    return {
        "data": target_data,
        "train_mask": train_mask.to(device),
        "val_mask": val_mask.to(device),
        "test_mask": test_mask.to(device),
    }


def supervised_pseudo_labels(features, labels, num_clusters, train_ratio=0.1, random_state=0):
    features_np = features.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)
    n_clusters = max(1, min(int(num_clusters), len(np.unique(labels_np))))
    try:
        from sklearn.svm import SVC

        count = max(n_clusters, int(features_np.shape[0] * train_ratio))
        count = min(count, features_np.shape[0])
        clf = SVC().fit(features_np[:count], labels_np[:count])
        pred = clf.predict(features_np)
    except Exception:
        pred, _ = cluster_embeddings(features, n_clusters, random_state=random_state)
    return np.asarray(pred, dtype=int)


def label_order_reallocate(features, labels, pseudo_labels, global_pseudo_labels, num_clusters):
    order = []
    for class_id in range(num_clusters):
        idx = torch.where(labels == class_id)[0].detach().cpu().tolist()
        order.extend(idx)
    if len(order) != labels.numel():
        seen = set(order)
        order.extend([idx for idx in range(labels.numel()) if idx not in seen])
    order_tensor = torch.tensor(order, dtype=torch.long, device=features.device)
    return (
        features[order_tensor],
        labels[order_tensor],
        np.asarray(pseudo_labels, dtype=int)[order],
        np.asarray(global_pseudo_labels, dtype=int)[order],
        order,
    )
