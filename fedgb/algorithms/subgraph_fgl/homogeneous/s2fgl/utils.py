import math

import torch
import torch.nn.functional as F


def dense_adjacency(edge_index, num_nodes, device=None, dtype=torch.float32):
    device = device or edge_index.device
    adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    if edge_index.numel() > 0:
        src, dst = edge_index.to(device)
        adj[src, dst] = 1.0
    return adj


def row_normalize(matrix, eps=1e-12):
    degree = matrix.sum(dim=1, keepdim=True).clamp_min(eps)
    return matrix / degree


def sparse_row_normalized_adjacency(edge_index, num_nodes, device=None, dtype=torch.float32, eps=1e-12):
    device = device or edge_index.device
    if edge_index.numel() > 0:
        edge_index = edge_index.to(device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    loops = torch.arange(num_nodes, dtype=torch.long, device=device)
    loop_index = torch.stack([loops, loops], dim=0)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    values = torch.ones(edge_index.size(1), dtype=dtype, device=device)
    row = edge_index[0]
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.scatter_add_(0, row, values)
    values = values / degree[row].clamp_min(eps)
    return torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=device).coalesce()


def build_s2fgl_adjs(data):
    device = data.x.device
    num_nodes = data.x.size(0)
    adj_low = sparse_row_normalized_adjacency(
        data.edge_index,
        num_nodes,
        device=device,
        dtype=data.x.dtype,
    )
    data.adj_low = adj_low
    data.adj_high = None
    if hasattr(data, "adj_low_un"):
        delattr(data, "adj_low_un")
    return data


def _degree_signal(edge_index, num_nodes, device, dtype):
    degree = torch.ones(num_nodes, dtype=dtype, device=device)
    if edge_index.numel() > 0:
        row = edge_index.to(device)[0]
        degree.scatter_add_(0, row, torch.ones(row.numel(), dtype=dtype, device=device))
    return degree


def pagerank_signal(edge_index, num_nodes, seed_signal, alpha=0.85, num_iter=20, device=None):
    device = device or edge_index.device
    seed_signal = seed_signal.to(device).float()
    if seed_signal.sum() <= 0:
        seed_signal = torch.ones(num_nodes, dtype=torch.float32, device=device)
    teleport = seed_signal / seed_signal.sum().clamp_min(1e-12)
    score = teleport.clone()
    adj = sparse_row_normalized_adjacency(
        edge_index,
        num_nodes,
        device=device,
        dtype=torch.float32,
    )
    adj_t = adj.transpose(0, 1).coalesce()
    for _ in range(max(1, int(num_iter))):
        score = alpha * teleport + (1.0 - alpha) * torch.sparse.mm(adj_t, score.view(-1, 1)).view(-1)
    return score


def personalized_pagerank(edge_index, num_nodes, alpha=0.85, eps=1e-6, device=None):
    device = device or edge_index.device
    scores = []
    eye = torch.eye(num_nodes, device=device)
    for seed in eye:
        scores.append(pagerank_signal(edge_index, num_nodes, seed, alpha=alpha, device=device))
    return torch.stack(scores, dim=1)


def select_important_nodes_lis(
    data,
    train_mask=None,
    alpha=0.85,
    ratio=1 / 3,
    device=None,
    num_iter=20,
    max_nodes=None,
):
    device = device or data.x.device
    edge_index = data.edge_index.to(device)
    if train_mask is None:
        train_mask = data.train_mask
    train_mask = train_mask.to(device).bool()
    num_nodes = data.x.size(0)
    train_signal = train_mask.float()

    lambda_s = pagerank_signal(
        edge_index,
        num_nodes,
        train_signal,
        alpha=alpha,
        num_iter=num_iter,
        device=device,
    )
    degree = _degree_signal(edge_index, num_nodes, device, torch.float32)
    lambda_l = degree / degree.sum().clamp_min(1e-12)
    salc = lambda_s + lambda_l

    top_k = max(1, int(math.ceil(num_nodes * ratio)))
    top_k = min(num_nodes, top_k)
    if max_nodes is not None and max_nodes > 0:
        top_k = min(top_k, int(max_nodes))
    _, top_indices = torch.topk(salc, top_k)
    return top_indices, data.y.to(device)[top_indices]


def select_important_nodes_lis_k(
    data,
    train_mask=None,
    alpha=0.85,
    ratio=0.2,
    device=None,
    num_iter=20,
    max_nodes=None,
):
    nodes, _ = select_important_nodes_lis(
        data,
        train_mask=train_mask,
        alpha=alpha,
        ratio=ratio,
        device=device,
        num_iter=num_iter,
        max_nodes=max_nodes,
    )
    return nodes


def compute_class_codebook_slot(node_features, important_nodes, important_labels, num_classes):
    feature_dim = node_features.size(-1)
    prototypes = torch.zeros(
        num_classes,
        feature_dim,
        device=node_features.device,
        dtype=node_features.dtype,
    )
    counts = torch.zeros(num_classes, device=node_features.device, dtype=node_features.dtype)
    for class_id in range(num_classes):
        selected = important_labels == class_id
        if selected.any():
            features = node_features[important_nodes[selected]]
            prototypes[class_id] = features.mean(dim=0)
            counts[class_id] = features.size(0)
    return prototypes, counts


def aggregate_s2fgl_codebook(payloads, num_classes, feature_dim, device, num_slots=4):
    codebook = torch.zeros(num_classes, num_slots, feature_dim, device=device)
    if not payloads:
        return codebook

    for slot_id in range(num_slots):
        proto_sum = torch.zeros(num_classes, feature_dim, device=device)
        count_sum = torch.zeros(num_classes, device=device)
        for payload in payloads:
            node_features = payload["node_features"].to(device)
            important_nodes = payload["important_nodes"].to(device)
            important_labels = payload["important_labels"].to(device)
            prototypes, counts = compute_class_codebook_slot(
                node_features,
                important_nodes,
                important_labels,
                num_classes,
            )
            proto_sum += prototypes * counts.unsqueeze(-1)
            count_sum += counts

        observed = count_sum > 0
        codebook[observed, slot_id] = proto_sum[observed] / count_sum[observed].unsqueeze(-1)
    return codebook


def federated_knowledge_distillation_loss(
    local_features,
    global_features,
    codebook_embeddings,
    temperature=1.0,
    lamb=10.0,
):
    if codebook_embeddings is None:
        return local_features.sum() * 0.0

    codebook = codebook_embeddings.to(local_features.device, dtype=local_features.dtype)
    codebook = codebook.reshape(-1, codebook.shape[-1])
    if codebook.numel() == 0 or codebook.shape[0] == 0:
        return local_features.sum() * 0.0

    valid = codebook.norm(dim=1) > 0
    if not valid.any():
        return local_features.sum() * 0.0
    codebook = codebook[valid]

    teacher_logits = F.cosine_similarity(
        global_features.detach().unsqueeze(1),
        codebook.unsqueeze(0),
        dim=-1,
    ) / temperature
    student_logits = F.cosine_similarity(
        local_features.unsqueeze(1),
        codebook.unsqueeze(0),
        dim=-1,
    ) / temperature
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature**2) * lamb


def _sparse_similarity_laplacian(features, top_k=3):
    num_nodes = features.size(0)
    if num_nodes == 0:
        return features.new_zeros((0, 0))

    k = min(top_k + 1, num_nodes)
    sim = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
    values, indices = sim.topk(k, dim=1)
    sparse_sim = torch.zeros_like(sim)
    sparse_sim.scatter_(1, indices, values)
    sparse_sim = 0.5 * (sparse_sim + sparse_sim.t())
    degree = sparse_sim.sum(dim=1)
    return torch.diag(degree) - sparse_sim


def _eigenvectors(laplacian, top_k):
    num_nodes = laplacian.size(0)
    if num_nodes <= 1:
        return laplacian.new_ones((num_nodes, 1)), laplacian.new_ones((num_nodes, 1))

    k = min(top_k, num_nodes)
    laplacian = laplacian.float()
    try:
        _, eigenvectors = torch.linalg.eigh(laplacian)
    except RuntimeError:
        stable_laplacian = 0.5 * (laplacian.detach().cpu().double() + laplacian.detach().cpu().double().t())
        scale = stable_laplacian.abs().max().clamp_min(1.0)
        eye = torch.eye(num_nodes, dtype=stable_laplacian.dtype, device=stable_laplacian.device)
        eigenvectors = None
        for jitter in (1e-10, 1e-8, 1e-6, 1e-4):
            try:
                _, eigenvectors = torch.linalg.eigh(stable_laplacian + eye * (scale * jitter))
                break
            except RuntimeError:
                continue
        if eigenvectors is None:
            eigenvectors = eye
        eigenvectors = eigenvectors.to(device=laplacian.device, dtype=laplacian.dtype)
    smallest = eigenvectors[:, :k].to(laplacian.device)
    largest = eigenvectors[:, -k:].to(laplacian.device)
    return smallest, largest


def _projection_energy(eigenvectors, features):
    return eigenvectors @ (eigenvectors.t() @ features)


def frequency_alignment_loss(local_features, global_features, top_k=10, similarity_top_k=3):
    if local_features.size(0) <= 1:
        return local_features.sum() * 0.0

    local_lap = _sparse_similarity_laplacian(local_features, top_k=similarity_top_k)
    global_lap = _sparse_similarity_laplacian(global_features.detach(), top_k=similarity_top_k)
    local_small, local_large = _eigenvectors(local_lap, top_k)
    global_small, global_large = _eigenvectors(global_lap, top_k)

    local_low = _projection_energy(local_small, local_features)
    global_low = _projection_energy(global_small, global_features.detach())
    local_high = _projection_energy(local_large, local_features)
    global_high = _projection_energy(global_large, global_features.detach())
    return F.mse_loss(local_low, global_low) + F.mse_loss(local_high, global_high)
