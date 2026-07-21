import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


def classwise_clean_mask(losses, labels, scale=1.0):
    clean = torch.zeros_like(losses, dtype=torch.bool)
    for class_id in torch.unique(labels):
        selected = labels == class_id
        class_losses = losses[selected]
        if class_losses.numel() == 0:
            continue
        std = class_losses.std(unbiased=False) if class_losses.numel() > 1 else torch.zeros((), device=losses.device)
        threshold = class_losses.mean() + scale * std
        clean[selected] = class_losses < threshold
    return clean


def two_stage_clean_noisy_split(
    train_mask,
    labels,
    stage1_losses,
    propagated_logits,
    stage1_scale=0.5,
    stage2_scale=0.5,
    only_stage1=False,
):
    train_idx = mask_indices(train_mask)
    if train_idx.numel() == 0:
        empty = torch.zeros_like(train_mask, dtype=torch.bool)
        return empty, empty

    if labels.shape[0] == train_mask.shape[0]:
        train_labels = labels[train_idx]
    else:
        train_labels = labels

    stage1_clean = classwise_clean_mask(stage1_losses, train_labels, scale=stage1_scale)
    if only_stage1:
        final_clean = stage1_clean
    else:
        if propagated_logits.shape[0] == train_mask.shape[0]:
            propagated_train = propagated_logits[train_idx]
        else:
            propagated_train = propagated_logits
        stage2_losses = F.cross_entropy(propagated_train, train_labels, reduction="none")
        stage2_clean = classwise_clean_mask(stage2_losses, train_labels, scale=stage2_scale)
        final_clean = stage1_clean & stage2_clean

    clean_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    clean_mask[train_idx] = final_clean
    noisy_mask = train_mask & ~clean_mask
    return clean_mask, noisy_mask


def label_propagation_soft_labels(
    edge_index,
    soft_labels,
    train_mask,
    high_mask=None,
    num_nodes=None,
    prop_steps=15,
    alpha=0.5,
):
    if num_nodes is None:
        num_nodes = soft_labels.shape[0]

    train_mask = train_mask.to(device=soft_labels.device, dtype=torch.bool)
    initial = soft_labels.detach().clone()
    if high_mask is not None:
        high_mask = high_mask.to(device=soft_labels.device, dtype=torch.bool)
        if high_mask.shape[0] != num_nodes:
            full_high = torch.zeros(num_nodes, dtype=torch.bool, device=soft_labels.device)
            full_high[mask_indices(train_mask)] = high_mask
            high_mask = full_high
        high_train = train_mask & high_mask
        if high_train.any():
            initial[high_train] = F.one_hot(
                initial[high_train].argmax(dim=-1),
                num_classes=initial.shape[1],
            ).to(dtype=initial.dtype)

    if edge_index.numel() == 0 or prop_steps <= 0:
        return _normalize_soft_labels(initial, soft_labels)

    edge_index = edge_index.to(soft_labels.device)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    keep = train_mask[edge_index[0]] & train_mask[edge_index[1]]
    edge_index = edge_index[:, keep]
    if edge_index.numel() == 0:
        return _normalize_soft_labels(initial, soft_labels)

    values = torch.ones(edge_index.shape[1], dtype=soft_labels.dtype, device=soft_labels.device)
    degree = torch.zeros(num_nodes, dtype=soft_labels.dtype, device=soft_labels.device)
    degree.index_add_(0, edge_index[0], values)
    values = values / degree[edge_index[0]].clamp_min(1.0)
    adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), device=soft_labels.device)

    propagated = initial
    for _ in range(prop_steps):
        propagated = (1.0 - alpha) * torch.sparse.mm(adj, propagated) + alpha * initial
    return _normalize_soft_labels(propagated, soft_labels)


def _normalize_soft_labels(labels, fallback):
    labels = labels.clamp_min(0.0)
    row_sum = labels.sum(dim=-1, keepdim=True)
    normalized = labels / row_sum.clamp_min(1e-12)
    fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(row_sum > 0, normalized, fallback)


def entropy_aggregation_weights(entropies, device):
    values = []
    for entropy in entropies:
        if torch.is_tensor(entropy):
            values.append(float(entropy.detach().cpu()))
        else:
            values.append(float(entropy))
    entropy = torch.as_tensor(values, dtype=torch.float32, device=device)
    inv_entropy = 1.0 / entropy.clamp_min(1e-9)
    total = inv_entropy.sum()
    if total <= 0 or torch.isnan(total):
        return torch.ones_like(inv_entropy) / max(inv_entropy.numel(), 1)
    return inv_entropy / total


def calculate_entropy(logits):
    if logits.numel() == 0:
        return torch.zeros((), device=logits.device)
    probabilities = F.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities.clamp_min(1e-9))
    return -(probabilities * log_probabilities).sum(dim=-1).mean()


def jensen_shannon_divergence(logits_p, logits_q):
    if logits_p.numel() == 0 or logits_q.numel() == 0:
        return logits_p.sum() * 0.0 + logits_q.sum() * 0.0
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(torch.log(p.clamp_min(1e-9)), m, reduction="batchmean")
        + F.kl_div(torch.log(q.clamp_min(1e-9)), m, reduction="batchmean")
    )


def contrastive_loss(z1, z2, temperature, max_nodes=None):
    if z1.numel() == 0 or z1.shape[0] <= 1:
        return z1.sum() * 0.0
    if max_nodes is not None and int(max_nodes) > 0 and z1.shape[0] > int(max_nodes):
        perm = torch.randperm(z1.shape[0], device=z1.device)[: int(max_nodes)]
        z1 = z1[perm]
        z2 = z2[perm]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    refl_sim = torch.exp(torch.mm(z1, z1.t()) / temperature)
    between_sim = torch.exp(torch.mm(z1, z2.t()) / temperature)
    numerator = between_sim.diag()
    denominator = refl_sim.sum(dim=1) + between_sim.sum(dim=1) - refl_sim.diag()
    return -torch.log((numerator / denominator.clamp_min(1e-12)).clamp_min(1e-12)).mean()


def drop_feature(x, drop_prob):
    if drop_prob <= 0:
        return x
    mask = torch.rand(x.shape[1], device=x.device) < drop_prob
    dropped = x.clone()
    dropped[:, mask] = 0
    return dropped


def drop_edge(edge_index, drop_prob):
    if drop_prob <= 0 or edge_index.numel() == 0:
        return edge_index
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) > drop_prob
    if keep.sum() == 0:
        keep[torch.randint(edge_index.shape[1], (1,), device=edge_index.device)] = True
    return edge_index[:, keep]


def prox_loss(local_model, global_weights, device):
    if global_weights is None:
        return torch.zeros((), device=device)
    total = torch.zeros((), device=device)
    for local_param, global_param in zip(local_model.parameters(), global_weights):
        total = total + torch.norm(local_param - global_param.to(device)).pow(2)
    return torch.sqrt(total.clamp_min(1e-12))


def mask_indices(mask):
    return mask.nonzero(as_tuple=True)[0]
