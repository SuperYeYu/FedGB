import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops


def compute_class_prototypes(embedding, labels, mask, num_classes):
    device = embedding.device
    feature_dim = embedding.shape[-1]
    prototypes = torch.zeros(num_classes, feature_dim, device=device, dtype=embedding.dtype)
    counts = torch.zeros(num_classes, device=device, dtype=embedding.dtype)

    labels = labels.to(device)
    mask = mask.to(device).bool()
    for class_id in range(num_classes):
        selected = mask & (labels == class_id)
        count = selected.sum()
        if count > 0:
            prototypes[class_id] = embedding[selected].mean(dim=0)
            counts[class_id] = count.to(dtype=embedding.dtype)
    return prototypes, counts


def aggregate_class_prototypes(
    local_payloads,
    num_classes,
    feature_dim,
    device,
    mode="sample_weighted",
):
    dtype = local_payloads[0][0].dtype if local_payloads else torch.float32
    prototype_sum = torch.zeros(num_classes, feature_dim, device=device, dtype=dtype)
    count_sum = torch.zeros(num_classes, device=device, dtype=dtype)

    for prototypes, counts in local_payloads:
        prototypes = prototypes.to(device)
        counts = counts.to(device, dtype=dtype)
        observed = counts > 0
        if mode == "client_mean":
            prototype_sum[observed] += prototypes[observed]
            count_sum[observed] += 1.0
        elif mode == "sample_weighted":
            prototype_sum += prototypes * counts.unsqueeze(-1)
            count_sum += counts
        else:
            raise ValueError("mode must be 'sample_weighted' or 'client_mean'.")

    aggregated = torch.zeros_like(prototype_sum)
    observed = count_sum > 0
    aggregated[observed] = prototype_sum[observed] / count_sum[observed].unsqueeze(-1)
    return aggregated


def initialize_prototype_representations(train_mask, global_prototypes, labels, num_nodes=None):
    device = global_prototypes.device
    if num_nodes is None:
        num_nodes = train_mask.numel()
    proto_reps = torch.zeros(
        num_nodes,
        global_prototypes.shape[-1],
        device=device,
        dtype=global_prototypes.dtype,
    )

    train_mask = train_mask.to(device).bool()
    labels = labels.to(device)
    if train_mask.any():
        proto_reps[train_mask] = global_prototypes[labels[train_mask]]
    return proto_reps


def prototype_propagation(proto_reps, edge_index, num_layers, alpha):
    edge_index = edge_index.to(proto_reps.device)
    edge_index, _ = add_self_loops(edge_index, num_nodes=proto_reps.shape[0])
    edge_index, edge_weight = gcn_norm(
        edge_index,
        num_nodes=proto_reps.shape[0],
        add_self_loops=False,
    )

    source, target = edge_index
    residual = (1.0 - alpha) * proto_reps
    out = proto_reps
    for _ in range(num_layers):
        propagated = torch.zeros_like(out)
        messages = out[source] * edge_weight.to(out.device).unsqueeze(-1)
        propagated.index_add_(0, target, messages)
        out = alpha * propagated + residual
    return out


def ppn_loss(model, data, embedding, logits, labels, mask, global_prototypes, config):
    if global_prototypes is None or global_prototypes.numel() == 0:
        return F.cross_entropy(logits[mask], labels[mask])

    proto_seed = initialize_prototype_representations(
        mask,
        global_prototypes.to(embedding.device),
        labels,
        num_nodes=embedding.shape[0],
    )
    proto_embedding = prototype_propagation(
        proto_seed,
        data.edge_index,
        num_layers=config["fedppn_lp_layers"],
        alpha=config["fedppn_lp_alpha"],
    )
    prototype_logits = model.head(proto_embedding)
    ensemble_logits = (
        config["fedppn_private_weight"] * logits
        + (1.0 - config["fedppn_private_weight"]) * prototype_logits
    )

    ce_loss = F.cross_entropy(logits[mask], labels[mask])
    ppn_ce_loss = F.cross_entropy(prototype_logits[mask], labels[mask])
    ensemble_loss = F.cross_entropy(ensemble_logits[mask], labels[mask])
    return (
        config["fedppn_loss_ce_weight"] * ce_loss
        + config["fedppn_loss_ppn_weight"] * ppn_ce_loss
        + config["fedppn_loss_ensemble_weight"] * ensemble_loss
    )
