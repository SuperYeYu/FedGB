import torch


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def pairwise_angles(sources):
    similarities = torch.zeros((len(sources), len(sources)), dtype=torch.float32)
    for i, source_i in enumerate(sources):
        flat_i = flatten(source_i)
        for j, source_j in enumerate(sources):
            flat_j = flatten(source_j)
            denom = max(torch.norm(flat_i) * torch.norm(flat_j), torch.tensor(1e-12, device=flat_i.device))
            similarities[i, j] = torch.sum(flat_i * flat_j) / denom + 1.0
    return similarities


def update_norm(update):
    return torch.norm(flatten(update)).item()


def aggregate_cluster_updates(targets, sources):
    total_size = sum(size for _, size in sources)
    if total_size <= 0:
        return [{name: value.detach().clone() for name, value in target.items()} for target in targets]

    averaged_update = {}
    for name in sources[0][0]:
        averaged_update[name] = torch.div(
            torch.sum(torch.stack([source[name].detach() * size for source, size in sources]), dim=0),
            total_size,
        )

    updated_targets = []
    for target in targets:
        updated = {}
        for name, value in target.items():
            updated[name] = value.detach().clone() + averaged_update[name].to(value.device)
        updated_targets.append(updated)
    return updated_targets


def sync_cluster_weights_to_clients(cluster_indices, cluster_weights, clients):
    with torch.no_grad():
        for cluster_id, client_ids in enumerate(cluster_indices):
            for member_id, client_id in enumerate(client_ids):
                client = clients[client_id]
                weights = cluster_weights[cluster_id][member_id]
                for name, source in weights.items():
                    if name in client.W:
                        client.W[name].data.copy_(source.detach().to(client.W[name].device))
