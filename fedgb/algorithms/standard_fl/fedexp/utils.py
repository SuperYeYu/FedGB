import torch


def parameter_list_to_vector(params, device):
    vectors = [param.detach().to(device).reshape(-1) for param in params]
    if not vectors:
        return torch.empty(0, device=device)
    return torch.cat(vectors)


def vector_to_parameter_list(vector, reference_params):
    params = []
    offset = 0
    for reference in reference_params:
        numel = reference.numel()
        params.append(vector[offset: offset + numel].view_as(reference).clone())
        offset += numel
    if offset != vector.numel():
        raise ValueError("FedExP vector size does not match model parameters.")
    return params


def sample_weight_tensor(sample_counts, device):
    counts = torch.tensor(sample_counts, dtype=torch.float32, device=device)
    total = counts.sum().clamp_min(1e-12)
    return counts / total


def fedexp_adaptive_eta(
    avg_delta,
    client_deltas,
    weights,
    num_sampled_clients,
    epsilon,
    min_eta=1.0,
    max_eta=None,
):
    grad_norm_avg = torch.stack(
        [weight * torch.linalg.norm(delta).pow(2) for weight, delta in zip(weights, client_deltas)]
    ).sum()
    avg_delta_norm_sq = torch.linalg.norm(avg_delta).pow(2)
    eta_g = 0.5 * grad_norm_avg / (
        avg_delta_norm_sq + float(num_sampled_clients) * float(epsilon)
    ).clamp_min(1e-12)
    if min_eta is not None:
        eta_g = torch.maximum(eta_g, eta_g.new_tensor(float(min_eta)))
    if max_eta is not None:
        eta_g = torch.minimum(eta_g, eta_g.new_tensor(float(max_eta)))
    return eta_g


def fedexp_aggregate_parameters(
    global_params,
    client_params,
    sample_counts,
    epsilon,
    min_eta,
    max_eta,
    device,
):
    global_vector = parameter_list_to_vector(global_params, device)
    client_vectors = [
        parameter_list_to_vector(params, device) for params in client_params
    ]
    weights = sample_weight_tensor(sample_counts, device)
    client_deltas = [client_vector - global_vector for client_vector in client_vectors]
    stacked_deltas = torch.stack(client_deltas, dim=0)
    avg_delta = torch.sum(weights.view(-1, 1) * stacked_deltas, dim=0)
    eta_g = fedexp_adaptive_eta(
        avg_delta=avg_delta,
        client_deltas=client_deltas,
        weights=weights,
        num_sampled_clients=len(client_deltas),
        epsilon=epsilon,
        min_eta=min_eta,
        max_eta=max_eta,
    )
    updated_vector = global_vector + eta_g * avg_delta
    stats = {
        "eta_g": eta_g.detach(),
        "avg_delta_norm_sq": torch.linalg.norm(avg_delta).pow(2).detach(),
        "client_delta_norm_avg": torch.stack(
            [weights[idx] * torch.linalg.norm(delta).pow(2) for idx, delta in enumerate(client_deltas)]
        ).sum().detach(),
    }
    return vector_to_parameter_list(updated_vector, global_params), eta_g.detach(), stats
