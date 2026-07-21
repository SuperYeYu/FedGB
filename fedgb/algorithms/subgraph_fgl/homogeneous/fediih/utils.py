from collections import OrderedDict

import torch


def gaussian_kl(mu_a, logvar_a, mu_b, logvar_b):
    var_a = torch.exp(logvar_a)
    var_b = torch.exp(logvar_b)
    return 0.5 * torch.sum(logvar_b - logvar_a + (var_a + (mu_a - mu_b).pow(2)) / var_b - 1)


def gaussian_js(mu_a, logvar_a, mu_b, logvar_b):
    var_a = torch.exp(logvar_a)
    var_b = torch.exp(logvar_b)
    mu_m = 0.5 * (mu_a + mu_b)
    var_m = 0.5 * (var_a + var_b)
    logvar_m = torch.log(var_m.clamp_min(1e-12))
    return 0.5 * (
        gaussian_kl(mu_a, logvar_a, mu_m, logvar_m)
        + gaussian_kl(mu_b, logvar_b, mu_m, logvar_m)
    )


def js_similarity_matrix(mu, logvar, norm="exp", norm_scale=1.0):
    n = mu.shape[0]
    sim = torch.empty(n, n, dtype=mu.dtype, device=mu.device)
    for i in range(n):
        for j in range(n):
            js = gaussian_js(mu[i], logvar[i], mu[j], logvar[j])
            sim[i, j] = 1.0 - js / torch.log(torch.tensor(2.0, dtype=mu.dtype, device=mu.device))
    if norm == "exp":
        sim = torch.exp(norm_scale * sim)
    return sim / sim.sum(dim=1, keepdim=True).clamp_min(1e-12)


def weighted_average_state(states, weights):
    aggregated = OrderedDict()
    for key in states[0].keys():
        value = None
        for state, weight in zip(states, weights):
            if key not in state:
                continue
            cur = state[key].detach() * weight.to(state[key].device, dtype=state[key].dtype)
            value = cur.clone() if value is None else value + cur
        if value is not None:
            aggregated[key] = value
    return aggregated


def aggregate_global_priors(semantic_mu, structure_mu):
    denom = semantic_mu.shape[0] + 0.25
    beta_mu = semantic_mu.sum(dim=0) / denom
    alpha_mu = structure_mu.sum(dim=0) / denom
    return alpha_mu.detach().clone(), beta_mu.detach().clone()


def aggregate_personalized_states(states, semantic_weights, structure_weights, num_factors=2):
    aggregated = OrderedDict()
    for key in states[0].keys():
        values = [state[key].detach() for state in states if key in state]
        if not values:
            continue
        if key in {"pca.weight", "clf.weight"} and values[0].dim() >= 2 and values[0].shape[1] % num_factors == 0:
            semantic_parts = []
            structure_parts = []
            for value, w_sem, w_str in zip(values, semantic_weights, structure_weights):
                left, right = torch.chunk(value, chunks=num_factors, dim=1)
                semantic_parts.append(left * w_sem.to(value.device, dtype=value.dtype))
                structure_parts.append(right * w_str.to(value.device, dtype=value.dtype))
            aggregated[key] = torch.cat([sum(semantic_parts), sum(structure_parts)], dim=1)
        elif key == "pca.bias" and values[0].dim() == 1 and values[0].shape[0] % num_factors == 0:
            semantic_parts = []
            structure_parts = []
            for value, w_sem, w_str in zip(values, semantic_weights, structure_weights):
                left, right = torch.chunk(value, chunks=num_factors, dim=0)
                semantic_parts.append(left * w_sem.to(value.device, dtype=value.dtype))
                structure_parts.append(right * w_str.to(value.device, dtype=value.dtype))
            aggregated[key] = torch.cat([sum(semantic_parts), sum(structure_parts)], dim=0)
        elif key == "clf.bias":
            weight = torch.ones(len(values), device=values[0].device, dtype=values[0].dtype) / len(values)
            aggregated[key] = sum(value * w for value, w in zip(values, weight))
        else:
            weights = (semantic_weights + structure_weights) / 2
            aggregated[key] = sum(
                value * weights[idx].to(value.device, dtype=value.dtype)
                for idx, value in enumerate(values)
            )
    return aggregated


def summarize_tensor_distribution(values, latent_dim):
    values = values.detach().float().flatten()
    if values.numel() == 0:
        values = torch.zeros(1)
    mean = values.mean()
    var = values.var(unbiased=False).clamp_min(1e-6)
    mu = torch.full((latent_dim,), mean.item(), dtype=torch.float32)
    logvar = torch.full((latent_dim,), torch.log(var).item(), dtype=torch.float32)
    return mu, logvar


def summarize_node_structure(edge_index, num_nodes, latent_dim):
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() > 0:
        degree.scatter_add_(0, edge_index[0].long(), torch.ones(edge_index.shape[1], device=edge_index.device))
    return summarize_tensor_distribution(degree.cpu(), latent_dim)


def summarize_hvae_distribution(z_mu, z_logvar):
    return z_mu.detach().mean(dim=0).cpu(), z_logvar.detach().mean(dim=0).cpu()
