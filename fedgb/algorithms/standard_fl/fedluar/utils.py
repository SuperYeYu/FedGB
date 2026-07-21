import math

import numpy as np
import torch


def kernel_parameter_indices(parameters):
    return [idx for idx, param in enumerate(parameters) if param.ndim > 1]


def weighted_average_parameters(client_parameters, sample_counts, device):
    total = float(sum(sample_counts))
    averaged = []
    for param_idx in range(len(client_parameters[0])):
        value = None
        for params, count in zip(client_parameters, sample_counts):
            weight = float(count) / total
            local = params[param_idx].detach().to(device)
            value = local * weight if value is None else value + local * weight
        averaged.append(value)
    return averaged


def apply_layerwise_update_recycling(averaged_params, prev_params, prev_updates, recycling_indices):
    result = []
    updates = []
    scores = {}
    for idx, averaged_param in enumerate(averaged_params):
        if averaged_param.ndim > 1 and idx in recycling_indices:
            new_param = prev_params[idx].to(averaged_param.device) + prev_updates[idx].to(averaged_param.device)
            update = prev_updates[idx].to(averaged_param.device)
        else:
            new_param = averaged_param
            update = new_param - prev_params[idx].to(averaged_param.device)
            if averaged_param.ndim > 1:
                denom = torch.norm(prev_params[idx].to(averaged_param.device)).clamp_min(1e-6)
                scores[idx] = float((torch.norm(update) / denom).detach().cpu())
        result.append(new_param.detach().clone())
        updates.append(update.detach().clone())
    return result, updates, scores


def inverse_score_probabilities(scores, candidate_indices):
    values = np.array([max(float(scores.get(idx, 0.0)), 1e-12) for idx in candidate_indices], dtype=np.float64)
    inverse = np.reciprocal(values)
    total = inverse.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(candidate_indices), dtype=np.float64) / max(len(candidate_indices), 1)
    return inverse / total


def select_recycling_layers(scores, kernel_indices, num_recycling_layers, rng=None):
    if rng is None:
        rng = np.random
    if num_recycling_layers <= 0 or not kernel_indices:
        return set()
    count = min(int(num_recycling_layers), len(kernel_indices))
    probabilities = inverse_score_probabilities(scores, kernel_indices)
    selected = rng.choice(np.array(kernel_indices), size=count, replace=False, p=probabilities)
    return set(int(idx) for idx in selected)


def num_recycling_layers_from_ratio(num_kernels, recycling_ratio):
    if num_kernels <= 0 or recycling_ratio <= 0:
        return 0
    return min(num_kernels, max(1, int(math.floor(num_kernels * recycling_ratio))))
