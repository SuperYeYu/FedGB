from collections import OrderedDict

import torch


def weighted_average_state_dicts(state_dicts, weights, shared_keys=None):
    """Average state_dict tensors while tolerating missing heterogeneous schema keys."""
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")
    if not state_dicts:
        return OrderedDict()

    keys = list(shared_keys) if shared_keys is not None else _ordered_union_keys(state_dicts)
    total_input_weight = float(sum(weights))
    if total_input_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    norm_weights = [float(weight) / total_input_weight for weight in weights]

    averaged = OrderedDict()
    for key in keys:
        available = [
            (state_dict[key], norm_weight)
            for state_dict, norm_weight in zip(state_dicts, norm_weights)
            if key in state_dict
        ]
        if not available:
            continue

        first_tensor = available[0][0].detach()
        if not (torch.is_floating_point(first_tensor) or torch.is_complex(first_tensor)):
            averaged[key] = first_tensor.clone()
            continue

        available_weight = sum(weight for _, weight in available)
        if available_weight <= 0:
            raise ValueError(f"available weights for key {key} must be positive")

        value = torch.zeros_like(first_tensor)
        for tensor, weight in available:
            value = value + tensor.detach().to(first_tensor.device) * (weight / available_weight)
        averaged[key] = value

    return averaged


def filter_private_state_dict(state_dict, ablation=None, is_encoder=True):
    """Remove FedHGN private parameters before server aggregation."""
    coeff_prefix = "basis_coeffs_encoder" if is_encoder else "basis_coeffs_decoder"
    filtered = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("embed_layer"):
            continue
        if key.startswith("feature_proj"):
            continue
        if (ablation is None or ablation == "B") and key.startswith(coeff_prefix):
            continue
        if ablation == "C" and "bases" in key:
            continue
        filtered[key] = value.detach().clone() if torch.is_tensor(value) else value
    return filtered


def basis_alignment_regularization(local_coeffs, other_coeffs):
    """Match each local relation basis coefficient to its nearest external relation."""
    if other_coeffs is None:
        return local_coeffs.new_tensor(0.0)

    if local_coeffs.numel() == 0 or other_coeffs.numel() == 0:
        return local_coeffs.new_tensor(0.0)

    diff = local_coeffs.unsqueeze(2) - other_coeffs.to(local_coeffs.device).unsqueeze(1)
    min_diff, _ = torch.min(torch.sum(torch.square(diff), dim=-1), dim=-1)
    return min_diff.sum()


def stack_basis_coefficients(basis_coeffs):
    """Convert FedHGN coefficient containers into a dense tensor."""
    if basis_coeffs is None:
        return None
    if torch.is_tensor(basis_coeffs):
        return basis_coeffs.detach()
    if isinstance(basis_coeffs, dict):
        return torch.stack([param.detach() for param in basis_coeffs.values()], dim=0)
    return torch.stack(
        [torch.stack([param.detach() for param in param_dict.values()], dim=0) for param_dict in basis_coeffs],
        dim=0,
    )


def _ordered_union_keys(state_dicts):
    keys = []
    seen = set()
    for state_dict in state_dicts:
        for key in state_dict.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys
