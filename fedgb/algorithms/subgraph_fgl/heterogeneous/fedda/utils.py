import random

import torch


def _clone_state(state):
    return {name: value.detach().clone().cpu() for name, value in state.items()}


def _is_float_tensor(value):
    return torch.is_tensor(value) and torch.is_floating_point(value)


def select_dynamic_param_names(state, keywords=None):
    names = []
    keywords = [keyword for keyword in (keywords or []) if keyword]
    for name, value in state.items():
        if not _is_float_tensor(value):
            continue
        if not keywords or any(keyword in name for keyword in keywords):
            names.append(name)
    return set(names)


def dynamic_parameter_aggregate(
    server_state,
    client_states,
    client_ids,
    returned_param_names,
    dynamic_names=None,
    aggregate_all=False,
):
    if not client_states:
        return _clone_state(server_state), dict(returned_param_names), {"updated_params": 0}

    dynamic_names = set(dynamic_names or select_dynamic_param_names(server_state))
    aggregated = {}
    next_returned = {
        client_id: set(returned_param_names.get(client_id, dynamic_names))
        for client_id in client_ids
    }
    updated_params = 0

    for name, server_value in server_state.items():
        server_value = server_value.detach().cpu()
        if not _is_float_tensor(server_value):
            aggregated[name] = server_value.clone()
            continue

        deltas = []
        candidate_client_ids = []
        for client_id, client_state in zip(client_ids, client_states):
            if name not in client_state:
                continue
            if name in dynamic_names and name not in returned_param_names.get(client_id, dynamic_names):
                continue
            client_value = client_state[name].detach().cpu()
            if client_value.shape != server_value.shape or not _is_float_tensor(client_value):
                continue
            deltas.append(client_value - server_value)
            candidate_client_ids.append(client_id)

        if not deltas:
            aggregated[name] = server_value.clone()
            continue

        if name not in dynamic_names or aggregate_all:
            selected_deltas = deltas
            selected_client_ids = candidate_client_ids
        else:
            magnitudes = torch.tensor([delta.abs().mean().item() for delta in deltas])
            threshold = magnitudes.mean()
            selected_deltas = []
            selected_client_ids = []
            for delta, magnitude, client_id in zip(deltas, magnitudes, candidate_client_ids):
                if magnitude > threshold:
                    selected_deltas.append(delta)
                    selected_client_ids.append(client_id)
                else:
                    next_returned.setdefault(client_id, set()).discard(name)

            if not selected_deltas:
                aggregated[name] = server_value.clone()
                continue

        avg_delta = torch.stack(selected_deltas).mean(dim=0)
        aggregated[name] = server_value + avg_delta
        if name in dynamic_names:
            updated_params += 1

        if name in dynamic_names:
            for client_id in selected_client_ids:
                next_returned.setdefault(client_id, set()).add(name)

    return aggregated, next_returned, {"updated_params": updated_params}


def update_client_activity(
    client_ids,
    current_active,
    removed_clients,
    returned_param_names,
    all_param_names,
    remove_client=True,
    explore=True,
    active_rate=1.0,
    client_threshold=0.5,
):
    client_ids = list(client_ids)
    current_active = list(current_active)
    removed_clients = list(dict.fromkeys(removed_clients))
    all_param_names = set(all_param_names)
    total_params = max(len(all_param_names), 1)

    active = []
    for client_id in current_active:
        keep_ratio = len(returned_param_names.get(client_id, all_param_names)) / total_params
        if remove_client and keep_ratio <= client_threshold:
            if client_id not in removed_clients:
                removed_clients.append(client_id)
        else:
            active.append(client_id)

    if explore and removed_clients:
        target_count = max(1, int(len(client_ids) * active_rate))
        needed = max(0, target_count - len(active))
        if needed > 0:
            rejoin = removed_clients[:needed]
            for client_id in rejoin:
                returned_param_names[client_id] = set(all_param_names)
                active.append(client_id)
                removed_clients.remove(client_id)

    active = sorted(set(active))
    removed_clients = sorted(set(removed_clients))
    return active, removed_clients, returned_param_names
