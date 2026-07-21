import copy

import torch
import torch.nn.functional as F


def clone_state_dict(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def move_state_dict_to_device(state_dict, device):
    return {key: value.to(device) for key, value in state_dict.items()}


def aggregate_proxy_states(state_payloads, device):
    total = float(sum(num_samples for _, num_samples in state_payloads))
    if total <= 0:
        raise ValueError("FedSpray state aggregation requires positive sample counts.")

    aggregated = {}
    for state_dict, num_samples in state_payloads:
        weight = float(num_samples) / total
        for key, value in state_dict.items():
            value = value.to(device)
            if key not in aggregated:
                aggregated[key] = weight * value
            else:
                aggregated[key] += weight * value
    return aggregated


def aggregate_structure_proxies(proxy_payloads, num_classes, proxy_dim, device, previous=None):
    proxy_sum = torch.zeros(num_classes, proxy_dim, device=device)
    count_sum = torch.zeros(num_classes, device=device)

    for local_proxy, local_counts in proxy_payloads:
        proxy_sum += local_proxy.to(device) * local_counts.to(device).view(-1, 1)
        count_sum += local_counts.to(device)

    aggregated = torch.zeros(num_classes, proxy_dim, device=device)
    observed = count_sum > 0
    aggregated[observed] = proxy_sum[observed] / count_sum[observed].view(-1, 1)

    if previous is not None:
        aggregated[~observed] = previous.to(device)[~observed]

    return aggregated


def build_proxy_teacher_logits(encoder, classifier, x, logits, global_proxy, node_proxy=None):
    class_prob = F.softmax(logits, dim=1)
    weighted_proxy = class_prob @ global_proxy.to(x.device)
    proxy_for_classifier = weighted_proxy if node_proxy is None else node_proxy.to(x.device)
    proxy_embedding = encoder(x)
    teacher_logits = classifier(proxy_embedding, proxy_for_classifier)
    return teacher_logits, weighted_proxy


def compute_local_structure_proxies(proxy, labels, train_mask, num_classes):
    train_idx = train_mask.nonzero(as_tuple=True)[0]
    proxy_dim = proxy.size(1)
    local_proxy = torch.zeros(num_classes, proxy_dim, device=proxy.device)
    local_counts = torch.zeros(num_classes, device=proxy.device)

    for class_id in range(num_classes):
        class_idx = train_idx[labels[train_idx] == class_id]
        local_counts[class_id] = float(class_idx.numel())
        if class_idx.numel() > 0:
            local_proxy[class_id] = proxy[class_idx].mean(dim=0)

    return local_proxy.detach(), local_counts.detach()
