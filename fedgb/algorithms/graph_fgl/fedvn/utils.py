import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


def weighted_average_state_dicts(state_dicts, weights, device):
    if not state_dicts:
        return {}
    averaged = {}
    for name in state_dicts[0]:
        total = torch.zeros_like(state_dicts[0][name], device=device)
        for state, weight in zip(state_dicts, weights):
            total = total + float(weight) * state[name].to(device)
        averaged[name] = total
    return averaged


def is_private_prediction_head(name, task_name):
    return task_name == "graph_reg" and name.startswith("classifier.")


def shared_parameter_names(model, task_name=None):
    return [
        name
        for name, _ in model.named_parameters()
        if not is_private_prediction_head(name, task_name)
    ]


def shared_parameter_payload(model, task_name=None):
    names = shared_parameter_names(model, task_name)
    named_params = dict(model.named_parameters())
    return names, [named_params[name] for name in names]


def load_shared_parameters(model, names, weights, device):
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, weight in zip(names, weights):
            if name in named_params:
                named_params[name].data.copy_(weight.to(device))


def virtual_node_decorrelation_loss(vn_embedding):
    centered = vn_embedding - vn_embedding.mean(dim=1, keepdim=True)
    normalized = F.normalize(centered, dim=1)
    correlation = normalized @ normalized.t()
    return correlation.pow(2).mean()


def score_contrastive_loss(graph_score, local_score, global_score, temperature):
    local = local_score.to(graph_score.device).view(1, -1).expand_as(graph_score)
    global_ = global_score.to(graph_score.device).view(1, -1).expand_as(graph_score)
    sim_local = F.cosine_similarity(graph_score, local, dim=1)
    sim_global = F.cosine_similarity(graph_score, global_, dim=1)
    logits = torch.stack((sim_local, sim_global), dim=1) / float(temperature)
    labels = torch.zeros(graph_score.shape[0], dtype=torch.long, device=graph_score.device)
    return F.cross_entropy(logits, labels)


def graph_level_score(score, batch):
    return global_add_pool(score, batch)
