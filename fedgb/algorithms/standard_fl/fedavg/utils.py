import torch


def is_private_prediction_head(name, task_name, private_head=False):
    graph_task_with_private_head = task_name == "graph_reg" or (task_name == "graph_cls" and private_head)
    return graph_task_with_private_head and name.startswith("head.")


def shared_parameter_names(model, task_name=None, private_head=False):
    return [
        name
        for name, _ in model.named_parameters()
        if not is_private_prediction_head(name, task_name, private_head)
    ]


def shared_parameter_payload(model, task_name=None, private_head=False):
    names = shared_parameter_names(model, task_name, private_head)
    named_params = dict(model.named_parameters())
    return names, [named_params[name] for name in names]


def load_shared_parameters(model, names, weights, device):
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, weight in zip(names, weights):
            if name in named_params:
                named_params[name].data.copy_(weight.to(device))
