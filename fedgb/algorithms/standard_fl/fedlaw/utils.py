import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - compatibility for older torch.
    from torch.nn.utils.stateless import functional_call


def model_parameter_vector(model):
    params = [param.detach().reshape(-1) for param in model.parameters()]
    if not params:
        return torch.empty(0)
    return torch.cat(params)


def parameter_dict_from_vector(model, vector):
    params = {}
    offset = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        params[name] = vector[offset: offset + numel].view_as(param)
        offset += numel
    if offset != vector.numel():
        raise ValueError("FedLAW parameter vector size does not match model parameters.")
    return params


def load_vector_to_model(model, vector):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            param.copy_(vector[offset: offset + numel].view_as(param))
            offset += numel
    if offset != vector.numel():
        raise ValueError("FedLAW parameter vector size does not match model parameters.")


def functional_forward_with_vector(model, data, vector):
    params = parameter_dict_from_vector(model, vector)
    return functional_call(model, params, (data,))


def output_to_logits(output):
    if isinstance(output, tuple):
        return output[-1]
    return output


def fedlaw_weights_and_gamma(optimizees, server_funct="exp"):
    if server_funct == "exp":
        weights = torch.softmax(optimizees[:-1], dim=0)
        gamma = torch.exp(optimizees[-1])
    elif server_funct == "quad":
        raw_weights = optimizees[:-1] * optimizees[:-1]
        weights = raw_weights / raw_weights.sum().clamp_min(1e-12)
        gamma = optimizees[-1] * optimizees[-1]
    else:
        raise ValueError(f"Unsupported FedLAW server_funct: {server_funct}")
    return weights, gamma


def init_fedlaw_optimizees(size_weights, server_funct, device):
    if server_funct == "exp":
        values = [math.log(max(float(weight), 1e-12)) for weight in size_weights] + [0.0]
    elif server_funct == "quad":
        cohort_size = len(size_weights)
        values = [math.sqrt(1.0 / cohort_size) for _ in size_weights] + [1.0]
    else:
        raise ValueError(f"Unsupported FedLAW server_funct: {server_funct}")
    return torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True)


def fedlaw_aggregate_vectors(gamma, weights, client_vectors):
    stacked = torch.stack(client_vectors, dim=0)
    weighted = torch.sum(weights.view(-1, *([1] * (stacked.dim() - 1))) * stacked, dim=0)
    return gamma * weighted


@contextmanager
def temporary_model_mode(model, training):
    previous = model.training
    model.train(training)
    try:
        yield
    finally:
        model.train(previous)


def _node_validation_loss(task, model, vector, device):
    data = task.splitted_data["data"].to(device)
    mask = task.splitted_data.get("val_mask", None)
    if mask is None or int(mask.sum()) == 0:
        mask = task.splitted_data["train_mask"]
    mask = mask.to(device).bool()
    logits = output_to_logits(functional_forward_with_vector(model, data, vector))
    return F.cross_entropy(logits[mask], data.y[mask])


def _graph_validation_loss(task, model, vector, device):
    losses = []
    sample_count = 0
    loader = task.splitted_data.get("val_dataloader", None)
    if loader is None:
        loader = task.splitted_data["train_dataloader"]
    for batch in loader:
        batch = batch.to(device)
        logits = output_to_logits(functional_forward_with_vector(model, batch, vector))
        loss = F.cross_entropy(logits, batch.y.long(), reduction="sum")
        losses.append(loss)
        sample_count += int(batch.y.numel())
    if sample_count == 0:
        return None
    return torch.stack(losses).sum() / sample_count


def _graph_reg_validation_loss(task, model, vector, device):
    losses = []
    sample_count = 0
    loader = task.splitted_data.get("val_dataloader") or task.splitted_data["train_dataloader"]
    for batch in loader:
        batch = batch.to(device)
        logits = output_to_logits(functional_forward_with_vector(model, batch, vector))
        labels = batch.y.float().view(-1, task.num_targets)
        logits = logits.view(-1, task.num_targets)
        target_mask = getattr(batch, "y_mask", None)
        if target_mask is None:
            target_mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            target_mask = target_mask.bool().view_as(labels)
        observed = int(target_mask.sum().item())
        if observed == 0:
            continue
        losses.append(torch.sum((logits[target_mask] - labels[target_mask]) ** 2))
        sample_count += observed
    if sample_count == 0:
        return None
    return torch.stack(losses).sum() / sample_count


def fedlaw_task_validation_loss(task, model, vector, device):
    if task.args.task == "node_cls":
        return _node_validation_loss(task, model, vector, device)
    if task.args.task == "graph_cls":
        return _graph_validation_loss(task, model, vector, device)
    if task.args.task == "graph_reg":
        return _graph_reg_validation_loss(task, model, vector, device)
    raise ValueError(f"FedLAW does not support task '{task.args.task}' in FedGB.")


def optimize_fedlaw_weights(task, model, client_vectors, size_weights, config, device):
    server_funct = config["fedlaw_server_funct"]
    optimizees = init_fedlaw_optimizees(size_weights, server_funct, device)
    server_lr = config["fedlaw_server_lr"]
    server_epochs = int(config["fedlaw_server_epochs"])
    if config["fedlaw_server_optimizer"] == "adam":
        optimizer = torch.optim.Adam([optimizees], lr=server_lr, betas=(0.5, 0.999))
    elif config["fedlaw_server_optimizer"] == "sgd":
        optimizer = torch.optim.SGD([optimizees], lr=server_lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported FedLAW server optimizer: {config['fedlaw_server_optimizer']}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    client_vectors = [vector.detach().to(device) for vector in client_vectors]
    with temporary_model_mode(model, training=config["fedlaw_update_bn_buffers"]):
        for _ in range(server_epochs):
            weights, gamma = fedlaw_weights_and_gamma(optimizees, server_funct)
            aggregate_vector = fedlaw_aggregate_vectors(gamma, weights, client_vectors)
            loss = fedlaw_task_validation_loss(task, model, aggregate_vector, device)
            if loss is None:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    weights, gamma = fedlaw_weights_and_gamma(optimizees, server_funct)
    aggregate_vector = fedlaw_aggregate_vectors(gamma, weights, client_vectors)
    return gamma.detach(), weights.detach(), aggregate_vector.detach()
