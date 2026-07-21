import hashlib

import torch
import torch.nn as nn


class GradWhere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, threshold):
        ctx.save_for_backward(input_tensor)
        return torch.where(
            input_tensor > threshold,
            torch.ones_like(input_tensor),
            torch.zeros_like(input_tensor),
        )

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class CWGGenerator(nn.Module):
    def __init__(self, max_nodes, layernum, trigger_size, dropout=0.05, prefix="my_seed"):
        super(CWGGenerator, self).__init__()
        self.max_nodes = int(max_nodes)
        self.trigger_size = int(trigger_size)
        self.prefix = prefix
        self.layers = _make_mlp(self.max_nodes, self.max_nodes, layernum, dropout)
        self.layers_id = _make_mlp(self.max_nodes, self.max_nodes, layernum, dropout)

    def forward(self, a_input, client_id, topomask=None, threshold=None):
        client_matrix = self._client_matrix(client_id, a_input.device, a_input.dtype)
        id_output = torch.sigmoid(self.layers_id(client_matrix))
        topo_logits = torch.sigmoid(self.layers(a_input * id_output))
        topo_logits = 0.5 * (topo_logits + topo_logits.t())
        if threshold is not None:
            topo_logits = GradWhere.apply(topo_logits, threshold)
        if topomask is not None:
            topo_logits = topo_logits * topomask.to(topo_logits.device)
        return topo_logits

    def _client_matrix(self, client_id, device, dtype):
        seed = _stable_seed(self.prefix, client_id)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        matrix = torch.rand(self.max_nodes, self.max_nodes, generator=generator, dtype=dtype)
        return matrix.to(device)


def _stable_seed(prefix, client_id):
    digest = hashlib.sha256(f"{prefix}{client_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _make_mlp(input_dim, output_dim, layernum, dropout):
    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    for _ in range(max(int(layernum) - 1, 0)):
        layers.append(nn.Linear(input_dim, input_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)
