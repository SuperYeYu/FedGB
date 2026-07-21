from types import SimpleNamespace

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from fedgb.algorithms.standard_fl.fedlaw.utils import (
    fedlaw_task_validation_loss,
    functional_forward_with_vector,
    model_parameter_vector,
    output_to_logits,
)


class TinyGraphRegressor(nn.Module):
    def __init__(self, num_targets=1):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.head = nn.Linear(4, num_targets)

    def forward(self, batch):
        node_embedding = torch.relu(self.encoder(batch.x))
        graph_embedding = global_mean_pool(node_embedding, batch.batch)
        return graph_embedding, self.head(graph_embedding)


def test_fedlaw_validation_loss_supports_graph_regression():
    graphs = [
        Data(x=torch.randn(4, 3), edge_index=torch.empty((2, 0), dtype=torch.long), y=torch.tensor([value]))
        for value in (0.5, 1.0, 1.5)
    ]
    loader = DataLoader(graphs, batch_size=2)
    task = SimpleNamespace(
        args=SimpleNamespace(task="graph_reg"),
        num_targets=1,
        splitted_data={"val_dataloader": loader, "train_dataloader": loader},
    )
    model = TinyGraphRegressor()
    vector = model_parameter_vector(model).requires_grad_(True)
    loss = fedlaw_task_validation_loss(task, model, vector, torch.device("cpu"))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert vector.grad is not None


def test_fedlaw_graph_regression_uses_only_observed_targets():
    graphs = [
        Data(
            x=torch.randn(3, 3),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.tensor([[1.0, 100.0]]),
            y_mask=torch.tensor([[True, False]]),
        ),
        Data(
            x=torch.randn(3, 3),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.tensor([[100.0, 2.0]]),
            y_mask=torch.tensor([[False, True]]),
        ),
    ]
    loader = DataLoader(graphs, batch_size=2)
    task = SimpleNamespace(
        args=SimpleNamespace(task="graph_reg"),
        num_targets=2,
        splitted_data={"val_dataloader": loader, "train_dataloader": loader},
    )
    model = TinyGraphRegressor(num_targets=2)
    vector = model_parameter_vector(model).requires_grad_(True)

    loss = fedlaw_task_validation_loss(task, model, vector, torch.device("cpu"))
    batch = next(iter(loader))
    logits = output_to_logits(functional_forward_with_vector(model, batch, vector)).view(-1, 2)
    expected = torch.mean((logits[batch.y_mask] - batch.y.view(-1, 2)[batch.y_mask]) ** 2)

    assert torch.allclose(loss, expected)


def test_fedlaw_graph_regression_returns_none_without_observed_targets():
    graph = Data(
        x=torch.randn(3, 3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=torch.zeros((1, 2)),
        y_mask=torch.zeros((1, 2), dtype=torch.bool),
    )
    loader = DataLoader([graph], batch_size=1)
    task = SimpleNamespace(
        args=SimpleNamespace(task="graph_reg"),
        num_targets=2,
        splitted_data={"val_dataloader": loader, "train_dataloader": loader},
    )
    model = TinyGraphRegressor(num_targets=2)
    vector = model_parameter_vector(model).requires_grad_(True)

    assert fedlaw_task_validation_loss(task, model, vector, torch.device("cpu")) is None
