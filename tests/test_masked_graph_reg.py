import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from types import SimpleNamespace
from pathlib import Path

from fedgb.data.fgl_graph_dataset import FGLGraphDataset
import fedgb.tasks.graph_reg as graph_reg
from fedgb.tasks.graph_reg import GraphRegTask
from fedgb.training.trainer import FGLTrainer
from fedgb.algorithms.standard_fl.fedavg.server import FedAvgServer


def graph(y, y_mask):
    return Data(
        x=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor(y, dtype=torch.float32),
        y_mask=torch.tensor(y_mask, dtype=torch.bool),
    )


def test_graph_dataset_stacks_target_masks_and_names():
    payload = FGLGraphDataset(
        [graph([[1.0, 0.0]], [[True, False]]), graph([[0.0, 2.0]], [[False, True]])],
        num_targets=2,
        target_names=["a", "b"],
    )

    assert payload.y.shape == (2, 2)
    assert payload.y_mask.tolist() == [[True, False], [False, True]]
    assert payload.target_names == ["a", "b"]
    copied = payload.copy([1])
    assert copied.target_names == ["a", "b"]
    assert copied.y_mask.tolist() == [[False, True]]


def test_graph_dataset_rejects_target_name_width_mismatch():
    with pytest.raises(ValueError, match="target_names"):
        FGLGraphDataset(
            [graph([[1.0, 0.0]], [[True, False]])],
            num_targets=2,
            target_names=["only_one"],
        )


def test_masked_mse_uses_only_observed_elements():
    assert hasattr(graph_reg, "masked_mse")
    logits = torch.tensor([[3.0, 100.0], [100.0, 5.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[True, False], [False, True]])

    assert graph_reg.masked_mse(logits, labels, mask).item() == pytest.approx(10.0)


def test_masked_mse_returns_none_without_observations():
    assert hasattr(graph_reg, "masked_mse")
    result = graph_reg.masked_mse(
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 3, dtype=torch.bool),
    )
    assert result is None


def test_masked_mse_keeps_legacy_unmasked_behavior():
    assert hasattr(graph_reg, "masked_mse")
    logits = torch.tensor([[3.0], [5.0]])
    labels = torch.tensor([[1.0], [1.0]])
    assert graph_reg.masked_mse(logits, labels).item() == pytest.approx(10.0)


def test_graph_reg_loss_combines_graph_and_target_masks():
    task = GraphRegTask.__new__(GraphRegTask)
    task.data = type("Payload", (), {"num_targets": 2})()
    logits = torch.tensor([[3.0, 100.0], [100.0, 5.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    graph_mask = torch.tensor([True, False])
    target_mask = torch.tensor([[True, False], [False, True]])

    loss = task.loss_fn(None, logits, labels, graph_mask, target_mask=target_mask)
    assert loss.item() == pytest.approx(4.0)


def test_masked_training_preserves_algorithm_loss_extensions():
    task = GraphRegTask.__new__(GraphRegTask)
    task.data = type("Payload", (), {"num_targets": 2})()
    calls = []

    def custom_loss(embedding, logits, labels, graph_mask):
        calls.append(graph_mask.clone())
        return task.default_loss_fn(logits[graph_mask], labels[graph_mask]) + 3.0

    task.loss_fn = custom_loss
    logits = torch.tensor([[3.0, 100.0], [100.0, 5.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_mask = torch.tensor([[True, False], [False, True]])

    loss = task._training_loss(None, logits, labels, target_mask)

    assert loss.item() == pytest.approx(13.0)
    assert calls[0].tolist() == [True, True]


def test_graph_regression_batch_loss_forwards_target_mask():
    assert hasattr(graph_reg, "graph_regression_batch_loss")
    captured = {}

    class FakeTask:
        num_targets = 2

        @staticmethod
        def _target(labels):
            return labels.float().view(-1, 2)

        @staticmethod
        def loss_fn(embedding, logits, labels, graph_mask, target_mask=None):
            captured["target_mask"] = target_mask
            return graph_reg.masked_mse(logits.view(-1, 2), labels, target_mask)

    batch = SimpleNamespace(
        y=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        y_mask=torch.tensor([[True, False], [False, True]]),
    )
    logits = torch.tensor([[3.0, 100.0], [100.0, 5.0]])

    loss = graph_reg.graph_regression_batch_loss(FakeTask(), None, logits, batch)

    assert loss.item() == pytest.approx(10.0)
    assert torch.equal(captured["target_mask"], batch.y_mask)

    batch.y_mask.zero_()
    assert graph_reg.graph_regression_batch_loss(FakeTask(), None, logits, batch) is None


def test_graph_fgl_custom_training_loops_use_masked_batch_loss():
    root = Path(__file__).resolve().parents[1]
    fedssp = (root / "fedgb/algorithms/graph_fgl/fedssp/client.py").read_text()
    fedvn = (root / "fedgb/algorithms/graph_fgl/fedvn/client.py").read_text()

    assert fedssp.count("graph_regression_batch_loss(") == 1
    assert "if loss is None" in fedssp
    assert fedvn.count("graph_regression_batch_loss(") == 2
    assert "if supervised_loss is None" in fedvn


def test_graph_reg_sample_weight_counts_observed_training_targets():
    payload = FGLGraphDataset(
        [graph([[1.0, 0.0]], [[True, False]]), graph([[0.0, 2.0]], [[False, True]])],
        num_targets=2,
        target_names=["a", "b"],
    )
    task = GraphRegTask.__new__(GraphRegTask)
    task.data = payload
    task.train_mask = torch.tensor([True, False])

    assert task.num_samples == 1


def test_graph_reg_split_masks_use_graph_count_not_observed_target_count():
    graphs = [
        graph([[1.0, 0.0]], [[True, False]]),
        graph([[0.0, 0.0]], [[False, False]]),
        graph([[0.0, 0.0]], [[False, False]]),
    ]
    for item, split in zip(graphs, ("train", "val", "test")):
        item.split = split
    payload = FGLGraphDataset(graphs, num_targets=2, target_names=["a", "b"])
    task = GraphRegTask.__new__(GraphRegTask)
    task.data = payload

    train_mask, val_mask, test_mask = task.local_graph_train_val_test_split(
        payload, "default_split"
    )

    assert train_mask.tolist() == [True, False, False]
    assert val_mask.tolist() == [False, True, False]
    assert test_mask.tolist() == [False, False, True]


def test_trainer_ignores_clients_without_observed_regression_targets():
    class FakeTask:
        def __init__(self, result, num_samples):
            self.result = result
            self.num_samples = num_samples

        def evaluate(self):
            return self.result

    trainer = FGLTrainer.__new__(FGLTrainer)
    trainer.args = SimpleNamespace(
        task="graph_reg",
        metrics=["mae"],
        evaluation_mode="local_model_on_local_data",
        num_clients=2,
    )
    trainer.message_pool = {"round": 0}
    trainer.clients = [
        SimpleNamespace(task=FakeTask({
            "loss_val": torch.tensor(1.0), "loss_test": torch.tensor(2.0),
            "mae_val": 1.0, "mae_test": 2.0,
            "num_observed_val": 3, "num_observed_test": 4,
        }, 5)),
        SimpleNamespace(task=FakeTask({
            "loss_val": torch.tensor(float("nan")), "loss_test": torch.tensor(float("nan")),
            "mae_val": float("nan"), "mae_test": float("nan"),
            "num_observed_val": 0, "num_observed_test": 0,
        }, 0)),
    ]
    trainer.server = SimpleNamespace(personalized=False)
    trainer.evaluation_result = {
        "best_round": 0,
        "best_val_loss": float("inf"),
        "best_test_loss": float("inf"),
        "best_val_mae": float("inf"),
        "best_test_mae": float("inf"),
    }
    trainer.logger = SimpleNamespace(add_log=lambda result: None)

    trainer.evaluate()

    assert trainer.evaluation_result["best_val_mae"] == pytest.approx(1.0)
    assert trainer.evaluation_result["best_test_mae"] == pytest.approx(2.0)


def test_fedavg_keeps_global_parameters_when_all_clients_are_unsupervised():
    server = FedAvgServer.__new__(FedAvgServer)
    model = nn.Linear(2, 1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    server.task = SimpleNamespace(model=model)
    server.message_pool = {"sampled_clients": [0, 1]}
    for client_id in (0, 1):
        server.message_pool[f"client_{client_id}"] = {
            "num_samples": 0,
            "weight_names": list(before),
            "weight": [value.clone() for value in before.values()],
        }

    server.execute()

    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])
