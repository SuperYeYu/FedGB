from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_fedproto_graph_reg_uses_single_task_prototype():
    client = (ROOT / "fedgb/algorithms/standard_fl/fedproto/client.py").read_text()
    server = (ROOT / "fedgb/algorithms/standard_fl/fedproto/server.py").read_text()
    assert "_prototype_ids" in client
    assert "return [0]" in client
    assert "_prototype_ids" in server
    assert "return [0]" in server


def test_fedtgp_graph_reg_uses_regression_prototype_loss():
    client = (ROOT / "fedgb/algorithms/standard_fl/fedtgp/client.py").read_text()
    server = (ROOT / "fedgb/algorithms/standard_fl/fedtgp/server.py").read_text()
    assert "_prototype_ids" in client
    assert "num_prototypes = 1" in server
    assert "nn.MSELoss()(global_prototypes[0], avg_proto[0])" in server


def test_feroma_regression_descriptor_has_stable_dimension():
    from fedgb.algorithms.standard_fl.feroma.utils import regression_latent_descriptor

    embedding = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    descriptor = regression_latent_descriptor(embedding, include_std=True)
    assert descriptor.shape == (4,)
    assert torch.allclose(descriptor[:2], torch.tensor([3.0, 6.0]))


def test_feroma_client_has_graph_reg_branch():
    client = (ROOT / "fedgb/algorithms/standard_fl/feroma/client.py").read_text()
    assert 'if self.args.task == "graph_reg"' in client
    assert "regression_latent_descriptor" in client
    assert "OpenFGL adapter" not in client


def test_tinyproto_graph_reg_uses_single_task_prototype():
    client = (ROOT / "fedgb/algorithms/standard_fl/tinyproto/client.py").read_text()
    server = (ROOT / "fedgb/algorithms/standard_fl/tinyproto/server.py").read_text()
    assert "self.num_prototypes = 1" in client
    assert "_regression_prototype_targets" in client
    assert "self.num_prototypes = 1" in server
    assert 'if self.args.task == "graph_reg":' in client


def test_trainer_infers_classes_from_task_before_data():
    from types import SimpleNamespace

    from fedgb.training.trainer import FGLTrainer

    trainer = FGLTrainer.__new__(FGLTrainer)
    trainer.args = SimpleNamespace(task="graph_cls")
    trainer.server = SimpleNamespace(task=SimpleNamespace(num_global_classes=0))
    trainer.clients = [
        SimpleNamespace(task=SimpleNamespace(num_global_classes=1)),
        SimpleNamespace(task=SimpleNamespace(num_global_classes=2)),
    ]
    assert trainer._infer_num_classes() == 2


def test_trainer_captures_classes_before_algorithm_initialization():
    trainer = (ROOT / "fedgb/training/trainer.py").read_text()
    dataset_line = trainer.index("fgl_dataset = FGLDataset(args)")
    capture_line = trainer.index("raw_num_classes = self._infer_dataset_num_classes")
    client_line = trainer.index("self.clients = [load_client")
    assert dataset_line < capture_line < client_line
