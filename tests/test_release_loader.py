import json

import pytest
import torch
from torch_geometric.data import Data

from fedgb.data.fgl_graph_dataset import FGLGraphDataset
from fedgb.data.release_loader import (
    DatasetBundle,
    iter_dataset_clients,
    load_client_payload,
    load_dataset_bundle,
)


def node_data(heterogeneous=False):
    data = Data(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        y=torch.tensor([0, 1, 0, 1, 0]),
        global_map={idx: idx for idx in range(5)},
    )
    if heterogeneous:
        data.node_type = torch.tensor([0, 0, 1, 1, 1])
        data.edge_type = torch.tensor([0, 1, 1, 0])
        data.target_node_type = "entity"
    return data


def write_variant(tmp_path, scenario, task, payload):
    root = tmp_path / "Demo"
    partition = root / "distrib" / "partition"
    split = partition / task / "default_split"
    split.mkdir(parents=True)
    torch.save(payload, partition / "data_0.pt")
    size = len(payload.graphs) if scenario == "graph" else payload.x.shape[0]
    train = torch.zeros(size, dtype=torch.bool)
    val = torch.zeros(size, dtype=torch.bool)
    test = torch.zeros(size, dtype=torch.bool)
    train[: max(1, size - 2)] = True
    val[-2:-1] = True
    test[-1:] = True
    for name, mask in {"train": train, "val": val, "test": test}.items():
        torch.save(mask, split / f"{name}_0.pt")
    manifest = {
        "name": "Demo",
        "schema_version": "1.0",
        "level": scenario,
        "task": task,
        "num_clients": 1,
        "dataset_id": "DEMO",
        "processed_partition": "partition",
    }
    (root / "fedgb_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def test_one_loader_handles_homogeneous_and_heterogeneous_subgraphs(tmp_path):
    for scenario in ("homo_subgraph", "hetero_subgraph"):
        root, spec = write_variant(tmp_path / scenario, scenario, "node_cls", node_data(scenario == "hetero_subgraph"))
        bundle = load_dataset_bundle(root, spec=spec, load_global=False)
        assert isinstance(bundle, DatasetBundle)
        assert bundle.scenario == scenario
        assert len(bundle.clients) == 1
        assert set(bundle.splits[0]) == {"train", "val", "test"}
        if scenario == "hetero_subgraph":
            assert hasattr(bundle.clients[0], "edge_type")
            assert bundle.clients[0].target_node_type == "entity"


def test_same_loader_handles_graph_classification_and_regression(tmp_path):
    for task, dtype in (("graph_cls", torch.long), ("graph_reg", torch.float32)):
        graphs = [
            Data(
                x=torch.randn(4, 3),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                y=torch.tensor([index % 2 if task == "graph_cls" else float(index)], dtype=dtype),
            )
            for index in range(4)
        ]
        payload = FGLGraphDataset(graphs, task_type=task, num_global_classes=2 if task == "graph_cls" else None)
        root, spec = write_variant(tmp_path / task, "graph", task, payload)
        bundle = load_dataset_bundle(root, spec=spec, load_global=False)
        assert isinstance(bundle.clients[0], FGLGraphDataset)


def test_load_client_payload_is_the_shared_file_boundary(tmp_path):
    path = tmp_path / "data_0.pt"
    torch.save(node_data(), path)
    loaded = load_client_payload(path, scenario="homo_subgraph", task="node_cls")
    assert loaded.x.dtype == torch.float32
    assert loaded.edge_index.dtype == torch.int64


def test_graph_loader_normalizes_target_masks(tmp_path):
    graph = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([[1.0, 0.0]]),
        y_mask=torch.tensor([[1, 0]], dtype=torch.uint8),
    )
    path = tmp_path / "data_0.pt"
    torch.save(FGLGraphDataset([graph], num_targets=2), path)

    loaded = load_client_payload(path, scenario="graph", task="graph_reg")
    assert loaded.graphs[0].y_mask.dtype == torch.bool
    assert loaded.y_mask.dtype == torch.bool


def test_iter_dataset_clients_supports_streaming_validation(tmp_path):
    root, spec = write_variant(tmp_path, "homo_subgraph", "node_cls", node_data())
    records = list(iter_dataset_clients(root, spec))
    assert len(records) == 1
    client_id, payload, splits = records[0]
    assert client_id == 0
    assert payload.x.shape == (5, 4)
    assert set(splits) == {"train", "val", "test"}


def test_loader_rejects_split_masks_with_wrong_length(tmp_path):
    root, spec = write_variant(tmp_path, "homo_subgraph", "node_cls", node_data())
    split = root / "distrib" / "partition" / "node_cls" / "default_split"
    torch.save(torch.tensor([True, False]), split / "train_0.pt")
    with pytest.raises(ValueError, match="length"):
        list(iter_dataset_clients(root, spec))
