import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from fedgb.config.datasets import get_dataset_spec
from fedgb.data.fgl_graph_dataset import FGLGraphDataset
from fedgb.data.global_dataset_loader import load_global_dataset
from fedgb.data.public_preparation import (
    deterministic_split_masks,
    prepare_public_dataset,
    write_graph_variant,
    write_homo_subgraph_variant,
)
from fedgb.data.release_loader import load_dataset_bundle


def _node_data():
    return Data(
        x=torch.arange(24, dtype=torch.float32).reshape(8, 3),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]),
        y=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )


def _local_node_data(global_ids):
    source = _node_data()
    node_ids = torch.tensor(global_ids)
    data = Data(
        x=source.x[node_ids],
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=source.y[node_ids],
        global_map={local_id: global_id for local_id, global_id in enumerate(global_ids)},
    )
    data.num_global_classes = 2
    return data


def _graph(label, offset):
    return Data(
        x=torch.tensor([[offset, 1.0], [offset + 1.0, 0.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([label], dtype=torch.long),
    )


def test_deterministic_split_masks_are_reproducible_and_complete():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    first = deterministic_split_masks(labels, (0.5, 0.2, 0.3), seed=7)
    second = deterministic_split_masks(labels, (0.5, 0.2, 0.3), seed=7)

    assert all(torch.equal(first[name], second[name]) for name in ("train", "val", "test"))
    assert all(mask.dtype == torch.bool for mask in first.values())
    assert torch.all(first["train"] | first["val"] | first["test"])
    assert not torch.any(first["train"] & first["val"])
    assert not torch.any(first["train"] & first["test"])
    assert not torch.any(first["val"] & first["test"])


def test_write_homo_subgraph_variant_produces_schema_1_bundle(tmp_path):
    root = tmp_path / "Cora-Louvain-2"
    manifest = write_homo_subgraph_variant(
        root=root,
        name="Cora-Louvain-2",
        source_dataset="Cora",
        partition="subgraph_fl_louvain_1_Cora_client_2",
        global_data=_node_data(),
        clients=[_local_node_data([0, 1, 4, 5]), _local_node_data([2, 3, 6, 7])],
        split=(0.5, 0.25, 0.25),
        seed=2024,
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["level"] == "homo_subgraph"
    assert manifest["source_dataset"] == "Cora"
    bundle = load_dataset_bundle(root)
    assert len(bundle.clients) == 2
    assert all(isinstance(client, Data) for client in bundle.clients)


def test_write_graph_variant_converts_clients_to_fgl_graph_dataset(tmp_path):
    root = tmp_path / "MUTAG-LabelSkew-2"
    graphs = [_graph(index % 2, float(index)) for index in range(12)]
    manifest = write_graph_variant(
        root=root,
        name="MUTAG-LabelSkew-2",
        source_dataset="MUTAG",
        partition="graph_fl_label_skew_10.00_MUTAG_client_2",
        global_graphs=graphs,
        clients=[graphs[:6], graphs[6:]],
        split=(0.6, 0.2, 0.2),
        seed=2024,
    )

    assert manifest["level"] == "graph"
    bundle = load_dataset_bundle(root)
    assert isinstance(bundle.global_data, FGLGraphDataset)
    assert all(isinstance(client, FGLGraphDataset) for client in bundle.clients)
    assert [len(client) for client in bundle.clients] == [6, 6]


def test_standard_tu_dataset_cache_uses_pyg_loader(tmp_path, monkeypatch):
    processed = tmp_path / "graph_fl" / "MUTAG" / "processed"
    processed.mkdir(parents=True)
    torch.save(({}, None, Data), processed / "data.pt")
    marker = object()

    monkeypatch.setattr("torch_geometric.datasets.TUDataset", lambda **kwargs: marker)

    assert load_global_dataset(str(tmp_path), "graph_fl", "MUTAG") is marker


def test_get_dataset_spec_discovers_generated_manifest(tmp_path):
    variant = tmp_path / "Cora-Louvain-2"
    variant.mkdir()
    manifest = {
        "name": "Cora-Louvain-2",
        "schema_version": "1.0",
        "level": "homo_subgraph",
        "task": "node_cls",
        "num_clients": 2,
        "dataset_id": "Cora-Louvain-2",
        "processed_partition": "subgraph_fl_louvain_1_Cora_client_2",
    }
    (variant / "fedgb_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert get_dataset_spec("Cora-Louvain-2", dataset_root=variant) == manifest


def test_prepare_public_dataset_reuses_existing_variant_without_downloading(tmp_path, monkeypatch):
    variant = tmp_path / "Cora-Louvain-2"
    variant.mkdir()
    manifest = {
        "name": "Cora-Louvain-2",
        "schema_version": "1.0",
        "level": "homo_subgraph",
        "task": "node_cls",
        "num_clients": 2,
        "dataset_id": "Cora-Louvain-2",
        "source_dataset": "Cora",
        "processed_partition": "subgraph_fl_louvain_1.0_Cora_client_2",
        "seed": 2024,
        "split": [0.2, 0.4, 0.4],
        "simulation": {
            "mode": "subgraph_fl_louvain",
            "louvain_resolution": 1.0,
            "louvain_delta": 20,
        },
    }
    (variant / "fedgb_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fail_if_downloading(*args, **kwargs):
        raise AssertionError("existing output should be reused before downloading")

    monkeypatch.setattr("fedgb.data.public_preparation.load_global_dataset", fail_if_downloading)
    result = prepare_public_dataset(
        {
            "dataset": "Cora",
            "partition": "louvain",
            "num_clients": 2,
            "output_name": "Cora-Louvain-2",
            "datasets_root": tmp_path,
            "seed": 2024,
            "split": (0.2, 0.4, 0.4),
        },
        scenario="homo_subgraph",
    )

    assert result == manifest


def test_prepare_public_dataset_rejects_reuse_with_changed_seed(tmp_path):
    variant = tmp_path / "Cora-Louvain-2"
    variant.mkdir()
    manifest = {
        "name": "Cora-Louvain-2",
        "schema_version": "1.0",
        "level": "homo_subgraph",
        "task": "node_cls",
        "num_clients": 2,
        "dataset_id": "Cora-Louvain-2",
        "source_dataset": "Cora",
        "processed_partition": "subgraph_fl_louvain_1.0_Cora_client_2",
        "seed": 2024,
        "split": [0.2, 0.4, 0.4],
        "simulation": {
            "mode": "subgraph_fl_louvain",
            "louvain_resolution": 1.0,
            "louvain_delta": 20,
        },
    }
    (variant / "fedgb_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FileExistsError, match="does not match"):
        prepare_public_dataset(
            {
                "dataset": "Cora",
                "partition": "louvain",
                "num_clients": 2,
                "output_name": "Cora-Louvain-2",
                "datasets_root": tmp_path,
                "seed": 9,
                "split": (0.2, 0.4, 0.4),
            },
            scenario="homo_subgraph",
        )


@pytest.mark.parametrize(
    ("filename", "scenario", "dataset"),
    [
        ("prepare_homo_subgraph.py", "homo_subgraph", "Cora"),
        ("prepare_graph.py", "graph", "MUTAG"),
    ],
)
def test_public_preparation_scripts_expose_editable_config(filename, scenario, dataset):
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.SCENARIO == scenario
    assert module.CONFIG["dataset"] == dataset
    assert module.CONFIG["num_clients"] > 1
