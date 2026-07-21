import json

import pytest
import torch
from torch_geometric.data import Data

from fedgb.data.fgl_graph_dataset import FGLGraphDataset
from fedgb.data.validation import validate_dataset_variant


def make_variant(tmp_path, *, overlap=False, missing_edge_type=False):
    root = tmp_path / "Demo-Het"
    partition = root / "distrib" / "partition"
    split = partition / "node_cls" / "default_split"
    split.mkdir(parents=True)
    data = Data(
        x=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        y=torch.tensor([0, 1, 0, 1, 0]),
        global_map={idx: idx for idx in range(5)},
        node_type=torch.tensor([0, 0, 1, 1, 1]),
        target_node_type="entity",
    )
    if not missing_edge_type:
        data.edge_type = torch.tensor([0, 1, 1])
    torch.save(data, partition / "data_0.pt")
    train = torch.tensor([True, True, False, False, False])
    val = torch.tensor([overlap, False, True, False, False])
    test = torch.tensor([False, False, False, True, True])
    for name, mask in {"train": train, "val": val, "test": test}.items():
        torch.save(mask, split / f"{name}_0.pt")
    manifest = {
        "name": "Demo-Het",
        "schema_version": "1.0",
        "level": "hetero_subgraph",
        "task": "node_cls",
        "num_clients": 1,
        "dataset_id": "DEMO",
        "processed_partition": "partition",
        "target_node": "entity",
    }
    (root / "fedgb_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    global_dir = root / "global" / "subgraph_fl" / "demo" / "processed"
    global_dir.mkdir(parents=True)
    torch.save(data, global_dir / "data.pt")
    return root, manifest


def test_validation_actually_deserializes_each_client(tmp_path):
    root, spec = make_variant(tmp_path)
    report = validate_dataset_variant(root, spec)
    assert report["clients"] == 1
    assert report["inspected_payloads"] == 1
    assert report["feature_dims"] == [3]
    assert report["schema_version"] == "1.0"


def test_validation_rejects_missing_required_relation_field(tmp_path):
    root, spec = make_variant(tmp_path, missing_edge_type=True)
    with pytest.raises(ValueError, match="edge_type"):
        validate_dataset_variant(root, spec)


def test_validation_rejects_overlapping_splits(tmp_path):
    root, spec = make_variant(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="overlapping"):
        validate_dataset_variant(root, spec)


def test_validation_rejects_internal_absolute_paths(tmp_path):
    root, spec = make_variant(tmp_path)
    (root / "description.txt").write_text(
        '{"source": "/data/zfzhu_nas/yyy/private"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="internal absolute path"):
        validate_dataset_variant(root, spec)


def test_validation_rejects_internal_path_in_binary_payload(tmp_path):
    root, spec = make_variant(tmp_path)
    (root / "source.pkl").write_bytes(b"metadata=/opt/data/private/yyy/source")
    with pytest.raises(ValueError, match="internal absolute path"):
        validate_dataset_variant(root, spec)


def test_validation_rejects_manifest_that_disagrees_with_registry(tmp_path):
    root, spec = make_variant(tmp_path)
    manifest = json.loads((root / "fedgb_manifest.json").read_text(encoding="utf-8"))
    manifest["num_clients"] = 2
    (root / "fedgb_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_dataset_variant(root, spec)


def test_graph_validation_rejects_wrong_edge_feature_width(tmp_path):
    root = tmp_path / "Demo-Graph"
    partition = root / "distrib" / "partition"
    split = partition / "graph_cls" / "default_split"
    split.mkdir(parents=True)
    graph = Data(
        x=torch.randn(3, 19),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 8),
        y=torch.tensor(1),
    )
    payload = FGLGraphDataset([graph], num_targets=1, global_map={0: 0})
    torch.save(payload, partition / "data_0.pt")
    for name, mask in {
        "train": torch.tensor([True]),
        "val": torch.tensor([False]),
        "test": torch.tensor([False]),
    }.items():
        torch.save(mask, split / f"{name}_0.pt")
    spec = {
        "name": "Demo-Graph",
        "schema_version": "1.0",
        "level": "graph",
        "task": "graph_cls",
        "num_clients": 1,
        "dataset_id": "DEMO",
        "processed_partition": "partition",
        "feature_dim": 19,
        "edge_feature_dim": 16,
    }
    disk_manifest = {key: value for key, value in spec.items() if key not in {"feature_dim", "edge_feature_dim"}}
    (root / "fedgb_manifest.json").write_text(json.dumps(disk_manifest), encoding="utf-8")
    global_dir = root / "global" / "graph_fl" / "DEMO" / "processed"
    global_dir.mkdir(parents=True)
    torch.save(payload, global_dir / "data.pt")

    with pytest.raises(ValueError, match="edge_feature_dim"):
        validate_dataset_variant(root, spec)
