import json

import torch
from torch_geometric.data import Data, HeteroData

from scripts.prepare_data.normalize_release_datasets import normalize_dataset_root, normalize_variant


def test_normalization_exports_embedded_masks_and_sanitizes_metadata(tmp_path):
    root = tmp_path / "Demo"
    partition = root / "distrib" / "partition"
    partition.mkdir(parents=True)
    data = Data(
        x=torch.randn(4, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, 0, 1]),
        global_map={idx: idx for idx in range(4)},
        train_mask=torch.tensor([True, True, False, False]),
        val_mask=torch.tensor([False, False, True, False]),
        test_mask=torch.tensor([False, False, False, True]),
    )
    torch.save(data, partition / "data_0.pt")
    (partition / "description.txt").write_text(
        '{"root": "/opt/data/private/yyy/source", "note": "paper split"}', encoding="utf-8"
    )
    spec = {
        "name": "Demo",
        "schema_version": "1.0",
        "level": "homo_subgraph",
        "task": "node_cls",
        "num_clients": 1,
        "dataset_id": "DEMO",
    }
    manifest = normalize_variant(root, spec)
    split = partition / "node_cls" / "default_split"
    assert torch.equal(torch.load(split / "train_0.pt", weights_only=False), data.train_mask)
    assert "/opt/data/private" not in (partition / "description.txt").read_text(encoding="utf-8")
    assert manifest["processed_partition"] == "partition"
    public_manifest = json.loads((root / "fedgb_manifest.json").read_text(encoding="utf-8"))
    assert public_manifest["schema_version"] == "1.0"


def test_normalization_converts_heterogeneous_global_graph_to_relation_data(tmp_path):
    root = tmp_path / "Demo-Het"
    partition = root / "distrib" / "partition"
    split = partition / "node_cls" / "default_split"
    split.mkdir(parents=True)
    client = Data(
        x=torch.randn(4, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, -1, -1]),
        global_map={idx: idx for idx in range(4)},
        node_type=torch.tensor([0, 0, 1, 1]),
        edge_type=torch.tensor([0, 1]),
        target_node_type="entity",
    )
    torch.save(client, partition / "data_0.pt")
    for name, mask in {
        "train": torch.tensor([True, False, False, False]),
        "val": torch.tensor([False, True, False, False]),
        "test": torch.tensor([False, False, False, False]),
    }.items():
        torch.save(mask, split / f"{name}_0.pt")

    global_dir = root / "global" / "subgraph_fl" / "demo" / "processed"
    global_dir.mkdir(parents=True)
    global_data = HeteroData()
    global_data["entity"].x = torch.randn(2, 2)
    global_data["entity"].y = torch.tensor([0, 1])
    global_data["other"].x = torch.randn(2, 2)
    global_data[("entity", "links", "other")].edge_index = torch.tensor([[0, 1], [0, 1]])
    torch.save((global_data.to_dict(), None, HeteroData), global_dir / "data.pt")
    spec = {
        "name": "Demo-Het",
        "schema_version": "1.0",
        "level": "hetero_subgraph",
        "task": "node_cls",
        "num_clients": 1,
        "dataset_id": "DEMO",
        "target_node": "entity",
    }
    normalize_variant(root, spec)
    normalized = torch.load(global_dir / "data.pt", weights_only=False)
    assert isinstance(normalized, Data)
    assert hasattr(normalized, "node_type")
    assert hasattr(normalized, "edge_type")


def test_dataset_root_normalization_sanitizes_copy_audit(tmp_path):
    (tmp_path / "copy_audit.json").write_text(
        '{"source": "/data/zfzhu_nas/yyy/private", "target": "/opt/data/private/yyy/FedGB"}',
        encoding="utf-8",
    )
    normalize_dataset_root(tmp_path)
    text = (tmp_path / "copy_audit.json").read_text(encoding="utf-8")
    assert "/data/zfzhu_nas" not in text
    assert "/opt/data/private" not in text
    assert "<private-source>/" in text
