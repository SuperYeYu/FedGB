import json
from pathlib import Path

import pytest

from scripts.release.build_dataset_archive import (
    audit_archive,
    build_archive,
    validate_paper_contract,
    write_checksum,
)


def write_variant(root, name, clients=2):
    variant = root / name
    variant.mkdir(parents=True)
    (variant / "fedgb_manifest.json").write_text(
        json.dumps(
            {
                "name": name,
                "num_clients": clients,
                "schema_version": "1.0",
                "level": "graph",
                "task": "graph_cls",
            }
        ),
        encoding="utf-8",
    )
    (variant / "description.txt").write_text(f"FedGB {name}\n", encoding="utf-8")


def test_archive_contains_only_downloadable_variants(tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    (dataset_root / "README.md").write_text("FedGB datasets\n", encoding="utf-8")
    write_variant(dataset_root, "Public-A")
    write_variant(dataset_root, "Public-B")
    write_variant(dataset_root, "EICU-Het")
    registry = {
        "variants": [
            {"name": "Public-A", "availability": "download", "num_clients": 2},
            {"name": "Public-B", "availability": "download", "num_clients": 2},
            {"name": "EICU-Het", "availability": "credentialed_build", "num_clients": 40},
        ]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    output = tmp_path / "FedGB-datasets-test.tar.zst"

    build_archive(dataset_root, registry_path, output, compression_level=1, threads=1)
    report = audit_archive(output, registry_path)

    assert report["variants"] == ["Public-A", "Public-B"]
    assert report["credentialed_variants"] == ["EICU-Het"]
    assert report["internal_path_offenders"] == []
    checksum = write_checksum(output)
    assert checksum.read_text(encoding="ascii").endswith("  FedGB-datasets-test.tar.zst\n")


def test_archive_audit_rejects_internal_paths(tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    write_variant(dataset_root, "Public-A")
    (dataset_root / "Public-A" / "description.txt").write_text(
        "source=/opt/data/private/yyy/secret\n", encoding="utf-8"
    )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"variants": [{"name": "Public-A", "availability": "download", "num_clients": 2}]}),
        encoding="utf-8",
    )
    output = tmp_path / "bad.tar.zst"
    build_archive(dataset_root, registry_path, output, compression_level=1, threads=1)

    try:
        audit_archive(output, registry_path)
    except ValueError as exc:
        assert "internal absolute paths" in str(exc)
    else:
        raise AssertionError("archive audit accepted an internal path")


def test_archive_audit_rejects_unregistered_dataset_root(tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    write_variant(dataset_root, "Public-A")
    write_variant(dataset_root, "EICU-Het")
    registry_path = tmp_path / "registry.json"
    build_registry = tmp_path / "build-registry.json"
    build_registry.write_text(
        json.dumps({
            "variants": [
                {"name": "Public-A", "availability": "download", "num_clients": 2},
                {"name": "EICU-Het", "availability": "download", "num_clients": 2},
            ]
        }),
        encoding="utf-8",
    )
    output = tmp_path / "bad-extra.tar.zst"
    build_archive(dataset_root, build_registry, output, compression_level=1, threads=1)

    registry_path.write_text(
        json.dumps({
            "variants": [
                {"name": "Public-A", "availability": "download", "num_clients": 2},
                {"name": "EICU-Het", "availability": "credentialed_build", "num_clients": 40},
            ]
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Credentialed variants"):
        audit_archive(output, registry_path)


def test_paper_contract_rejects_wrong_graph_count(tmp_path):
    dataset_root = tmp_path / "datasets"
    root = dataset_root / "Demo"
    partition = root / "distrib" / "partition"
    partition.mkdir(parents=True)
    manifest = {"name": "Demo", "processed_partition": "partition"}
    (root / "fedgb_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (partition / "description.txt").write_text("demo_task\n", encoding="utf-8")
    from fedgb.data.fgl_graph_dataset import FGLGraphDataset
    from torch_geometric.data import Data
    import torch

    torch.save(FGLGraphDataset([Data(x=torch.ones(3, 2), edge_index=torch.empty(2, 0, dtype=torch.long), y=torch.tensor(0))]), partition / "data_0.pt")
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"variants": [{
        "name": "Demo",
        "availability": "download",
        "level": "graph",
        "num_clients": 1,
        "num_graphs": 2,
        "avg_nodes": 3.0,
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="graph count"):
        validate_paper_contract(dataset_root, contract)
