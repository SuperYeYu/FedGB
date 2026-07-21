import json
from pathlib import Path

from fedgb.config.datasets import dataset_manifest, dataset_registry


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_source_manifest_contains_the_18_paper_variants():
    path = ROOT / "scripts" / "prepare_data" / "dataset_sources.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = payload["variants"]
    assert len(variants) == 18
    assert len({item["name"] for item in variants}) == 18
    assert len({item["source_group"] for item in variants}) == 8
    for item in variants:
        assert item["level"] in {"homo_subgraph", "hetero_subgraph", "graph"}
        assert item["task"] in {"node_cls", "graph_cls", "graph_reg"}
        assert item["num_clients"] > 0
        assert "source_host" not in item
        assert "source_root" not in item
        assert "dataset_root" not in item

    for name in ["TCGA", "TCGA-S1", "TCGA-S2", "TCGA-S3"]:
        item = next(item for item in variants if item["name"] == name)
        assert item["feature_dim"] == 64


def test_packaged_manifest_is_the_runtime_source_of_truth():
    payload = dataset_manifest()
    assert payload["schema_version"] == "1.0"
    assert payload["version"] == 4
    assert len(dataset_registry()) == 18
    for item in payload["variants"]:
        assert item["schema_version"] == "1.0"
        assert "source_root" not in item
        assert "source_host" not in item


def test_dataset_manifest_matches_paper_client_counts():
    payload = json.loads((ROOT / "scripts" / "prepare_data" / "dataset_sources.json").read_text())
    counts = {item["name"]: item["num_clients"] for item in payload["variants"]}
    assert counts == {
        "EICU-Het": 40,
        "EICU-Hom": 40,
        "ICIJ-Het": 20,
        "ICIJ-Hom": 20,
        "ICIJ-Het-Cross": 20,
        "ICIJ-Hom-Cross": 20,
        "AML-HI": 29,
        "AML-HI-Cross": 29,
        "XS-Video": 5,
        "PubChem": 13,
        "TCGA": 27,
        "TCGA-S1": 17,
        "TCGA-S2": 4,
        "TCGA-S3": 4,
        "NOMAD": 6,
        "OPTIMADE": 8,
        "OPTIMADE-S1": 4,
        "OPTIMADE-S2": 3,
    }
