import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def contract():
    path = ROOT / "fedgb" / "config" / "paper_dataset_contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_covers_downloadable_and_credentialed_variants():
    variants = contract()["variants"]
    assert len(variants) == 18
    assert sum(item["availability"] == "download" for item in variants) == 16
    assert sum(item["availability"] == "credentialed_build" for item in variants) == 2


def test_eicu_contract_matches_the_validated_rebuild():
    variants = {item["name"]: item for item in contract()["variants"]}
    for name, level in [("EICU-Het", "hetero_subgraph"), ("EICU-Hom", "homo_subgraph")]:
        item = variants[name]
        assert item["availability"] == "credentialed_build"
        assert item["level"] == level
        assert item["num_clients"] == 40
        assert item["num_tasks"] == 1
        assert item["num_classes"] == 3
        assert item["feature_dim"] == 315
        assert item["task_name"] == "length-of-stay prediction"
        assert item["split"] == {"train": 0.05, "val": 0.15, "test": 0.80, "seed": 42}


def test_tcga_contract_uses_validated_counts_and_scenario_labels():
    variants = {item["name"]: item for item in contract()["variants"]}
    assert variants["TCGA"] | {
        "num_graphs": 293786,
        "num_clients": 27,
        "feature_dim": 64,
        "num_tasks": 5,
    } == variants["TCGA"]

    assert variants["TCGA-S2"]["task_name"] == "tumor grade"
    assert variants["TCGA-S2"]["num_graphs"] == 35719
    assert variants["TCGA-S2"]["avg_nodes"] == 85.62
    assert variants["TCGA-S3"]["task_name"] == "progression/recurrence"
    assert variants["TCGA-S3"]["num_graphs"] == 40116
    assert variants["TCGA-S3"]["avg_nodes"] == 83.88


def test_optimade_contract_distinguishes_targets_from_task_families():
    item = next(item for item in contract()["variants"] if item["name"] == "OPTIMADE")
    assert item["num_target_fields"] == 19
    assert item["num_task_families"] == 2
