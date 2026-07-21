import json
from pathlib import Path

import torch

from fedgb.config.datasets import dataset_registry
from fedgb.data.release_loader import iter_dataset_clients


ROOT = Path(__file__).resolve().parents[1]

OPTIMADE_CLIENTS = [
    ("alexandria_pbe", 30000, 10.24, 72.11, 5),
    ("alexandria_pbesol", 14159, 8.00, 49.94, 7),
    ("matterverse", 22894, 25.79, 205.81, 0),
    ("mp", 8073, 30.78, 242.70, 3),
    ("mpdd", 7500, 8.59, 58.16, 4),
    ("nmd", 4598, 23.02, 182.50, 0),
    ("oqmd", 11200, 8.16, 54.10, 5),
    ("twodmatpedia", 6351, 10.44, 74.77, 0),
]


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_all_release_variants_have_contiguous_client_files():
    payload = json.loads((ROOT / "scripts" / "prepare_data" / "dataset_sources.json").read_text())
    for spec in payload["variants"]:
        if spec["availability"] != "download":
            continue
        root = ROOT / "datasets" / spec["name"]
        client_dirs = [path for path in (root / "distrib").iterdir() if path.is_dir()]
        assert len(client_dirs) == 1, spec["name"]
        client_dir = client_dirs[0]
        expected = {f"data_{idx}.pt" for idx in range(spec["num_clients"])}
        actual = {path.name for path in client_dir.glob("data_*.pt")}
        assert actual == expected, spec["name"]


def test_graph_level_payloads_use_fedgb_serialization():
    payload = json.loads((ROOT / "scripts" / "prepare_data" / "dataset_sources.json").read_text())
    for spec in payload["variants"]:
        if spec["availability"] != "download" or spec["level"] != "graph":
            continue
        root = ROOT / "datasets" / spec["name"]
        client_dir = next(path for path in (root / "distrib").iterdir() if path.is_dir())
        dataset = load(client_dir / "data_0.pt")
        assert dataset.__class__.__module__.startswith("fedgb."), spec["name"]


def test_tcga_variants_use_64_dimensional_node_features():
    for name in ["TCGA", "TCGA-S1", "TCGA-S2", "TCGA-S3"]:
        root = ROOT / "datasets" / name
        client_dir = next(path for path in (root / "distrib").iterdir() if path.is_dir())
        dataset = load(client_dir / "data_0.pt")
        assert dataset.graphs[0].x.shape[1] == 64, name


def test_tcga_single_task_payloads_match_the_named_contract():
    expected = {
        "TCGA-S2": (35719, "clinical_grade_high_vs_low"),
        "TCGA-S3": (40116, "progression_or_recurrence_vs_free"),
    }
    for name, (num_graphs, task_name) in expected.items():
        root = ROOT / "datasets" / name
        client_dir = next(path for path in (root / "distrib").iterdir() if path.is_dir())
        payloads = [load(client_dir / f"data_{client_id}.pt") for client_id in range(4)]
        assert sum(len(payload.graphs) for payload in payloads) == num_graphs
        description = (client_dir / "description.txt").read_text(encoding="utf-8")
        assert task_name in description


def test_optimade_matches_paper_client_contract():
    spec = dataset_registry()["OPTIMADE"]
    assert spec["num_clients"] == 8
    assert spec["feature_dim"] == 12
    assert spec["edge_feature_dim"] == 24
    assert spec["num_targets"] == 19

    root = ROOT / "datasets" / "OPTIMADE"
    for client_id, payload, _ in iter_dataset_clients(root, spec):
        provider, graphs, avg_nodes, avg_edges, tasks = OPTIMADE_CLIENTS[client_id]
        assert payload.client_name == provider
        assert len(payload.graphs) == graphs
        actual_avg_nodes = sum(graph.num_nodes for graph in payload.graphs) / graphs
        actual_avg_edges = sum(graph.edge_index.shape[1] for graph in payload.graphs) / graphs
        assert abs(actual_avg_nodes - avg_nodes) <= 0.011
        assert abs(actual_avg_edges - avg_edges) <= 0.011
        assert len(payload.active_target_names) == tasks
        assert payload.y.shape == (graphs, 19)
        assert payload.y_mask.shape == (graphs, 19)


def test_pubchem_uses_paper_feature_width():
    spec = dataset_registry()["PubChem"]
    assert spec["feature_dim"] == 19

    root = ROOT / "datasets" / "PubChem"
    for _, payload, _ in iter_dataset_clients(root, spec):
        assert {int(graph.x.shape[1]) for graph in payload.graphs} == {19}
