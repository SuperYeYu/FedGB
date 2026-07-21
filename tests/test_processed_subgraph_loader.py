from pathlib import Path

import pytest
from torch_geometric.data import Data

from fedgb.data.global_dataset_loader import load_global_dataset


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("XS-Video", Data),
        ("ICIJ-Hom", Data),
        ("ICIJ-Het", Data),
        ("AML-HI", Data),
        ("AML-HI-Cross", Data),
    ],
)
def test_processed_subgraph_dataset_loads_without_downloading(name, expected_type):
    root = ROOT / "datasets" / name / "global"
    dataset = load_global_dataset(str(root), scenario="subgraph_fl", dataset="ACM")
    assert isinstance(dataset.data, expected_type)


def test_processed_heterogeneous_dataset_preserves_typed_relations():
    root = ROOT / "datasets" / "ICIJ-Het" / "global"
    dataset = load_global_dataset(str(root), scenario="subgraph_fl", dataset="ACM")
    assert dataset.data.target_node_type == "entity"
    assert dataset.data.node_type.numel() == dataset.data.x.shape[0]
    assert dataset.data.edge_type.numel() == dataset.data.edge_index.shape[1]
