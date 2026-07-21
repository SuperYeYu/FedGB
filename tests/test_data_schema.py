import pytest
import torch
from torch_geometric.data import Data

from fedgb.data.fgl_graph_dataset import FGLGraphDataset
from fedgb.data.schema import (
    SCHEMA_VERSION,
    schema_for,
    validate_client_payload,
    validate_global_payload,
)


def test_each_public_scenario_has_one_versioned_schema():
    assert SCHEMA_VERSION == "1.0"
    assert schema_for("homo_subgraph").required_fields == frozenset(
        {"x", "edge_index", "y", "global_map"}
    )
    assert schema_for("hetero_subgraph").required_fields == frozenset(
        {
            "x",
            "edge_index",
            "y",
            "global_map",
            "node_type",
            "edge_type",
            "target_node_type",
        }
    )
    assert schema_for("graph").payload_type == "FGLGraphDataset"


def test_homogeneous_payload_requires_common_tensor_contract():
    payload = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, 0, 1]),
        global_map={0: 10, 1: 11, 2: 12, 3: 13},
    )
    validate_client_payload(payload, "homo_subgraph", task="node_cls")


def test_heterogeneous_payload_rejects_missing_relation_fields():
    payload = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, 0, 1]),
        global_map={0: 10, 1: 11, 2: 12, 3: 13},
    )
    with pytest.raises(ValueError, match="edge_type"):
        validate_client_payload(payload, "hetero_subgraph", task="node_cls")


def test_heterogeneous_global_payload_uses_the_same_relation_data_format():
    payload = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([0, 1, -1, -1]),
        node_type=torch.tensor([0, 0, 1, 1]),
        edge_type=torch.tensor([0, 1]),
        target_node_type="entity",
    )
    validate_global_payload(payload, "hetero_subgraph", task="node_cls")


def test_graph_payload_rejects_non_boolean_target_mask():
    graph = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([[1.0, 0.0]]),
        y_mask=torch.tensor([[1.0, 0.0]]),
    )
    payload = FGLGraphDataset([graph], num_targets=2)
    with pytest.raises(ValueError, match="y_mask.*boolean"):
        validate_client_payload(payload, "graph", task="graph_reg")


def test_graph_payload_rejects_target_mask_shape_mismatch():
    graph = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([[1.0, 0.0]]),
        y_mask=torch.tensor([[True, False]]),
    )
    payload = FGLGraphDataset([graph], num_targets=2)
    graph.y_mask = torch.tensor([[True]])
    with pytest.raises(ValueError, match="y_mask.*shape"):
        validate_client_payload(payload, "graph", task="graph_reg")
