"""Versioned data contracts for public FedGB dataset payloads."""

from dataclasses import dataclass

import torch
from torch_geometric.data import Data


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class DatasetSchema:
    scenario: str
    payload_type: str
    required_fields: frozenset[str]


SCHEMAS = {
    "homo_subgraph": DatasetSchema(
        scenario="homo_subgraph",
        payload_type="Data",
        required_fields=frozenset({"x", "edge_index", "y", "global_map"}),
    ),
    "hetero_subgraph": DatasetSchema(
        scenario="hetero_subgraph",
        payload_type="Data",
        required_fields=frozenset(
            {"x", "edge_index", "y", "global_map", "node_type", "edge_type", "target_node_type"}
        ),
    ),
    "graph": DatasetSchema(
        scenario="graph",
        payload_type="FGLGraphDataset",
        required_fields=frozenset({"graphs"}),
    ),
}


def schema_for(scenario: str) -> DatasetSchema:
    try:
        return SCHEMAS[scenario]
    except KeyError as exc:
        raise ValueError(f"Unknown FedGB dataset scenario '{scenario}'.") from exc


def _missing_fields(payload, required_fields):
    return sorted(field for field in required_fields if not hasattr(payload, field))


def _validate_graph(data: Data, task: str, context: str) -> None:
    missing = _missing_fields(data, {"x", "edge_index", "y"})
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")
    if not torch.is_tensor(data.x) or data.x.ndim != 2:
        raise ValueError(f"{context}.x must be a two-dimensional tensor.")
    if not torch.is_tensor(data.edge_index) or data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        raise ValueError(f"{context}.edge_index must have shape [2, num_edges].")
    if data.edge_index.numel() and (data.edge_index.min() < 0 or data.edge_index.max() >= data.x.shape[0]):
        raise ValueError(f"{context}.edge_index contains an out-of-range node id.")
    if not torch.is_tensor(data.y):
        raise ValueError(f"{context}.y must be a tensor.")
    if task in {"node_cls", "graph_cls"} and data.y.dtype.is_floating_point:
        raise ValueError(f"{context}.y must use an integer dtype for classification.")
    if task == "graph_reg" and not data.y.dtype.is_floating_point:
        raise ValueError(f"{context}.y must use a floating dtype for regression.")
    if hasattr(data, "y_mask"):
        if not torch.is_tensor(data.y_mask) or data.y_mask.dtype != torch.bool:
            raise ValueError(f"{context}.y_mask must be a boolean tensor.")
        if data.y_mask.shape != data.y.shape:
            raise ValueError(f"{context}.y_mask shape must match y shape.")


def validate_client_payload(payload, scenario: str, task: str, context: str = "client payload") -> None:
    """Validate one released client payload against its scenario contract."""

    schema = schema_for(scenario)
    missing = _missing_fields(payload, schema.required_fields)
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")

    if scenario in {"homo_subgraph", "hetero_subgraph"}:
        if not isinstance(payload, Data):
            raise ValueError(f"{context} must be a torch_geometric.data.Data instance.")
        _validate_graph(payload, task, context)
        if payload.y.reshape(-1).shape[0] != payload.x.shape[0]:
            raise ValueError(f"{context}.y must contain one label per node.")
        if scenario == "hetero_subgraph":
            if not torch.is_tensor(payload.node_type) or payload.node_type.numel() != payload.x.shape[0]:
                raise ValueError(f"{context}.node_type must contain one value per node.")
            if not torch.is_tensor(payload.edge_type) or payload.edge_type.numel() != payload.edge_index.shape[1]:
                raise ValueError(f"{context}.edge_type must contain one value per edge.")
            if not isinstance(payload.target_node_type, str) or not payload.target_node_type:
                raise ValueError(f"{context}.target_node_type must be a non-empty string.")
        return

    from fedgb.data.fgl_graph_dataset import FGLGraphDataset

    if not isinstance(payload, FGLGraphDataset):
        raise ValueError(f"{context} must be an FGLGraphDataset instance.")
    if not payload.graphs:
        raise ValueError(f"{context} must contain at least one graph.")
    for graph_id, graph in enumerate(payload.graphs):
        _validate_graph(graph, task, f"{context}.graphs[{graph_id}]")


def validate_global_payload(payload, scenario: str, task: str, context: str = "global payload") -> None:
    """Validate a released global payload without requiring a client global_map."""

    if scenario == "graph":
        validate_client_payload(payload, scenario, task, context=context)
        return
    required = {"x", "edge_index", "y"}
    if scenario == "hetero_subgraph":
        required.update({"node_type", "edge_type", "target_node_type"})
    missing = _missing_fields(payload, required)
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")
    if not isinstance(payload, Data):
        raise ValueError(f"{context} must be a torch_geometric.data.Data instance.")
    _validate_graph(payload, task, context)
    if payload.y.reshape(-1).shape[0] != payload.x.shape[0]:
        raise ValueError(f"{context}.y must contain one label per node.")
    if scenario == "hetero_subgraph":
        if payload.node_type.numel() != payload.x.shape[0]:
            raise ValueError(f"{context}.node_type must contain one value per node.")
        if payload.edge_type.numel() != payload.edge_index.shape[1]:
            raise ValueError(f"{context}.edge_type must contain one value per edge.")
