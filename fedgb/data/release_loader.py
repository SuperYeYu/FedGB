"""Single loading boundary for all released FedGB dataset scenarios."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from fedgb.data.schema import SCHEMA_VERSION, validate_client_payload


@dataclass
class DatasetBundle:
    name: str
    scenario: str
    task: str
    manifest: dict
    clients: list
    splits: list[dict[str, torch.Tensor]]
    global_data: object | None = None


def normalize_client_payload(payload, scenario: str, task: str, target_node: str | None = None):
    """Convert legacy payloads and normalize public tensor dtypes."""

    if isinstance(payload, HeteroData) or hasattr(payload, "node_types"):
        from fedgb.data.heterogeneous import hetero_to_relation_data

        payload = hetero_to_relation_data(payload, target_node)

    if scenario == "graph":
        for graph in payload.graphs:
            graph.x = graph.x.to(torch.float32)
            graph.edge_index = graph.edge_index.to(torch.int64)
            if task == "graph_reg":
                graph.y = graph.y.to(torch.float32).view(1, -1)
            else:
                graph.y = graph.y.squeeze()
            if hasattr(graph, "y_mask"):
                graph.y_mask = graph.y_mask.bool().view_as(graph.y)
        payload.y = payload._stack_y()
        if hasattr(payload, "_stack_y_mask"):
            payload.y_mask = payload._stack_y_mask()
        validate_client_payload(payload, scenario, task)
        return payload

    validate_client_payload(payload, scenario, task)

    if scenario in {"homo_subgraph", "hetero_subgraph"}:
        payload.x = payload.x.to(torch.float32)
        payload.edge_index = payload.edge_index.to(torch.int64)
        payload.y = payload.y.squeeze()
        if scenario == "hetero_subgraph":
            payload.node_type = payload.node_type.to(torch.int64)
            payload.edge_type = payload.edge_type.to(torch.int64)
    return payload


def load_client_payload(path, scenario: str, task: str, target_node: str | None = None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return normalize_client_payload(payload, scenario, task, target_node=target_node)


def _partition_path(root: Path, spec: dict) -> Path:
    partition = spec.get("processed_partition")
    if partition:
        return root / "distrib" / partition
    candidates = sorted(path for path in (root / "distrib").iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"{root.name}: expected one processed partition, found {len(candidates)}.")
    return candidates[0]


def _load_splits(partition: Path, task: str, client_id: int, expected_length: int) -> dict[str, torch.Tensor]:
    split_root = partition / task / "default_split"
    result = {}
    for name in ("train", "val", "test"):
        path = split_root / f"{name}_{client_id}.pt"
        if not path.is_file():
            raise ValueError(f"Missing fixed split file: {path}")
        mask = torch.load(path, map_location="cpu", weights_only=False)
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError(f"{path} must contain a one-dimensional boolean tensor.")
        if mask.numel() != expected_length:
            raise ValueError(
                f"{path} has length {mask.numel()}, expected {expected_length} for client {client_id}."
            )
        result[name] = mask
    if torch.any(result["train"] & result["val"]) or torch.any(result["train"] & result["test"]) or torch.any(result["val"] & result["test"]):
        raise ValueError(f"Client {client_id} has overlapping train/validation/test splits.")
    return result


def iter_dataset_clients(root, spec: dict | None = None):
    """Yield validated client payloads one at a time for bounded-memory audits."""

    root = Path(root)
    manifest_path = root / "fedgb_manifest.json"
    manifest = dict(spec) if spec is not None else json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{root.name}: expected schema_version {SCHEMA_VERSION}, found {manifest.get('schema_version')!r}."
        )
    scenario = manifest["level"]
    task = manifest["task"]
    partition = _partition_path(root, manifest)
    for client_id in range(int(manifest["num_clients"])):
        payload = load_client_payload(
            partition / f"data_{client_id}.pt",
            scenario=scenario,
            task=task,
            target_node=manifest.get("target_node"),
        )
        expected_length = len(payload.graphs) if scenario == "graph" else int(payload.x.shape[0])
        yield client_id, payload, _load_splits(partition, task, client_id, expected_length)


def load_dataset_bundle(root, spec: dict | None = None, load_global: bool = True) -> DatasetBundle:
    root = Path(root)
    manifest_path = root / "fedgb_manifest.json"
    manifest = dict(spec) if spec is not None else json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{root.name}: expected schema_version {SCHEMA_VERSION}, found {manifest.get('schema_version')!r}."
        )
    scenario = manifest["level"]
    task = manifest["task"]
    partition = _partition_path(root, manifest)
    num_clients = int(manifest["num_clients"])
    records = list(iter_dataset_clients(root, manifest))
    clients = [payload for _, payload, _ in records]
    splits = [client_splits for _, _, client_splits in records]

    global_data = None
    if load_global:
        from fedgb.data.global_dataset_loader import load_global_dataset

        internal_scenario = "graph_fl" if scenario == "graph" else "subgraph_fl"
        global_data = load_global_dataset(
            str(root / "global"),
            scenario=internal_scenario,
            dataset=manifest["dataset_id"],
        )
    return DatasetBundle(
        name=manifest["name"],
        scenario=scenario,
        task=task,
        manifest=manifest,
        clients=clients,
        splits=splits,
        global_data=global_data,
    )
