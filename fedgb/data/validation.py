"""Validation helpers for public FedGB dataset archives."""

import json
from pathlib import Path
import re

from fedgb.data.release_loader import iter_dataset_clients
from fedgb.data.global_dataset_loader import load_global_dataset
from fedgb.data.schema import validate_global_payload


INTERNAL_PATH_PATTERN = re.compile(r"/(?:opt/data/private|data/zfzhu_nas)/yyy(?:/|\b)")
REGISTRY_ONLY_FIELDS = {
    "availability",
    "builder",
    "edge_feature_dim",
    "feature_dim",
    "num_classes",
    "num_tasks",
    "task_name",
}


def find_internal_paths(root: Path) -> list[str]:
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt", ".md", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if INTERNAL_PATH_PATTERN.search(text):
            offenders.append(str(path.relative_to(root)))
    return sorted(offenders)


def find_internal_paths_in_binary(root: Path) -> list[str]:
    markers = (b"/opt/data/private" + b"/yyy", b"/data/zfzhu_nas" + b"/yyy")
    offenders = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".pt", ".pkl", ".pickle"}:
            continue
        overlap = b""
        with path.open("rb") as stream:
            while chunk := stream.read(16 * 1024 * 1024):
                content = overlap + chunk
                if any(marker in content for marker in markers):
                    offenders.append(str(path.relative_to(root)))
                    break
                overlap = content[-max(len(marker) for marker in markers) :]
    return sorted(offenders)


def validate_dataset_variant(root, spec: dict | None = None) -> dict:
    root = Path(root)
    disk_manifest = json.loads((root / "fedgb_manifest.json").read_text(encoding="utf-8"))
    manifest = dict(spec) if spec is not None else disk_manifest
    if spec is not None:
        payload_spec = {key: value for key, value in spec.items() if key not in REGISTRY_ONLY_FIELDS}
        compared = {key: disk_manifest.get(key) for key in payload_spec}
        if compared != payload_spec:
            raise ValueError(f"{root.name}: manifest mismatch between dataset and packaged registry.")
    internal_paths = sorted(set(find_internal_paths(root) + find_internal_paths_in_binary(root)))
    if internal_paths:
        raise ValueError(
            f"{manifest.get('name', root.name)} contains an internal absolute path in: {internal_paths}"
        )
    feature_dims = set()
    edge_feature_dims = set()
    inspected = 0
    for _, payload, _ in iter_dataset_clients(root, manifest):
        inspected += 1
        if manifest["level"] == "graph":
            feature_dims.update(int(graph.x.shape[1]) for graph in payload.graphs)
            edge_feature_dims.update(
                int(graph.edge_attr.shape[1])
                for graph in payload.graphs
                if getattr(graph, "edge_attr", None) is not None
            )
        else:
            feature_dims.add(int(payload.x.shape[1]))
    expected_dim = manifest.get("feature_dim")
    if expected_dim is not None and feature_dims != {int(expected_dim)}:
        raise ValueError(
            f"{manifest['name']}: expected feature_dim {expected_dim}, found {sorted(feature_dims)}."
        )
    expected_edge_dim = manifest.get("edge_feature_dim")
    if expected_edge_dim is not None and edge_feature_dims != {int(expected_edge_dim)}:
        raise ValueError(
            f"{manifest['name']}: expected edge_feature_dim {expected_edge_dim}, "
            f"found {sorted(edge_feature_dims)}."
        )
    internal_scenario = "graph_fl" if manifest["level"] == "graph" else "subgraph_fl"
    global_dataset = load_global_dataset(
        str(root / "global"),
        scenario=internal_scenario,
        dataset=manifest["dataset_id"],
    )
    global_payload = global_dataset if manifest["level"] == "graph" else global_dataset.data
    validate_global_payload(global_payload, manifest["level"], manifest["task"])
    return {
        "dataset": manifest["name"],
        "schema_version": manifest["schema_version"],
        "scenario": manifest["level"],
        "task": manifest["task"],
        "clients": inspected,
        "inspected_payloads": inspected,
        "feature_dims": sorted(feature_dims),
        "edge_feature_dims": sorted(edge_feature_dims),
        "global_payload": f"{global_payload.__class__.__module__}.{global_payload.__class__.__name__}",
    }
