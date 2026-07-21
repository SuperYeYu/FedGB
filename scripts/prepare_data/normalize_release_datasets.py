#!/usr/bin/env python3
"""Normalize FedGB dataset metadata and fixed split layout in place."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import shutil

import torch

from fedgb.config.datasets import dataset_registry
from fedgb.data.schema import SCHEMA_VERSION
from fedgb.data.validation import INTERNAL_PATH_PATTERN


ROOT = Path(__file__).resolve().parents[2]


def _partition(root: Path) -> Path:
    candidates = sorted(path for path in (root / "distrib").iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"{root.name}: expected one processed partition, found {len(candidates)}.")
    return candidates[0]


def _sanitize_text_files(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt", ".md", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        sanitized = INTERNAL_PATH_PATTERN.sub("<private-source>/", text)
        if sanitized != text:
            path.write_text(sanitized, encoding="utf-8")


def normalize_dataset_root(root: Path) -> None:
    """Sanitize release-level metadata outside individual variant directories."""

    _sanitize_text_files(Path(root))


def _copy_external_split(source: Path, target: Path, num_clients: int) -> None:
    if not source.is_dir():
        raise ValueError(f"External split directory does not exist: {source}")
    target.mkdir(parents=True, exist_ok=True)
    for client_id in range(num_clients):
        for prefix, suffix in (
            ("train", ".pt"), ("val", ".pt"), ("test", ".pt"),
            ("glb_train", ".pkl"), ("glb_val", ".pkl"), ("glb_test", ".pkl"),
        ):
            source_file = source / f"{prefix}_{client_id}{suffix}"
            if not source_file.is_file():
                raise ValueError(f"Missing external split file: {source_file}")
            shutil.copy2(source_file, target / source_file.name)


def _export_embedded_splits(partition: Path, task: str, num_clients: int) -> None:
    split_root = partition / task / "default_split"
    split_root.mkdir(parents=True, exist_ok=True)
    for client_id in range(num_clients):
        data = torch.load(partition / f"data_{client_id}.pt", map_location="cpu", weights_only=False)
        if hasattr(data, "graphs"):
            split_values = [getattr(graph, "split", None) for graph in data.graphs]
            if not all(value in {"train", "val", "test"} for value in split_values):
                continue
            masks = {
                name: torch.tensor([value == name for value in split_values], dtype=torch.bool)
                for name in ("train", "val", "test")
            }
        elif all(hasattr(data, f"{name}_mask") for name in ("train", "val", "test")):
            masks = {name: getattr(data, f"{name}_mask").bool().cpu() for name in ("train", "val", "test")}
        else:
            continue
        for name, mask in masks.items():
            path = split_root / f"{name}_{client_id}.pt"
            if not path.exists():
                torch.save(mask, path)
            global_ids = []
            global_map = getattr(data, "global_map", {})
            for local_id in mask.nonzero(as_tuple=True)[0].tolist():
                if isinstance(global_map, dict):
                    global_ids.append(int(global_map.get(local_id, local_id)))
                else:
                    global_ids.append(int(global_map[local_id]))
            glb_path = split_root / f"glb_{name}_{client_id}.pkl"
            if not glb_path.exists():
                with glb_path.open("wb") as stream:
                    pickle.dump(global_ids, stream)


def _normalize_global_payload(root: Path, spec: dict) -> None:
    if spec["level"] != "hetero_subgraph":
        return
    path = root / "global" / "subgraph_fl" / spec["dataset_id"].lower() / "processed" / "data.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, tuple):
        raw_data = payload[0]
        data_cls = payload[2] if len(payload) >= 3 and isinstance(payload[2], type) else None
        if isinstance(raw_data, dict) and data_cls is not None:
            payload = data_cls.from_dict(raw_data)
    if hasattr(payload, "node_types"):
        from fedgb.data.heterogeneous import hetero_to_relation_data

        payload = hetero_to_relation_data(payload, spec.get("target_node"))
        torch.save(payload, path)


def normalize_variant(root, spec: dict, external_split: Path | None = None) -> dict:
    root = Path(root)
    partition = _partition(root)
    task = spec["task"]
    num_clients = int(spec["num_clients"])
    split_root = partition / task / "default_split"
    if external_split is not None:
        _copy_external_split(Path(external_split), split_root, num_clients)
    _export_embedded_splits(partition, task, num_clients)
    _normalize_global_payload(root, spec)
    missing = [
        str(split_root / f"{name}_{client_id}.pt")
        for client_id in range(num_clients)
        for name in ("train", "val", "test")
        if not (split_root / f"{name}_{client_id}.pt").is_file()
    ]
    if missing:
        raise ValueError(f"{spec['name']}: fixed split normalization is incomplete: {missing[:3]}")

    manifest = dict(spec)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "processed_partition": partition.name,
            "split": f"{task}/default_split",
        }
    )
    for key in ("source_host", "source_root", "resolved_source_root", "dataset_root", "destination_root", "files"):
        manifest.pop(key, None)
    (root / "fedgb_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _sanitize_text_files(root)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--dataset", action="append")
    parser.add_argument(
        "--external-split",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Copy an exact fixed split from PATH for a named dataset.",
    )
    opts = parser.parse_args()
    external = {}
    for item in opts.external_split:
        name, path = item.split("=", 1)
        external[name] = Path(path)
    registry = dataset_registry()
    names = opts.dataset or sorted(registry)
    for name in names:
        manifest = normalize_variant(
            opts.dataset_root / name,
            registry[name],
            external_split=external.get(name),
        )
        print(name, manifest["processed_partition"])
    normalize_dataset_root(opts.dataset_root)


if __name__ == "__main__":
    main()
