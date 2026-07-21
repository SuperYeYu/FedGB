#!/usr/bin/env python3
"""Validate eICU_2 intermediates and final FedGB release directories."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eicu2_core import FORBIDDEN_STAY_FEATURES, max_split_ratio_error


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def validate_intermediate(root: Path) -> dict:
    schema = json.loads((root / "schema.json").read_text(encoding="utf-8"))
    stay_columns = {str(value).lower() for value in schema["stay_feature_columns"]}
    forbidden = sorted(stay_columns & FORBIDDEN_STAY_FEATURES)
    discharge_named = sorted(value for value in stay_columns if "discharge" in value)
    if forbidden or discharge_named:
        raise ValueError(f"forbidden intermediate features: {forbidden + discharge_named}")
    if set(schema["tasks"]) != {"icu_los_3class"}:
        raise ValueError("intermediate must contain only icu_los_3class")

    stays = read_rows(root / "stays.csv")
    splits = read_rows(root / "splits.csv")
    clients = read_rows(root / "clients.csv")
    if len(stays) != len(splits):
        raise ValueError("each stay must have one fixed split")
    labels = Counter(int(row["icu_los_3class"]) for row in stays)
    if set(labels) != {0, 1, 2}:
        raise ValueError(f"expected all three LOS classes, found {dict(labels)}")
    if any("mortality" in field.lower() or "discharge" in field.lower() for field in stays[0]):
        raise ValueError("stays.csv contains a forbidden task or discharge field")

    split_by_stay = {int(row["stay_nid"]): row["split"] for row in splits}
    patient_splits = defaultdict(set)
    per_client_class_split = defaultdict(Counter)
    for row in stays:
        key = (row["hospitalid"], row["uniquepid"])
        split = split_by_stay[int(row["stay_nid"])]
        patient_splits[key].add(split)
        per_client_class_split[(row["hospitalid"], int(row["icu_los_3class"]))][split] += 1
    leaks = [key for key, values in patient_splits.items() if len(values) != 1]
    if leaks:
        raise ValueError(f"patients crossing splits: {leaks[:5]}")

    ratios = {}
    for key, counts in per_client_class_split.items():
        total = sum(counts.values())
        ratios[f"{key[0]}:class{key[1]}"] = {
            split: counts[split] / total for split in ("train", "val", "test")
        }
        if total >= 40 and counts["train"] == 0:
            raise ValueError(f"client/class {key} has no training sample")
        error = max_split_ratio_error(counts, 0.05, 0.15)
        if error > 0.0200001:
            raise ValueError(f"client/class {key} split ratio error {error:.4f} exceeds 0.02")
    order = [row["hospitalid"] for row in clients]
    expected_order = [row["hospitalid"] for row in sorted(clients, key=lambda row: (-int(row["n_stays"]), int(row["hospitalid"])))]
    if order != expected_order:
        raise ValueError("clients.csv is not ordered by descending stay count")
    return {"stays": len(stays), "clients": len(clients), "labels": dict(labels), "ratios": ratios}


def _validate_variant(root: Path, expected_level: str):
    import torch

    manifest = json.loads((root / "fedgb_manifest.json").read_text(encoding="utf-8"))
    if manifest["level"] != expected_level:
        raise ValueError(f"{root}: unexpected level {manifest['level']}")
    partition = root / "distrib" / manifest["processed_partition"]
    client_records = []
    for client_id in range(int(manifest["num_clients"])):
        data = torch.load(partition / f"data_{client_id}.pt", map_location="cpu", weights_only=False)
        required = {"x", "edge_index", "y", "global_map"}
        if expected_level == "hetero_subgraph":
            required.update({"node_type", "edge_type", "target_node_type"})
        missing = sorted(field for field in required if not hasattr(data, field))
        if missing:
            raise ValueError(f"client {client_id} missing {missing}")
        if hasattr(data, "mortality_3class") or hasattr(data, "y2"):
            raise ValueError(f"client {client_id} contains a legacy task")
        if data.x.ndim != 2 or data.edge_index.shape[0] != 2 or data.y.numel() != data.x.shape[0]:
            raise ValueError(f"client {client_id} has invalid tensor shapes")
        if data.edge_index.numel() and (data.edge_index.min() < 0 or data.edge_index.max() >= data.x.shape[0]):
            raise ValueError(f"client {client_id} has out-of-range edges")
        if expected_level == "hetero_subgraph" and data.edge_type.numel() != data.edge_index.shape[1]:
            raise ValueError(f"client {client_id} edge_type length mismatch")
        split_root = partition / "node_cls" / "default_split"
        masks = {}
        global_sets = {}
        for split in ("train", "val", "test"):
            mask = torch.load(split_root / f"{split}_{client_id}.pt", map_location="cpu", weights_only=False)
            if mask.dtype != torch.bool or mask.ndim != 1 or mask.numel() != data.x.shape[0]:
                raise ValueError(f"client {client_id} invalid {split} mask")
            masks[split] = mask
            with (split_root / f"glb_{split}_{client_id}.pkl").open("rb") as handle:
                global_sets[split] = set(pickle.load(handle))
        if torch.any(masks["train"] & masks["val"]) or torch.any(masks["train"] & masks["test"]) or torch.any(masks["val"] & masks["test"]):
            raise ValueError(f"client {client_id} has overlapping masks")
        if any(global_sets[left] & global_sets[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"client {client_id} has overlapping global split IDs")
        selected = masks["train"] | masks["val"] | masks["test"]
        if not torch.all((data.y >= 0) == selected):
            raise ValueError(f"client {client_id} split masks do not exactly cover target nodes")
        if set(data.y[data.y >= 0].tolist()) - {0, 1, 2}:
            raise ValueError(f"client {client_id} has invalid labels")
        client_records.append(
            {
                "client_id": client_id,
                "hospitalid": int(data.hospitalid),
                "nodes": int(data.x.shape[0]),
                "edges": int(data.edge_index.shape[1]),
                "train": int(masks["train"].sum()),
                "val": int(masks["val"].sum()),
                "test": int(masks["test"].sum()),
            }
        )
    global_path = root / "global" / "subgraph_fl" / manifest["dataset_id"].lower() / "processed" / "data.pt"
    global_data = torch.load(global_path, map_location="cpu", weights_only=False)
    if hasattr(global_data, "mortality_3class") or hasattr(global_data, "y2"):
        raise ValueError("global payload contains a legacy task")
    return manifest, client_records


def validate_fedgb(output_root: Path, fedgb_root: Path | None = None) -> dict:
    import torch

    het_manifest, het_clients = _validate_variant(output_root / "het", "hetero_subgraph")
    hom_manifest, hom_clients = _validate_variant(output_root / "hom", "homo_subgraph")
    if [row["hospitalid"] for row in het_clients] != [row["hospitalid"] for row in hom_clients]:
        raise ValueError("Het/Hom client order differs")
    for het, hom in zip(het_clients, hom_clients):
        if (het["train"], het["val"], het["test"]) != (hom["train"], hom["val"], hom["test"]):
            raise ValueError(f"Het/Hom split mismatch for client {het['client_id']}")

    loader_results = {}
    if fedgb_root is not None:
        sys.path.insert(0, str(fedgb_root))
        from fedgb.data.release_loader import load_dataset_bundle

        for variant, manifest in (("het", het_manifest), ("hom", hom_manifest)):
            bundle = load_dataset_bundle(output_root / variant, load_global=True)
            loader_results[variant] = {
                "clients": len(bundle.clients),
                "scenario": bundle.scenario,
                "global_loaded": bundle.global_data is not None,
                "manifest": manifest["name"],
            }
    global_hom = torch.load(
        output_root / "hom" / "global" / "subgraph_fl" / hom_manifest["dataset_id"].lower() / "processed" / "data.pt",
        map_location="cpu",
        weights_only=False,
    )
    global_het = torch.load(
        output_root / "het" / "global" / "subgraph_fl" / het_manifest["dataset_id"].lower() / "processed" / "data.pt",
        map_location="cpu",
        weights_only=False,
    )
    metadata = json.loads((output_root / "het" / "metadata" / "manifest.json").read_text(encoding="utf-8"))
    class_counts = Counter(int(value) for value in global_hom.y.tolist())
    aggregate = {
        "num_clients": len(het_clients),
        "num_stays": sum(row["train"] + row["val"] + row["test"] for row in hom_clients),
        "num_patients": int(metadata["global_counts"]["patients"]),
        "feature_dim": int(global_hom.x.shape[1]),
        "num_classes": len(class_counts),
        "num_tasks": 1,
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "split_counts": {
            split: sum(row[split] for row in hom_clients) for split in ("train", "val", "test")
        },
        "heterogeneous": {
            "num_unique_nodes": int(global_het.x.shape[0]),
            "num_node_types": len(getattr(global_het, "hetero_node_types", [])),
            "num_directed_relation_types": len(getattr(global_het, "hetero_edge_types", [])),
            "num_forward_semantic_edges": int(global_het.edge_index.shape[1]) // 2,
        },
        "homogeneous": {
            "num_nodes": sum(row["nodes"] for row in hom_clients),
            "num_undirected_edges": sum(row["edges"] for row in hom_clients) // 2,
        },
    }
    return {"het": het_clients, "hom": hom_clients, "aggregate": aggregate, "fedgb_loader": loader_results}


def validate_expected_contract(report: dict, expected_path: Path) -> None:
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    aggregate = report["aggregate"]
    for key in ("num_clients", "num_stays", "num_patients", "feature_dim", "num_classes", "num_tasks"):
        if aggregate[key] != expected[key]:
            raise ValueError(f"EICU contract mismatch for {key}: {aggregate[key]} != {expected[key]}")
    if aggregate["split_counts"] != expected["split_counts"]:
        raise ValueError("EICU split counts differ from expected_contract.json")
    if aggregate["class_counts"] != expected["class_counts"]:
        raise ValueError("EICU class counts differ from expected_contract.json")
    for section, keys in {
        "heterogeneous": (
            "num_unique_nodes",
            "num_node_types",
            "num_directed_relation_types",
            "num_forward_semantic_edges",
        ),
        "homogeneous": ("num_nodes", "num_undirected_edges"),
    }.items():
        for key in keys:
            if aggregate[section][key] != expected[section][key]:
                raise ValueError(
                    f"EICU contract mismatch for {section}.{key}: "
                    f"{aggregate[section][key]} != {expected[section][key]}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fedgb-root", type=Path)
    parser.add_argument("--expected-contract", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {}
    if args.intermediate_root:
        report["intermediate"] = validate_intermediate(args.intermediate_root)
    if args.output_root:
        report["fedgb"] = validate_fedgb(args.output_root, args.fedgb_root)
        if args.expected_contract:
            validate_expected_contract(report["fedgb"], args.expected_contract)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
