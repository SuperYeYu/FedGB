#!/usr/bin/env python3
"""Export eICU_2 intermediates to the public FedGB subgraph contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data


NODE_TYPES = ("patient", "stay", "diagnosis_concept", "treatment_concept", "medication_concept")
FORWARD_RELATIONS = (
    ("patient", "has_stay", "stay"),
    ("stay", "has_diagnosis", "diagnosis_concept"),
    ("stay", "has_treatment", "treatment_concept"),
    ("stay", "has_medication", "medication_concept"),
)
EDGE_TYPES = tuple(
    relation
    for forward in FORWARD_RELATIONS
    for relation in (forward, (forward[2], "rev_" + forward[1], forward[0]))
)
CONCEPT_SPECS = (
    ("diagnosis_concept", "diagnosis_concepts.csv", "diagnosis_nid", "edges_stay_diagnosis.csv"),
    ("treatment_concept", "treatment_concepts.csv", "treatment_nid", "edges_stay_treatment.csv"),
    ("medication_concept", "medication_concepts.csv", "medication_nid", "edges_stay_medication.csv"),
)


def read_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def read_edge_index(path: Path, source_column: str, target_column: str) -> torch.Tensor:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        pairs = [(int(row[source_column]), int(row[target_column])) for row in reader]
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def parse_number(value: object):
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def encode_structured(rows: Sequence[dict], columns: Sequence[str]) -> torch.Tensor:
    values = np.zeros((len(rows), len(columns)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            number = parse_number(row.get(column))
            if number is not None:
                values[row_index, column_index] = number
    return torch.from_numpy(values)


def _tokens(row: Mapping[str, object]) -> Iterable[str]:
    for field, raw in sorted(row.items()):
        if field.endswith("_nid"):
            continue
        text = str(raw or "").strip().lower()
        if not text:
            continue
        yield f"field:{field}"
        for token in re.findall(r"[a-z0-9]+", text):
            yield f"{field}:{token}"
        parts = [part.strip() for part in re.split(r"[|>/]", text) if part.strip()]
        for depth, part in enumerate(parts[:8]):
            yield f"{field}:path:{depth}:{part}"


def encode_concepts(rows: Sequence[dict], hash_dim: int = 128) -> torch.Tensor:
    values = np.zeros((len(rows), hash_dim + 4), dtype=np.float32)
    for row_index, row in enumerate(rows):
        seen = Counter(_tokens(row))
        for token, count in seen.items():
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) % hash_dim
            values[row_index, bucket] += float(count)
        text = " ".join(str(value or "") for key, value in row.items() if not key.endswith("_nid"))
        values[row_index, hash_dim] = math.log1p(len(text))
        values[row_index, hash_dim + 1] = min(8, text.count("|") + 1)
        values[row_index, hash_dim + 2] = float(bool(row.get("icd9code") or row.get("drughiclseqno")))
        values[row_index, hash_dim + 3] = 1.0
        norm = float(np.linalg.norm(values[row_index, :hash_dim]))
        if norm > 0:
            values[row_index, :hash_dim] /= norm
    return torch.from_numpy(values)


def pad_features(features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    width = max(int(tensor.shape[1]) for tensor in features.values())
    padded = {}
    for node_type, tensor in features.items():
        if tensor.shape[1] < width:
            tensor = torch.cat((tensor, torch.zeros((tensor.shape[0], width - tensor.shape[1]))), dim=1)
        padded[node_type] = tensor.to(torch.float32)
    return padded


def load_intermediate(source_root: Path):
    schema = json.loads((source_root / "schema.json").read_text(encoding="utf-8"))
    if schema.get("schema_version") != "eicu2-intermediate-1.0":
        raise ValueError("unsupported intermediate schema")
    patients = sorted(read_rows(source_root / "patients.csv"), key=lambda row: int(row["patient_nid"]))
    stays = sorted(read_rows(source_root / "stays.csv"), key=lambda row: int(row["stay_nid"]))
    stay_features = sorted(read_rows(source_root / "stay_features.csv"), key=lambda row: int(row["stay_nid"]))
    clients = sorted(read_rows(source_root / "clients.csv"), key=lambda row: int(row["client_id"]))
    splits = {int(row["stay_nid"]): row["split"] for row in read_rows(source_root / "splits.csv")}
    if len(stays) != len(stay_features) or [row["stay_nid"] for row in stays] != [row["stay_nid"] for row in stay_features]:
        raise ValueError("stays.csv and stay_features.csv are misaligned")

    concept_rows = {}
    relation_edges = {}
    for node_type, concept_file, id_column, edge_file in CONCEPT_SPECS:
        concept_rows[node_type] = sorted(read_rows(source_root / concept_file), key=lambda row: int(row[id_column]))
        relation_edges[node_type] = read_edge_index(source_root / edge_file, "stay_nid", id_column)
    relation_edges["patient"] = read_edge_index(source_root / "edges_patient_stay.csv", "patient_nid", "stay_nid")

    features = {
        "patient": encode_structured(patients, schema["patient_feature_columns"]),
        "stay": encode_structured(stay_features, schema["stay_feature_columns"]),
        **{node_type: encode_concepts(rows) for node_type, rows in concept_rows.items()},
    }
    features = pad_features(features)
    labels = torch.tensor([int(row["icu_los_3class"]) for row in stays], dtype=torch.long)
    if labels.numel() == 0 or labels.min().item() < 0 or labels.max().item() > 2:
        raise ValueError("LOS labels must be non-empty and in {0,1,2}")
    return schema, patients, stays, clients, splits, concept_rows, relation_edges, features, labels


def global_offsets(features: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    offsets = {}
    cursor = 0
    for node_type in NODE_TYPES:
        offsets[node_type] = cursor
        cursor += int(features[node_type].shape[0])
    return offsets


def filter_medication_hubs(edge: torch.Tensor, quantile: float = 0.99) -> Tuple[torch.Tensor, dict]:
    if edge.numel() == 0 or quantile >= 1.0:
        return edge, {"quantile": quantile, "cutoff": None, "removed_edges": 0}
    concept_ids, counts = torch.unique(edge[1], return_counts=True)
    cutoff = float(torch.quantile(counts.to(torch.float64), quantile).item())
    kept_concepts = concept_ids[counts.to(torch.float64) <= cutoff]
    keep = torch.isin(edge[1], kept_concepts)
    return edge[:, keep], {
        "quantile": quantile,
        "cutoff": cutoff,
        "removed_edges": int((~keep).sum().item()),
        "removed_concepts": int((counts.to(torch.float64) > cutoff).sum().item()),
    }


def global_relation_edges(relation_edges: Mapping[str, torch.Tensor]) -> List[torch.Tensor]:
    patient_stay = relation_edges["patient"]
    result = [patient_stay, patient_stay.flip(0)]
    for node_type, _, id_column, _ in CONCEPT_SPECS:
        edge = relation_edges[node_type]
        result.extend((edge, edge.flip(0)))
    return result


def flatten_graph(
    node_ids: Mapping[str, Sequence[int]],
    features: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    relation_edges: Sequence[torch.Tensor],
    offsets: Mapping[str, int],
) -> Data:
    local_offsets = {}
    global_to_local = {}
    xs, ys, node_types = [], [], []
    cursor = 0
    global_map = {}
    for type_id, node_type in enumerate(NODE_TYPES):
        ids = list(node_ids[node_type])
        local_offsets[node_type] = cursor
        mapping = torch.full((features[node_type].shape[0],), -1, dtype=torch.long)
        if ids:
            mapping[torch.tensor(ids, dtype=torch.long)] = torch.arange(len(ids), dtype=torch.long)
        global_to_local[node_type] = mapping
        for local_type_id, global_type_id in enumerate(ids):
            global_map[cursor + local_type_id] = int(offsets[node_type]) + int(global_type_id)
        tensor_ids = torch.tensor(ids, dtype=torch.long)
        xs.append(features[node_type][tensor_ids] if ids else features[node_type].new_empty((0, features[node_type].shape[1])))
        if node_type == "stay":
            ys.append(labels[tensor_ids] if ids else torch.empty((0,), dtype=torch.long))
        else:
            ys.append(torch.full((len(ids),), -1, dtype=torch.long))
        node_types.append(torch.full((len(ids),), type_id, dtype=torch.long))
        cursor += len(ids)

    edge_parts = []
    edge_type_parts = []
    for edge_type_id, (edge_type, edge_index) in enumerate(zip(EDGE_TYPES, relation_edges)):
        src_type, _, dst_type = edge_type
        if edge_index.numel() == 0:
            continue
        src_local = global_to_local[src_type][edge_index[0]]
        dst_local = global_to_local[dst_type][edge_index[1]]
        keep = (src_local >= 0) & (dst_local >= 0)
        if torch.any(keep):
            shifted = torch.stack(
                (src_local[keep] + local_offsets[src_type], dst_local[keep] + local_offsets[dst_type])
            )
            edge_parts.append(shifted.contiguous())
            edge_type_parts.append(torch.full((shifted.shape[1],), edge_type_id, dtype=torch.long))

    data = Data(
        x=torch.cat(xs, dim=0),
        y=torch.cat(ys, dim=0),
        edge_index=torch.cat(edge_parts, dim=1) if edge_parts else torch.empty((2, 0), dtype=torch.long),
    )
    data.node_type = torch.cat(node_types, dim=0)
    data.edge_type = torch.cat(edge_type_parts) if edge_type_parts else torch.empty((0,), dtype=torch.long)
    data.target_node_type = "stay"
    data.hetero_node_types = list(NODE_TYPES)
    data.hetero_edge_types = list(EDGE_TYPES)
    data.global_map = global_map
    data.num_global_classes = 3
    return data


def client_node_ids(
    hospital_id: str,
    stays: Sequence[dict],
    relation_edges: Mapping[str, torch.Tensor],
) -> Dict[str, List[int]]:
    stay_ids = sorted(int(row["stay_nid"]) for row in stays if row["hospitalid"] == hospital_id)
    selected_stays = set(stay_ids)
    patient_ids = sorted({int(row["patient_nid"]) for row in stays if int(row["stay_nid"]) in selected_stays})
    result = {"patient": patient_ids, "stay": stay_ids}
    for node_type, _, id_column, _ in CONCEPT_SPECS:
        edge = relation_edges[node_type]
        if edge.numel() == 0:
            result[node_type] = []
            continue
        stay_mask = torch.zeros(max(int(edge[0].max().item()) + 1, max(stay_ids, default=-1) + 1), dtype=torch.bool)
        if stay_ids:
            stay_mask[torch.tensor(stay_ids, dtype=torch.long)] = True
        result[node_type] = torch.unique(edge[1, stay_mask[edge[0]]], sorted=True).tolist()
    return result


def save_fixed_splits(
    partition_root: Path,
    client_id: int,
    data: Data,
    stay_ids: Sequence[int],
    splits: Mapping[int, str],
    global_stay_offset: int,
    hetero: bool,
) -> Dict[str, int]:
    split_root = partition_root / "node_cls" / "default_split"
    split_root.mkdir(parents=True, exist_ok=True)
    masks = {name: torch.zeros(data.x.shape[0], dtype=torch.bool) for name in ("train", "val", "test")}
    target_positions = (data.node_type == NODE_TYPES.index("stay")).nonzero(as_tuple=False).view(-1).tolist() if hetero else list(range(len(stay_ids)))
    global_ids = {name: [] for name in masks}
    for local_stay_index, global_stay_id in enumerate(stay_ids):
        split = splits[int(global_stay_id)]
        masks[split][target_positions[local_stay_index]] = True
        global_ids[split].append(global_stay_offset + int(global_stay_id))
    for name, mask in masks.items():
        torch.save(mask, split_root / f"{name}_{client_id}.pt")
        with (split_root / f"glb_{name}_{client_id}.pkl").open("wb") as handle:
            pickle.dump(global_ids[name], handle)
    return {name: int(mask.sum().item()) for name, mask in masks.items()}


def homogeneous_edges(
    stay_ids: Sequence[int],
    relation_edges: Mapping[str, torch.Tensor],
    threshold: int,
) -> torch.Tensor:
    stay_to_local = {int(stay_id): index for index, stay_id in enumerate(stay_ids)}
    row_parts, column_parts = [], []
    concept_offset = 0
    for node_type, _, id_column, _ in CONCEPT_SPECS[1:]:
        edge = relation_edges[node_type]
        if edge.numel() == 0:
            continue
        map_size = max(int(edge[0].max().item()) + 1, max(stay_ids, default=-1) + 1)
        local_map = torch.full((map_size,), -1, dtype=torch.long)
        local_map[torch.tensor(stay_ids, dtype=torch.long)] = torch.arange(len(stay_ids), dtype=torch.long)
        local_stays = local_map[edge[0]]
        keep = local_stays >= 0
        if not torch.any(keep):
            continue
        concepts, inverse = torch.unique(edge[1, keep], sorted=True, return_inverse=True)
        row_parts.append(local_stays[keep].numpy())
        column_parts.append(inverse.numpy() + concept_offset)
        concept_offset += int(concepts.numel())
    if not row_parts or concept_offset == 0:
        return torch.empty((2, 0), dtype=torch.long)
    rows = np.concatenate(row_parts)
    columns = np.concatenate(column_parts)
    incidence = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, columns)),
        shape=(len(stay_ids), concept_offset),
    ).tocsr()
    incidence.sum_duplicates()
    incidence.data[:] = 1
    projected = (incidence @ incidence.T).tocsr()
    projected.setdiag(0)
    projected.eliminate_zeros()
    projected.data = (projected.data > threshold).astype(np.uint8)
    projected.eliminate_zeros()
    src, dst = projected.nonzero()
    return torch.from_numpy(np.vstack((src, dst)).astype(np.int64, copy=False))


def _save_global(path: Path, data: Data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def _write_manifest(root: Path, name: str, level: str, num_clients: int) -> str:
    partition = f"subgraph_fl_louvain_1_ACM_client_{num_clients}"
    manifest = {
        "dataset_id": "ACM",
        "level": level,
        "name": name,
        "num_clients": num_clients,
        "processed_partition": partition,
        "schema_version": "1.0",
        "source_group": "EICU",
        "split": "node_cls/default_split",
        "target_node": "stay" if level == "hetero_subgraph" else None,
        "task": "node_cls",
    }
    if manifest["target_node"] is None:
        del manifest["target_node"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "fedgb_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return partition


def export_fedgb(source_root: Path, output_root: Path, hom_threshold: int = 3) -> dict:
    source_root = Path(source_root)
    output_root = Path(output_root)
    schema, patients, stays, clients, splits, concept_rows, relation_edges, features, labels = load_intermediate(source_root)
    num_clients = len(clients)
    offsets = global_offsets(features)
    het_relation_edges = dict(relation_edges)
    het_relation_edges["medication_concept"], medication_filter = filter_medication_hubs(
        relation_edges["medication_concept"], quantile=0.99
    )
    flattened_relation_edges = global_relation_edges(het_relation_edges)
    all_ids = {node_type: list(range(features[node_type].shape[0])) for node_type in NODE_TYPES}

    het_root = output_root / "het"
    hom_root = output_root / "hom"
    het_partition_name = _write_manifest(het_root, "EICU-Het", "hetero_subgraph", num_clients)
    hom_partition_name = _write_manifest(hom_root, "EICU-Hom", "homo_subgraph", num_clients)
    het_partition = het_root / "distrib" / het_partition_name
    hom_partition = hom_root / "distrib" / hom_partition_name
    het_partition.mkdir(parents=True, exist_ok=True)
    hom_partition.mkdir(parents=True, exist_ok=True)

    global_het = flatten_graph(all_ids, features, labels, flattened_relation_edges, offsets)
    _save_global(het_root / "global" / "subgraph_fl" / "acm" / "processed" / "data.pt", global_het)
    global_hom = Data(
        x=features["stay"].clone(),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        y=labels.clone(),
    )
    global_hom.global_map = {index: index for index in range(len(stays))}
    global_hom.num_global_classes = 3
    _save_global(hom_root / "global" / "subgraph_fl" / "acm" / "processed" / "data.pt", global_hom)

    client_stats = []
    for client in clients:
        client_id = int(client["client_id"])
        hospital_id = str(client["hospitalid"])
        ids = client_node_ids(hospital_id, stays, het_relation_edges)
        het_data = flatten_graph(ids, features, labels, flattened_relation_edges, offsets)
        het_data.hospitalid = int(hospital_id)
        het_data.client_name = f"client_{hospital_id}"
        torch.save(het_data, het_partition / f"data_{client_id}.pt")
        het_splits = save_fixed_splits(
            het_partition, client_id, het_data, ids["stay"], splits, offsets["stay"], True
        )

        stay_tensor_ids = torch.tensor(ids["stay"], dtype=torch.long)
        hom_data = Data(
            x=features["stay"][stay_tensor_ids],
            edge_index=homogeneous_edges(ids["stay"], relation_edges, hom_threshold),
            y=labels[stay_tensor_ids],
        )
        hom_data.global_map = {index: int(stay_id) for index, stay_id in enumerate(ids["stay"])}
        hom_data.num_global_classes = 3
        hom_data.hospitalid = int(hospital_id)
        hom_data.client_name = f"client_{hospital_id}"
        torch.save(hom_data, hom_partition / f"data_{client_id}.pt")
        hom_splits = save_fixed_splits(hom_partition, client_id, hom_data, ids["stay"], splits, 0, False)
        if het_splits != hom_splits:
            raise ValueError(f"Het/Hom split mismatch for client {client_id}")
        client_stats.append(
            {
                "client_id": client_id,
                "hospitalid": int(hospital_id),
                "num_stays": len(ids["stay"]),
                "het_nodes": int(het_data.x.shape[0]),
                "het_edges": int(het_data.edge_index.shape[1]),
                "hom_edges": int(hom_data.edge_index.shape[1]),
                "split_counts": het_splits,
                "label_counts": {str(label): int((hom_data.y == label).sum()) for label in range(3)},
            }
        )

    metadata = {
        "schema_version": "eicu2-fedgb-1.0",
        "source_intermediate_schema": schema,
        "task": "icu_los_3class",
        "num_classes": 3,
        "feature_time_window_hours": schema.get("feature_time_window_hours", 24),
        "relation_time_window": schema.get("relation_time_window", "full ICU stay"),
        "homogeneous_projection": {
            "concept_types": ["treatment", "medication"],
            "rule": f"shared distinct concepts > {hom_threshold}",
        },
        "heterogeneous_medication_hub_filter": medication_filter,
        "node_types": list(NODE_TYPES),
        "edge_types": [list(value) for value in EDGE_TYPES],
        "feature_dim": int(global_het.x.shape[1]),
        "global_counts": {
            "patients": len(patients), "stays": len(stays),
            **{node_type: len(concept_rows[node_type]) for node_type, *_ in CONCEPT_SPECS},
        },
        "client_stats": client_stats,
    }
    for root in (het_root, hom_root):
        (root / "metadata").mkdir(parents=True, exist_ok=True)
        (root / "metadata" / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (root / "distrib" / (het_partition_name if root == het_root else hom_partition_name) / "description.txt").write_text(
            json.dumps(
                {
                    "task": "node_cls",
                    "primary_label": "icu_los_3class",
                    "train_val_test": "default_split (patient-grouped stratified 0.05/0.15/0.80)",
                    "num_clients": num_clients,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hom-threshold", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_fedgb(args.source_root, args.output_root, args.hom_threshold)


if __name__ == "__main__":
    main()
