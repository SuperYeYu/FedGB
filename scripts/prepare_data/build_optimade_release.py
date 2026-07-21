#!/usr/bin/env python3
"""Build the public eight-client OPTIMADE release in the FedGB schema."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedgb.data.fgl_graph_dataset import FGLGraphDataset


TARGET_NAMES = [
    "alexandria_pbe.band_gap.ev",
    "alexandria_pbe.formation_energy.ev_per_atom",
    "alexandria_pbe.hull_distance.ev_per_atom",
    "alexandria_pbesol.band_gap.ev",
    "alexandria_pbesol.formation_energy.ev_per_atom",
    "alexandria_pbesol.hull_distance.ev_per_atom",
    "alexandria_pbesol.scan_band_gap.ev",
    "alexandria_pbesol.scan_hull_distance.ev_per_atom",
    "common.band_gap.ev",
    "common.thermo_stability.ev_per_atom",
    "mp.energy_above_hull.ev_per_atom",
    "mp.formation_energy.ev_per_atom",
    "mpdd.formation_energy_sipfenn_light.ev_per_atom",
    "mpdd.formation_energy_sipfenn_novel.ev_per_atom",
    "mpdd.formation_energy_sipfenn_standard.ev_per_atom",
    "mpdd.stability_sipfenn.ev_per_atom",
    "oqmd.band_gap.ev",
    "oqmd.delta_e.ev_per_atom",
    "oqmd.stability.ev_per_atom",
]

PROVIDER_SPECS = [
    (
        "alexandria_pbe",
        [
            "alexandria_pbe.band_gap.ev",
            "alexandria_pbe.formation_energy.ev_per_atom",
            "alexandria_pbe.hull_distance.ev_per_atom",
            "common.band_gap.ev",
            "common.thermo_stability.ev_per_atom",
        ],
    ),
    (
        "alexandria_pbesol",
        [
            "alexandria_pbesol.band_gap.ev",
            "alexandria_pbesol.formation_energy.ev_per_atom",
            "alexandria_pbesol.hull_distance.ev_per_atom",
            "alexandria_pbesol.scan_band_gap.ev",
            "alexandria_pbesol.scan_hull_distance.ev_per_atom",
            "common.band_gap.ev",
            "common.thermo_stability.ev_per_atom",
        ],
    ),
    ("matterverse", []),
    (
        "mp",
        [
            "mp.energy_above_hull.ev_per_atom",
            "mp.formation_energy.ev_per_atom",
            "common.thermo_stability.ev_per_atom",
        ],
    ),
    (
        "mpdd",
        [
            "mpdd.formation_energy_sipfenn_light.ev_per_atom",
            "mpdd.formation_energy_sipfenn_novel.ev_per_atom",
            "mpdd.formation_energy_sipfenn_standard.ev_per_atom",
            "mpdd.stability_sipfenn.ev_per_atom",
        ],
    ),
    ("nmd", []),
    (
        "oqmd",
        [
            "oqmd.band_gap.ev",
            "oqmd.delta_e.ev_per_atom",
            "oqmd.stability.ev_per_atom",
            "common.band_gap.ev",
            "common.thermo_stability.ev_per_atom",
        ],
    ),
    ("twodmatpedia", []),
]

PROCESSED_PARTITION = "graph_fl_label_skew_10.00_NOMAD_FGL_client_8"
REMOVED_SOURCE_ATTRIBUTES = {
    "provider_targets",
    "common_targets",
    "task_names",
    "common_task_names",
    "node_feature_names",
    "edge_feature_names",
}


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _iter_provider_graphs(source_root: Path, provider: str):
    shard_root = source_root / "clients" / provider / "shards"
    shards = sorted(shard_root.glob("*.pt"))
    if not shards:
        raise ValueError(f"No source shards found for provider {provider}: {shard_root}")
    for shard in shards:
        graphs = _load(shard)
        if not isinstance(graphs, (list, tuple)):
            raise TypeError(f"{shard} must contain a list or tuple of graphs")
        yield from graphs
        del graphs
        gc.collect()


def _target_value(graph, target_name: str):
    source = graph.common_targets if target_name.startswith("common.") else graph.provider_targets
    value = source.get(target_name)
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _convert_graph(graph, client_id: int, provider: str, local_id: int, global_id: int, active_names):
    split = getattr(graph, "split", None)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"{provider} graph {local_id} has invalid split {split!r}")

    graph.x = graph.x.to(torch.float32)
    graph.edge_index = graph.edge_index.to(torch.int64)
    if getattr(graph, "edge_attr", None) is not None:
        graph.edge_attr = graph.edge_attr.to(torch.float32)
    if getattr(graph, "pos", None) is not None:
        graph.pos = graph.pos.to(torch.float32)

    y = torch.zeros((1, len(TARGET_NAMES)), dtype=torch.float32)
    y_mask = torch.zeros_like(y, dtype=torch.bool)
    for target_name in active_names:
        value = _target_value(graph, target_name)
        if value is not None:
            target_id = TARGET_NAMES.index(target_name)
            y[0, target_id] = value
            y_mask[0, target_id] = True
    graph.y = y
    graph.y_mask = y_mask
    graph.fgl_client_id = client_id
    graph.client_name = provider
    graph.source_provider_id = provider
    graph.local_id = local_id
    graph.global_id = global_id
    graph.sample_id = str(getattr(graph, "structure_id", f"{provider}-{local_id}"))
    for attribute in REMOVED_SOURCE_ATTRIBUTES:
        if attribute in graph:
            del graph[attribute]
    return graph


def _save_splits(graphs, split_root: Path, client_id: int, global_map: dict[int, int]):
    split_root.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        mask = torch.tensor(
            [getattr(graph, "split", None) == split_name for graph in graphs], dtype=torch.bool
        )
        torch.save(mask, split_root / f"{split_name}_{client_id}.pt")
        global_ids = [global_map[local_id] for local_id in mask.nonzero(as_tuple=True)[0].tolist()]
        with (split_root / f"glb_{split_name}_{client_id}.pkl").open("wb") as stream:
            pickle.dump(global_ids, stream)


def build_release(source_root, output_root):
    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    partition = output_root / "distrib" / PROCESSED_PARTITION
    split_root = partition / "graph_reg" / "default_split"
    partition.mkdir(parents=True, exist_ok=True)
    representatives = []
    provider_rows = []
    global_offset = 0

    for client_id, (provider, active_names) in enumerate(PROVIDER_SPECS):
        graphs = []
        for local_id, graph in enumerate(_iter_provider_graphs(source_root, provider)):
            graphs.append(
                _convert_graph(
                    graph,
                    client_id=client_id,
                    provider=provider,
                    local_id=local_id,
                    global_id=global_offset + local_id,
                    active_names=active_names,
                )
            )
        if not graphs:
            raise ValueError(f"Provider {provider} contains no graphs")
        global_map = {local_id: global_offset + local_id for local_id in range(len(graphs))}
        payload = FGLGraphDataset(
            graphs=graphs,
            num_targets=len(TARGET_NAMES),
            global_map=global_map,
            client_name=provider,
            task_type="graph_regression",
            target_names=TARGET_NAMES,
            active_target_names=active_names,
        )
        torch.save(payload, partition / f"data_{client_id}.pt")
        _save_splits(graphs, split_root, client_id, global_map)
        representatives.append(graphs[0])
        split_counts = {
            name: sum(getattr(graph, "split", None) == name for graph in graphs)
            for name in ("train", "val", "test")
        }
        provider_rows.append(
            {
                "client_id": client_id,
                "provider": provider,
                "num_graphs": len(graphs),
                "active_target_names": list(active_names),
                **split_counts,
            }
        )
        global_offset += len(graphs)
        del payload, graphs
        gc.collect()

    global_payload = FGLGraphDataset(
        graphs=representatives,
        num_targets=len(TARGET_NAMES),
        global_map={i: i for i in range(len(representatives))},
        client_name="OPTIMADE_GLOBAL",
        task_type="graph_regression",
        target_names=TARGET_NAMES,
        active_target_names=TARGET_NAMES,
    )
    global_path = output_root / "global" / "graph_fl" / "NOMAD_FGL" / "processed" / "data.pt"
    global_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(global_payload, global_path)

    manifest = {
        "dataset_id": "NOMAD_FGL",
        "edge_feature_dim": 24,
        "feature_dim": 12,
        "level": "graph",
        "masked_targets": True,
        "name": "OPTIMADE",
        "num_clients": len(PROVIDER_SPECS),
        "num_targets": len(TARGET_NAMES),
        "processed_partition": PROCESSED_PARTITION,
        "schema_version": "1.0",
        "source_group": "OPTIMADE",
        "split": "graph_reg/default_split",
        "target_names": TARGET_NAMES,
        "task": "graph_reg",
    }
    (output_root / "fedgb_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = {
        "providers": provider_rows,
        "target_names": TARGET_NAMES,
        "unlabeled_clients": [2, 5, 7],
    }
    (output_root / "client_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (partition / "description.txt").write_text(
        "FedGB OPTIMADE release: eight natural provider clients with a shared "
        "19-target masked graph-regression schema.\n",
        encoding="utf-8",
    )
    return {"num_clients": len(PROVIDER_SPECS), "num_graphs": global_offset, "num_targets": len(TARGET_NAMES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and args.force:
        shutil.rmtree(args.output_root)
    report = build_release(args.source_root, args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
