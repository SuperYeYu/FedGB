"""Prepare downloadable public datasets as FedGB schema 1.0 variants."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

from fedgb.data.fgl_graph_dataset import FGLGraphDataset
from fedgb.data.global_dataset_loader import load_global_dataset
from fedgb.data.schema import SCHEMA_VERSION, validate_client_payload, validate_global_payload


HOMO_DATASETS = frozenset(
    {
        "Cora", "CiteSeer", "PubMed", "CS", "Physics", "Computers", "Photo",
        "Chameleon", "Squirrel", "Tolokers", "Actor", "Amazon-ratings",
        "Roman-empire", "Questions", "Minesweeper", "Reddit", "Flickr",
    }
)
GRAPH_DATASETS = frozenset(
    {
        "AIDS", "BZR", "COLLAB", "COX2", "DD", "DHFR", "ENZYMES",
        "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "NCI1", "PROTEINS", "PTC_MR",
    }
)
HOMO_MODES = {
    "louvain": "subgraph_fl_louvain",
    "louvain_plus": "subgraph_fl_louvain_plus",
    "metis": "subgraph_fl_metis",
    "metis_plus": "subgraph_fl_metis_plus",
    "label_skew": "subgraph_fl_label_skew",
}
GRAPH_MODES = {
    "label_skew": "graph_fl_label_skew",
    "topology_skew": "graph_fl_topology_skew",
    "feature_skew": "graph_fl_feature_skew",
}


def deterministic_split_masks(labels, split, seed):
    """Create deterministic, stratified and non-overlapping boolean masks."""

    ratios = tuple(float(value) for value in split)
    if len(ratios) != 3 or any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split must contain three non-negative ratios that sum to 1.")
    labels = torch.as_tensor(labels).view(-1).cpu()
    masks = {name: torch.zeros(labels.numel(), dtype=torch.bool) for name in ("train", "val", "test")}
    generator = torch.Generator().manual_seed(int(seed))
    for label in torch.unique(labels, sorted=True):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        train_end = int(indices.numel() * ratios[0])
        val_end = train_end + int(indices.numel() * ratios[1])
        masks["train"][indices[:train_end]] = True
        masks["val"][indices[train_end:val_end]] = True
        masks["test"][indices[val_end:]] = True
    return masks


def _reset_root(root: Path, force: bool) -> None:
    if root.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {root}. Use --force to replace it.")
        shutil.rmtree(root)
    root.mkdir(parents=True)


def _write_splits(partition_root, task, clients, split, seed, graph_level):
    split_root = partition_root / task / "default_split"
    split_root.mkdir(parents=True)
    for client_id, client in enumerate(clients):
        labels = client.y if graph_level else client.y.view(-1)
        masks = deterministic_split_masks(labels, split, seed + client_id)
        for name, mask in masks.items():
            torch.save(mask, split_root / f"{name}_{client_id}.pt")


def _write_manifest(root, manifest):
    (root / "fedgb_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _normalize_node_data(data):
    data = data.clone()
    data.x = data.x.to(torch.float32)
    data.y = data.y.view(-1).to(torch.long)
    data.edge_index = data.edge_index.to(torch.long)
    if data.edge_index.numel() == 0:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
    if isinstance(data.global_map, dict):
        data.global_map = torch.tensor(
            [data.global_map[index] for index in range(data.x.shape[0])], dtype=torch.long
        )
    else:
        data.global_map = torch.as_tensor(data.global_map, dtype=torch.long)
    data.num_global_classes = int(data.y.max().item()) + 1
    validate_client_payload(data, "homo_subgraph", "node_cls")
    return data


def _normalize_graph(graph):
    graph = graph.clone()
    if getattr(graph, "x", None) is None:
        node_degree = degree(graph.edge_index[1], num_nodes=graph.num_nodes).view(-1, 1)
        graph.x = node_degree
    graph.x = graph.x.to(torch.float32)
    graph.edge_index = graph.edge_index.to(torch.long)
    graph.y = graph.y.view(-1)[:1].to(torch.long)
    return graph


def _as_graph_dataset(graphs, *, global_map=None, num_classes=None):
    if isinstance(graphs, FGLGraphDataset):
        source_graphs = graphs.graphs
        global_map = graphs.global_map if global_map is None else global_map
        num_classes = graphs.num_classes if num_classes is None else num_classes
    else:
        source_graphs = list(graphs)
        if global_map is None:
            global_map = getattr(graphs, "global_map", None)
        if num_classes is None:
            num_classes = getattr(graphs, "num_global_classes", None)
            if num_classes is None:
                num_classes = getattr(graphs, "num_classes", None)
    normalized = [_normalize_graph(graph) for graph in source_graphs]
    if num_classes is None:
        num_classes = max(int(graph.y.item()) for graph in normalized) + 1
    return FGLGraphDataset(
        normalized,
        global_map=global_map,
        task_type="graph_cls",
        num_global_classes=int(num_classes),
    )


def write_homo_subgraph_variant(
    *, root, name, source_dataset, partition, global_data, clients, split, seed,
    simulation=None, force=False
):
    root = Path(root)
    _reset_root(root, force)
    normalized_clients = [_normalize_node_data(client) for client in clients]
    global_data = global_data.clone()
    global_data.x = global_data.x.to(torch.float32)
    global_data.edge_index = global_data.edge_index.to(torch.long)
    global_data.y = global_data.y.view(-1).to(torch.long)
    validate_global_payload(global_data, "homo_subgraph", "node_cls")

    partition_root = root / "distrib" / partition
    partition_root.mkdir(parents=True)
    for client_id, client in enumerate(normalized_clients):
        torch.save(client, partition_root / f"data_{client_id}.pt")
    _write_splits(partition_root, "node_cls", normalized_clients, split, seed, graph_level=False)
    global_path = root / "global" / "subgraph_fl" / name.lower() / "processed"
    global_path.mkdir(parents=True)
    torch.save(global_data, global_path / "data.pt")
    manifest = {
        "name": name,
        "source_group": source_dataset,
        "source_dataset": source_dataset,
        "schema_version": SCHEMA_VERSION,
        "level": "homo_subgraph",
        "task": "node_cls",
        "num_clients": len(normalized_clients),
        "dataset_id": name,
        "processed_partition": partition,
        "feature_dim": int(global_data.x.shape[1]),
        "seed": int(seed),
        "split": list(map(float, split)),
        "simulation": dict(simulation or {}),
    }
    return _write_manifest(root, manifest)


def write_graph_variant(
    *, root, name, source_dataset, partition, global_graphs, clients, split, seed,
    simulation=None, force=False
):
    root = Path(root)
    _reset_root(root, force)
    global_dataset = _as_graph_dataset(global_graphs)
    normalized_clients = []
    for client in clients:
        normalized_clients.append(
            _as_graph_dataset(
                client,
                global_map=getattr(client, "global_map", None),
                num_classes=global_dataset.num_classes,
            )
        )
    if any(len(client) == 0 for client in normalized_clients):
        raise ValueError("Simulation produced an empty graph client; adjust num_clients or skew settings.")
    validate_global_payload(global_dataset, "graph", "graph_cls")
    partition_root = root / "distrib" / partition
    partition_root.mkdir(parents=True)
    for client_id, client in enumerate(normalized_clients):
        validate_client_payload(client, "graph", "graph_cls")
        torch.save(client, partition_root / f"data_{client_id}.pt")
    _write_splits(partition_root, "graph_cls", normalized_clients, split, seed, graph_level=True)
    global_path = root / "global" / "graph_fl" / name / "processed"
    global_path.mkdir(parents=True)
    torch.save(global_dataset, global_path / "data.pt")
    manifest = {
        "name": name,
        "source_group": source_dataset,
        "source_dataset": source_dataset,
        "schema_version": SCHEMA_VERSION,
        "level": "graph",
        "task": "graph_cls",
        "num_clients": len(normalized_clients),
        "dataset_id": name,
        "processed_partition": partition,
        "feature_dim": int(global_dataset.num_features),
        "seed": int(seed),
        "split": list(map(float, split)),
        "simulation": dict(simulation or {}),
    }
    return _write_manifest(root, manifest)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _output_name(config, scenario):
    if config.get("output_name"):
        return config["output_name"]
    mode = "".join(part.title() for part in config["partition"].split("_"))
    suffix = "Graph" if scenario == "graph" else "Subgraph"
    return f"{config['dataset']}-{mode}-{config['num_clients']}-{suffix}"


def _simulation_args(config, mode):
    return Namespace(
        num_clients=int(config["num_clients"]),
        dirichlet_alpha=float(config.get("dirichlet_alpha", 10.0)),
        skew_alpha=float(config.get("dirichlet_alpha", 10.0)),
        dirichlet_try_cnt=int(config.get("dirichlet_try_cnt", 100)),
        least_samples=int(config.get("least_samples", 5)),
        louvain_resolution=float(config.get("louvain_resolution", 1.0)),
        louvain_delta=int(config.get("louvain_delta", 20)),
        metis_num_coms=int(config.get("metis_num_coms", 100)),
        simulation_mode=mode,
    )


def _simulation_metadata(args, mode):
    metadata = {"mode": mode}
    if "louvain" in mode:
        metadata.update(
            louvain_resolution=args.louvain_resolution,
            louvain_delta=args.louvain_delta,
        )
    elif "metis" in mode:
        metadata["metis_num_coms"] = args.metis_num_coms
    if mode.endswith("label_skew"):
        metadata.update(
            dirichlet_alpha=args.dirichlet_alpha,
            least_samples=args.least_samples,
            dirichlet_try_cnt=args.dirichlet_try_cnt,
        )
    return metadata


def prepare_public_dataset(config, *, scenario, dry_run=False, force=False):
    """Download, simulate and write one public FedGB dataset variant."""

    config = dict(config)
    dataset = config["dataset"]
    modes = GRAPH_MODES if scenario == "graph" else HOMO_MODES
    supported = GRAPH_DATASETS if scenario == "graph" else HOMO_DATASETS
    if dataset not in supported:
        raise ValueError(f"Unsupported {scenario} source dataset '{dataset}'. Supported: {', '.join(sorted(supported))}")
    partition_key = config["partition"]
    if partition_key not in modes:
        raise ValueError(f"Unsupported partition '{partition_key}'. Supported: {', '.join(modes)}")
    if int(config["num_clients"]) < 2:
        raise ValueError("num_clients must be at least 2.")

    repo_root = Path(__file__).resolve().parents[2]
    datasets_root = Path(config.get("datasets_root", repo_root / "datasets"))
    download_root = Path(config.get("download_root", datasets_root / ".cache"))
    name = _output_name(config, scenario)
    output_root = datasets_root / name
    mode = modes[partition_key]
    args = _simulation_args(config, mode)
    seed = int(config.get("seed", 2024))
    split = tuple(
        map(float, config.get("split", (0.2, 0.4, 0.4) if scenario != "graph" else (0.8, 0.1, 0.1)))
    )
    simulation_metadata = _simulation_metadata(args, mode)
    summary = {
        "scenario": scenario,
        "source_dataset": dataset,
        "output_name": name,
        "output_root": str(output_root),
        "simulation_mode": mode,
        "num_clients": args.num_clients,
    }
    if dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    manifest_path = output_root / "fedgb_manifest.json"
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_level = "graph" if scenario == "graph" else "homo_subgraph"
        matches = (
            manifest.get("name") == name
            and manifest.get("source_dataset") == dataset
            and manifest.get("level") == expected_level
            and int(manifest.get("num_clients", -1)) == args.num_clients
            and manifest.get("schema_version") == SCHEMA_VERSION
            and int(manifest.get("seed", -1)) == seed
            and tuple(map(float, manifest.get("split", ()))) == split
            and manifest.get("simulation") == simulation_metadata
        )
        if not matches:
            raise FileExistsError(
                f"Existing output manifest does not match the requested configuration: {manifest_path}. "
                "Use --force to replace it."
            )
        print(f"Reusing existing FedGB dataset: {output_root}")
        return manifest

    _seed_everything(seed)
    internal_scenario = "graph_fl" if scenario == "graph" else "subgraph_fl"
    global_dataset = load_global_dataset(str(download_root), internal_scenario, dataset)
    from fedgb.data import simulation

    clients = getattr(simulation, mode)(args, global_dataset)
    partition = (
        f"{mode}_{args.dirichlet_alpha:.2f}_{dataset}_client_{args.num_clients}"
        if mode.endswith("label_skew")
        else f"{mode}_{getattr(args, 'louvain_resolution', 1.0)}_{dataset}_client_{args.num_clients}"
        if "louvain" in mode
        else f"{mode}_{args.metis_num_coms}_{dataset}_client_{args.num_clients}"
        if "metis_plus" in mode
        else f"{mode}_{dataset}_client_{args.num_clients}"
    )
    common = dict(
        root=output_root,
        name=name,
        source_dataset=dataset,
        partition=partition,
        clients=clients,
        split=split,
        seed=seed,
        simulation=simulation_metadata,
        force=force,
    )
    if scenario == "graph":
        manifest = write_graph_variant(global_graphs=global_dataset, **common)
    else:
        manifest = write_homo_subgraph_variant(global_data=global_dataset[0], **common)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
