#!/usr/bin/env python3
"""Build compact deterministic datasets for FedGB release smoke tests."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import torch
from torch_geometric.data import Data

from fedgb.data.fgl_graph_dataset import FGLGraphDataset


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / ".smoke_fixtures"
NUM_CLIENTS = 3


def ring_edges(num_nodes):
    src = torch.arange(num_nodes)
    dst = torch.roll(src, shifts=-1)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])


def masks(num_items):
    train = torch.zeros(num_items, dtype=torch.bool)
    val = torch.zeros(num_items, dtype=torch.bool)
    test = torch.zeros(num_items, dtype=torch.bool)
    train[: max(2, num_items // 2)] = True
    val[max(2, num_items // 2) : max(3, 3 * num_items // 4)] = True
    test[~(train | val)] = True
    return train, val, test


def save_node_fixture(name, heterogeneous=False):
    root = FIXTURE_ROOT / name
    partition = root / "distrib" / f"subgraph_fl_louvain_1_ACM_client_{NUM_CLIENTS}"
    split = partition / "node_cls" / "default_split"
    global_dir = root / "global" / "subgraph_fl" / "acm" / "processed"
    split.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    global_x, global_y, edge_parts = [], [], []
    offset = 0
    for client_id in range(NUM_CLIENTS):
        num_nodes = 30
        generator = torch.Generator().manual_seed(100 + client_id)
        x = torch.randn(num_nodes, 16, generator=generator)
        y = torch.arange(num_nodes) % 3
        edge_index = ring_edges(num_nodes)
        data = Data(x=x, y=y, edge_index=edge_index)
        data.global_map = {idx: offset + idx for idx in range(num_nodes)}
        data.num_global_classes = 3
        if heterogeneous:
            data.edge_type = torch.arange(edge_index.shape[1]) % 4
            data.node_type = torch.arange(num_nodes) % 3
            data.target_node_type = "stay"
            data.hetero_node_types = ["stay", "diagnosis", "treatment"]
            data.hetero_edge_types = [
                ("stay", "self", "stay"),
                ("stay", "diagnosis", "diagnosis"),
                ("diagnosis", "rev", "stay"),
                ("stay", "treatment", "treatment"),
            ]
        torch.save(data, partition / f"data_{client_id}.pt")
        train, val, test = masks(num_nodes)
        for split_name, mask in [("train", train), ("val", val), ("test", test)]:
            torch.save(mask, split / f"{split_name}_{client_id}.pt")
            with (split / f"glb_{split_name}_{client_id}.pkl").open("wb") as stream:
                pickle.dump([offset + idx for idx in mask.nonzero(as_tuple=True)[0].tolist()], stream)

        global_x.append(x)
        global_y.append(y)
        edge_parts.append(edge_index + offset)
        offset += num_nodes

    global_data = Data(x=torch.cat(global_x), y=torch.cat(global_y), edge_index=torch.cat(edge_parts, dim=1))
    global_data.num_global_classes = 3
    if heterogeneous:
        global_data.edge_type = torch.arange(global_data.edge_index.shape[1]) % 4
    torch.save(global_data, global_dir / "data.pt")


def graph_object(client_id, graph_id, regression):
    num_nodes = 6 + graph_id % 3
    generator = torch.Generator().manual_seed(1000 + 100 * client_id + graph_id)
    graph = Data(x=torch.randn(num_nodes, 8, generator=generator), edge_index=ring_edges(num_nodes))
    graph.y = torch.tensor([float((client_id + graph_id) % 7) / 3]) if regression else torch.tensor([(client_id + graph_id) % 2])
    graph.split = "train" if graph_id < 6 else "val" if graph_id < 9 else "test"
    return graph


def save_graph_fixture(name, regression):
    root = FIXTURE_ROOT / name
    dataset_id = "NOMAD_FGL" if regression else "PUBCHEM_FGL"
    partition = root / "distrib" / f"graph_fl_label_skew_10.00_{dataset_id}_client_{NUM_CLIENTS}"
    task = "graph_reg" if regression else "graph_cls"
    split = partition / task / "default_split"
    global_dir = root / "global" / "graph_fl" / dataset_id / "processed"
    split.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    global_id = 0
    representatives = []
    for client_id in range(NUM_CLIENTS):
        graphs = [graph_object(client_id, graph_id, regression) for graph_id in range(12)]
        dataset = FGLGraphDataset(
            graphs,
            num_targets=1,
            global_map={idx: global_id + idx for idx in range(len(graphs))},
            client_name=f"smoke-client-{client_id}",
            task_type="graph_regression" if regression else "graph_cls",
            num_global_classes=None if regression else 2,
        )
        torch.save(dataset, partition / f"data_{client_id}.pt")
        for split_name in ("train", "val", "test"):
            mask = torch.tensor([graph.split == split_name for graph in graphs], dtype=torch.bool)
            torch.save(mask, split / f"{split_name}_{client_id}.pt")
            with (split / f"glb_{split_name}_{client_id}.pkl").open("wb") as stream:
                pickle.dump([dataset.global_map[idx] for idx in mask.nonzero(as_tuple=True)[0].tolist()], stream)
        representatives.append(graphs[0])
        global_id += len(graphs)

    global_dataset = FGLGraphDataset(
        representatives,
        num_targets=1,
        task_type="graph_regression" if regression else "graph_cls",
        num_global_classes=None if regression else 2,
    )
    torch.save(global_dataset, global_dir / "data.pt")


def main():
    if FIXTURE_ROOT.exists():
        shutil.rmtree(FIXTURE_ROOT)
    save_node_fixture("SMOKE-HOMO", heterogeneous=False)
    save_node_fixture("SMOKE-HETERO", heterogeneous=True)
    save_graph_fixture("SMOKE-GRAPH-CLS", regression=False)
    save_graph_fixture("SMOKE-GRAPH-REG", regression=True)
    (FIXTURE_ROOT / "manifest.json").write_text(
        json.dumps({"num_clients": NUM_CLIENTS, "fixtures": sorted(path.name for path in FIXTURE_ROOT.iterdir())}, indent=2),
        encoding="utf-8",
    )
    print(FIXTURE_ROOT)


if __name__ == "__main__":
    main()

