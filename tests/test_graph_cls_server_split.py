from argparse import Namespace
import pickle

import torch
from torch_geometric.data import Data

from fedgb.data.fgl_graph_dataset import FGLGraphDataset
from fedgb.tasks.graph_cls import GraphClsTask


def test_graph_cls_server_split_falls_back_when_global_ids_exceed_server_data(tmp_path):
    graphs = []
    for graph_id in range(6):
        graph = Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
            y=torch.tensor(graph_id % 2),
        )
        graphs.append(graph)
    data = FGLGraphDataset(graphs, num_global_classes=2, global_map={idx: idx for idx in range(6)})

    split_dir = tmp_path / "graph_cls" / "default_split"
    split_dir.mkdir(parents=True)
    for client_id in range(3):
        for split_name, ids in (("train", [100 + client_id]), ("val", [200 + client_id]), ("test", [300 + client_id])):
            with (split_dir / f"glb_{split_name}_{client_id}.pkl").open("wb") as stream:
                pickle.dump(ids, stream)

    args = Namespace(
        dataset=["PUBCHEM_FGL"], num_clients=3, train_val_test="default_split",
        model=["gin"], hid_dim=8, num_layers=2, dropout=0.5, optim="adam",
        lr=0.01, weight_decay=5e-4, batch_size=4, task="graph_cls",
        metrics=["accuracy", "f1"], dp_mech="no_dp", processing="raw",
        gin_pooling="sum", gin_num_mlp_layers=2, gin_learn_eps=False,
    )
    task = GraphClsTask(args, client_id=None, data=data, data_dir=str(tmp_path), device=torch.device("cpu"))
    assert task.train_mask.numel() == len(data)
    assert task.val_mask.numel() == len(data)
    assert task.test_mask.numel() == len(data)
