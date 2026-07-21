from fedgb.config.runtime import build_run_config
from fedgb.data.distributed_dataset_loader import FGLDataset
from pathlib import Path


def test_subgraph_runtime_uses_release_partition_directory():
    args = build_run_config(
        family="standard_fl",
        scenario="homo_subgraph",
        algorithm="fedavg",
        dataset="XS-Video",
        model="gcn",
    )
    dataset = object.__new__(FGLDataset)
    dataset.args = args
    dataset.root = args.root
    assert Path(dataset.processed_dir).parts[-2:] == (
        "distrib",
        "subgraph_fl_louvain_1_ACM_client_5",
    )


def test_graph_runtime_uses_release_partition_directory():
    args = build_run_config(
        family="standard_fl",
        scenario="graph",
        algorithm="fedavg",
        dataset="NOMAD",
        model="gin",
        task="graph_reg",
    )
    dataset = object.__new__(FGLDataset)
    dataset.args = args
    dataset.root = args.root
    assert Path(dataset.processed_dir).parts[-2:] == (
        "distrib",
        "graph_fl_label_skew_10.00_NOMAD_FGL_client_6",
    )

