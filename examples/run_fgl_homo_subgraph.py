"""Run a homogeneous subgraph-level FGL baseline."""

from fedgb.training.entrypoint import run_example

CONFIG = {"algorithm": "fedgta", "dataset": "AML-HI", "model": "gcn", "num_clients": 29}

if __name__ == "__main__":
    run_example(CONFIG, family="subgraph_fgl", scenario="homo_subgraph")
