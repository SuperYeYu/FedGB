"""Run a standard FL baseline on homogeneous client subgraphs."""

from fedgb.training.entrypoint import run_example

CONFIG = {"algorithm": "fedavg", "dataset": "AML-HI", "model": "gcn", "num_clients": 29}

if __name__ == "__main__":
    run_example(CONFIG, family="standard_fl", scenario="homo_subgraph")
