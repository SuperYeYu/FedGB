"""Run a standard FL baseline on heterogeneous client subgraphs."""

from fedgb.training.entrypoint import run_example

CONFIG = {"algorithm": "fedavg", "dataset": "ICIJ-Het", "model": "rgcn", "num_clients": 20}

if __name__ == "__main__":
    run_example(CONFIG, family="standard_fl", scenario="hetero_subgraph")
