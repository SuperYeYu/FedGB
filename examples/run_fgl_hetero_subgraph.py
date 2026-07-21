"""Run a heterogeneous subgraph-level FGL baseline."""

from fedgb.training.entrypoint import run_example

CONFIG = {"algorithm": "fedhgn", "dataset": "ICIJ-Het", "model": "rgcn", "num_clients": 20}

if __name__ == "__main__":
    run_example(CONFIG, family="subgraph_fgl", scenario="hetero_subgraph")
