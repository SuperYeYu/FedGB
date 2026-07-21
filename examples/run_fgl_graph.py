"""Run a graph-level FGL baseline."""

from fedgb.training.entrypoint import run_example

CONFIG = {"algorithm": "gcfl_plus", "dataset": "PubChem", "model": "gin", "num_clients": 13}

if __name__ == "__main__":
    run_example(CONFIG, family="graph_fgl", scenario="graph")
