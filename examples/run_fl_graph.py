"""Run a standard FL baseline on client graph collections."""

from fedgb.training.entrypoint import run_example

CONFIG = {"algorithm": "fedavg", "dataset": "PubChem", "model": "gin", "num_clients": 13}

if __name__ == "__main__":
    run_example(CONFIG, family="standard_fl", scenario="graph")
