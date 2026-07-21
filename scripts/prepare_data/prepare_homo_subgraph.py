#!/usr/bin/env python3
"""Download and partition a public homogeneous node-classification graph."""

import argparse

from fedgb.data.public_preparation import prepare_public_dataset


SCENARIO = "homo_subgraph"
CONFIG = {
    "dataset": "Cora",
    "partition": "louvain",
    "num_clients": 10,
    "output_name": "Cora-Louvain-10",
    "seed": 2024,
    "split": (0.2, 0.4, 0.4),
    "louvain_resolution": 1.0,
    "louvain_delta": 20,
    "dirichlet_alpha": 10.0,
    "least_samples": 5,
    "metis_num_coms": 100,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_public_dataset(CONFIG, scenario=SCENARIO, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
