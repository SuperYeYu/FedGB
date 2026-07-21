#!/usr/bin/env python3
"""Download and partition a public graph-classification dataset."""

import argparse

from fedgb.data.public_preparation import prepare_public_dataset


SCENARIO = "graph"
CONFIG = {
    "dataset": "MUTAG",
    "partition": "label_skew",
    "num_clients": 5,
    "output_name": "MUTAG-LabelSkew-5",
    "seed": 2024,
    "split": (0.8, 0.1, 0.1),
    "dirichlet_alpha": 10.0,
    "least_samples": 5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_public_dataset(CONFIG, scenario=SCENARIO, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
