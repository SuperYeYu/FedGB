"""Shared command-line execution for public FedGB examples."""

import argparse
import datetime
import json
from pathlib import Path

from fedgb.config.runtime import build_run_config
from fedgb.training.trainer import FGLTrainer
from fedgb.utils.basic_utils import seed_everything


def prepare_run_directory(args, timestamp=None):
    timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        Path(args.results_root)
        / args.public_scenario
        / args.public_dataset
        / args.fl_algorithm
        / f"{timestamp}-seed{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    args.log_root = str(run_dir)
    args.log_name = "training"
    args.debug = True
    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def run_example(config, family, scenario):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    args = build_run_config(family=family, scenario=scenario, **config)
    if cli.dry_run:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return
    run_dir = prepare_run_directory(args)
    print(f"FedGB results: {run_dir}")
    seed_everything(args.seed)
    FGLTrainer(args).train()

