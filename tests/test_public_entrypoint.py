import json
from argparse import Namespace

from fedgb.training.entrypoint import prepare_run_directory


def test_prepare_run_directory_writes_resolved_config(tmp_path):
    args = Namespace(
        results_root=str(tmp_path),
        public_dataset="AML-HI",
        public_scenario="homo_subgraph",
        fl_algorithm="fedavg",
        seed=2024,
        log_root=None,
        log_name=None,
        debug=False,
    )
    run_dir = prepare_run_directory(args, timestamp="20260712-120000")
    assert run_dir == tmp_path / "homo_subgraph" / "AML-HI" / "fedavg" / "20260712-120000-seed2024"
    assert args.log_root == str(run_dir)
    assert args.log_name == "training"
    assert args.debug is True
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["fl_algorithm"] == "fedavg"

