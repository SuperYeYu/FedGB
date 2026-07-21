#!/usr/bin/env python3
"""Run one FedGB end-to-end smoke case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedgb.config.runtime import build_run_config
from fedgb.training.trainer import FGLTrainer
from fedgb.utils.basic_utils import seed_everything


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = {
    "SMOKE-HOMO": ("AML-HI", "ACM", "gcn", "subgraph_fl_louvain_1_ACM_client_3"),
    "SMOKE-HETERO": ("EICU-Het", "ACM", "rgcn", "subgraph_fl_louvain_1_ACM_client_3"),
    "SMOKE-GRAPH-CLS": ("PubChem", "PUBCHEM_FGL", "gin", "graph_fl_label_skew_10.00_PUBCHEM_FGL_client_3"),
    "SMOKE-GRAPH-REG": ("NOMAD", "NOMAD_FGL", "gin", "graph_fl_label_skew_10.00_NOMAD_FGL_client_3"),
}


def family_for(case):
    if case["scenario"] == "graph":
        return "standard_fl" if case["algorithm"] in {
            "fedavg", "fedprox", "scaffold", "moon", "feddc", "fedproto", "fedexp", "fedlaw",
            "fedala", "fedtgp", "fedluar", "feroma", "pfed1bs", "tinyproto",
        } else "graph_fgl"
    if case["algorithm"] in {
        "fedavg", "fedprox", "scaffold", "moon", "feddc", "fedproto", "fedexp", "fedlaw",
        "fedala", "fedtgp", "fedluar", "feroma", "pfed1bs", "tinyproto",
    }:
        return "standard_fl"
    return "subgraph_fgl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()

    matrix = json.loads((ROOT / "scripts" / "verify" / "smoke_matrix.json").read_text())
    case = next(item for item in matrix["cases"] if item["id"] == opts.case_id)
    public_name, dataset_id, model, partition = FIXTURES[case["dataset"]]
    args = build_run_config(
        family=family_for(case),
        scenario=case["scenario"],
        algorithm=case["algorithm"],
        dataset=public_name,
        dataset_root=ROOT / ".smoke_fixtures" / case["dataset"],
        model=model,
        task=case["task"],
        num_clients=3,
        num_rounds=1,
        num_epochs=1,
        use_cuda=not opts.cpu,
        gpuid=opts.gpuid,
        processed_partition=partition,
        batch_size=8,
        rgcn_num_relations=4,
        log_root=str(opts.output / "fedgb_logs"),
        log_name=case["id"],
        debug=False,
        comm_cost=True,
    )
    args.dataset = [dataset_id]
    seed_everything(args.seed)
    trainer = FGLTrainer(args)
    trainer.train()
    opts.output.mkdir(parents=True, exist_ok=True)
    result = {"case": case, "evaluation": trainer.evaluation_result}
    (opts.output / "result.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps(result, default=float))


if __name__ == "__main__":
    main()

