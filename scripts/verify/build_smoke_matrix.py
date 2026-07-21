#!/usr/bin/env python3
"""Generate the complete FedGB end-to-end smoke matrix."""

import json
from pathlib import Path

from fedgb.config.registry import GRAPH_FGL_METHODS, STANDARD_FL_METHODS, SUBGRAPH_FGL_METHODS


ROOT = Path(__file__).resolve().parents[2]
HETERO = {"fedlit", "fedda", "fedhgn"}


def add(cases, algorithm, scenario, task, dataset):
    cases.append(
        {
            "id": f"{algorithm}__{scenario}__{task}",
            "algorithm": algorithm,
            "scenario": scenario,
            "task": task,
            "dataset": dataset,
        }
    )


def main():
    cases = []
    for method in sorted(STANDARD_FL_METHODS):
        add(cases, method, "homo_subgraph", "node_cls", "SMOKE-HOMO")
        add(cases, method, "hetero_subgraph", "node_cls", "SMOKE-HETERO")
        add(cases, method, "graph", "graph_cls", "SMOKE-GRAPH-CLS")
        add(cases, method, "graph", "graph_reg", "SMOKE-GRAPH-REG")
    for method in sorted(SUBGRAPH_FGL_METHODS):
        scenario = "hetero_subgraph" if method in HETERO else "homo_subgraph"
        add(cases, method, scenario, "node_cls", "SMOKE-HETERO" if method in HETERO else "SMOKE-HOMO")
    for method in sorted(GRAPH_FGL_METHODS):
        add(cases, method, "graph", "graph_cls", "SMOKE-GRAPH-CLS")
        add(cases, method, "graph", "graph_reg", "SMOKE-GRAPH-REG")
    path = ROOT / "scripts" / "verify" / "smoke_matrix.json"
    path.write_text(json.dumps({"version": 1, "cases": cases}, indent=2), encoding="utf-8")
    print(path, len(cases))


if __name__ == "__main__":
    main()

