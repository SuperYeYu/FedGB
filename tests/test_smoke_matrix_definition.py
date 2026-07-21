import json
from pathlib import Path

from fedgb.config.registry import ALL_METHODS, STANDARD_FL_METHODS
from scripts.verify.run_smoke_matrix import assign_gpus


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_matrix_covers_all_required_method_scenarios():
    payload = json.loads((ROOT / "scripts" / "verify" / "smoke_matrix.json").read_text())
    cases = payload["cases"]
    assert len(cases) == 91
    assert len({case["id"] for case in cases}) == 91
    assert {case["algorithm"] for case in cases} == set(ALL_METHODS)

    for method in STANDARD_FL_METHODS:
        scenarios = {case["scenario"] for case in cases if case["algorithm"] == method}
        scenarios = {(case["scenario"], case["task"]) for case in cases if case["algorithm"] == method}
        assert scenarios == {
            ("homo_subgraph", "node_cls"),
            ("hetero_subgraph", "node_cls"),
            ("graph", "graph_cls"),
            ("graph", "graph_reg"),
        }

    graph_methods = {"gcfl_plus", "fedstar", "fedssp", "optgdba", "fedgmark", "nigdba", "fedvn"}
    for method in graph_methods:
        tasks = {case["task"] for case in cases if case["algorithm"] == method}
        assert tasks == {"graph_cls", "graph_reg"}


def test_smoke_matrix_uses_the_four_release_fixtures():
    payload = json.loads((ROOT / "scripts" / "verify" / "smoke_matrix.json").read_text())
    fixtures = {case["dataset"] for case in payload["cases"]}
    assert fixtures == {"SMOKE-HOMO", "SMOKE-HETERO", "SMOKE-GRAPH-CLS", "SMOKE-GRAPH-REG"}


def test_smoke_cases_are_distributed_round_robin_across_gpus():
    assignments = assign_gpus([{"id": str(index)} for index in range(5)], [0, 1])
    assert [gpu for _, gpu in assignments] == [0, 1, 0, 1, 0]
