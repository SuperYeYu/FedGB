import importlib.util
from pathlib import Path

import pytest

from fedgb.config.runtime import build_run_config


ROOT = Path(__file__).resolve().parents[1]


def load_example(name):
    path = ROOT / "examples" / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("filename", "family", "scenario"),
    [
        ("run_fgl_homo_subgraph.py", "subgraph_fgl", "homo_subgraph"),
        ("run_fgl_hetero_subgraph.py", "subgraph_fgl", "hetero_subgraph"),
        ("run_fgl_graph.py", "graph_fgl", "graph"),
        ("run_fl_homo_subgraph.py", "standard_fl", "homo_subgraph"),
        ("run_fl_hetero_subgraph.py", "standard_fl", "hetero_subgraph"),
        ("run_fl_graph.py", "standard_fl", "graph"),
    ],
)
def test_examples_build_valid_configs(filename, family, scenario):
    module = load_example(filename)
    args = build_run_config(family=family, scenario=scenario, **module.CONFIG)
    assert args.fl_algorithm == module.CONFIG["algorithm"]
    assert args.num_rounds == 100
    assert args.hid_dim == 64
    assert args.dropout == 0.5


def test_standard_fl_heterogeneous_subgraph_forces_rgcn():
    args = build_run_config(
        family="standard_fl",
        scenario="hetero_subgraph",
        algorithm="fedavg",
        dataset="ICIJ-Het",
        model="gcn",
    )
    assert args.model == ["rgcn"]


def test_public_dataset_name_maps_to_internal_processed_id():
    args = build_run_config(
        family="standard_fl",
        scenario="homo_subgraph",
        algorithm="fedavg",
        dataset="AML-HI",
        model="gcn",
    )
    assert args.public_dataset == "AML-HI"
    assert args.dataset == ["ACM"]


def test_graph_dataset_uses_registered_task_when_task_is_omitted():
    args = build_run_config(
        family="standard_fl",
        scenario="graph",
        algorithm="fedavg",
        dataset="NOMAD",
        model="gin",
    )
    assert args.task == "graph_reg"


def test_incompatible_model_is_rejected_before_training():
    with pytest.raises(ValueError, match="model"):
        build_run_config(
            family="graph_fgl",
            scenario="graph",
            algorithm="fedstar",
            dataset="PubChem",
            model="gcn",
        )


def test_invalid_method_scenario_is_rejected():
    with pytest.raises(ValueError, match="not available"):
        build_run_config(
            family="graph_fgl",
            scenario="graph",
            algorithm="fedgta",
            dataset="PubChem",
            model="gin",
        )
