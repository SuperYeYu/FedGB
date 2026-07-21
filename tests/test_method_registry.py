from fedgb.config.registry import GRAPH_FGL_METHODS, STANDARD_FL_METHODS, SUBGRAPH_FGL_METHODS
from fedgb.config.method_specs import METHOD_SPECS


def test_public_registry_contains_exactly_the_paper_baselines():
    assert len(STANDARD_FL_METHODS) == 14
    assert len(SUBGRAPH_FGL_METHODS) == 21
    assert len(GRAPH_FGL_METHODS) == 7
    assert len(STANDARD_FL_METHODS | SUBGRAPH_FGL_METHODS | GRAPH_FGL_METHODS) == 42


def test_public_method_names_are_canonical():
    assert "spp_fgc" in SUBGRAPH_FGL_METHODS
    assert "fedgraph" not in SUBGRAPH_FGL_METHODS
    assert "gcfl_plus" in GRAPH_FGL_METHODS
    assert "gcfl" not in GRAPH_FGL_METHODS


def test_each_method_declares_public_compatibility_contract():
    for name, spec in METHOD_SPECS.items():
        assert spec["family"] in {"standard_fl", "subgraph_fgl", "graph_fgl"}, name
        assert spec["scenarios"], name
        assert spec["tasks"], name
        assert set(spec["models"]) == set(spec["scenarios"]), name
        for scenario in spec["scenarios"]:
            assert spec["models"][scenario], (name, scenario)

