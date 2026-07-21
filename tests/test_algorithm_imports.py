import importlib

from fedgb.config.registry import GRAPH_FGL_METHODS, STANDARD_FL_METHODS, SUBGRAPH_FGL_METHODS


HETERO_METHODS = {"fedlit", "fedda", "fedhgn"}


def method_package(method):
    if method in STANDARD_FL_METHODS:
        return f"fedgb.algorithms.standard_fl.{method}"
    if method in GRAPH_FGL_METHODS:
        return f"fedgb.algorithms.graph_fgl.{method}"
    level = "heterogeneous" if method in HETERO_METHODS else "homogeneous"
    return f"fedgb.algorithms.subgraph_fgl.{level}.{method}"


def test_all_public_algorithm_client_and_server_modules_import():
    for method in sorted(STANDARD_FL_METHODS | SUBGRAPH_FGL_METHODS | GRAPH_FGL_METHODS):
        package = method_package(method)
        importlib.import_module(f"{package}.client")
        importlib.import_module(f"{package}.server")

