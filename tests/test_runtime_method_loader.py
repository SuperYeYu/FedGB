import pytest

from fedgb.config.method_specs import METHOD_SPECS, resolve_method_class
from fedgb.config.registry import ALL_METHODS


def test_runtime_method_specs_match_public_registry():
    assert set(METHOD_SPECS) == set(ALL_METHODS)


def test_runtime_method_specs_resolve_client_and_server_classes():
    for method in sorted(ALL_METHODS):
        assert resolve_method_class(method, "client").__name__ == METHOD_SPECS[method]["client_class"]
        assert resolve_method_class(method, "server").__name__ == METHOD_SPECS[method]["server_class"]


def test_runtime_loader_rejects_non_public_methods():
    with pytest.raises(ValueError, match="not a public FedGB baseline"):
        resolve_method_class("gcfl", "client")

