import importlib


def test_core_runtime_modules_import():
    modules = [
        "fedgb.data.distributed_dataset_loader",
        "fedgb.models.gcn",
        "fedgb.models.gin",
        "fedgb.models.rgcn",
        "fedgb.tasks.node_cls",
        "fedgb.tasks.graph_cls",
        "fedgb.tasks.graph_reg",
        "fedgb.training.base",
        "fedgb.training.trainer",
        "fedgb.utils.basic_utils",
        "fedgb.utils.metrics",
    ]
    for module in modules:
        importlib.import_module(module)

