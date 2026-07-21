"""Public runtime configuration for the six FedGB entry points."""

from argparse import Namespace
from pathlib import Path

from fedgb.config.datasets import get_dataset_spec
from fedgb.config.method_specs import METHOD_SPECS
from fedgb.config.registry import GRAPH_FGL_METHODS, STANDARD_FL_METHODS, SUBGRAPH_FGL_METHODS


HETERO_SUBGRAPH_FGL_METHODS = frozenset({"fedlit", "fedda", "fedhgn"})
HOMO_SUBGRAPH_FGL_METHODS = SUBGRAPH_FGL_METHODS - HETERO_SUBGRAPH_FGL_METHODS


def _supported_methods(family, scenario):
    if family == "standard_fl":
        return STANDARD_FL_METHODS
    if family == "subgraph_fgl" and scenario == "homo_subgraph":
        return HOMO_SUBGRAPH_FGL_METHODS
    if family == "subgraph_fgl" and scenario == "hetero_subgraph":
        return HETERO_SUBGRAPH_FGL_METHODS
    if family == "graph_fgl" and scenario == "graph":
        return GRAPH_FGL_METHODS
    raise ValueError(f"Unsupported family/scenario pair: {family}/{scenario}")


def build_run_config(
    *,
    family,
    scenario,
    algorithm,
    dataset,
    model,
    dataset_root=None,
    task=None,
    num_clients=None,
    num_rounds=100,
    num_epochs=None,
    gpuid=0,
    seed=2024,
    **overrides,
):
    supported = _supported_methods(family, scenario)
    if algorithm not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Algorithm '{algorithm}' is not available for {family}/{scenario}. Supported: {choices}")

    is_graph = scenario == "graph"
    is_hetero = scenario == "hetero_subgraph"
    if family == "standard_fl" and is_hetero:
        model = "rgcn"

    root = Path(dataset_root) if dataset_root else Path(__file__).resolve().parents[2] / "datasets" / dataset
    dataset_spec = get_dataset_spec(dataset, dataset_root=root)
    if dataset_spec.get("availability") == "credentialed_build" and not (root / "fedgb_manifest.json").is_file():
        builder = dataset_spec["builder"]
        raise FileNotFoundError(
            f"Dataset '{dataset}' requires a credentialed local build and is not included in the public archive. "
            f"Follow {builder}, then set dataset_root to the generated dataset directory."
        )
    if dataset_spec["level"] != scenario:
        raise ValueError(
            f"Dataset '{dataset}' is registered for {dataset_spec['level']}, not {scenario}."
        )
    method_spec = METHOD_SPECS[algorithm]
    selected_task = task or dataset_spec["task"]
    if selected_task not in method_spec["tasks"]:
        raise ValueError(
            f"Algorithm '{algorithm}' does not support task '{selected_task}'. "
            f"Supported tasks: {', '.join(method_spec['tasks'])}."
        )
    if model not in method_spec["models"][scenario]:
        choices = ", ".join(method_spec["models"][scenario])
        raise ValueError(
            f"Algorithm '{algorithm}' does not support model '{model}' in {scenario}. "
            f"Supported models: {choices}."
        )
    default_clients = 5 if is_graph else 10
    default_epochs = 1 if is_graph else 2

    values = dict(
        root=str(root),
        scenario="graph_fl" if is_graph else "subgraph_fl",
        public_scenario=scenario,
        public_dataset=dataset,
        dataset=[dataset_spec["dataset_id"]],
        processed_partition=dataset_spec.get(
            "processed_partition",
            (
                f"graph_fl_label_skew_10.00_{dataset_spec['dataset_id']}_client_{dataset_spec['num_clients']}"
                if is_graph
                else f"subgraph_fl_louvain_1_{dataset_spec['dataset_id']}_client_{dataset_spec['num_clients']}"
            ),
        ),
        simulation_mode="graph_fl_label_skew" if is_graph else "subgraph_fl_louvain",
        task=selected_task,
        train_val_test="default_split",
        num_clients=num_clients or dataset_spec.get("num_clients", default_clients),
        num_rounds=int(num_rounds),
        client_frac=1.0,
        fl_algorithm=algorithm,
        model=[model],
        num_layers=2,
        hid_dim=64,
        dropout=0.5,
        num_epochs=default_epochs if num_epochs is None else int(num_epochs),
        lr=0.01,
        optim="adam",
        weight_decay=5e-4,
        batch_size=128,
        metrics=[],
        evaluation_mode="local_model_on_local_data",
        use_cuda=True,
        gpuid=int(gpuid),
        seed=int(seed),
        dp_mech="no_dp",
        dirichlet_alpha=10.0,
        louvain_resolution=1.0,
        metis_num_coms=100,
        log_root=None,
        log_name=None,
        results_root=str(Path(__file__).resolve().parents[2] / "results"),
        comm_cost=True,
        model_param=False,
        debug=False,
        processing="raw",
        processing_percentage=0.1,
        feature_mask_prob=0.1,
        dp_epsilon=0.0,
        homo_injection_ratio=0.0,
        hete_injection_ratio=0.0,
        dirichlet_try_cnt=100,
        least_samples=5,
        louvain_delta=20,
        num_clusters=7,
        noise_scale=1.0,
        grad_clip=1.0,
        dp_q=0.1,
        max_degree=5,
        max_epsilon=20,
        target_node=overrides.pop("target_node", dataset_spec.get("target_node", "stay" if is_hetero else None)),
        rgcn_num_relations=overrides.pop("rgcn_num_relations", 8),
    )
    values.update(overrides)
    return Namespace(**values)
