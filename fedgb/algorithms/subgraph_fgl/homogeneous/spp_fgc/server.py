import numpy as np
import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.spp_fgc.fedgraph_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.spp_fgc.utils import (
    add_noise_to_graph,
    allocate_privacy_budget,
    build_global_similarity_graph,
    clr_graph_learning,
    global_s_slices,
    laplace_noise_addition,
    match_clusters_greedy,
)


def _cfg(args, name):
    return getattr(args, name, config[name])


class FedGraphServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedGraphServer, self).__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self._apply_defaults()
        self._global_slices = {}
        self._personalized = {}
        self._last_global_s = None

    def _apply_defaults(self):
        for name, value in config.items():
            if not hasattr(self.args, name):
                setattr(self.args, name, value)

    def _set_global_parameters(self, params):
        with torch.no_grad():
            for global_param, param in zip(self.task.model.parameters(), params):
                global_param.data.copy_(param.to(self.device))

    def _aggregate_model(self, sampled):
        with torch.no_grad():
            num_total = sum(self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in sampled)
            aggregated = []
            for idx, client_id in enumerate(sampled):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / max(num_total, 1)
                client_params = self.message_pool[f"client_{client_id}"]["weight"]
                if idx == 0:
                    aggregated = [weight * param.detach().clone().to(self.device) for param in client_params]
                else:
                    for param_idx, param in enumerate(client_params):
                        aggregated[param_idx] += weight * param.detach().to(self.device)
            if aggregated:
                self._set_global_parameters(aggregated)
            global_weights = [param.detach().clone() for param in self.task.model.parameters()]
            self._personalized = {client_id: global_weights for client_id in sampled}

    def _apply_structural_privacy(self, payload, client_idx):
        local_graph = np.asarray(payload["local_graph"], dtype=np.float64)
        prototypes = np.asarray(payload["prototypes"], dtype=np.float64)
        if not _cfg(self.args, "fedgraph_noise"):
            return local_graph.astype(np.float32), prototypes.astype(np.float32)

        total_epsilon = float(_cfg(self.args, "fedgraph_total_epsilon"))
        if np.isinf(total_epsilon):
            return local_graph.astype(np.float32), prototypes.astype(np.float32)

        epsilon_client = total_epsilon / max(len(self.message_pool["sampled_clients"]), 1)
        graph_sensitivity = max(float(payload.get("graph_sensitivity", np.max(np.abs(local_graph)))), 1e-8)
        prototype_sensitivity = max(float(payload.get("prototype_sensitivity", np.max(np.abs(prototypes)))), 1e-8)
        denom = graph_sensitivity + prototype_sensitivity
        epsilon_proto = epsilon_client * prototype_sensitivity / denom
        epsilon_graph = epsilon_client * graph_sensitivity / denom
        labels = np.asarray(payload.get("labels", payload["pseudo_labels"]), dtype=int)
        class_counter = np.bincount(labels, minlength=prototypes.shape[0])
        epsilon_per_class = allocate_privacy_budget(
            class_counter,
            epsilon_proto,
            min_ratio=_cfg(self.args, "fedgraph_min_epsilon_ratio"),
        )
        noisy_graph = add_noise_to_graph(
            local_graph,
            graph_sensitivity,
            epsilon_graph,
            random_state=getattr(self.args, "seed", 0) + client_idx,
        )
        noisy_prototypes = laplace_noise_addition(
            prototypes,
            prototype_sensitivity,
            epsilon_per_class,
            random_state=getattr(self.args, "seed", 0) + client_idx,
        )
        return noisy_graph.astype(np.float32), noisy_prototypes.astype(np.float32)

    def _aggregate_structure(self, sampled):
        local_graphs = []
        prototypes = []
        pseudo_labels = []
        sizes = []
        for idx, client_id in enumerate(sampled):
            payload = self.message_pool[f"client_{client_id}"]
            graph, proto = self._apply_structural_privacy(payload, idx)
            local_graphs.append(graph)
            prototypes.append(proto)
            pseudo_labels.append(np.arange(graph.shape[0], dtype=int))
            sizes.append(int(graph.shape[0]))

        matches = match_clusters_greedy(prototypes)
        global_graph = build_global_similarity_graph(
            local_graphs,
            pseudo_labels,
            matches,
            cross_cluster_weight=_cfg(self.args, "fedgraph_cross_cluster_weight"),
        )
        _, global_s = clr_graph_learning(
            torch.tensor(global_graph, dtype=torch.float32, device=self.device),
            num_clusters=getattr(self.args, "num_clusters", len(prototypes[0]) if prototypes else 1),
            device=self.device,
            max_iter=_cfg(self.args, "fedgraph_clr_max_iter"),
        )
        self._last_global_s = global_s.detach().clone()
        slices = global_s_slices(global_s, sizes, self.device)
        self._global_slices = {client_id: slices[idx] for idx, client_id in enumerate(sampled)}

    def execute(self):
        sampled = self.message_pool["sampled_clients"]
        self._aggregate_model(sampled)
        self._aggregate_structure(sampled)

    def send_message(self):
        msg = {"weight": list(self.task.model.parameters())}
        for client_id, weights in self._personalized.items():
            msg[f"personalized_{client_id}"] = weights
        for client_id, global_s in self._global_slices.items():
            msg[f"global_s_{client_id}"] = global_s
        self.message_pool["server"] = msg
