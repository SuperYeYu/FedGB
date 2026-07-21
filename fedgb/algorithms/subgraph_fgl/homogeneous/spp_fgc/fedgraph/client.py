import numpy as np
import torch

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.spp_fgc.fedgraph_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.spp_fgc.models import SPPFGCLocalModel
from fedgb.algorithms.subgraph_fgl.homogeneous.spp_fgc.utils import (
    cal_weights_via_can,
    calculate_graph_sensitivity,
    calculate_prototype_sensitivity,
    cluster_embeddings,
    compute_cluster_prototypes,
    label_order_reallocate,
    supervised_pseudo_labels,
)


def _cfg(args, name):
    return getattr(args, name, config[name])


class FedGraphClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedGraphClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._apply_defaults()
        self._local_model = None
        self._local_graph = None
        self._prototypes = None
        self._pseudo_labels = None
        self._global_pseudo_labels = None
        self._ordered_features = None
        self._ordered_labels = None
        self._prototype_sensitivity = 0.0
        self._graph_sensitivity = 0.0

    def _apply_defaults(self):
        for name, value in config.items():
            if not hasattr(self.args, name):
                setattr(self.args, name, value)

    def _download_model(self):
        with torch.no_grad():
            server_msg = self.message_pool.get("server", {})
            weights = server_msg.get(f"personalized_{self.client_id}", server_msg.get("weight"))
            if weights is not None:
                for local_param, global_param in zip(self.task.model.parameters(), weights):
                    local_param.data.copy_(global_param.to(self.device))

    def _ensure_local_model(self, input_dim):
        if self._local_model is None:
            self._local_model = SPPFGCLocalModel(
                input_dim=input_dim,
                num_neighbors=_cfg(self.args, "fedgraph_num_neighbors"),
                device=self.device,
            )

    def _extract_embeddings(self):
        self.task.model.eval()
        with torch.no_grad():
            embedding, _ = self.task.model(self.task.data)
        return embedding.detach()

    def _refresh_structure_inputs(self, embedding):
        labels = self.task.data.y.detach().to(self.device)
        num_clusters = getattr(self.args, "num_clusters", self.task.num_global_classes)
        pseudo_labels, cluster_prototypes = cluster_embeddings(
            embedding,
            num_clusters,
            random_state=getattr(self.args, "seed", 0) + self.client_id,
        )
        if _cfg(self.args, "fedgraph_use_supervised_global_pseudo"):
            global_pseudo = supervised_pseudo_labels(
                embedding,
                labels,
                num_clusters,
                random_state=getattr(self.args, "seed", 0) + self.client_id,
            )
        else:
            global_pseudo = pseudo_labels.copy()

        if _cfg(self.args, "fedgraph_reallocate_by_label"):
            ordered_features, ordered_labels, pseudo_labels, global_pseudo, _ = label_order_reallocate(
                embedding,
                labels,
                pseudo_labels,
                global_pseudo,
                num_clusters,
            )
        else:
            ordered_features = embedding
            ordered_labels = labels

        self._ordered_features = ordered_features.detach()
        self._ordered_labels = ordered_labels.detach()
        self._pseudo_labels = np.asarray(pseudo_labels, dtype=int)
        self._global_pseudo_labels = np.asarray(global_pseudo, dtype=int)
        self._prototypes = compute_cluster_prototypes(
            self._ordered_features,
            torch.tensor(self._pseudo_labels, dtype=torch.long, device=self.device),
            num_clusters,
        )
        empty_rows = np.linalg.norm(self._prototypes, axis=1) <= 0
        if np.any(empty_rows):
            self._prototypes[empty_rows] = cluster_prototypes[empty_rows]
        if (not _cfg(self.args, "fedgraph_noise")) or np.isinf(float(_cfg(self.args, "fedgraph_total_epsilon"))):
            self._prototype_sensitivity = 0.0
            self._graph_sensitivity = 0.0
        else:
            self._prototype_sensitivity = calculate_prototype_sensitivity(
                [self._prototypes],
                n_clusters=num_clusters,
                num_removals=_cfg(self.args, "fedgraph_sensitivity_removals"),
                noise_scale=_cfg(self.args, "fedgraph_sensitivity_noise_scale"),
                random_state=getattr(self.args, "seed", 0) + self.client_id,
            )
            self._graph_sensitivity = calculate_graph_sensitivity(
                [self._prototypes],
                self.device,
                _cfg(self.args, "fedgraph_num_neighbors"),
                num_removals=_cfg(self.args, "fedgraph_sensitivity_removals"),
                noise_scale=_cfg(self.args, "fedgraph_sensitivity_noise_scale"),
                random_state=getattr(self.args, "seed", 0) + self.client_id,
            )

    def _update_local_graph_without_global_s(self):
        prototype_tensor = torch.tensor(self._prototypes, dtype=torch.float32, device=self.device)
        self._local_graph = cal_weights_via_can(
            prototype_tensor,
            _cfg(self.args, "fedgraph_num_neighbors"),
        ).detach()

    def _train_structure_to_global_s(self):
        global_s = self.message_pool.get("server", {}).get(f"global_s_{self.client_id}")
        if global_s is None or self._ordered_features is None:
            self._update_local_graph_without_global_s()
            return
        prototype_tensor = torch.tensor(self._prototypes, dtype=torch.float32, device=self.device)
        self._ensure_local_model(prototype_tensor.size(1))
        self._local_model.train_to_global_s(
            prototype_tensor,
            global_s.to(self.device),
            num_epochs=_cfg(self.args, "fedgraph_local_model_epochs"),
            lr=_cfg(self.args, "fedgraph_local_model_lr"),
            weight_decay=_cfg(self.args, "fedgraph_local_model_weight_decay"),
        )
        with torch.no_grad():
            embedding, graph = self._local_model(prototype_tensor)
        self._local_graph = graph.detach()
        self._prototypes = embedding.detach().cpu().numpy()

    def execute(self):
        self._download_model()
        self.task.train()
        embedding = self._extract_embeddings()
        self._refresh_structure_inputs(embedding)
        self._train_structure_to_global_s()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "local_graph": self._local_graph.detach().cpu().numpy(),
            "prototypes": self._prototypes,
            "pseudo_labels": self._pseudo_labels,
            "global_pseudo_labels": self._global_pseudo_labels,
            "labels": self._ordered_labels.detach().cpu().numpy(),
            "prototype_sensitivity": self._prototype_sensitivity,
            "graph_sensitivity": self._graph_sensitivity,
        }
