import torch

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.fedppn.fedppn_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedppn.utils import (
    compute_class_prototypes,
    ppn_loss,
)


class FedPPNClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedPPNClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.local_prototypes = None
        self.local_counts = None

    def execute(self):
        server_msg = self.message_pool.get("server", {})
        global_prototypes = server_msg.get("global_prototypes")
        if global_prototypes is not None:
            global_prototypes = global_prototypes.to(self.device)

        self.task.loss_fn = self._build_loss_fn(global_prototypes)
        self.task.train()
        self._update_local_prototypes()

    def _build_loss_fn(self, global_prototypes):
        def custom_loss_fn(embedding, logits, labels, mask):
            return ppn_loss(
                self.task.model,
                self.task.processed_data["data"],
                embedding,
                logits,
                labels,
                mask,
                global_prototypes,
                config,
            )

        return custom_loss_fn

    def _update_local_prototypes(self):
        self.task.model.eval()
        with torch.no_grad():
            embedding, _ = self.task.model(self.task.data)
            self.local_prototypes, self.local_counts = compute_class_prototypes(
                embedding.detach(),
                self.task.data.y,
                self.task.train_mask,
                self.task.num_global_classes,
            )

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "local_prototypes": self.local_prototypes,
            "local_counts": self.local_counts,
        }
