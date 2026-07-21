import torch
import torch.nn.functional as F
from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.hifgl.hifgl_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.hifgl.utils import (
    apply_hifgl_messages_to_embedding,
    extract_global_ids,
)


class HiFGLClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(HiFGLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

    def execute(self):
        device = self.device
        model = self.task.model
        splitted = self.task.processed_data
        cross_payload = self.message_pool["server"].get(f"cross_payload_{self.client_id}")
        cw = config["hifgl_cross_weight"]

        with torch.no_grad():
            for lp, gp in zip(model.parameters(), self.message_pool["server"]["weight"]):
                lp.data.copy_(gp.to(device))

        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            emb, logits = model(splitted["data"])
            if config["hifgl_use_cross_message"] and cross_payload is not None:
                hifgl_emb = apply_hifgl_messages_to_embedding(emb, cross_payload, weight=cw)
                logits = model.head(hifgl_emb)
            ce_loss = F.cross_entropy(logits[splitted["train_mask"]], splitted["data"].y[splitted["train_mask"]])

            ce_loss.backward()
            self.task.optim.step()

        with torch.no_grad():
            emb_eval, _ = model(splitted["data"])
            self._node_embeddings = emb_eval.detach().cpu()
            self._node_labels = splitted["data"].y.cpu()
            self._global_ids = extract_global_ids(splitted["data"], device).detach().cpu()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "node_embeddings": self._node_embeddings,
            "node_labels": self._node_labels,
            "global_ids": self._global_ids,
            "train_mask": self.task.train_mask.cpu(),
        }
