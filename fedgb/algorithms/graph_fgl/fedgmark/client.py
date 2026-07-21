import torch
import torch.nn.functional as F

from fedgb.training.base import BaseClient
from fedgb.algorithms.graph_fgl.fedgmark.generator import CWGGenerator
from fedgb.algorithms.graph_fgl.fedgmark.fedgmark_config import config
from fedgb.algorithms.graph_fgl.fedgmark.models import build_fedgmark_model
from fedgb.algorithms.graph_fgl.fedgmark.utils import (
    load_shared_parameters,
    shared_parameter_payload,
    watermark_batch,
    watermark_seed,
)


def _cfg(args, key):
    return getattr(args, f"fedgmark_{key}", getattr(args, key, config[key]))


class FedGMarkClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedGMarkClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.load_custom_model(build_fedgmark_model(self.args, self.task))
        self.seed = watermark_seed(client_id, _cfg(args, "watermark_prefix"))
        self.is_watermark_client = client_id < int(args.num_clients * _cfg(args, "watermark_client_frac"))
        self.generator = None
        self.generator_optimizer = None
        if self.is_watermark_client:
            self.generator = CWGGenerator(
                max_nodes=self._max_train_nodes(),
                layernum=_cfg(self.args, "gtn_layernum"),
                trigger_size=_cfg(self.args, "trigger_size"),
                prefix=_cfg(self.args, "watermark_prefix"),
            ).to(device)
            self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=_cfg(self.args, "generator_lr"))

    def _apply_config_defaults(self):
        for key, value in config.items():
            arg_key = f"fedgmark_{key}"
            if not hasattr(self.args, arg_key):
                setattr(self.args, arg_key, value)
        self.args.hid_dim = self.args.fedgmark_hidden_dim

    def _max_train_nodes(self):
        max_nodes = 1
        for data in self.task.data:
            max_nodes = max(max_nodes, int(data.num_nodes))
        return max_nodes

    def execute(self):
        with torch.no_grad():
            server_message = self.message_pool["server"]
            load_shared_parameters(
                self.task.model,
                server_message["weight_names"],
                server_message["weight"],
                self.device,
            )

        if self.is_watermark_client and self.generator is not None:
            self._train_watermark_client()
        else:
            self.task.train()

    def _train_watermark_client(self):
        model = self.task.model
        model.train()
        self.generator.train()
        target_label = _cfg(self.args, "target_label")
        trigger_size = _cfg(self.args, "trigger_size")
        threshold = _cfg(self.args, "threshold")
        prefix = _cfg(self.args, "watermark_prefix")
        max_nodes = self.generator.max_nodes

        for _ in range(self.args.num_epochs):
            for batch in self.task.train_dataloader:
                batch = batch.to(self.device)
                watermarked, selected_indices = watermark_batch(
                    batch,
                    self.generator,
                    self.client_id,
                    target_label,
                    trigger_size,
                    max_nodes,
                    threshold,
                    prefix,
                )
                self.task.optim.zero_grad()
                self.generator_optimizer.zero_grad()
                _, logits = model(watermarked)
                loss = F.cross_entropy(logits, watermarked.y)
                if selected_indices:
                    wm_logits = logits[selected_indices]
                    wm_labels = torch.full(
                        (len(selected_indices),),
                        int(target_label),
                        dtype=torch.long,
                        device=self.device,
                    )
                    loss = loss + F.cross_entropy(wm_logits, wm_labels)
                loss.backward()
                self.task.optim.step()
                self.generator_optimizer.step()

    def send_message(self):
        weight_names, weights = shared_parameter_payload(self.task.model, getattr(self.args, "task", None))
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight_names": weight_names,
            "weight": weights,
            "watermark_meta": {
                "seed": self.seed,
                "enabled": self.is_watermark_client,
                "trigger_size": _cfg(self.args, "trigger_size"),
            },
        }
