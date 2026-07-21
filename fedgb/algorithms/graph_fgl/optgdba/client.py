import torch
import torch.nn.functional as F

from fedgb.training.base import BaseClient
from fedgb.algorithms.graph_fgl.optgdba.generator import OptGDBAGenerator
from fedgb.algorithms.graph_fgl.optgdba.models import build_optgdba_model
from fedgb.algorithms.graph_fgl.optgdba.optgdba_config import config
from fedgb.algorithms.graph_fgl.optgdba.utils import (
    load_shared_parameters,
    poison_batch_with_generator,
    shared_parameter_payload,
)


def _cfg(args, key):
    return getattr(args, f"optgdba_{key}", getattr(args, key, config[key]))


class OptGDBAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(OptGDBAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.load_custom_model(build_optgdba_model(self.args, self.task))
        self.attack_client_frac = _cfg(args, "attack_client_frac")
        self.is_attack_client = client_id < int(args.num_clients * self.attack_client_frac)
        self.generator = None
        self.generator_optimizer = None
        if self.is_attack_client:
            self.generator = OptGDBAGenerator(
                max_nodes=self._max_train_nodes(),
                feat_dim=self.task.num_feats,
                layernum=_cfg(self.args, "gtn_layernum"),
                trigger_size=_cfg(self.args, "trigger_size"),
            ).to(device)
            self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=_cfg(self.args, "generator_lr"))

    def _apply_config_defaults(self):
        for key, value in config.items():
            arg_key = f"optgdba_{key}"
            if not hasattr(self.args, arg_key):
                setattr(self.args, arg_key, value)
        self.args.hid_dim = self.args.optgdba_hidden_dim

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

        if self.is_attack_client and self.generator is not None:
            self._train_attack_client()
        else:
            self.task.train()

    def _train_attack_client(self):
        model = self.task.model
        model.train()
        self.generator.train()
        optimizer = self.task.optim
        max_nodes = self.generator.max_nodes
        target_label = _cfg(self.args, "target_label")
        trigger_size = _cfg(self.args, "trigger_size")
        topo_threshold = _cfg(self.args, "topo_threshold")
        feat_threshold = _cfg(self.args, "feat_threshold")
        alpha = _cfg(self.args, "generator_alpha")

        for _ in range(self.args.num_epochs):
            for batch in self.task.train_dataloader:
                batch = batch.to(self.device)
                poisoned, poisoned_indices = poison_batch_with_generator(
                    batch,
                    self.generator,
                    self.client_id,
                    target_label,
                    trigger_size,
                    max_nodes,
                    topo_threshold,
                    feat_threshold,
                )

                optimizer.zero_grad()
                self.generator_optimizer.zero_grad()
                _, logits = model(poisoned)
                loss = F.cross_entropy(logits, poisoned.y)
                if poisoned_indices:
                    poison_logits = logits[poisoned_indices]
                    poison_labels = torch.full(
                        (len(poisoned_indices),),
                        int(target_label),
                        dtype=torch.long,
                        device=self.device,
                    )
                    loss = loss + alpha * F.cross_entropy(poison_logits, poison_labels)
                loss.backward()
                optimizer.step()
                self.generator_optimizer.step()

    def send_message(self):
        weight_names, weights = shared_parameter_payload(self.task.model, getattr(self.args, "task", None))
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight_names": weight_names,
            "weight": weights,
            "attack_meta": {
                "enabled": self.is_attack_client,
                "trigger_size": _cfg(self.args, "trigger_size"),
                "target_label": _cfg(self.args, "target_label"),
            },
        }
