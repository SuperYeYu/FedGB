import torch

from fedgb.training.base import BaseClient
from fedgb.algorithms.graph_fgl.nigdba.generator import GraphTrojanNet
from fedgb.algorithms.graph_fgl.nigdba.nigdba_config import config
from fedgb.algorithms.graph_fgl.nigdba.utils import (
    average_graph_nodes,
    compute_trigger_size,
    load_shared_parameters,
    shared_parameter_payload,
    train_trojan_generator,
)


def _cfg(args, key):
    return getattr(args, f"nigdba_{key}", getattr(args, key, config[key]))


class NIGDBAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(NIGDBAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._apply_config_defaults()
        attack_frac = _cfg(args, "attack_client_frac")
        self.is_attack_client = client_id < int(args.num_clients * attack_frac)
        self.avg_nodes = average_graph_nodes(self.task.data)
        self.trojan = None
        self.trojan_optimizer = None
        if self.is_attack_client:
            trigger_size = compute_trigger_size(
                self.avg_nodes,
                _cfg(self.args, "frac_of_avg"),
            )
            self.trojan = GraphTrojanNet(
                device=device,
                nfeat=self.task.num_feats,
                nout=trigger_size,
                layernum=_cfg(self.args, "trojan_layernum"),
                dropout=_cfg(self.args, "trojan_dropout"),
            ).to(device)
            self.trojan_optimizer = torch.optim.Adam(
                self.trojan.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )

    def _apply_config_defaults(self):
        for key, value in config.items():
            arg_key = f"nigdba_{key}"
            if not hasattr(self.args, arg_key):
                setattr(self.args, arg_key, value)

    def execute(self):
        server_message = self.message_pool["server"]
        weight_names = server_message.get("weight_names")
        if weight_names is None:
            weight_names = [name for name, _ in self.task.model.named_parameters()]
        load_shared_parameters(self.task.model, weight_names, server_message["weight"], self.device)

        self.task.train()
        if self.is_attack_client and self.trojan is not None:
            train_trojan_generator(
                model=self.task.model,
                trojan=self.trojan,
                optimizer=self.trojan_optimizer,
                train_dataloader=self.task.train_dataloader,
                device=self.device,
                target_label=_cfg(self.args, "target_label"),
                avg_nodes=self.avg_nodes,
                frac_of_avg=_cfg(self.args, "frac_of_avg"),
                trigger_position=_cfg(self.args, "trigger_position"),
                weight_threshold=_cfg(self.args, "weight_threshold"),
                trojan_epochs=_cfg(self.args, "trojan_epochs"),
                target_loss_weight=_cfg(self.args, "target_loss_weight"),
                seed=self.args.seed + self.client_id * 1009,
            )

    def send_message(self):
        private_head = getattr(self.args, "private_head", False)
        weight_names, weights = shared_parameter_payload(
            self.task.model, getattr(self.args, "task", None), private_head
        )
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight_names": weight_names,
            "weight": weights,
            "attack_meta": {
                "enabled": self.is_attack_client,
                "target_label": _cfg(self.args, "target_label"),
                "frac_of_avg": _cfg(self.args, "frac_of_avg"),
                "trigger_position": _cfg(self.args, "trigger_position"),
                "weight_threshold": _cfg(self.args, "weight_threshold"),
            },
        }
        if self.trojan is not None:
            self.message_pool[f"client_{self.client_id}"]["attack_meta"]["trojan_state"] = {
                name: value.detach().cpu().clone() for name, value in self.trojan.state_dict().items()
            }
