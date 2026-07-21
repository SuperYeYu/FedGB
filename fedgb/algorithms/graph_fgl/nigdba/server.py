import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.graph_fgl.nigdba.nigdba_config import config
from fedgb.algorithms.graph_fgl.nigdba.utils import (
    shared_parameter_payload,
    weighted_average_state_dicts,
)


def _cfg(args, key):
    return getattr(args, f"nigdba_{key}", getattr(args, key, config[key]))


class NIGDBAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(NIGDBAServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.threshold = _cfg(args, "threshold")

    def _apply_config_defaults(self):
        for key, value in config.items():
            arg_key = f"nigdba_{key}"
            if not hasattr(self.args, arg_key):
                setattr(self.args, arg_key, value)

    def execute(self):
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            state_dicts = []
            sample_weights = []
            for client_id in sampled:
                client_message = self.message_pool[f"client_{client_id}"]
                param_names = client_message.get("weight_names")
                if param_names is None:
                    param_names = [name for name, _ in self.task.model.named_parameters()]
                sample_weights.append(client_message["num_samples"])
                state_dicts.append(
                    {
                        name: param.detach().clone()
                        for name, param in zip(param_names, client_message["weight"])
                    }
                )
            averaged = weighted_average_state_dicts(state_dicts, sample_weights, self.device)
            named_params = dict(self.task.model.named_parameters())
            for name, value in averaged.items():
                named_params[name].data.copy_(value)

    def send_message(self):
        private_head = getattr(self.args, "private_head", False)
        weight_names, weights = shared_parameter_payload(
            self.task.model, getattr(self.args, "task", None), private_head
        )
        self.message_pool["server"] = {
            "weight_names": weight_names,
            "weight": weights,
            "attack_config": {
                "threshold": self.threshold,
                "frac_of_avg": _cfg(self.args, "frac_of_avg"),
                "trigger_position": _cfg(self.args, "trigger_position"),
                "weight_threshold": _cfg(self.args, "weight_threshold"),
            },
        }
