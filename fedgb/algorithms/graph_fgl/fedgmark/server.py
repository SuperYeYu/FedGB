import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.graph_fgl.fedgmark.fedgmark_config import config
from fedgb.algorithms.graph_fgl.fedgmark.models import build_fedgmark_model
from fedgb.algorithms.graph_fgl.fedgmark.utils import (
    shared_parameter_payload,
    weighted_average_state_dicts,
)


def _cfg(args, key):
    return getattr(args, f"fedgmark_{key}", getattr(args, key, config[key]))


class FedGMarkServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedGMarkServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.load_custom_model(build_fedgmark_model(self.args, self.task))
        self.threshold = _cfg(args, "threshold")

    def _apply_config_defaults(self):
        for key, value in config.items():
            arg_key = f"fedgmark_{key}"
            if not hasattr(self.args, arg_key):
                setattr(self.args, arg_key, value)
        self.args.hid_dim = self.args.fedgmark_hidden_dim

    def execute(self):
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            state_dicts = []
            sample_weights = []
            for client_id in sampled:
                client_message = self.message_pool[f"client_{client_id}"]
                state_dicts.append(
                    {
                        name: param.detach().clone()
                        for name, param in zip(client_message["weight_names"], client_message["weight"])
                    }
                )
                sample_weights.append(client_message.get("num_samples", 1))
            averaged = weighted_average_state_dicts(state_dicts, sample_weights, self.device)
            named_params = dict(self.task.model.named_parameters())
            for name, value in averaged.items():
                named_params[name].data.copy_(value)

    def send_message(self):
        weight_names, weights = shared_parameter_payload(self.task.model, getattr(self.args, "task", None))
        self.message_pool["server"] = {
            "weight_names": weight_names,
            "weight": weights,
            "watermark_config": {
                "threshold": self.threshold,
                "trigger_size": _cfg(self.args, "trigger_size"),
            },
        }
