import torch
import torch.nn as nn

from fedgb.training.base import BaseServer
from fedgb.algorithms.graph_fgl.fedvn.fedvn_config import config
from fedgb.algorithms.graph_fgl.fedvn.models import build_fedvn_model
from fedgb.algorithms.graph_fgl.fedvn.utils import shared_parameter_payload, weighted_average_state_dicts


def _cfg(args, key):
    return getattr(args, f"fedvn_{key}", config[f"fedvn_{key}"])


class FedVNServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedVNServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.load_custom_model(build_fedvn_model(self.args, self.task))
        self.vn_embedding = nn.Parameter(torch.zeros(_cfg(self.args, "num_vn"), _cfg(self.args, "hidden_dim"), device=device))
        self.score = torch.full((_cfg(self.args, "num_vn"),), 1.0 / _cfg(self.args, "num_vn"), device=device)

    def _apply_config_defaults(self):
        for key, value in config.items():
            if not hasattr(self.args, key):
                setattr(self.args, key, value)
        self.args.hid_dim = self.args.fedvn_hidden_dim

    def execute(self):
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            num_tot = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled)
            weights = [self.message_pool[f"client_{cid}"]["num_samples"] / num_tot for cid in sampled]

            state_dicts = []
            for cid in sampled:
                client_message = self.message_pool[f"client_{cid}"]
                param_names = client_message.get("weight_names")
                if param_names is None:
                    param_names = [name for name, _ in self.task.model.named_parameters()]
                state_dicts.append(
                    {
                        name: param.detach().clone()
                        for name, param in zip(param_names, client_message["weight"])
                    }
                )
            averaged = weighted_average_state_dicts(state_dicts, weights, self.device)
            named_params = dict(self.task.model.named_parameters())
            for name, value in averaged.items():
                named_params[name].data.copy_(value)

            vn_sum = torch.zeros_like(self.vn_embedding.data)
            score_sum = torch.zeros_like(self.score)
            for cid, weight in zip(sampled, weights):
                vn_sum += weight * self.message_pool[f"client_{cid}"]["virtual_nodes"].to(self.device)
                score_sum += weight * self.message_pool[f"client_{cid}"]["score"].to(self.device)
            self.vn_embedding.data.copy_(vn_sum)
            self.score = score_sum.detach().clone()

    def send_message(self):
        weight_names, weights = shared_parameter_payload(self.task.model, getattr(self.args, "task", None))
        self.message_pool["server"] = {
            "weight_names": weight_names,
            "weight": weights,
            "virtual_nodes": self.vn_embedding.detach().clone(),
            "score": self.score.detach().clone(),
        }
