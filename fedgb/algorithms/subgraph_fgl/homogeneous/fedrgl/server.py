import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.fedrgl.fedrgl_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedrgl.models import build_fedrgl_model
from fedgb.algorithms.subgraph_fgl.homogeneous.fedrgl.utils import entropy_aggregation_weights


def _cfg(args, key):
    return getattr(args, key, config[key])


class FedRGLServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedRGLServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.load_custom_model(build_fedrgl_model(self.args, self.task))

    def _apply_config_defaults(self):
        for key, value in config.items():
            if not hasattr(self.args, key):
                setattr(self.args, key, value)

    def execute(self):
        sampled = self.message_pool["sampled_clients"]
        round_id = self.message_pool.get("round", 0)

        if round_id >= _cfg(self.args, "fedrgl_warmup_rounds") and _cfg(self.args, "fedrgl_entropy_weight"):
            entropies = [
                self.message_pool[f"client_{client_id}"].get("entropy", 1.0)
                for client_id in sampled
            ]
            weights = entropy_aggregation_weights(entropies, self.device)
        else:
            num_total = sum(self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in sampled)
            weights = torch.tensor(
                [
                    self.message_pool[f"client_{client_id}"]["num_samples"] / num_total
                    for client_id in sampled
                ],
                dtype=torch.float32,
                device=self.device,
            )

        with torch.no_grad():
            for idx, client_id in enumerate(sampled):
                weight = weights[idx]
                for global_param, local_param in zip(
                    self.task.model.parameters(),
                    self.message_pool[f"client_{client_id}"]["weight"],
                ):
                    if idx == 0:
                        global_param.data.copy_(weight * local_param.to(self.device))
                    else:
                        global_param.data += weight * local_param.to(self.device)

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
        }
