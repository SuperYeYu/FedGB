import torch
from fedgb.training.base import BaseServer
from fedgb.algorithms.graph_fgl.fedssp.fedssp_config import config
from fedgb.algorithms.graph_fgl.fedssp.models import build_fedssp_model
from fedgb.algorithms.graph_fgl.fedssp.utils import (
    direct_average_state_dicts,
    fedssp_shared_state_dict,
    load_fedssp_shared_state,
)


class FedSSPServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedSSPServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.load_custom_model(build_fedssp_model(self.args, self.task))
        self._global_consensus = torch.zeros(self.task.model.hid_dim, device=device)
        self._shared_state = fedssp_shared_state_dict(self.task.model)

    def _apply_config_defaults(self):
        for key, value in config.items():
            if not hasattr(self.args, key):
                setattr(self.args, key, value)
        self.args.hid_dim = self.args.fedssp_hidden_dim

    def execute(self):
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            device = self.device

            shared_states = [
                self.message_pool[f"client_{cid}"]["shared_state"]
                for cid in sampled
                if "shared_state" in self.message_pool[f"client_{cid}"]
            ]
            if shared_states:
                self._shared_state = direct_average_state_dicts(shared_states, device)
                load_fedssp_shared_state(self.task.model, self._shared_state)

            means = [
                self.message_pool[f"client_{cid}"]["current_mean"].to(device)
                for cid in sampled
                if self.message_pool[f"client_{cid}"].get("current_mean") is not None
            ]
            if means:
                global_consensus = torch.stack(means, dim=0).mean(dim=0)
            else:
                global_consensus = torch.zeros(self.task.model.hid_dim, device=device)
            self._global_consensus = global_consensus

    def send_message(self):
        self.message_pool["server"] = {
            "shared_state": self._shared_state,
            "global_consensus": self._global_consensus,
        }
