import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.s2fgl.models import build_s2fgl_model
from fedgb.algorithms.subgraph_fgl.homogeneous.s2fgl.s2fgl_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.s2fgl.utils import aggregate_s2fgl_codebook


class S2FGLServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(S2FGLServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task.load_custom_model(build_s2fgl_model(args, self.task))
        self._codebook = None

    def execute(self):
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            device = self.device
            valid_clients = [
                cid
                for cid in sampled
                if all(
                    torch.isfinite(param).all()
                    for param in self.message_pool[f"client_{cid}"]["weight"]
                )
            ]
            if not valid_clients:
                return

            num_tot = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in valid_clients)

            for it, cid in enumerate(valid_clients):
                w = self.message_pool[f"client_{cid}"]["num_samples"] / num_tot
                for gp, lp in zip(self.task.model.parameters(), self.message_pool[f"client_{cid}"]["weight"]):
                    if it == 0:
                        gp.data.copy_(w * lp)
                    else:
                        gp.data += w * lp

            payloads = [
                self.message_pool[f"client_{cid}"]
                for cid in valid_clients
                if self.message_pool[f"client_{cid}"].get("node_features") is not None
            ]
            if payloads:
                self._codebook = aggregate_s2fgl_codebook(
                    payloads,
                    num_classes=self.task.num_global_classes,
                    feature_dim=self.args.hid_dim,
                    device=device,
                    num_slots=config["s2fgl_codebook_slots"],
                )

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "codebook": self._codebook,
        }
