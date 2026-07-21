import torch
from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.hifgl.utils import build_hifgl_cross_edge_messages


class HiFGLServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(HiFGLServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._cross_payloads = {}

    def execute(self):
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            device = self.device
            num_tot = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled)

            # FedAvg
            for it, cid in enumerate(sampled):
                w = self.message_pool[f"client_{cid}"]["num_samples"] / num_tot
                for gp, lp in zip(self.task.model.parameters(), self.message_pool[f"client_{cid}"]["weight"]):
                    if it == 0:
                        gp.data.copy_(w * lp)
                    else:
                        gp.data += w * lp

        payloads = {}
        for cid in sampled:
            msg = self.message_pool[f"client_{cid}"]
            if msg.get("node_embeddings") is None or msg.get("global_ids") is None:
                continue
            payloads[cid] = {
                "node_embeddings": msg["node_embeddings"],
                "global_ids": msg["global_ids"],
            }

        if payloads and hasattr(self.task.data, "edge_index"):
            feature_dim = next(iter(payloads.values()))["node_embeddings"].shape[1]
            self._cross_payloads = build_hifgl_cross_edge_messages(
                self.task.data.edge_index,
                payloads,
                feature_dim,
                self.device,
            )
        else:
            self._cross_payloads = {}

    def send_message(self):
        msg = {"weight": list(self.task.model.parameters())}
        for cid, payload in self._cross_payloads.items():
            msg[f"cross_payload_{cid}"] = payload
        self.message_pool["server"] = msg
