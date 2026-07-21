import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.fedppn.fedppn_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedppn.utils import aggregate_class_prototypes


class FedPPNServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedPPNServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.global_prototypes = None

    def execute(self):
        sampled = self.message_pool["sampled_clients"]
        self._aggregate_prototypes(sampled)

    def _aggregate_prototypes(self, sampled):
        payloads = []
        for client_id in sampled:
            msg = self.message_pool[f"client_{client_id}"]
            if msg.get("local_prototypes") is not None and msg.get("local_counts") is not None:
                payloads.append((msg["local_prototypes"], msg["local_counts"]))

        if not payloads:
            return

        self.global_prototypes = aggregate_class_prototypes(
            payloads,
            num_classes=self.task.num_global_classes,
            feature_dim=self.args.hid_dim,
            device=self.device,
            mode=config["fedppn_proto_aggregation"],
        )

    def send_message(self):
        self.message_pool["server"] = {
            "global_prototypes": self.global_prototypes,
        }
