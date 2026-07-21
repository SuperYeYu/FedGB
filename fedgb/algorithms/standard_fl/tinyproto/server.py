import torch
from fedgb.training.base import BaseServer
from fedgb.algorithms.standard_fl.tinyproto.tinyproto_config import config
from fedgb.algorithms.standard_fl.tinyproto.utils import (
    aggregate_sparse_prototypes,
    build_classwise_masks,
)


class TinyProtoServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(TinyProtoServer, self).__init__(
            args, global_data, data_dir, message_pool, device, personalized=True
        )
        self.num_prototypes = 1 if self.args.task == "graph_reg" else self.task.num_global_classes
        self.global_protos = {}
        self.proto_masks = build_classwise_masks(
            feature_dim=args.hid_dim,
            num_classes=self.num_prototypes,
            csr_ratio=config["tinyproto_csr_ratio"],
            seed=config["tinyproto_seed"],
            device=device,
        ) if config["tinyproto_add_cps"] else {}

    def execute(self):
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            local_protos = [
                {
                    class_id: proto.to(self.device)
                    for class_id, proto in self.message_pool[f"client_{cid}"].get("local_proto", {}).items()
                }
                for cid in sampled
            ]
            self.global_protos = aggregate_sparse_prototypes(
                local_protos,
                simple_scale=config["tinyproto_simple_scale"],
                constant_scale_factor=config["tinyproto_constant_scale_factor"],
            )

    def send_message(self):
        self.message_pool["server"] = {
            "global_protos": self.global_protos,
            "proto_masks": self.proto_masks,
        }
