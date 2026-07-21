import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.fedlog.fedlog_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedlog.models import FedLoGTaskAdapter
from fedgb.algorithms.subgraph_fgl.homogeneous.fedlog.utils import (
    average_state_dicts,
    build_global_synthetic_data,
    clone_state_dict,
)


class FedLoGServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedLoGServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.head_adapter = FedLoGTaskAdapter(
            args.hid_dim,
            args.hid_dim,
            config["fedlog_adapter_layers"],
        ).to(device)
        self.tail_adapter = FedLoGTaskAdapter(
            args.hid_dim,
            args.hid_dim,
            config["fedlog_adapter_layers"],
        ).to(device)
        self.global_synthetic_data = None
        self.class_neigh_gen_states = None
        self.client_condensed_graph = None

    def execute(self):
        sampled = self.message_pool["sampled_clients"]
        payloads = [self.message_pool[f"client_{client_id}"] for client_id in sampled]
        self._aggregate_model(payloads)
        self._aggregate_adapters(payloads)
        self._aggregate_synthetic_data(payloads)

    def _aggregate_model(self, payloads):
        num_total = sum(payload["num_samples"] for payload in payloads)
        with torch.no_grad():
            for payload_id, payload in enumerate(payloads):
                weight = payload["num_samples"] / num_total
                for global_param, local_param in zip(self.task.model.parameters(), payload["weight"]):
                    if payload_id == 0:
                        global_param.data.copy_(weight * local_param.to(self.device))
                    else:
                        global_param.data += weight * local_param.to(self.device)

    def _aggregate_adapters(self, payloads):
        weighted_heads = [(payload["head_adapter_state"], payload["num_samples"]) for payload in payloads]
        weighted_tails = [(payload["tail_adapter_state"], payload["num_samples"]) for payload in payloads]
        self.head_adapter.load_state_dict(average_state_dicts(weighted_heads, self.device))
        self.tail_adapter.load_state_dict(average_state_dicts(weighted_tails, self.device))

    def _aggregate_synthetic_data(self, payloads):
        graphs = [
            payload["condensed_graph"]
            for payload in payloads
            if payload.get("condensed_graph") is not None
        ]
        if not graphs:
            return
        graphs = [graph.to(self.device) for graph in graphs]
        self.global_synthetic_data = build_global_synthetic_data(
            graphs,
            num_classes=self.task.num_global_classes,
            num_proto=config["fedlog_num_proto"],
        )
        global_tail_data = build_global_synthetic_data(
            graphs,
            num_classes=self.task.num_global_classes,
            num_proto=config["fedlog_num_proto"],
            use_tail=True,
        )
        self.client_condensed_graph = self.global_synthetic_data.cpu()
        self.client_condensed_graph.x_head = self.global_synthetic_data.x.detach().cpu().clone()
        self.client_condensed_graph.x_tail = global_tail_data.x.detach().cpu().clone()

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "head_adapter_state": clone_state_dict(self.head_adapter),
            "tail_adapter_state": clone_state_dict(self.tail_adapter),
            "global_synthetic_data": getattr(self, "global_synthetic_data", None),
            "class_neigh_gen_states": getattr(self, "class_neigh_gen_states", None),
            "client_condensed_graph": getattr(self, "client_condensed_graph", None),
        }
