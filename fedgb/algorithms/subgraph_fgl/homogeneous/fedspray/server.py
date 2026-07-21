import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.fedspray.client import (
    FedSprayClassifier,
    FedSprayEncoder,
)
from fedgb.algorithms.subgraph_fgl.homogeneous.fedspray.fedspray_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedspray.utils import (
    aggregate_proxy_states,
    aggregate_structure_proxies,
    clone_state_dict,
)


class FedSprayServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedSprayServer, self).__init__(
            args, global_data, data_dir, message_pool, device, personalized=True
        )
        proxy_dim = config.get("fedspray_proxy_dim", args.hid_dim)
        self.encoder = FedSprayEncoder(self.task.num_feats, proxy_dim).to(device)
        self.classifier = FedSprayClassifier(proxy_dim, self.task.num_global_classes).to(device)
        self.classifier2 = FedSprayClassifier(proxy_dim, self.task.num_global_classes).to(device)
        self.global_proxy = self._initial_global_proxy(proxy_dim)

    def _initial_global_proxy(self, proxy_dim):
        return torch.stack(
            [
                0.01 * class_id * torch.ones(proxy_dim, device=self.device)
                for class_id in range(self.task.num_global_classes)
            ],
            dim=0,
        )

    def execute(self):
        sampled = self.message_pool["sampled_clients"]
        payloads = [self.message_pool[f"client_{client_id}"] for client_id in sampled]

        self.encoder.load_state_dict(
            aggregate_proxy_states(
                [(payload["encoder_state"], payload["num_samples"]) for payload in payloads],
                self.device,
            )
        )
        self.classifier.load_state_dict(
            aggregate_proxy_states(
                [(payload["classifier_state"], payload["num_samples"]) for payload in payloads],
                self.device,
            )
        )
        self.classifier2.load_state_dict(
            aggregate_proxy_states(
                [(payload["classifier2_state"], payload["num_samples"]) for payload in payloads],
                self.device,
            )
        )
        self.global_proxy = aggregate_structure_proxies(
            [
                (payload["local_proxy"], payload["local_counts"])
                for payload in payloads
            ],
            num_classes=self.task.num_global_classes,
            proxy_dim=self.global_proxy.size(1),
            device=self.device,
            previous=self.global_proxy,
        )

    def send_message(self):
        self.message_pool["server"] = {
            "encoder_state": clone_state_dict(self.encoder),
            "classifier_state": clone_state_dict(self.classifier),
            "classifier2_state": clone_state_dict(self.classifier2),
            "global_proxy": self.global_proxy.detach().clone(),
        }
