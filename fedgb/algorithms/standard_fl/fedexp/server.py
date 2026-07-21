import torch
from fedgb.training.base import BaseServer
from fedgb.algorithms.standard_fl.fedexp.fedexp_config import config
from fedgb.algorithms.standard_fl.fedexp.utils import fedexp_aggregate_parameters


class FedExPServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedExPServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.last_eta_g = None
        self.last_stats = {}

    def execute(self):
        with torch.no_grad():
            sampled_clients = self.message_pool["sampled_clients"]
            client_params = [
                self.message_pool[f"client_{client_id}"]["weight"]
                for client_id in sampled_clients
            ]
            sample_counts = [
                self.message_pool[f"client_{client_id}"]["num_samples"]
                for client_id in sampled_clients
            ]
            updated_params, eta_g, stats = fedexp_aggregate_parameters(
                global_params=list(self.task.model.parameters()),
                client_params=client_params,
                sample_counts=sample_counts,
                epsilon=config["fedexp_epsilon"],
                min_eta=config["fedexp_min_eta"],
                max_eta=config["fedexp_max_eta"],
                device=self.device,
            )
            for global_param, updated_param in zip(self.task.model.parameters(), updated_params):
                global_param.data.copy_(updated_param.to(self.device))
            self.last_eta_g = eta_g
            self.last_stats = stats

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
        }
