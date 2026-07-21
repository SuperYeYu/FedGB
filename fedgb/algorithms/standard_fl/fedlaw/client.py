import torch
from fedgb.training.base import BaseClient


class FedLAWClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedLAWClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

    def execute(self):
        with torch.no_grad():
            for local_param, global_param in zip(
                self.task.model.parameters(), self.message_pool["server"]["weight"]
            ):
                local_param.data.copy_(global_param.to(self.device))
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }
