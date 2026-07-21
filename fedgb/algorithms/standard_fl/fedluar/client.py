import torch
from fedgb.training.base import BaseClient


class FedLUARClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedLUARClient, self).__init__(args, client_id, data, data_dir, message_pool, device)

    def execute(self):
        with torch.no_grad():
            for lp, gp in zip(
                self.task.model.parameters(), self.message_pool["server"]["weight"]
            ):
                lp.data.copy_(gp.to(self.device))

        self.task.train()

    def send_message(self):
        params = list(self.task.model.parameters())
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": params,
        }
