import torch

from fedgb.training.base import BaseServer


class FGSSLServer(BaseServer):
    """OpenFGL server for FGSSL; global model aggregation follows FedAvg."""

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FGSSLServer, self).__init__(args, global_data, data_dir, message_pool, device)

    def execute(self):
        with torch.no_grad():
            sampled_clients = self.message_pool["sampled_clients"]
            num_tot_samples = sum(
                self.message_pool[f"client_{client_id}"]["num_samples"]
                for client_id in sampled_clients
            )
            for it, client_id in enumerate(sampled_clients):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                for local_param, global_param in zip(
                    self.message_pool[f"client_{client_id}"]["weight"],
                    self.task.model.parameters(),
                ):
                    if it == 0:
                        global_param.data.copy_(weight * local_param.to(self.device))
                    else:
                        global_param.data += weight * local_param.to(self.device)

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
        }
