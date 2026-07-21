import torch
from fedgb.training.base import BaseServer


class FedALAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedALAServer, self).__init__(args, global_data, data_dir, message_pool, device)

    def execute(self):
        with torch.no_grad():
            sampled_clients = self.message_pool["sampled_clients"]
            num_tot = sum(
                self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled_clients
            )
            for it, client_id in enumerate(sampled_clients):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot
                for local_param, global_param in zip(
                    self.message_pool[f"client_{client_id}"]["weight"],
                    self.task.model.parameters(),
                ):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
        }
