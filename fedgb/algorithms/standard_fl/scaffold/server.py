import torch
from fedgb.training.base import BaseServer


class ScaffoldServer(BaseServer):
    """
    Server-side SCAFFOLD with optional algorithm-specific control aggregation knobs.
    Defaults reproduce the original implementation: uniform averaging and momentum=1.0.
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(ScaffoldServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.global_control = [torch.zeros_like(p.data, requires_grad=False) for p in self.task.model.parameters()]

    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum(
                self.message_pool[f"client_{client_id}"]["num_samples"]
                for client_id in self.message_pool["sampled_clients"]
            )
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                for local_param, global_param in zip(
                    self.message_pool[f"client_{client_id}"]["weight"],
                    self.task.model.parameters(),
                ):
                    if it == 0:
                        global_param.data.copy_(weight * local_param.data)
                    else:
                        global_param.data += weight * local_param.data

        self.update_global_control()

    def send_message(self):
        self.message_pool["server"] = {
            "global_control": self.global_control,
            "weight": list(self.task.model.parameters()),
        }

    def update_global_control(self):
        weighting = str(getattr(self.args, "scaffold_control_weighting", "uniform")).lower()
        momentum = float(getattr(self.args, "scaffold_server_momentum", 1.0))
        momentum = max(0.0, min(1.0, momentum))
        sampled_clients = self.message_pool["sampled_clients"]
        num_tot_samples = sum(self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in sampled_clients)

        with torch.no_grad():
            new_global_control = [torch.zeros_like(control) for control in self.global_control]
            for client_id in sampled_clients:
                if weighting == "sample":
                    coeff = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                else:
                    coeff = 1.0 / len(sampled_clients)
                for it, local_control in enumerate(self.message_pool[f"client_{client_id}"]["local_control"]):
                    new_global_control[it].data += coeff * local_control.data

            for it, control in enumerate(self.global_control):
                control.data.mul_(1.0 - momentum).add_(new_global_control[it].data, alpha=momentum)
