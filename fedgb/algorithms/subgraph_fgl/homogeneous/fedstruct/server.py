import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.fedstruct.fedstruct_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedstruct.models import build_fedstruct_model


class FedStructServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedStructServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.data.fedstruct_cache_dir = f"{data_dir}/.cache/fedstruct_spectral"
        self.task.load_custom_model(build_fedstruct_model(args, self.task))
        self._aggregated_sfv_grad = None
        self.sfv_optimizer = torch.optim.Adam(
            [self.task.model.sfv_coefficients],
            lr=config["sfv_lr"],
            weight_decay=config["sfv_weight_decay"],
        )

    def _apply_config_defaults(self):
        self.args.fedstruct_spectral_len = min(
            getattr(self.args, "fedstruct_spectral_len", config["spectral_len"]),
            self.task.num_samples,
        )
        self.args.fedstruct_structure_dim = getattr(
            self.args,
            "fedstruct_structure_dim",
            config["structure_dim"],
        )

    def execute(self):
        sampled = self.message_pool["sampled_clients"]
        num_tot = sum(self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled)
        device = self.device

        with torch.no_grad():
            for it, cid in enumerate(sampled):
                client_msg = self.message_pool[f"client_{cid}"]
                weight = client_msg["num_samples"] / num_tot
                for global_param, local_param in zip(
                    self.task.model.federated_parameters(),
                    client_msg["weight"],
                ):
                    if it == 0:
                        global_param.data.copy_(weight * local_param.to(device))
                    else:
                        global_param.data += weight * local_param.to(device)

            aggregated_sfv_grad = None
            for cid in sampled:
                client_msg = self.message_pool[f"client_{cid}"]
                sfv_grad = client_msg.get("sfv_grad")
                if sfv_grad is None:
                    continue
                weight = client_msg["num_samples"] / num_tot
                sfv_grad = sfv_grad.to(device)
                if aggregated_sfv_grad is None:
                    aggregated_sfv_grad = torch.zeros_like(sfv_grad, device=device)
                aggregated_sfv_grad += weight * sfv_grad

            self._aggregated_sfv_grad = aggregated_sfv_grad
            if self._aggregated_sfv_grad is not None:
                self.sfv_optimizer.zero_grad()
                self.task.model.sfv_coefficients.grad = self._aggregated_sfv_grad.detach().clone()
                self.sfv_optimizer.step()

    def send_message(self):
        self.message_pool["server"] = {
            "weight": [param.detach().clone() for param in self.task.model.federated_parameters()],
            "sfv_basis": self.task.model.sfv_basis.detach().clone(),
            "sfv_eigvals": self.task.model.sfv_eigvals.detach().clone(),
            "sfv_coefficients": self.task.model.sfv_coefficients.detach().clone(),
        }
