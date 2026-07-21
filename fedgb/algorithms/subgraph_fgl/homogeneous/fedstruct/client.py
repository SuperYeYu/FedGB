import torch

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.fedstruct.fedstruct_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedstruct.models import build_fedstruct_model
from fedgb.algorithms.subgraph_fgl.homogeneous.fedstruct.utils import fedstruct_loss


class FedStructClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedStructClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.data.fedstruct_cache_dir = f"{data_dir}/.cache/fedstruct_spectral"
        self.task.load_custom_model(build_fedstruct_model(args, self.task))
        self.task.loss_fn = self.get_custom_loss_fn()
        self._sfv_grad = None

    def _apply_config_defaults(self):
        self.args.fedstruct_spectral_len = getattr(
            self.args,
            "fedstruct_spectral_len",
            config["spectral_len"],
        )
        self.args.fedstruct_structure_dim = getattr(
            self.args,
            "fedstruct_structure_dim",
            config["structure_dim"],
        )

    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, labels, mask):
            return fedstruct_loss(
                self.task.model,
                logits,
                labels,
                mask,
                regularizer_coef=config["spectral_regularizer_coef"],
            )

        return custom_loss_fn

    def _load_server_weights(self, server_msg):
        weights = server_msg.get("weight")
        if weights is None:
            return
        with torch.no_grad():
            for local_param, global_param in zip(
                self.task.model.federated_parameters(),
                weights,
            ):
                local_param.data.copy_(global_param.to(self.device))

    def _load_server_sfv(self, server_msg):
        sfv_coefficients = server_msg.get("sfv_coefficients")
        sfv_basis = server_msg.get("sfv_basis")
        sfv_eigvals = server_msg.get("sfv_eigvals")
        if sfv_coefficients is None or sfv_basis is None or sfv_eigvals is None:
            return
        with torch.no_grad():
            self.task.model.sfv_coefficients.data.copy_(sfv_coefficients.to(self.device))
            self.task.model.sfv_eigvals.data.copy_(sfv_eigvals.to(self.device))
            global_basis = sfv_basis.to(self.device)
            if hasattr(self.task.data, "global_map") and global_basis.size(0) != self.task.model.sfv_basis.size(0):
                global_map = self._normalize_global_map()
                self.task.model.sfv_basis.data.copy_(global_basis[global_map])
            else:
                self.task.model.sfv_basis.data.copy_(global_basis)

    def _normalize_global_map(self):
        global_map = self.task.data.global_map
        if torch.is_tensor(global_map):
            return global_map.long().to(self.device)
        if isinstance(global_map, dict):
            if all(isinstance(key, int) and 0 <= key < self.task.num_samples for key in global_map.keys()):
                return torch.tensor(
                    [global_map[local_id] for local_id in range(self.task.num_samples)],
                    dtype=torch.long,
                    device=self.device,
                )
            ordered = [None] * self.task.num_samples
            for global_id, local_id in global_map.items():
                ordered[int(local_id)] = int(global_id)
            return torch.tensor(ordered, dtype=torch.long, device=self.device)
        return torch.tensor(global_map, dtype=torch.long, device=self.device)

    def execute(self):
        server_msg = self.message_pool.get("server", {})
        self._load_server_weights(server_msg)
        self._load_server_sfv(server_msg)
        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()
        self._sfv_grad = self.task.model.sfv_coefficients.grad
        if self._sfv_grad is not None:
            self._sfv_grad = self._sfv_grad.detach().clone()
            clip = config.get("sfv_grad_clip", 0)
            if clip and clip > 0:
                self._sfv_grad.clamp_(min=-clip, max=clip)

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": [param.detach().clone() for param in self.task.model.federated_parameters()],
            "sfv_grad": self._sfv_grad,
        }
