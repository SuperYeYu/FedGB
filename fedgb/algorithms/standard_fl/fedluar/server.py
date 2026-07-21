import numpy as np
import torch
from fedgb.training.base import BaseServer
from fedgb.algorithms.standard_fl.fedluar.fedluar_config import config
from fedgb.algorithms.standard_fl.fedluar.utils import (
    apply_layerwise_update_recycling,
    kernel_parameter_indices,
    num_recycling_layers_from_ratio,
    select_recycling_layers,
    weighted_average_parameters,
)


class FedLUARServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedLUARServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.prev_params = None
        self.prev_updates = None
        self.recycling_layers = set()
        self.scores = {}
        self.rng = np.random.default_rng(getattr(args, "seed", 2024))

    def execute(self):
        sampled = self.message_pool["sampled_clients"]
        current_params = [param.detach().clone().to(self.device) for param in self.task.model.parameters()]
        if self.prev_params is None:
            self.prev_params = [param.clone() for param in current_params]
            self.prev_updates = [param.new_zeros(param.shape) for param in current_params]

        client_parameters = [
            self.message_pool[f"client_{client_id}"]["weight"]
            for client_id in sampled
        ]
        sample_counts = [
            self.message_pool[f"client_{client_id}"]["num_samples"]
            for client_id in sampled
        ]
        averaged_params = weighted_average_parameters(client_parameters, sample_counts, self.device)
        next_params, updates, scores = apply_layerwise_update_recycling(
            averaged_params=averaged_params,
            prev_params=self.prev_params,
            prev_updates=self.prev_updates,
            recycling_indices=self.recycling_layers,
        )

        with torch.no_grad():
            for global_param, new_param in zip(self.task.model.parameters(), next_params):
                global_param.data.copy_(new_param)

        self.prev_params = [param.detach().clone().to(self.device) for param in self.task.model.parameters()]
        self.prev_updates = [update.detach().clone().to(self.device) for update in updates]
        self.scores.update(scores)
        self.recycling_layers = self._sample_next_recycling_layers(current_params)

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "fedluar_recycling_layers": sorted(self.recycling_layers),
        }

    def _sample_next_recycling_layers(self, current_params):
        kernel_indices = kernel_parameter_indices(current_params)
        if config["fedluar_recycling_layers"] is None:
            num_recycling = num_recycling_layers_from_ratio(
                len(kernel_indices),
                config["fedluar_recycling_ratio"],
            )
        else:
            num_recycling = int(config["fedluar_recycling_layers"])
        if self.message_pool.get("round", 0) < 0:
            return set()
        return select_recycling_layers(
            scores=self.scores,
            kernel_indices=kernel_indices,
            num_recycling_layers=num_recycling,
            rng=self.rng,
        )
