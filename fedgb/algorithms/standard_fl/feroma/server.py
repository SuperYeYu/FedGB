import numpy as np
import torch
from fedgb.training.base import BaseServer
from fedgb.algorithms.standard_fl.feroma.feroma_config import config
from fedgb.algorithms.standard_fl.feroma.utils import (
    GroupwiseMinMaxScaler,
    personalized_weighted_aggregate,
    profile_distance_weights,
    weighted_average,
)


class FEROMAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FEROMAServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._per_client_weight = {}
        self.parent_descriptors = None
        self.descriptor_scaler = GroupwiseMinMaxScaler()
        self.client_descriptors = {}
        self.distance_weights = None

    def execute(self):
        sampled_clients = self.message_pool["sampled_clients"]
        client_params = [
            self.message_pool[f"client_{client_id}"]["weight"]
            for client_id in sampled_clients
        ]
        sample_counts = [
            self.message_pool[f"client_{client_id}"]["num_samples"]
            for client_id in sampled_clients
        ]
        descriptors = np.vstack(
            [
                self.message_pool[f"client_{client_id}"]["descriptor"].detach().cpu().numpy()
                for client_id in sampled_clients
            ]
        )
        scaled_descriptors = self.descriptor_scaler.scale(descriptors)
        for row, client_id in zip(scaled_descriptors, sampled_clients):
            self.client_descriptors[client_id] = row

        if self.message_pool["round"] < config["feroma_warmup_rounds"]:
            global_params = weighted_average(client_params, sample_counts, self.device)
            self._load_params(global_params)
            self._per_client_weight = {}
            self.parent_descriptors = scaled_descriptors
            self.distance_weights = None
            return

        self.distance_weights = profile_distance_weights(
            current_descriptors=scaled_descriptors,
            parent_descriptors=self.parent_descriptors,
            distance=config["feroma_distance"],
        )
        personalized_params = personalized_weighted_aggregate(
            client_params=client_params,
            sample_counts=sample_counts,
            distance_weights=self.distance_weights,
            device=self.device,
        )
        self._per_client_weight = {
            client_id: params
            for client_id, params in zip(sampled_clients, personalized_params)
        }
        global_params = weighted_average(client_params, sample_counts, self.device)
        self._load_params(global_params)
        self.parent_descriptors = scaled_descriptors

    def _load_params(self, params):
        with torch.no_grad():
            for global_param, new_param in zip(self.task.model.parameters(), params):
                global_param.data.copy_(new_param.to(self.device))

    def send_message(self):
        if self._per_client_weight:
            for client_id, params in self._per_client_weight.items():
                self.message_pool[f"server_{client_id}"] = {"weight": params}
            self.message_pool["server"] = {
                "weight": list(self.task.model.parameters()),
                "feroma_personalized": True,
            }
        else:
            self.message_pool["server"] = {
                "weight": list(self.task.model.parameters()),
                "feroma_personalized": False,
            }
