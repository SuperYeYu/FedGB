from fedgb.training.base import BaseServer
from fedgb.algorithms.standard_fl.fedlaw.fedlaw_config import config
from fedgb.algorithms.standard_fl.fedlaw.utils import (
    load_vector_to_model,
    model_parameter_vector,
    optimize_fedlaw_weights,
)


class FedLAWServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedLAWServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.last_gamma = None
        self.last_aggregation_weights = None

    def execute(self):
        sampled_clients = self.message_pool["sampled_clients"]
        num_tot_samples = sum(
            self.message_pool[f"client_{client_id}"]["num_samples"]
            for client_id in sampled_clients
        )
        size_weights = [
            self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
            for client_id in sampled_clients
        ]
        client_vectors = [
            self._client_parameter_vector(client_id)
            for client_id in sampled_clients
        ]
        gamma, aggregation_weights, aggregate_vector = optimize_fedlaw_weights(
            task=self.task,
            model=self.task.model,
            client_vectors=client_vectors,
            size_weights=size_weights,
            config=config,
            device=self.device,
        )
        load_vector_to_model(self.task.model, aggregate_vector)
        self.last_gamma = float(gamma.cpu())
        self.last_aggregation_weights = aggregation_weights.cpu().tolist()
        self.message_pool["server"].update(
            {
                "fedlaw_gamma": self.last_gamma,
                "fedlaw_aggregation_weights": self.last_aggregation_weights,
            }
        )

    def _client_parameter_vector(self, client_id):
        weights = self.message_pool[f"client_{client_id}"]["weight"]
        offset = 0
        vector = model_parameter_vector(self.task.model).to(self.device)
        for source_param, target_param in zip(weights, self.task.model.parameters()):
            numel = target_param.numel()
            vector[offset: offset + numel].copy_(source_param.detach().to(self.device).reshape(-1))
            offset += numel
        return vector

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
        }
