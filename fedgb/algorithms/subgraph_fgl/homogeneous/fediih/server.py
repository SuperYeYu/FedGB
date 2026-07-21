import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.fediih.fediih_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fediih.models import FedIIHDisentangledGNN
from fedgb.algorithms.subgraph_fgl.homogeneous.fediih.utils import (
    aggregate_global_priors,
    aggregate_personalized_states,
    js_similarity_matrix,
    weighted_average_state,
)


class FedIIHServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedIIHServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._apply_official_defaults()
        self.task.load_custom_model(
            FedIIHDisentangledGNN(
                input_dim=self.task.num_feats,
                output_dim=self.task.num_global_classes,
                args=args,
            )
        )
        self.norm = getattr(args, "fediih_similarity_norm", config["similarity_norm"])
        self.norm_scale = getattr(args, "fediih_norm_scale", config["norm_scale"])
        self.num_factors = getattr(args, "fediih_num_factors", config["num_factors"])
        self.global_state_dict = {
            key: value.detach().clone().cpu()
            for key, value in self.task.model.state_dict().items()
        }
        self.personalized_states = {}
        self.alpha_mu = torch.zeros(getattr(args, "fediih_latent_dim", config["n_latentdims"]))
        self.beta_mu = torch.zeros(getattr(args, "fediih_latent_dim", config["n_latentdims"]))

    def _apply_official_defaults(self):
        self.args.n_factors = getattr(self.args, "fediih_num_factors", config["num_factors"])
        self.args.n_latentdims = getattr(self.args, "fediih_latent_dim", config["n_latentdims"])
        self.args.n_layers = getattr(self.args, "fediih_n_layers", config["n_layers"])
        self.args.n_routit = getattr(self.args, "fediih_n_routit", config["n_routit"])
        self.args.dropout = getattr(self.args, "fediih_dropout", config["dropout"])
        self.args.edge_chunk_size = getattr(self.args, "fediih_edge_chunk_size", config["edge_chunk_size"])

    def execute(self):
        sampled = list(self.message_pool["sampled_clients"])
        states = [self.message_pool[f"client_{client_id}"]["state_dict"] for client_id in sampled]
        sizes = torch.tensor(
            [self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in sampled],
            dtype=torch.float32,
        )
        fedavg_weights = sizes / sizes.sum().clamp_min(1.0)
        self.global_state_dict = weighted_average_state(states, fedavg_weights)
        self.task.model.load_state_dict(
            {key: value.to(self.device) for key, value in self.global_state_dict.items()},
            strict=False,
        )

        semantic_mu = torch.stack([self.message_pool[f"client_{client_id}"]["semantic_mu"] for client_id in sampled])
        semantic_logvar = torch.stack([self.message_pool[f"client_{client_id}"]["semantic_logvar"] for client_id in sampled])
        structure_mu = torch.stack([self.message_pool[f"client_{client_id}"]["structure_mu"] for client_id in sampled])
        structure_logvar = torch.stack([self.message_pool[f"client_{client_id}"]["structure_logvar"] for client_id in sampled])
        semantic_sim = js_similarity_matrix(semantic_mu, semantic_logvar, self.norm, self.norm_scale)
        structure_sim = js_similarity_matrix(structure_mu, structure_logvar, self.norm, self.norm_scale)
        self.alpha_mu, self.beta_mu = aggregate_global_priors(semantic_mu, structure_mu)

        self.personalized_states = {}
        for idx, client_id in enumerate(sampled):
            self.personalized_states[client_id] = aggregate_personalized_states(
                states,
                semantic_sim[idx],
                structure_sim[idx],
                num_factors=self.num_factors,
            )

    def send_message(self):
        payload = {
            "weight": list(self.task.model.parameters()),
            "global_state_dict": {
                key: value.detach().clone().to(self.device)
                for key, value in self.global_state_dict.items()
            },
            "Alpha_mu": self.alpha_mu.detach().clone().to(self.device),
            "Beta_mu": self.beta_mu.detach().clone().to(self.device),
        }
        for client_id, state in self.personalized_states.items():
            payload[f"personalized_{client_id}"] = {
                key: value.detach().clone().to(self.device)
                for key, value in state.items()
            }
        self.message_pool["server"] = payload
