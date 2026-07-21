import torch

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.fediih.fediih_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fediih.models import (
    FedIIHDisentangledGNN,
    FedIIHHVAE,
)
from fedgb.algorithms.subgraph_fgl.homogeneous.fediih.utils import (
    summarize_hvae_distribution,
)


class FedIIHClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedIIHClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.latent_dim = getattr(args, "fediih_latent_dim", config["n_latentdims"])
        self.num_factors = getattr(args, "fediih_num_factors", config["num_factors"])
        self._apply_official_defaults()
        self.task.load_custom_model(
            FedIIHDisentangledGNN(
                input_dim=self.task.num_feats,
                output_dim=self.task.num_global_classes,
                args=self._model_args(),
            )
        )
        self.vae = FedIIHHVAE(self.latent_dim, self._model_args()).to(device)
        self.optimizer_vae = torch.optim.Adam(
            self.vae.parameters(),
            lr=config["hvae_lr"],
            weight_decay=config["hvae_weight_decay"],
        )
        self.alpha_mu = torch.randn(self.latent_dim, device=device)
        self.beta_mu = torch.randn(self.latent_dim, device=device)
        self.semantic_mu = torch.zeros(self.latent_dim)
        self.semantic_logvar = torch.zeros(self.latent_dim)
        self.structure_mu = torch.zeros(self.latent_dim)
        self.structure_logvar = torch.zeros(self.latent_dim)

    def _apply_official_defaults(self):
        self.args.n_factors = self.num_factors
        self.args.n_latentdims = self.latent_dim
        self.args.n_layers = getattr(self.args, "fediih_n_layers", config["n_layers"])
        self.args.n_routit = getattr(self.args, "fediih_n_routit", config["n_routit"])
        self.args.dropout = getattr(self.args, "fediih_dropout", config["dropout"])
        self.args.edge_chunk_size = getattr(self.args, "fediih_edge_chunk_size", config["edge_chunk_size"])

    def _model_args(self):
        return self.args

    def execute(self):
        server_msg = self.message_pool.get("server", {})
        state_dict = server_msg.get(f"personalized_{self.client_id}", server_msg.get("global_state_dict"))
        if state_dict is not None:
            self.task.model.load_state_dict(
                {key: value.to(self.device) for key, value in state_dict.items()},
                strict=False,
            )
        elif "weight" in server_msg:
            with torch.no_grad():
                for local_param, global_param in zip(self.task.model.parameters(), server_msg["weight"]):
                    local_param.data.copy_(global_param.to(self.device))

        if "Alpha_mu" in server_msg:
            self.alpha_mu = server_msg["Alpha_mu"].to(self.device)
        if "Beta_mu" in server_msg:
            self.beta_mu = server_msg["Beta_mu"].to(self.device)

        self.task.train()
        self._train_hvae()
        self._collect_latent_summaries()

    def _train_hvae(self):
        data = self.task.processed_data["data"]
        for _ in range(config["hvae_epochs"]):
            self.task.model.eval()
            self.vae.train()
            self.optimizer_vae.zero_grad()
            with torch.no_grad():
                z_mu_n, z_logvar_n, z_mu_e, z_logvar_e = self.task.model.encode_for_hvae(data)
            z_n, z_e = self.vae.sampling(z_mu_n, z_logvar_n, z_mu_e, z_logvar_e)
            if data.num_nodes > config.get("hvae_dense_node_limit", 12000):
                edge_logits, edge_labels = self.vae.decode_edges(
                    z_n,
                    z_e,
                    data,
                    negative_ratio=config["hvae_negative_ratio"],
                    max_edges=config["hvae_max_edges"],
                )
            else:
                edge_logits = self.vae.decode(z_n, z_e)
                edge_labels = None
            loss = self.vae.loss_function(
                data,
                z_mu_n,
                z_logvar_n,
                z_mu_e,
                z_logvar_e,
                self.alpha_mu,
                self.beta_mu,
                edge_logits,
                edge_labels,
            )
            loss.backward()
            self.optimizer_vae.step()

    def _collect_latent_summaries(self):
        self.task.model.eval()
        with torch.no_grad():
            z_mu_n, z_logvar_n, z_mu_e, z_logvar_e = self.task.model.encode_for_hvae(self.task.data)
        self.semantic_mu, self.semantic_logvar = summarize_hvae_distribution(z_mu_n, z_logvar_n)
        self.structure_mu, self.structure_logvar = summarize_hvae_distribution(z_mu_e, z_logvar_e)

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "state_dict": {
                key: value.detach().clone().cpu()
                for key, value in self.task.model.state_dict().items()
            },
            "semantic_mu": self.semantic_mu,
            "semantic_logvar": self.semantic_logvar,
            "structure_mu": self.structure_mu,
            "structure_logvar": self.structure_logvar,
        }
