import torch

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedhgn.fedhgn_config import config
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedhgn.models import PyGFedHGNModel, infer_fedhgn_metadata
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedhgn.utils import (
    basis_alignment_regularization,
    filter_private_state_dict,
    stack_basis_coefficients,
)


class FedHGNClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedHGNClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._load_fedhgn_model()
        self.ablation = getattr(args, "fedhgn_ablation", config["fedhgn_ablation"])
        self.align_reg = getattr(args, "fedhgn_align_reg", config["fedhgn_align_reg"])
        self.others_basis_coeffs_encoder = None

    def execute(self):
        server_msg = self.message_pool.get("server", {})
        state_dict = server_msg.get("state_dict")
        if state_dict is not None:
            self.task.model.load_state_dict(
                {key: value.to(self.device) for key, value in state_dict.items()},
                strict=False,
            )
        else:
            weights = server_msg.get("weight")
            if weights is not None:
                with torch.no_grad():
                    for local_param, global_param in zip(self.task.model.parameters(), weights):
                        local_param.data.copy_(global_param.to(self.device))

        self._set_others_basis_coeffs(server_msg)
        self._train_fedhgn()

    def send_message(self):
        private_state = filter_private_state_dict(
            self.task.model.state_dict(),
            ablation=self.ablation,
            is_encoder=True,
        )
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "state_dict": private_state,
            "basis_coeffs_encoder": self._basis_coeffs_encoder(),
            "basis_coeffs_decoder": self._basis_coeffs_decoder(),
        }

    def _basis_coeffs_encoder(self):
        encoder_coeffs = getattr(self.task.model, "basis_coeffs_encoder", None)
        return stack_basis_coefficients(encoder_coeffs)

    def _basis_coeffs_decoder(self):
        decoder_coeffs = getattr(self.task.model, "basis_coeffs_decoder", None)
        return stack_basis_coefficients(decoder_coeffs)

    def _load_fedhgn_model(self):
        num_node_types, num_edge_types = infer_fedhgn_metadata(self.task.data)
        max_nodes = getattr(self.args, "fedhgn_max_nodes", config["fedhgn_max_nodes"])
        if max_nodes is None:
            max_nodes = int(self.task.data.x.size(0))
        model = PyGFedHGNModel(
            input_dim=self.task.num_feats,
            hidden_dim=getattr(self.args, "fedhgn_hidden_dim", getattr(self.args, "hid_dim", config["fedhgn_hidden_dim"])),
            output_dim=self.task.num_global_classes,
            num_node_types=getattr(self.args, "fedhgn_num_node_types", num_node_types),
            num_edge_types=getattr(self.args, "fedhgn_num_edge_types", num_edge_types),
            num_bases=getattr(self.args, "fedhgn_num_bases", config["fedhgn_num_bases"]),
            num_layers=getattr(self.args, "fedhgn_num_layers", getattr(self.args, "num_layers", config["fedhgn_num_layers"])),
            dropout=getattr(self.args, "fedhgn_dropout", getattr(self.args, "dropout", config["fedhgn_dropout"])),
            max_nodes=max_nodes,
            use_self_loop=getattr(self.args, "fedhgn_use_self_loop", config["fedhgn_use_self_loop"]),
        )
        self.task.load_custom_model(model)

    def _set_others_basis_coeffs(self, server_msg):
        self.others_basis_coeffs_encoder = None
        if self.ablation is not None:
            return
        basis_by_client = server_msg.get("basis_coeffs_encoder", {})
        coeffs = [
            value.to(self.device)
            for client_id, value in basis_by_client.items()
            if client_id != self.client_id and value is not None
        ]
        if coeffs:
            self.others_basis_coeffs_encoder = torch.cat(coeffs, dim=1)

    def _train_fedhgn(self):
        splitted_data = self.task.processed_data
        self.task.model.train()
        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            embedding, logits = self.task.model.forward(splitted_data["data"])
            loss_train = self.task.loss_fn(
                embedding,
                logits,
                splitted_data["data"].y,
                splitted_data["train_mask"],
            )
            if self.ablation is None:
                local_coeffs = stack_basis_coefficients(self.task.model.basis_coeffs_encoder)
                align_loss = basis_alignment_regularization(local_coeffs, self.others_basis_coeffs_encoder)
                loss_train = loss_train + self.align_reg * align_loss
            loss_train.backward()
            self.task.optim.step()
