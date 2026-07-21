import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedhgn.fedhgn_config import config
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedhgn.models import PyGFedHGNModel, infer_fedhgn_metadata
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedhgn.utils import (
    filter_private_state_dict,
    weighted_average_state_dicts,
)


class FedHGNServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedHGNServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._load_fedhgn_model()
        self.ablation = getattr(args, "fedhgn_ablation", config["fedhgn_ablation"])
        self.global_state_dict = filter_private_state_dict(
            self.task.model.state_dict(),
            ablation=self.ablation,
            is_encoder=True,
        )
        self.all_clients_basis_coeffs_encoder = {}
        self.all_clients_basis_coeffs_decoder = {}

    def execute(self):
        sampled = self.message_pool["sampled_clients"]
        state_dicts = []
        weights = []
        for client_id in sampled:
            msg = self.message_pool[f"client_{client_id}"]
            state_dict = msg.get("state_dict")
            if state_dict is None:
                continue
            state_dicts.append(state_dict)
            weights.append(msg["num_samples"])
            if msg.get("basis_coeffs_encoder") is not None:
                self.all_clients_basis_coeffs_encoder[client_id] = msg["basis_coeffs_encoder"].detach().cpu()
            if msg.get("basis_coeffs_decoder") is not None:
                self.all_clients_basis_coeffs_decoder[client_id] = msg["basis_coeffs_decoder"].detach().cpu()

        if not state_dicts:
            return

        averaged = weighted_average_state_dicts(state_dicts, weights)
        self.global_state_dict.update(averaged)
        self.task.model.load_state_dict(
            {key: value.to(self.device) for key, value in self.global_state_dict.items()},
            strict=False,
        )

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "state_dict": {
                key: value.detach().clone().to(self.device)
                for key, value in self.global_state_dict.items()
            },
            "basis_coeffs_encoder": self.all_clients_basis_coeffs_encoder,
            "basis_coeffs_decoder": self.all_clients_basis_coeffs_decoder,
        }

    def _load_fedhgn_model(self):
        num_node_types, num_edge_types = infer_fedhgn_metadata(self.task.data)
        num_node_types = getattr(self.args, "fedhgn_num_node_types", num_node_types)
        num_edge_types = getattr(self.args, "fedhgn_num_edge_types", num_edge_types)
        self.args.fedhgn_num_node_types = num_node_types
        self.args.fedhgn_num_edge_types = num_edge_types
        max_nodes = getattr(self.args, "fedhgn_max_nodes", config["fedhgn_max_nodes"])
        if max_nodes is None:
            max_nodes = int(self.task.data.x.size(0))
        self.args.fedhgn_max_nodes = max_nodes
        model = PyGFedHGNModel(
            input_dim=self.task.num_feats,
            hidden_dim=getattr(self.args, "fedhgn_hidden_dim", getattr(self.args, "hid_dim", config["fedhgn_hidden_dim"])),
            output_dim=self.task.num_global_classes,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            num_bases=getattr(self.args, "fedhgn_num_bases", config["fedhgn_num_bases"]),
            num_layers=getattr(self.args, "fedhgn_num_layers", getattr(self.args, "num_layers", config["fedhgn_num_layers"])),
            dropout=getattr(self.args, "fedhgn_dropout", getattr(self.args, "dropout", config["fedhgn_dropout"])),
            max_nodes=max_nodes,
            use_self_loop=getattr(self.args, "fedhgn_use_self_loop", config["fedhgn_use_self_loop"]),
        )
        self.task.load_custom_model(model)
