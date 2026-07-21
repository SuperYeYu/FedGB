from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedda.fedda_config import config
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedda.models import PyGFedDAModel, infer_fedda_metadata


class FedDAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedDAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._load_fedda_model()
        self._active = True

    def execute(self):
        server_msg = self.message_pool.get("server", {})
        state_dict = server_msg.get("state_dict")
        if state_dict is not None:
            self.task.model.load_state_dict(
                {name: value.to(self.device) for name, value in state_dict.items()},
                strict=False,
            )

        active_clients = server_msg.get("fedda_active_clients")
        self._active = active_clients is None or self.client_id in active_clients
        if self._active:
            self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "active": self._active,
            "num_samples": self.task.num_samples,
            "state_dict": {
                name: value.detach().clone().cpu()
                for name, value in self.task.model.state_dict().items()
            },
            "weight": list(self.task.model.parameters()),
        }

    def _load_fedda_model(self):
        num_node_types, num_edge_types = infer_fedda_metadata(self.task.data)
        model = PyGFedDAModel(
            input_dim=self.task.num_feats,
            hid_dim=getattr(self.args, "hid_dim", 64),
            output_dim=self.task.num_global_classes,
            num_layers=getattr(self.args, "num_layers", 2),
            num_heads=getattr(self.args, "fedda_num_heads", config["num_heads"]),
            num_node_types=getattr(self.args, "fedda_num_node_types", num_node_types),
            num_edge_types=getattr(self.args, "fedda_num_edge_types", num_edge_types),
            edge_dim=getattr(self.args, "fedda_edge_dim", config["edge_dim"]),
            dropout=getattr(self.args, "dropout", 0.5),
            attn_dropout=getattr(self.args, "fedda_attn_dropout", config["attn_dropout"]),
            negative_slope=getattr(self.args, "fedda_negative_slope", config["negative_slope"]),
            residual=getattr(self.args, "fedda_residual", config["residual"]),
            residual_attention_alpha=getattr(
                self.args,
                "fedda_residual_attention_alpha",
                config["residual_attention_alpha"],
            ),
        )
        self.task.load_custom_model(model)
