from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedda.fedda_config import config
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedda.models import PyGFedDAModel, infer_fedda_metadata
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedda.utils import (
    dynamic_parameter_aggregate,
    select_dynamic_param_names,
    update_client_activity,
)


class FedDAServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedDAServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._load_fedda_model()
        self.aggregate_all = getattr(args, "fedda_aggregate_all", config["aggregate_all"])
        self.partially_return = getattr(args, "fedda_partially_return", config["partially_return"])
        self.remove_client = getattr(args, "fedda_remove_client", config["remove_client"])
        self.explore = getattr(args, "fedda_explore", config["explore"])
        self.active_rate = getattr(args, "fedda_active_rate", getattr(args, "client_frac", config["active_rate"]))
        self.client_threshold = getattr(args, "fedda_client_threshold", config["client_threshold"])
        self.round_threshold = getattr(args, "fedda_round_threshold", config["round_threshold"])
        self.dynamic_keywords = getattr(args, "fedda_dynamic_keywords", config["dynamic_keywords"])

        self.global_state_dict = {
            name: value.detach().clone().cpu()
            for name, value in self.task.model.state_dict().items()
        }
        self.dynamic_param_names = select_dynamic_param_names(self.global_state_dict, self.dynamic_keywords)
        self.returned_param_names = {
            client_id: set(self.dynamic_param_names)
            for client_id in range(getattr(args, "num_clients", 0))
        }
        self.active_clients = list(range(getattr(args, "num_clients", 0)))
        self.removed_clients = []

    def execute(self):
        sampled = list(self.message_pool["sampled_clients"])
        active_sampled = [
            client_id
            for client_id in sampled
            if client_id in self.active_clients
            and self.message_pool.get(f"client_{client_id}", {}).get("active", True)
        ]
        if not active_sampled:
            active_sampled = sampled

        client_states = [
            self.message_pool[f"client_{client_id}"]["state_dict"]
            for client_id in active_sampled
        ]
        returned = self.returned_param_names if self.partially_return else {
            client_id: set(self.dynamic_param_names)
            for client_id in active_sampled
        }
        self.global_state_dict, self.returned_param_names, stats = dynamic_parameter_aggregate(
            self.global_state_dict,
            client_states,
            active_sampled,
            returned,
            dynamic_names=self.dynamic_param_names,
            aggregate_all=self.aggregate_all,
        )
        self.message_pool["fedda_stats"] = stats

        if len(self.dynamic_param_names) > 0:
            updated_ratio = stats["updated_params"] / len(self.dynamic_param_names)
        else:
            updated_ratio = 1.0
        if updated_ratio < self.round_threshold:
            self.active_clients = list(range(getattr(self.args, "num_clients", 0)))
            self.removed_clients = []
            self.returned_param_names = {
                client_id: set(self.dynamic_param_names)
                for client_id in self.active_clients
            }
        else:
            self.active_clients, self.removed_clients, self.returned_param_names = update_client_activity(
                client_ids=range(getattr(self.args, "num_clients", 0)),
                current_active=self.active_clients,
                removed_clients=self.removed_clients,
                returned_param_names=self.returned_param_names,
                all_param_names=self.dynamic_param_names,
                remove_client=self.remove_client,
                explore=self.explore,
                active_rate=self.active_rate,
                client_threshold=self.client_threshold,
            )

        self.task.model.load_state_dict(
            {name: value.to(self.device) for name, value in self.global_state_dict.items()},
            strict=False,
        )

    def send_message(self):
        self.message_pool["server"] = {
            "state_dict": {
                name: value.detach().clone().to(self.device)
                for name, value in self.global_state_dict.items()
            },
            "weight": list(self.task.model.parameters()),
            "fedda_active_clients": list(self.active_clients),
            "fedda_returned_param_names": {
                client_id: sorted(names)
                for client_id, names in self.returned_param_names.items()
            },
        }

    def _load_fedda_model(self):
        num_node_types, num_edge_types = infer_fedda_metadata(self.task.data)
        num_node_types = getattr(self.args, "fedda_num_node_types", num_node_types)
        num_edge_types = getattr(self.args, "fedda_num_edge_types", num_edge_types)
        self.args.fedda_num_node_types = num_node_types
        self.args.fedda_num_edge_types = num_edge_types
        model = PyGFedDAModel(
            input_dim=self.task.num_feats,
            hid_dim=getattr(self.args, "hid_dim", 64),
            output_dim=self.task.num_global_classes,
            num_layers=getattr(self.args, "num_layers", 2),
            num_heads=getattr(self.args, "fedda_num_heads", config["num_heads"]),
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
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
