import torch

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.fedlog.fedlog_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedlog.models import (
    FedLoGNeighborGenerator,
    FedLoGTaskAdapter,
)
from fedgb.algorithms.subgraph_fgl.homogeneous.fedlog.utils import (
    build_condensed_graph,
    class_prototype_init,
    clone_state_dict,
    condensed_graph_loss,
    fedlog_metric_logits,
    load_state_dict_to_module,
    parameter_norm_loss,
    synthetic_data_loss,
)
from fedgb.utils.metrics import compute_supervised_metrics


class FedLoGClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedLoGClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.num_proto = config["fedlog_num_proto"]
        self.head_adapter = FedLoGTaskAdapter(
            args.hid_dim,
            args.hid_dim,
            config["fedlog_adapter_layers"],
        ).to(device)
        self.tail_adapter = FedLoGTaskAdapter(
            args.hid_dim,
            args.hid_dim,
            config["fedlog_adapter_layers"],
        ).to(device)
        self.neigh_gen = FedLoGNeighborGenerator(self.task.num_feats).to(device)
        self.syn_feat_head = torch.nn.Parameter(
            class_prototype_init(
                self.task.data,
                self.task.num_global_classes,
                self.num_proto,
                self.task.num_feats,
                self.task.train_mask,
                device,
            )
        )
        self.syn_feat_tail = torch.nn.Parameter(self.syn_feat_head.detach().clone())
        self.syn_optimizer = torch.optim.Adam([self.syn_feat_head, self.syn_feat_tail], lr=args.lr)
        self.aux_optimizer = torch.optim.Adam(
            list(self.head_adapter.parameters())
            + list(self.tail_adapter.parameters())
            + list(self.neigh_gen.parameters()),
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0),
        )
        self.condensed_graph = None
        self._install_metric_evaluator()

    def _install_metric_evaluator(self):
        def evaluate(splitted_data=None, mute=False):
            if splitted_data is None:
                splitted_data = self.task.splitted_data
            data = splitted_data["data"]
            train_mask = splitted_data["train_mask"]
            val_mask = splitted_data["val_mask"]
            test_mask = splitted_data["test_mask"]
            self.task.model.eval()
            self.head_adapter.eval()
            self.tail_adapter.eval()
            with torch.no_grad():
                logits = fedlog_metric_logits(
                    self.task.model,
                    self.head_adapter,
                    self.tail_adapter,
                    data,
                    self.syn_feat_head,
                    self.syn_feat_tail,
                    self.task.num_global_classes,
                    self.num_proto,
                    config["fedlog_head_deg_thres"],
                )
                embedding, model_logits = self.task.model(data)
                round_id = self.message_pool.get("round", 0)
                if round_id < config.get("fedlog_ce_warmup_rounds", 0):
                    metric_weight = 0.0
                else:
                    metric_weight = config.get("fedlog_metric_eval_weight", 1.0)
                if metric_weight < 1.0:
                    model_log_probs = torch.nn.functional.log_softmax(model_logits, dim=-1)
                    metric_probs = logits.exp()
                    model_probs = model_log_probs.exp()
                    mixed_probs = metric_weight * metric_probs + (1.0 - metric_weight) * model_probs
                    logits = torch.log(mixed_probs.clamp_min(1e-12))
                loss_train = torch.nn.functional.nll_loss(logits[train_mask], data.y[train_mask])
                loss_val = torch.nn.functional.nll_loss(logits[val_mask], data.y[val_mask])
                loss_test = torch.nn.functional.nll_loss(logits[test_mask], data.y[test_mask])

            eval_output = {
                "embedding": embedding,
                "logits": logits,
                "loss_train": loss_train,
                "loss_val": loss_val,
                "loss_test": loss_test,
            }
            eval_output.update(
                compute_supervised_metrics(
                    metrics=self.args.metrics,
                    logits=logits[train_mask],
                    labels=data.y[train_mask],
                    suffix="train",
                )
            )
            eval_output.update(
                compute_supervised_metrics(
                    metrics=self.args.metrics,
                    logits=logits[val_mask],
                    labels=data.y[val_mask],
                    suffix="val",
                )
            )
            eval_output.update(
                compute_supervised_metrics(
                    metrics=self.args.metrics,
                    logits=logits[test_mask],
                    labels=data.y[test_mask],
                    suffix="test",
                )
            )
            if not mute:
                info = ""
                for key, val in eval_output.items():
                    try:
                        info += f"\t{key}: {val:.4f}"
                    except Exception:
                        continue
                print(f"[client {self.client_id}]" + info)
            return eval_output

        self.task.override_evaluate = evaluate

    def execute(self):
        server_msg = self.message_pool.get("server", {})
        self._load_server_state(server_msg)
        data = self.task.processed_data["data"]
        train_mask = self.task.processed_data["train_mask"]
        global_syn_data = server_msg.get("global_synthetic_data")

        self.task.model.train()
        self.head_adapter.train()
        self.tail_adapter.train()
        self.neigh_gen.train()
        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            self.syn_optimizer.zero_grad()
            self.aux_optimizer.zero_grad()

            metric_loss = condensed_graph_loss(
                self.task.model,
                self.head_adapter,
                self.tail_adapter,
                data,
                train_mask,
                self.syn_feat_head,
                self.syn_feat_tail,
                self.task.num_global_classes,
                self.num_proto,
                config["fedlog_head_deg_thres"],
            )
            syn_norm_loss = parameter_norm_loss(self.syn_feat_head, self.syn_feat_tail)
            global_loss = synthetic_data_loss(
                self.task.model,
                self.head_adapter,
                self.tail_adapter,
                global_syn_data,
                self.syn_feat_head,
                self.syn_feat_tail,
                self.task.num_global_classes,
                self.num_proto,
            )
            _, model_logits = self.task.model(data)
            ce_loss = torch.nn.functional.cross_entropy(model_logits[train_mask], data.y[train_mask])
            round_id = self.message_pool.get("round", 0)
            if round_id < config.get("fedlog_ce_warmup_rounds", 0):
                loss = ce_loss
            else:
                loss = (
                    config["fedlog_metric_weight"] * metric_loss
                    + config.get("fedlog_ce_weight", 0.0) * ce_loss
                    + config["fedlog_syn_norm_weight"] * syn_norm_loss
                    + config["fedlog_synthetic_weight"] * global_loss
                )
            loss.backward()
            self.task.optim.step()
            self.syn_optimizer.step()
            self.aux_optimizer.step()

        self.condensed_graph = build_condensed_graph(
            self.syn_feat_head.detach(),
            self.syn_feat_tail.detach(),
            self.task.num_global_classes,
            self.num_proto,
            self.task.data.y,
            self.task.train_mask,
        )

    def _load_server_state(self, server_msg):
        weights = server_msg.get("weight")
        if weights is not None:
            with torch.no_grad():
                for local_param, global_param in zip(self.task.model.parameters(), weights):
                    local_param.data.copy_(global_param.to(self.device))
        load_state_dict_to_module(self.head_adapter, server_msg.get("head_adapter_state"), self.device)
        load_state_dict_to_module(self.tail_adapter, server_msg.get("tail_adapter_state"), self.device)
        graph = server_msg.get("client_condensed_graph")
        if graph is not None:
            self.syn_feat_head = torch.nn.Parameter(graph.x_head.to(self.device).detach().clone())
            self.syn_feat_tail = torch.nn.Parameter(graph.x_tail.to(self.device).detach().clone())
            self.syn_optimizer = torch.optim.Adam([self.syn_feat_head, self.syn_feat_tail], lr=self.args.lr)

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "head_adapter_state": clone_state_dict(self.head_adapter),
            "tail_adapter_state": clone_state_dict(self.tail_adapter),
            "neigh_gen_state": clone_state_dict(self.neigh_gen),
            "condensed_graph": self.condensed_graph,
        }
