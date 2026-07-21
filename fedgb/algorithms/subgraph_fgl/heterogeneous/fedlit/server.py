import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedlit.fedlit_config import config
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedlit.models import PyGFedLITModel
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedlit.utils import (
    aggregate_fedlit_state,
    fedlit_forward_with_centroids,
    group_centroids,
    is_fedlit_state,
)


class FedLITServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(FedLITServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self._load_fedlit_model()
        self.nlinktype = getattr(args, "fedlit_nlinktype", config["fedlit_nlinktype"])
        self.groups = None
        self.centers = None
        self.global_state_dict = {
            key: value.detach().clone()
            for key, value in self.task.model.state_dict().items()
        }
        self.task.override_evaluate = self._evaluate_fedlit

    def _load_fedlit_model(self):
        nlinktype = getattr(self.args, "fedlit_nlinktype", config["fedlit_nlinktype"])
        model = PyGFedLITModel(
            nlinktype=nlinktype,
            input_dim=self.task.num_feats,
            hid_dim=getattr(self.args, "hid_dim", config["fedlit_hidden_dim"]),
            output_dim=self.task.num_global_classes,
            num_layers=getattr(self.args, "num_layers", 2),
            dropout=getattr(self.args, "dropout", config["fedlit_dropout"]),
        )
        self.task.load_custom_model(model)

    def execute(self):
        sampled = self.message_pool["sampled_clients"]
        payloads = {}
        centroids = {}
        for client_id in sampled:
            msg = self.message_pool[f"client_{client_id}"]
            payloads[client_id] = {
                "state_dict": msg.get("state_dict"),
                "total_train_size": msg.get("total_train_size", msg["num_samples"]),
                "cluster_train_size": msg.get("cluster_train_size", [1] * self.nlinktype),
            }
            if msg.get("centroids") is not None:
                centroids[client_id] = msg["centroids"].to(self.device)

        if centroids and is_fedlit_state(self.global_state_dict):
            self.groups, self.centers = group_centroids(centroids, self.nlinktype, self.centers)
            self.global_state_dict = aggregate_fedlit_state(
                self.global_state_dict,
                payloads,
                self.groups,
                self.nlinktype,
            )
            self.task.model.load_state_dict(
                {key: value.to(self.device) for key, value in self.global_state_dict.items()},
                strict=False,
            )
        else:
            self._fallback_fedavg(sampled)

    def _fallback_fedavg(self, sampled):
        with torch.no_grad():
            num_total = sum(self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in sampled)
            for idx, client_id in enumerate(sampled):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_total
                for global_param, local_param in zip(
                    self.task.model.parameters(),
                    self.message_pool[f"client_{client_id}"]["weight"],
                ):
                    if idx == 0:
                        global_param.data.copy_(weight * local_param.to(self.device))
                    else:
                        global_param.data += weight * local_param.to(self.device)
        self.global_state_dict = {
            key: value.detach().clone()
            for key, value in self.task.model.state_dict().items()
        }

    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "state_dict": {
                key: value.detach().clone().to(self.device)
                for key, value in self.global_state_dict.items()
            },
            "groups": self.groups,
            "centers": self.centers,
        }

    def _evaluate_fedlit(self, splitted_data=None, mute=False):
        eval_data = self.task.splitted_data if splitted_data is None else splitted_data
        data = eval_data["data"]
        nlinktype = getattr(self.args, "fedlit_nlinktype", config["fedlit_nlinktype"])
        num_iter = getattr(self.args, "fedlit_num_iter_em", config["fedlit_num_iter_em"])
        centroids = self.centers
        self.task.model.eval()
        with torch.no_grad():
            embedding, logits, next_centroids, clusters, subgraphs = fedlit_forward_with_centroids(
                model=self.task.model,
                data=data,
                centroids=centroids,
                nlinktype=nlinktype,
                num_iter=num_iter,
                device=self.device,
            )
            if self.centers is None:
                self.centers = next_centroids.detach()
            data.fedlit_clusters = clusters.detach()
            data.fedlit_subgraphs = subgraphs
            loss_train = self.task.loss_fn(embedding, logits, data.y, eval_data["train_mask"])
            loss_val = self.task.loss_fn(embedding, logits, data.y, eval_data["val_mask"])
            loss_test = self.task.loss_fn(embedding, logits, data.y, eval_data["test_mask"])

        from fedgb.utils.metrics import compute_supervised_metrics

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
                logits=logits[eval_data["train_mask"]],
                labels=data.y[eval_data["train_mask"]],
                suffix="train",
            )
        )
        eval_output.update(
            compute_supervised_metrics(
                metrics=self.args.metrics,
                logits=logits[eval_data["val_mask"]],
                labels=data.y[eval_data["val_mask"]],
                suffix="val",
            )
        )
        eval_output.update(
            compute_supervised_metrics(
                metrics=self.args.metrics,
                logits=logits[eval_data["test_mask"]],
                labels=data.y[eval_data["test_mask"]],
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
            print("[server]" + info)
        return eval_output
