import torch

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedlit.fedlit_config import config
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedlit.models import PyGFedLITModel
from fedgb.algorithms.subgraph_fgl.heterogeneous.fedlit.utils import (
    assign_or_update_edge_clusters,
    fedlit_forward_with_centroids,
    pyg_subgraphs_by_linktype,
)


class FedLITClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedLITClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._load_fedlit_model()
        self.centroids = None
        self.clusters = None
        self.cluster_train_size = None
        self.task.override_evaluate = self._evaluate_fedlit

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

        self._train_fedlit()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "total_train_size": self._total_train_size(),
            "weight": list(self.task.model.parameters()),
            "state_dict": {
                key: value.detach().clone().cpu()
                for key, value in self.task.model.state_dict().items()
            },
            "centroids": None if self.centroids is None else self.centroids.detach().clone().cpu(),
            "cluster_train_size": self.cluster_train_size,
        }

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

    def _train_fedlit(self):
        data = self.task.processed_data["data"]
        data.train_mask = self.task.train_mask
        data.val_mask = self.task.val_mask
        data.test_mask = self.task.test_mask
        nlinktype = getattr(self.args, "fedlit_nlinktype", config["fedlit_nlinktype"])
        num_iter = getattr(self.args, "fedlit_num_iter_em", config["fedlit_num_iter_em"])
        self.task.model.train()
        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            node_embeddings = self.task.model.feature_projection(data.x)
            self.centroids, self.clusters = assign_or_update_edge_clusters(
                node_embeddings=node_embeddings.detach(),
                edge_index=data.edge_index,
                nlinktype=nlinktype,
                centroids=self.centroids,
                num_iter=num_iter,
            )
            subgraphs = pyg_subgraphs_by_linktype(data, self.clusters, nlinktype)
            self.cluster_train_size = []
            for subg in subgraphs:
                subg.x = node_embeddings[subg.orig_node_ids].clone()
                subg = subg.to(self.device)
                self.cluster_train_size.append(
                    int(subg.train_mask.sum().item()) if hasattr(subg, "train_mask") else 0
                )
            data.fedlit_clusters = self.clusters.detach()
            data.fedlit_subgraphs = subgraphs
            branch_embedding = self.task.model.split_forward(subgraphs, data.x.size(0), self.device)
            logits = self.task.model.classify(branch_embedding)
            loss = self.task.loss_fn(branch_embedding, logits, data.y, self.task.train_mask)
            loss.backward()
            self.task.optim.step()

    def _evaluate_fedlit(self, splitted_data=None, mute=False):
        eval_data = self.task.splitted_data if splitted_data is None else splitted_data
        data = eval_data["data"]
        nlinktype = getattr(self.args, "fedlit_nlinktype", config["fedlit_nlinktype"])
        num_iter = getattr(self.args, "fedlit_num_iter_em", config["fedlit_num_iter_em"])
        self.task.model.eval()
        with torch.no_grad():
            embedding, logits, next_centroids, clusters, subgraphs = fedlit_forward_with_centroids(
                model=self.task.model,
                data=data,
                centroids=self.centroids,
                nlinktype=nlinktype,
                num_iter=num_iter,
                device=self.device,
            )
            self.centroids = next_centroids.detach()
            self.clusters = clusters.detach()
            data.fedlit_clusters = self.clusters
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
            print(f"[client {self.client_id}]" + info)
        return eval_output

    def _total_train_size(self):
        mask = getattr(self.task, "train_mask", None)
        if mask is None:
            return self.task.num_samples
        return int(mask.sum().item())
