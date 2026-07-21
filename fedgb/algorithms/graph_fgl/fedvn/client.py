import torch
import torch.nn as nn
import torch.nn.functional as F

from fedgb.training.base import BaseClient
from fedgb.algorithms.graph_fgl.fedvn.fedvn_config import config
from fedgb.algorithms.graph_fgl.fedvn.models import build_fedvn_model, build_fedvn_scoring_model
from fedgb.algorithms.graph_fgl.fedvn.utils import (
    graph_level_score,
    load_shared_parameters,
    score_contrastive_loss,
    shared_parameter_payload,
    virtual_node_decorrelation_loss,
)
from fedgb.tasks.graph_reg import graph_regression_batch_loss


def _cfg(args, key):
    return getattr(args, f"fedvn_{key}", config[f"fedvn_{key}"])


class FedVNClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedVNClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.load_custom_model(build_fedvn_model(self.args, self.task))
        self.vn_embedding = nn.Parameter(torch.zeros(_cfg(self.args, "num_vn"), _cfg(self.args, "hidden_dim"), device=device))
        self.optimizer_vn = torch.optim.SGD([self.vn_embedding], lr=self.args.lr)
        self.scoring_model = build_fedvn_scoring_model(self.args, self.task).to(device)
        self.optimizer_per = torch.optim.SGD(self.scoring_model.parameters(), lr=self.args.lr)
        self.local_score = torch.full((_cfg(self.args, "num_vn"),), 0.5, device=device)
        self._refresh_model_context()

    def _apply_config_defaults(self):
        for key, value in config.items():
            if not hasattr(self.args, key):
                setattr(self.args, key, value)
        self.args.hid_dim = self.args.fedvn_hidden_dim

    def execute(self):
        self._load_server_state()
        self._train_scoring_generator()
        self._train_model_and_virtual_nodes()

    def _load_server_state(self):
        server_message = self.message_pool["server"]
        weight_names = server_message.get("weight_names")
        if weight_names is None:
            weight_names = [name for name, _ in self.task.model.named_parameters()]
        load_shared_parameters(self.task.model, weight_names, server_message["weight"], self.device)
        with torch.no_grad():
            server_vn = server_message.get("virtual_nodes")
            if server_vn is not None:
                self.vn_embedding.data.copy_(server_vn.to(self.device))
        self._refresh_model_context()

    def _train_scoring_generator(self):
        self.scoring_model.train()
        self.task.model.eval()
        global_score = self.message_pool["server"].get("score")
        if global_score is None:
            global_score = torch.full_like(self.local_score, 1.0 / self.local_score.numel())
        temperature = _cfg(self.args, "temperature")
        lam2 = _cfg(self.args, "lambda2")
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.args.num_epochs):
            for batch in self.task.train_dataloader:
                batch = batch.to(self.device)
                score, _ = self.scoring_model(batch)
                score = torch.sigmoid(score)
                embedding, logits = self.task.model(batch, self.vn_embedding, score)
                if getattr(self.args, "task", None) == "graph_reg":
                    loss = graph_regression_batch_loss(self.task, embedding, logits, batch)
                else:
                    loss = criterion(logits, batch.y.squeeze().long())
                graph_score = graph_level_score(score, batch.batch)
                contrastive_loss = lam2 * score_contrastive_loss(
                    graph_score, self.local_score, global_score, temperature
                )
                loss = contrastive_loss if loss is None else loss + contrastive_loss

                self.optimizer_per.zero_grad()
                self.task.optim.zero_grad()
                self.optimizer_vn.zero_grad()
                loss.backward()
                self.optimizer_per.step()
                self.local_score = graph_score.mean(dim=0).detach()

    def _train_model_and_virtual_nodes(self):
        self.task.model.train()
        self.scoring_model.eval()
        lam1 = _cfg(self.args, "lambda1")
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.args.num_epochs):
            for batch in self.task.train_dataloader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    score, _ = self.scoring_model(batch)
                    score = torch.sigmoid(score)
                embedding, logits = self.task.model(batch, self.vn_embedding, score)
                if getattr(self.args, "task", None) == "graph_reg":
                    supervised_loss = graph_regression_batch_loss(
                        self.task, embedding, logits, batch
                    )
                else:
                    supervised_loss = criterion(logits, batch.y.squeeze().long())
                decorrelation_loss = lam1 * virtual_node_decorrelation_loss(self.vn_embedding)
                loss = (
                    decorrelation_loss
                    if supervised_loss is None
                    else supervised_loss + decorrelation_loss
                )

                self.optimizer_per.zero_grad()
                self.task.optim.zero_grad()
                self.optimizer_vn.zero_grad()
                loss.backward()
                self.task.optim.step()
                self.optimizer_vn.step()
        self._refresh_model_context()

    def _refresh_model_context(self):
        if hasattr(self.task.model, "set_virtual_node_context"):
            self.task.model.set_virtual_node_context(self.vn_embedding, self.scoring_model)

    def send_message(self):
        weight_names, weights = shared_parameter_payload(self.task.model, getattr(self.args, "task", None))
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight_names": weight_names,
            "weight": weights,
            "virtual_nodes": self.vn_embedding.detach().clone(),
            "score": self.local_score.detach().clone(),
        }
