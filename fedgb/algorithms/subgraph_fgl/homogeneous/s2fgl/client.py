import copy

import torch
import torch.nn.functional as F

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.s2fgl.models import build_s2fgl_model
from fedgb.algorithms.subgraph_fgl.homogeneous.s2fgl.s2fgl_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.s2fgl.utils import (
    build_s2fgl_adjs,
    federated_knowledge_distillation_loss,
    frequency_alignment_loss,
    select_important_nodes_lis,
    select_important_nodes_lis_k,
)


class S2FGLClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(S2FGLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.load_custom_model(build_s2fgl_model(args, self.task))
        self.global_model = copy.deepcopy(self.task.model).to(device)
        build_s2fgl_adjs(self.task.data)
        build_s2fgl_adjs(self.task.processed_data["data"])
        self.important_nodes, self.important_labels = select_important_nodes_lis(
            self.task.data,
            train_mask=self.task.train_mask,
            alpha=config["s2fgl_ppr_alpha"],
            ratio=config["s2fgl_important_ratio"],
            device=device,
            num_iter=config["s2fgl_ppr_iters"],
            max_nodes=config["s2fgl_max_important_nodes"],
        )
        self.important_nodes_k = select_important_nodes_lis_k(
            self.task.data,
            train_mask=self.task.train_mask,
            alpha=config["s2fgl_ppr_alpha"],
            ratio=config["s2fgl_k_important_ratio"],
            device=device,
            num_iter=config["s2fgl_ppr_iters"],
            max_nodes=config["s2fgl_max_k_important_nodes"],
        )
        self.node_features = None

    def execute(self):
        server_msg = self.message_pool["server"]
        self._sync_model(server_msg["weight"])
        codebook = server_msg.get("codebook")
        if codebook is not None:
            codebook = codebook.to(self.device)

        self.global_model.eval()
        data = self.task.processed_data["data"]
        train_mask = self.task.processed_data["train_mask"]
        labels = data.y

        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            embedding, logits = self.task.model(data)
            ce_loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            with torch.no_grad():
                global_embedding, _ = self.global_model(data)

            loss = ce_loss
            if codebook is not None and codebook.numel() > 0:
                nlir_loss = federated_knowledge_distillation_loss(
                    embedding,
                    global_embedding,
                    codebook,
                    temperature=config["s2fgl_temperature"],
                    lamb=config["s2fgl_kd_lamb"],
                )
                loss = loss + config["s2fgl_kd_weight"] * nlir_loss

                selected = self.important_nodes_k
                if selected.numel() > 1:
                    fgma_loss = frequency_alignment_loss(
                        embedding[selected],
                        global_embedding[selected],
                        top_k=config["s2fgl_fgma_top_k"],
                        similarity_top_k=config["s2fgl_fgma_similarity_top_k"],
                    )
                    loss = loss + config["s2fgl_fgma_weight"] * fgma_loss

            self._backward_step_if_finite(loss)

        with torch.no_grad():
            self.node_features, _ = self.task.model(self.task.data)
            self.node_features = torch.nan_to_num(self.node_features)

    def _backward_step_if_finite(self, loss):
        if not torch.isfinite(loss):
            return
        old_params = [param.detach().clone() for param in self.task.model.parameters()]
        loss.backward()
        grads_are_finite = all(
            param.grad is None or torch.isfinite(param.grad).all()
            for param in self.task.model.parameters()
        )
        if not grads_are_finite:
            self.task.optim.zero_grad()
            return
        torch.nn.utils.clip_grad_norm_(self.task.model.parameters(), max_norm=5.0)
        self.task.optim.step()
        params_are_finite = all(torch.isfinite(param).all() for param in self.task.model.parameters())
        if not params_are_finite:
            with torch.no_grad():
                for param, old_param in zip(self.task.model.parameters(), old_params):
                    param.copy_(old_param)

    def _sync_model(self, weights):
        with torch.no_grad():
            for local_param, global_param in zip(self.task.model.parameters(), weights):
                local_param.data.copy_(global_param.to(self.device))
            for teacher_param, global_param in zip(self.global_model.parameters(), weights):
                teacher_param.data.copy_(global_param.to(self.device))

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "node_features": self.node_features.detach().clone(),
            "important_nodes": self.important_nodes.detach().clone(),
            "important_labels": self.important_labels.detach().clone(),
        }
