import copy

import torch
import torch.nn.functional as F

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.fgssl import augment as A
from fedgb.algorithms.subgraph_fgl.homogeneous.fgssl.fgssl_config import config


def fgssl_semantic_contrastive_loss(
    local_embedding,
    global_embedding,
    labels,
    mask,
    temperature,
):
    """Node semantic contrast: same-class local/global pairs positive, other classes negative."""
    if mask.sum() <= 1:
        return local_embedding.sum() * 0.0

    local_z = F.normalize(local_embedding[mask], dim=-1)
    global_z = F.normalize(global_embedding[mask].detach(), dim=-1)
    labels = labels[mask]

    logits = local_z @ global_z.t() / temperature
    pos_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0))
    pos_mask.fill_diagonal_(True)

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * pos_mask.float()).sum(dim=1).div(pos_mask.sum(dim=1).clamp_min(1)).mean()


def fgssl_structure_distillation_loss(
    local_embedding,
    global_embedding,
    edge_index,
    temperature,
    edge_sample_size=None,
):
    """Graph structure distillation over adjacent edge-wise relation distributions."""
    if edge_index.numel() == 0:
        return local_embedding.sum() * 0.0

    if edge_sample_size is not None and edge_index.size(1) > edge_sample_size:
        perm = torch.randperm(edge_index.size(1), device=edge_index.device)[:edge_sample_size]
        edge_index = edge_index[:, perm]

    src, dst = edge_index.to(local_embedding.device)
    local_relation = torch.abs(local_embedding[src] - local_embedding[dst])
    global_relation = torch.abs(global_embedding.detach()[src] - global_embedding.detach()[dst])

    student_log_prob = F.log_softmax(local_relation / temperature, dim=-1)
    teacher_prob = F.softmax(global_relation / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)


class FGSSLClient(BaseClient):
    """OpenFGL client for Federated Graph Semantic and Structural Learning."""

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FGSSLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.global_model = copy.deepcopy(self.task.model).to(self.device)
        self.weak_aug = A.Compose(
            [
                A.EdgeRemoving(pe=config["weak_edge_drop"]),
                A.FeatureMasking(pf=config["weak_feature_drop"]),
            ]
        )
        self.strong_aug = A.Compose(
            [
                A.EdgeRemoving(pe=config["strong_edge_drop"]),
                A.FeatureMasking(pf=config["strong_feature_drop"]),
            ]
        )

    def _sync_with_server(self):
        with torch.no_grad():
            for local_param, global_param in zip(
                self.task.model.parameters(),
                self.message_pool["server"]["weight"],
            ):
                local_param.data.copy_(global_param.to(self.device))
            for local_param, global_param in zip(
                self.global_model.parameters(),
                self.message_pool["server"]["weight"],
            ):
                local_param.data.copy_(global_param.to(self.device))

    def _augmented_data(self, data, augmentor):
        aug_data = copy.copy(data)
        x, edge_index, edge_weight = augmentor(data.x, data.edge_index)
        aug_data.x = x
        aug_data.edge_index = edge_index
        if hasattr(aug_data, "edge_attr"):
            aug_data.edge_attr = edge_weight
        return aug_data

    def get_custom_loss_fn(self):
        def custom_loss_fn(embedding, logits, label, mask):
            ce_loss = F.cross_entropy(logits[mask], label[mask])
            if not self.task.model.training:
                return ce_loss

            data = self.task.processed_data["data"]
            weak_data = self._augmented_data(data, self.weak_aug)
            strong_data = self._augmented_data(data, self.strong_aug)

            self.global_model.eval()
            with torch.no_grad():
                global_embedding, _ = self.global_model(weak_data)

            local_embedding_weak, _ = self.task.model(weak_data)
            local_embedding_strong, _ = self.task.model(strong_data)

            semantic_loss = fgssl_semantic_contrastive_loss(
                local_embedding_strong,
                global_embedding,
                label,
                mask,
                config["semantic_temperature"],
            )
            consistency_loss = fgssl_semantic_contrastive_loss(
                local_embedding_strong,
                local_embedding_weak.detach(),
                label,
                mask,
                config["semantic_temperature"],
            )
            structure_loss = fgssl_structure_distillation_loss(
                local_embedding_strong,
                global_embedding,
                data.edge_index,
                config["structure_temperature"],
                config.get("structure_edge_sample_size"),
            )

            return (
                ce_loss
                + config["semantic_weight"] * 0.5 * (semantic_loss + consistency_loss)
                + config["structure_weight"] * structure_loss
            )

        return custom_loss_fn

    def execute(self):
        self._sync_with_server()
        self.task.loss_fn = self.get_custom_loss_fn()
        self.task.train()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }
