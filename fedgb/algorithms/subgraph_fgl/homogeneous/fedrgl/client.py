import torch
import torch.nn.functional as F

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.fedrgl.fedrgl_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedrgl.models import build_fedrgl_model
from fedgb.algorithms.subgraph_fgl.homogeneous.fedrgl.utils import (
    calculate_entropy,
    contrastive_loss,
    jensen_shannon_divergence,
    label_propagation_soft_labels,
    mask_indices,
    prox_loss,
    two_stage_clean_noisy_split,
)


def _cfg(args, key):
    return getattr(args, key, config[key])


class FedRGLClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedRGLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.load_custom_model(build_fedrgl_model(self.args, self.task))
        self._client_entropy = None

    def execute(self):
        server_weights = self.message_pool.get("server", {}).get("weight")
        if server_weights is not None:
            with torch.no_grad():
                for local_param, global_param in zip(self.task.model.parameters(), server_weights):
                    local_param.data.copy_(global_param.to(self.device))

        global_weights = [param.detach().clone() for param in self.task.model.parameters()]
        round_id = self.message_pool.get("round", 0)

        if round_id < _cfg(self.args, "fedrgl_warmup_rounds"):
            self._train_warmup(global_weights)
        else:
            clean_mask, noisy_mask = self._split_clean_noisy()
            self._train_robust(clean_mask, noisy_mask, global_weights)

        self._update_entropy()

    def _apply_config_defaults(self):
        for key, value in config.items():
            if not hasattr(self.args, key):
                setattr(self.args, key, value)

    def _train_warmup(self, global_weights):
        data = self.task.processed_data["data"]
        train_idx = mask_indices(self.task.processed_data["train_mask"])
        labels = data.y

        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            embedding, logits = self.task.model(data)
            emb1, emb2, _, _ = self.task.model.forward_two_views(data)
            ce_loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            con_loss = 0.5 * (
                contrastive_loss(
                    emb1,
                    emb2,
                    _cfg(self.args, "fedrgl_temperature"),
                    max_nodes=_cfg(self.args, "fedrgl_contrastive_max_nodes"),
                )
                + contrastive_loss(
                    emb2,
                    emb1,
                    _cfg(self.args, "fedrgl_temperature"),
                    max_nodes=_cfg(self.args, "fedrgl_contrastive_max_nodes"),
                )
            )
            loss = ce_loss + _cfg(self.args, "fedrgl_alpha_warmup") * con_loss
            loss = loss + _cfg(self.args, "fedrgl_mu_warmup") * prox_loss(self.task.model, global_weights, self.device)
            loss.backward()
            self.task.optim.step()

    def _split_clean_noisy(self):
        data = self.task.processed_data["data"]
        train_mask = self.task.processed_data["train_mask"]
        train_idx = mask_indices(train_mask)

        self.task.model.eval()
        with torch.no_grad():
            _, logits = self.task.model(data)
            losses = F.cross_entropy(logits[train_idx], data.y[train_idx], reduction="none")
            soft_labels = F.softmax(logits, dim=-1)
            predictions = soft_labels.argmax(dim=-1)
            high_mask = predictions == data.y
            propagated = label_propagation_soft_labels(
                data.edge_index,
                soft_labels,
                train_mask,
                high_mask=high_mask if _cfg(self.args, "fedrgl_use_lp_filter") else None,
                num_nodes=data.x.shape[0],
                prop_steps=_cfg(self.args, "fedrgl_lp_prop"),
                alpha=_cfg(self.args, "fedrgl_lp_alpha"),
            )
            clean_mask, noisy_mask = two_stage_clean_noisy_split(
                train_mask,
                data.y,
                losses,
                propagated,
                stage1_scale=_cfg(self.args, "fedrgl_clean_scale"),
                stage2_scale=_cfg(self.args, "fedrgl_clean_scale_lp"),
                only_stage1=_cfg(self.args, "fedrgl_only_stage1"),
            )

        return clean_mask, noisy_mask

    def _train_robust(self, clean_mask, noisy_mask, global_weights):
        data = self.task.processed_data["data"]
        clean_idx = mask_indices(clean_mask)
        noisy_idx = mask_indices(noisy_mask)
        labels = data.y

        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            _, logits = self.task.model(data)
            emb1, emb2, logits1, logits2 = self.task.model.forward_two_views(data)

            loss = torch.zeros((), device=self.device)
            if clean_idx.numel() > 0:
                loss = loss + F.cross_entropy(logits[clean_idx], labels[clean_idx])

            if noisy_idx.numel() > 0:
                pred_noisy = (F.softmax(logits1[noisy_idx], dim=-1) + F.softmax(logits2[noisy_idx], dim=-1)) * 0.5
                pseudo_conf, pseudo_label = pred_noisy.max(dim=-1)
                confident = pseudo_conf > _cfg(self.args, "fedrgl_confidence_threshold")
                if confident.any():
                    selected = noisy_idx[confident]
                    pseudo = pseudo_label[confident]
                    noisy_loss = 0.5 * (
                        F.cross_entropy(logits1[selected], pseudo)
                        + F.cross_entropy(logits2[selected], pseudo)
                    )
                    js_loss = (
                        jensen_shannon_divergence(logits1[selected], logits[selected])
                        + jensen_shannon_divergence(logits2[selected], logits[selected])
                    )
                    loss = loss + _cfg(self.args, "fedrgl_noisy_beta") * noisy_loss
                    loss = loss + _cfg(self.args, "fedrgl_js_weight") * js_loss

            con_loss = 0.5 * (
                contrastive_loss(
                    emb1,
                    emb2,
                    _cfg(self.args, "fedrgl_temperature"),
                    max_nodes=_cfg(self.args, "fedrgl_contrastive_max_nodes"),
                )
                + contrastive_loss(
                    emb2,
                    emb1,
                    _cfg(self.args, "fedrgl_temperature"),
                    max_nodes=_cfg(self.args, "fedrgl_contrastive_max_nodes"),
                )
            )
            loss = loss + _cfg(self.args, "fedrgl_alpha_robust") * con_loss
            loss = loss + _cfg(self.args, "fedrgl_mu_robust") * prox_loss(self.task.model, global_weights, self.device)
            loss.backward()
            self.task.optim.step()

    def _update_entropy(self):
        self.task.model.eval()
        with torch.no_grad():
            _, logits = self.task.model(self.task.splitted_data["data"])
            val_entropy = calculate_entropy(logits[self.task.splitted_data["val_mask"]])
            test_entropy = calculate_entropy(logits[self.task.splitted_data["test_mask"]])
            self._client_entropy = (val_entropy + test_entropy).detach()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "entropy": self._client_entropy,
        }
