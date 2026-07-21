import torch
import torch.nn as nn
import torch.nn.functional as F

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.fedspray.fedspray_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fedspray.utils import (
    build_proxy_teacher_logits,
    clone_state_dict,
    compute_local_structure_proxies,
    move_state_dict_to_device,
)


class FedSprayEncoder(nn.Module):
    def __init__(self, input_dim, proxy_dim):
        super(FedSprayEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, proxy_dim)

    def forward(self, x):
        return F.relu(self.linear(x))


class FedSprayClassifier(nn.Module):
    def __init__(self, proxy_dim, num_classes):
        super(FedSprayClassifier, self).__init__()
        self.linear = nn.Linear(proxy_dim, num_classes)

    def forward(self, embedding, proxy=None):
        if proxy is not None:
            embedding = embedding + proxy
        return self.linear(embedding)


class FedSprayClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedSprayClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device, personalized=True
        )
        proxy_dim = config.get("fedspray_proxy_dim", args.hid_dim)
        self.encoder = FedSprayEncoder(self.task.num_feats, proxy_dim).to(device)
        self.classifier = FedSprayClassifier(proxy_dim, self.task.num_global_classes).to(device)
        self.classifier2 = FedSprayClassifier(proxy_dim, self.task.num_global_classes).to(device)
        self.proxy = nn.Parameter(torch.full((self.task.num_samples, proxy_dim), 0.1, device=device))
        self.proxy_optimizer = torch.optim.Adam([self.proxy], lr=config["fedspray_proxy_lr"])
        self.proxy_optimizer_net = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.classifier.parameters())
            + list(self.classifier2.parameters()),
            lr=args.lr,
            weight_decay=getattr(args, "weight_decay", 0.0),
        )
        self.local_proxy = torch.zeros(
            self.task.num_global_classes, proxy_dim, device=device
        )
        self.local_counts = torch.zeros(self.task.num_global_classes, device=device)

    def execute(self):
        server_msg = self.message_pool.get("server", {})
        self._load_global_proxy_modules(server_msg)
        global_proxy = self._global_proxy_from_message(server_msg)

        data = self.task.processed_data["data"]
        train_mask = self.task.processed_data["train_mask"]
        val_mask = self.task.processed_data["val_mask"]
        test_mask = self.task.processed_data["test_mask"]
        train_idx = train_mask.nonzero(as_tuple=True)[0]
        eval_idx = (val_mask | test_mask).nonzero(as_tuple=True)[0]

        self._train_personalized_gnn(data, train_idx, eval_idx, global_proxy)
        self._train_proxy_modules(data, train_idx)
        self.local_proxy, self.local_counts = compute_local_structure_proxies(
            self.proxy,
            data.y,
            train_mask,
            self.task.num_global_classes,
        )

    def _load_global_proxy_modules(self, server_msg):
        module_pairs = [
            ("encoder_state", self.encoder),
            ("classifier_state", self.classifier),
            ("classifier2_state", self.classifier2),
        ]
        for key, module in module_pairs:
            state_dict = server_msg.get(key)
            if state_dict is not None:
                module.load_state_dict(move_state_dict_to_device(state_dict, self.device))

    def _global_proxy_from_message(self, server_msg):
        global_proxy = server_msg.get("global_proxy")
        if global_proxy is None:
            proxy_dim = self.proxy.size(1)
            values = [
                0.01 * class_id * torch.ones(proxy_dim, device=self.device)
                for class_id in range(self.task.num_global_classes)
            ]
            return torch.stack(values, dim=0)
        return global_proxy.to(self.device)

    def _reliable_indices(self, proxy_logits, train_idx, eval_idx):
        with torch.no_grad():
            confidence, _ = F.softmax(proxy_logits[eval_idx], dim=1).max(dim=1)
            confident_eval_idx = eval_idx[
                confidence > config["fedspray_confidence_threshold"]
            ]
            return torch.cat([train_idx, confident_eval_idx])

    def _train_personalized_gnn(self, data, train_idx, eval_idx, global_proxy):
        criterion_kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.task.model.train()
        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            _, logits = self.task.model(data)
            with torch.no_grad():
                teacher_logits, weighted_proxy = build_proxy_teacher_logits(
                    self.encoder,
                    self.classifier,
                    data.x,
                    logits,
                    global_proxy,
                )
                if eval_idx.numel() > 0:
                    reliable_idx = self._reliable_indices(teacher_logits, train_idx, eval_idx)
                else:
                    reliable_idx = train_idx
                self.proxy.data[eval_idx] = weighted_proxy[eval_idx]

            ce_loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
            dist_loss = criterion_kl(
                F.log_softmax(logits[reliable_idx], dim=1),
                F.log_softmax(teacher_logits[reliable_idx], dim=1).detach(),
            )
            loss = ce_loss + config["fedspray_lambda1"] * dist_loss
            loss.backward()
            self.task.optim.step()

    def _train_proxy_modules(self, data, train_idx):
        criterion_kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.task.model.eval()
        with torch.no_grad():
            _, gnn_logits = self.task.model(data)
            gnn_logits = gnn_logits.detach()

        self.encoder.train()
        self.classifier.train()
        self.classifier2.train()
        for _ in range(self.args.num_epochs):
            self.proxy_optimizer.zero_grad()
            self.proxy_optimizer_net.zero_grad()

            proxy_embedding = self.encoder(data.x)
            proxy_logits = self.classifier(proxy_embedding, self.proxy)
            classifier2_logits = self.classifier2(proxy_embedding)
            ce_loss = F.cross_entropy(classifier2_logits[train_idx], data.y[train_idx])
            dist_loss = criterion_kl(
                F.log_softmax(proxy_logits[train_idx], dim=1),
                F.log_softmax(gnn_logits[train_idx], dim=1).detach(),
            )
            loss = ce_loss + config["fedspray_proxy_distill_weight"] * dist_loss
            loss.backward()
            self.proxy_optimizer_net.step()
            self.proxy_optimizer.step()

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "encoder_state": clone_state_dict(self.encoder),
            "classifier_state": clone_state_dict(self.classifier),
            "classifier2_state": clone_state_dict(self.classifier2),
            "local_proxy": self.local_proxy.detach().clone(),
            "local_counts": self.local_counts.detach().clone(),
        }
