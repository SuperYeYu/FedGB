import torch
import torch.nn.functional as F
from fedgb.training.base import BaseClient
from fedgb.algorithms.graph_fgl.fedssp.fedssp_config import config
from fedgb.algorithms.graph_fgl.fedssp.models import build_fedssp_model
from fedgb.algorithms.graph_fgl.fedssp.utils import fedssp_shared_state_dict, load_fedssp_shared_state
from fedgb.tasks.graph_reg import graph_regression_batch_loss


def _cfg(args, key):
    return getattr(args, key, config[key])


class FedSSPClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedSSPClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._apply_config_defaults()
        self.task.load_custom_model(build_fedssp_model(self.args, self.task))
        self._current_mean = torch.zeros(self.task.model.hid_dim, device=device)

    def _apply_config_defaults(self):
        for key, value in config.items():
            if not hasattr(self.args, key):
                setattr(self.args, key, value)
        self.args.hid_dim = self.args.fedssp_hidden_dim

    def execute(self):
        device = self.device
        model = self.task.model
        global_consensus = self.message_pool["server"].get("global_consensus")

        with torch.no_grad():
            shared_state = self.message_pool["server"].get("shared_state")
            if shared_state is not None:
                load_fedssp_shared_state(model, shared_state)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=_cfg(self.args, "fedssp_optimizer_lr"),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=_cfg(self.args, "fedssp_optimizer_weight_decay"),
        )
        momentum = _cfg(self.args, "fedssp_momentum")
        tau = _cfg(self.args, "fedssp_tau_weight")

        model.train()
        current_mean = torch.zeros(model.hid_dim, device=device)
        batches_tracked = 0

        for _ in range(self.args.num_epochs):
            for batch in self.task.train_dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                embedding, logits = model(batch, use_preference=global_consensus is not None)

                batch_mean = embedding.mean(dim=0)
                if batches_tracked == 0:
                    current_mean = batch_mean.detach()
                else:
                    current_mean = (1.0 - momentum) * current_mean + momentum * batch_mean.detach()
                batches_tracked += 1

                if getattr(self.args, "task", None) == "graph_reg":
                    loss = graph_regression_batch_loss(self.task, embedding, logits, batch)
                else:
                    loss = F.cross_entropy(logits, batch.y)
                if global_consensus is not None:
                    consensus_loss = 0.5 * torch.mean((current_mean - global_consensus.to(device)) ** 2)
                    consensus_loss = tau * consensus_loss
                    loss = consensus_loss if loss is None else loss + consensus_loss
                if loss is None:
                    continue

                loss.backward()
                optimizer.step()

        self._current_mean = self._compute_current_mean()

    def _compute_current_mean(self):
        model = self.task.model
        model.eval()
        means = []
        with torch.no_grad():
            for batch in self.task.train_dataloader:
                batch = batch.to(self.device)
                embedding, _ = model(batch)
                means.append(embedding.mean(dim=0))
        if not means:
            return torch.zeros(model.hid_dim, device=self.device)
        return torch.stack(means, dim=0).mean(dim=0)

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "shared_state": fedssp_shared_state_dict(self.task.model),
            "current_mean": self._current_mean.detach().clone(),
        }
