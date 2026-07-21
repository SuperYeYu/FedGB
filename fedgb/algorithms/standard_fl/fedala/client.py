import torch
import torch.nn as nn
import copy
import numpy as np
from fedgb.training.base import BaseClient
from fedgb.algorithms.standard_fl.fedala.fedala_config import config


class FedALAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FedALAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.ala_weights = None
        self.prev_local_params = None
        self.start_phase = True

    def execute(self):
        global_params = [p.to(self.device) for p in self.message_pool["server"]["weight"]]
        local_params = list(self.task.model.parameters())

        if self.prev_local_params is not None:
            self._ala_aggregation(global_params, local_params)
        else:
            for lp, gp in zip(local_params, global_params):
                lp.data.copy_(gp)

        self.task.train()

    def _ala_aggregation(self, global_params, local_params):
        device = self.device
        data = self.task.data
        train_mask = self.task.train_mask
        idx = train_mask.nonzero(as_tuple=True)[0]
        num_train = len(idx)
        rand_pct = config["fedala_rand_percent"] / 100
        rand_num = max(int(rand_pct * num_train), 4)
        rand_start = torch.randint(0, max(num_train - rand_num, 1), (1,)).item()
        sample_idx = idx[rand_start : rand_start + rand_num]

        # Init per-parameter weights (same shape as each param)
        if self.ala_weights is None:
            self.ala_weights = [
                torch.ones_like(p.data).to(device) for p in local_params
            ]

        eta = config["fedala_eta"]
        prev = self.prev_local_params

        # Temp model: copy of current local model
        model_t = copy.deepcopy(self.task.model)
        params_t = list(model_t.parameters())

        # Set temp model to: temp = prev + w * (global - prev)
        for pt, pp, pg, w in zip(params_t, prev, global_params, self.ala_weights):
            pt.data.copy_(pp.to(device) + (pg.to(device) - pp.to(device)) * w)

        # Weight learning loop
        x_full = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y_full = data.y.to(device)

        losses_log = []
        cnt = 0
        max_iter = 100 if self.start_phase else 1

        while True:
            # Set temp model params from current weights
            for pt, pp, pg, w in zip(params_t, prev, global_params, self.ala_weights):
                pt.data.copy_(pp.to(device) + (pg.to(device) - pp.to(device)) * w)

            # Forward through temp model on full graph
            emb_t, logits_t = model_t(data)
            loss = nn.CrossEntropyLoss()(logits_t[sample_idx], y_full[sample_idx])
            loss.backward()

            # Update per-parameter weights: w -= eta * grad_t * (global - prev)
            with torch.no_grad():
                for pt, pp, pg, w in zip(params_t, prev, global_params, self.ala_weights):
                    delta = pg.to(device) - pp.to(device)
                    w.sub_(eta * pt.grad * delta).clamp_(0, 1)

            # Zero all grads in temp model
            model_t.zero_grad()

            losses_log.append(loss.item())
            cnt += 1

            if not self.start_phase or cnt >= max_iter:
                break
            if len(losses_log) > 10 and np.std(losses_log[-10:]) < config["fedala_threshold"]:
                break

        self.start_phase = False

        # Apply final mix to local model
        for lp, pp, pg, w in zip(local_params, prev, global_params, self.ala_weights):
            lp.data.copy_(pp.to(device) + (pg.to(device) - pp.to(device)) * w)

        self.prev_local_params = [p.detach().clone().to(device) for p in local_params]

    def send_message(self):
        if self.prev_local_params is None:
            self.prev_local_params = [
                p.detach().clone().to(self.device) for p in self.task.model.parameters()
            ]
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
        }
