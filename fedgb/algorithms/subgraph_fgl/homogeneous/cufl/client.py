import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

from fedgb.training.base import BaseClient
from fedgb.algorithms.subgraph_fgl.homogeneous.cufl.cufl_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.cufl.models import build_cufl_model
from fedgb.algorithms.subgraph_fgl.homogeneous.cufl.spcl import SPCL
from fedgb.algorithms.subgraph_fgl.homogeneous.cufl.utils import (
    VectorGSS,
    compute_edge_confidence,
    copy_state_dict_skip_mask,
    ensure_edge_attr,
    make_masked_split,
    transfer_proxy_reconstruction,
    update_personalization_degree,
)


def _cfg(args, name):
    return getattr(args, name, config[name])


class CUFLClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(CUFLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._apply_cufl_defaults()
        self.task.load_custom_model(build_cufl_model(self.args, self.task))
        self._cl_model = None
        self._proxy_cl_model = None
        self._spcl_optimizer = None
        self._proxy_spcl_optimizer = None
        self._proxy_recon = np.ones(1)
        self._scale_scheduler = self._build_scale_scheduler()
        self._edge_confidence = None
        self._pd = _cfg(self.args, "cufl_spcl_warmup_pd")
        self._prev_state = None
        self._initialized = False

    def _apply_cufl_defaults(self):
        for name, value in config.items():
            if not hasattr(self.args, name):
                setattr(self.args, name, value)

    def _build_scale_scheduler(self):
        return VectorGSS(
            init_scale=_cfg(self.args, "cufl_sim_scale"),
            window_size=_cfg(self.args, "cufl_scheduler_window_size"),
            patience=_cfg(self.args, "cufl_scheduler_patience"),
            varying_factor=_cfg(self.args, "cufl_scheduler_varying_factor"),
            max_scale=_cfg(self.args, "cufl_scheduler_max_scale"),
            min_scale=_cfg(self.args, "cufl_scheduler_min_scale"),
            prefer_larger=_cfg(self.args, "cufl_scheduler_prefer_larger"),
        )

    def _current_round(self):
        return int(self.message_pool.get("round", 0))

    def _state_dict(self):
        return {name: tensor.detach().clone() for name, tensor in self.task.model.state_dict().items()}

    def _load_server_state(self):
        personalized = self.message_pool["server"].get(f"personalized_{self.client_id}")
        global_state = self.message_pool["server"].get("state_dict")
        if global_state is None:
            global_state = {
                name: param.detach().clone()
                for name, param in zip(self.task.model.state_dict().keys(), self.message_pool["server"]["weight"])
            }
        incoming = personalized if personalized is not None else global_state
        self._prev_state = {name: tensor.detach().clone().to(self.device) for name, tensor in incoming.items()}
        copy_state_dict_skip_mask(
            self.task.model,
            self._prev_state,
            skip_mask=_cfg(self.args, "cufl_use_mask"),
        )

    def _ensure_data_attrs(self):
        ensure_edge_attr(self.task.processed_data["data"], self.device)

    def _ensure_spcl(self, data, proxy_data):
        if self._cl_model is None or self._cl_model.num_edges != data.edge_index.size(1):
            self._cl_model = SPCL(
                max_edges=data.edge_index.size(1),
                num_edges=data.edge_index.size(1),
                device=self.device,
                pd=_cfg(self.args, "cufl_spcl_warmup_pd"),
                beta=_cfg(self.args, "cufl_spcl_beta"),
            ).to(self.device)
            self._spcl_optimizer = torch.optim.Adam(
                self._cl_model.parameters(),
                lr=_cfg(self.args, "cufl_spcl_lr"),
            )

        if proxy_data is not None and (
            self._proxy_cl_model is None
            or self._proxy_cl_model.num_edges != proxy_data.edge_index.size(1)
        ):
            self._proxy_cl_model = SPCL(
                max_edges=proxy_data.edge_index.size(1),
                num_edges=proxy_data.edge_index.size(1),
                device=self.device,
                pd=_cfg(self.args, "cufl_spcl_warmup_pd"),
                beta=_cfg(self.args, "cufl_spcl_beta"),
            ).to(self.device)
            self._proxy_spcl_optimizer = torch.optim.Adam(
                self._proxy_cl_model.parameters(),
                lr=_cfg(self.args, "cufl_proxy_spcl_lr"),
            )

    def _build_pretrained_model(self, data):
        pretrained = copy.deepcopy(self.task.model).to(self.device)
        pretrained_dir = _cfg(self.args, "cufl_pretrained_state_dir")
        if pretrained_dir:
            candidate = os.path.join(pretrained_dir, f"{self.client_id}.pt")
            if os.path.exists(candidate):
                state = torch.load(candidate, map_location=self.device)
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                pretrained.load_state_dict(state, strict=False)
        pretrain_epochs = int(_cfg(self.args, "cufl_pretrain_epochs"))
        if pretrain_epochs > 0:
            optimizer = torch.optim.Adam(pretrained.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            for _ in range(pretrain_epochs):
                pretrained.train()
                optimizer.zero_grad()
                _, logits = pretrained(data)
                loss = F.cross_entropy(logits[self.task.train_mask], data.y[self.task.train_mask])
                loss.backward()
                optimizer.step()
        pretrained.eval()
        return pretrained

    def _initialize_curriculum(self, data):
        pretrained = self._build_pretrained_model(data)
        with torch.no_grad():
            _, logits = pretrained(data)
            self._edge_confidence = compute_edge_confidence(
                logits,
                data.y,
                data.edge_index,
                self.task.train_mask,
                norm_scale=_cfg(self.args, "cufl_confidence_norm_scale"),
                norm_method=_cfg(self.args, "cufl_confidence_norm_method"),
            ).to(self.device)
            z = pretrained(data, get_feature=True)

        full_adj = torch.ones(data.edge_index.size(1), device=self.device)
        self._pd = _cfg(self.args, "cufl_spcl_warmup_pd")
        for _ in range(int(_cfg(self.args, "cufl_spcl_warmup_epochs"))):
            self._train_spcl(
                self._cl_model,
                self._spcl_optimizer,
                z,
                data.edge_index,
                full_adj,
                self._pd,
            )
        self._initialized = True

    def _train_spcl(self, spcl, optimizer, embedding, edge_index, full_adj, pd):
        return spcl.train_step(
            optimizer,
            embedding.detach(),
            edge_index,
            pd=pd,
            full_adj=full_adj,
            loss_type=_cfg(self.args, "cufl_spcl_loss_type"),
            beta=_cfg(self.args, "cufl_spcl_beta"),
        )

    def _predict_spcl(self, spcl, edge_index, update_accumul):
        masked_edge_index, masked_edge_weight = spcl.predict(
            edge_index,
            threshold=_cfg(self.args, "cufl_predict_mask_threshold"),
            update_accumul=update_accumul,
        )
        if masked_edge_index.size(1) == 0:
            masked_edge_index = edge_index
            masked_edge_weight = torch.ones(edge_index.size(1), device=self.device)
        return masked_edge_index, masked_edge_weight

    def _evaluate_local_loss(self):
        self.task.model.eval()
        with torch.no_grad():
            _, logits = self.task.model(self.task.splitted_data["data"])
            loss = F.cross_entropy(logits[self.task.val_mask], self.task.data.y[self.task.val_mask])
        return float(loss.detach().cpu())

    def _train_task_with_curriculum(self, data):
        masked_edge_index, masked_edge_weight = self._predict_spcl(
            self._cl_model,
            data.edge_index,
            update_accumul=False,
        )
        if _cfg(self.args, "cufl_use_edge_confidence") and self._edge_confidence is not None:
            edge_mask = self._cl_model.s_mask > _cfg(self.args, "cufl_predict_mask_threshold")
            if edge_mask.sum().item() == masked_edge_weight.numel():
                masked_edge_weight = masked_edge_weight / self._edge_confidence[edge_mask].clamp_min(1e-12)

        masked_split = make_masked_split(
            self.task.processed_data,
            masked_edge_index,
            self.device,
            masked_edge_weight,
        )

        self.task.model.train()
        for _ in range(self.args.num_epochs):
            self.task.optim.zero_grad()
            embedding, logits = self.task.model(masked_split["data"])
            loss = F.cross_entropy(logits[masked_split["train_mask"]], data.y[masked_split["train_mask"]])
            loss = loss + self._mask_l1_loss()
            if self._current_round() > 0 and self._prev_state is not None:
                loss = loss + self._local_l2_loss()
            loss.backward()
            self.task.optim.step()

    def _mask_l1_loss(self):
        if not _cfg(self.args, "cufl_use_mask"):
            return torch.tensor(0.0, device=self.device)
        reg = torch.tensor(0.0, device=self.device)
        for name, param in self.task.model.named_parameters():
            if "mask" in name:
                reg = reg + param.float().norm(1) * _cfg(self.args, "cufl_l1")
        return reg

    def _local_l2_loss(self):
        reg = torch.tensor(0.0, device=self.device)
        for name, param in self.task.model.named_parameters():
            if name not in self._prev_state:
                continue
            if "mask" in name:
                continue
            if "conv" in name or "convs" in name or "classifier" in name:
                reg = reg + (param.float() - self._prev_state[name].float()).norm(2) * _cfg(self.args, "cufl_loc_l2")
        return reg

    def _update_pd(self, curr_round):
        self._pd = update_personalization_degree(
            curr_round,
            self.args.num_rounds,
            _cfg(self.args, "cufl_spcl_base_pd"),
            _cfg(self.args, "cufl_spcl_warmup_pd"),
            _cfg(self.args, "cufl_pd_update_rule"),
        )

    def _train_curriculum_model(self, spcl, optimizer, data, is_proxy, curr_round):
        cycle = _cfg(self.args, "cufl_proxy_spcl_train_cycle") if is_proxy else _cfg(self.args, "cufl_spcl_train_cycle")
        if curr_round % int(cycle) != 0:
            return

        with torch.no_grad():
            masked_edge_index, masked_edge_weight = self._predict_spcl(
                spcl,
                data.edge_index,
                update_accumul=False,
            )
            if (not is_proxy) and _cfg(self.args, "cufl_use_edge_confidence") and self._edge_confidence is not None:
                edge_mask = spcl.s_mask > _cfg(self.args, "cufl_predict_mask_threshold")
                if edge_mask.sum().item() == masked_edge_weight.numel():
                    masked_edge_weight = masked_edge_weight / self._edge_confidence[edge_mask].clamp_min(1e-12)
            masked_data = copy.copy(data)
            masked_data.edge_index = masked_edge_index
            masked_data.edge_attr = masked_edge_weight
            z = self.task.model(masked_data, get_feature=True)

        full_adj = torch.ones(data.edge_index.size(1), device=self.device)
        epochs = _cfg(self.args, "cufl_proxy_spcl_epochs") if is_proxy else _cfg(self.args, "cufl_spcl_epochs")
        for _ in range(int(epochs)):
            self._train_spcl(spcl, optimizer, z, data.edge_index, full_adj, self._pd)
        self._predict_spcl(spcl, data.edge_index, update_accumul=True)

    def execute(self):
        self._load_server_state()
        self._ensure_data_attrs()
        data = self.task.processed_data["data"]
        proxy_data = self.message_pool["server"].get("proxy_data")
        proxy_data = ensure_edge_attr(proxy_data.to(self.device), self.device) if proxy_data is not None else None
        self._ensure_spcl(data, proxy_data)

        if not self._initialized:
            self._initialize_curriculum(data)

        self._scale_scheduler.evaluate(self._evaluate_local_loss())
        self._train_task_with_curriculum(data)

        curr_round = self._current_round()
        self._update_pd(curr_round)
        self._train_curriculum_model(self._cl_model, self._spcl_optimizer, data, False, curr_round)
        if proxy_data is not None:
            self._train_curriculum_model(
                self._proxy_cl_model,
                self._proxy_spcl_optimizer,
                proxy_data,
                True,
                curr_round,
            )
            self._proxy_recon = transfer_proxy_reconstruction(
                self._proxy_cl_model,
                method=_cfg(self.args, "cufl_transfer_mask_method"),
                ratio=_cfg(self.args, "cufl_transfer_mask_ratio"),
                threshold=_cfg(self.args, "cufl_transfer_mask_threshold"),
            )
        else:
            self._proxy_recon = np.ones(1)

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "state_dict": self._state_dict(),
            "proxy_recon_edge": self._proxy_recon,
            "scale": self._scale_scheduler.scale,
        }
