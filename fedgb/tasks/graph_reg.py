import os
import pickle
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from fedgb.data.processing import processing
from fedgb.tasks.base import BaseTask
from fedgb.utils.basic_utils import extract_floats, idx_to_mask_tensor
from fedgb.utils.metrics import compute_regression_metrics
from fedgb.utils.task_utils import load_graph_cls_default_model


def masked_mse(logits, labels, target_mask=None):
    """Return MSE over observed target elements, or None when none are observed."""

    if target_mask is None:
        return F.mse_loss(logits, labels)
    target_mask = target_mask.bool()
    if not torch.any(target_mask):
        return None
    return F.mse_loss(logits[target_mask], labels[target_mask])


def graph_regression_batch_loss(task, embedding, logits, batch):
    """Evaluate a task loss while forwarding an optional element-level target mask."""

    labels = task._target(batch.y)
    target_mask = getattr(batch, "y_mask", None)
    if target_mask is not None:
        target_mask = target_mask.bool().view(-1, task.num_targets)
    graph_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
    return task.loss_fn(
        embedding,
        logits,
        labels,
        graph_mask,
        target_mask=target_mask,
    )


class GraphRegTask(BaseTask):
    """Graph regression task following the OpenFGL graph task interface."""

    def __init__(self, args, client_id, data, data_dir, device):
        super(GraphRegTask, self).__init__(args, client_id, data, data_dir, device)

    def _move_batch(self, batch):
        return batch.to(self.device)

    def _target(self, label):
        label = label.float()
        if self.num_targets == 1:
            return label.view(-1, 1)
        return label.view(-1, self.num_targets)

    def _training_loss(self, embedding, logits, labels, target_mask=None):
        logits = logits.view(-1, self.num_targets)
        labels = labels.view(-1, self.num_targets).float()
        base_loss = masked_mse(logits, labels, target_mask)
        if base_loss is None:
            return None

        loss_function = getattr(self.loss_fn, "__func__", None)
        if loss_function is GraphRegTask.loss_fn:
            return base_loss

        # Legacy algorithm hooks add their regularizer to the unmasked task loss.
        graph_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
        custom_loss = self.loss_fn(embedding, logits, labels, graph_mask)
        unmasked_base = self.default_loss_fn(logits, labels)
        return base_loss + (custom_loss - unmasked_base)

    def _indices_from_mask(self, mask):
        idx = mask.detach().cpu().nonzero(as_tuple=True)[0].tolist()
        return idx

    def _graphs_from_mask(self, mask):
        return [self.data[i] for i in self._indices_from_mask(mask)]

    def train(self, splitted_data=None):
        if splitted_data is None:
            splitted_data = self.processed_data
        else:
            names = ["data", "train_dataloader", "val_dataloader", "test_dataloader", "train_mask", "val_mask", "test_mask"]
            for name in names:
                assert name in splitted_data

        self.model.train()
        for _ in range(self.args.num_epochs):
            for batch in splitted_data["train_dataloader"]:
                batch = self._move_batch(batch)
                self.optim.zero_grad()
                embedding, logits = self.model.forward(batch)
                labels = self._target(batch.y)
                target_mask = getattr(batch, "y_mask", None)
                if target_mask is not None:
                    target_mask = target_mask.bool().view(-1, self.num_targets)
                loss_train = self._training_loss(embedding, logits, labels, target_mask)
                if loss_train is None:
                    continue
                loss_train.backward()
                if self.step_preprocess is not None:
                    self.step_preprocess()
                self.optim.step()

    def evaluate(self, splitted_data=None, mute=False):
        if splitted_data is None:
            splitted_data = self.splitted_data
        else:
            names = ["data", "train_dataloader", "val_dataloader", "test_dataloader", "train_mask", "val_mask", "test_mask"]
            for name in names:
                assert name in splitted_data

        eval_output = {}
        self.model.eval()

        num_samples = len(splitted_data["data"])
        embedding_all = torch.zeros((num_samples, self.args.hid_dim), device=self.device)
        logits_all = torch.zeros((num_samples, self.num_targets), device=self.device)
        label_all = torch.zeros((num_samples, self.num_targets), device=self.device)
        target_mask_all = torch.zeros((num_samples, self.num_targets), dtype=torch.bool, device=self.device)

        split_specs = [
            ("train", splitted_data["train_dataloader"], splitted_data["train_mask"]),
            ("val", splitted_data["val_dataloader"], splitted_data["val_mask"]),
            ("test", splitted_data["test_dataloader"], splitted_data["test_mask"]),
        ]

        with torch.no_grad():
            for _, dataloader, mask in split_specs:
                indices = self._indices_from_mask(mask)
                offset = 0
                for batch in dataloader:
                    batch = self._move_batch(batch)
                    embedding, logits = self.model.forward(batch)
                    labels = self._target(batch.y)
                    cur_indices = indices[offset: offset + batch.num_graphs]
                    embedding_all[cur_indices] = embedding
                    logits_all[cur_indices] = logits.view(-1, self.num_targets)
                    label_all[cur_indices] = labels
                    batch_target_mask = getattr(batch, "y_mask", None)
                    if batch_target_mask is None:
                        batch_target_mask = torch.ones_like(labels, dtype=torch.bool)
                    target_mask_all[cur_indices] = batch_target_mask.bool().view(-1, self.num_targets)
                    offset += batch.num_graphs

            losses = {}
            observed_masks = {}
            for split_name in ("train", "val", "test"):
                graph_mask = splitted_data[f"{split_name}_mask"].view(-1, 1)
                observed = graph_mask & target_mask_all
                observed_masks[split_name] = observed
                losses[split_name] = masked_mse(logits_all, label_all, observed)

        eval_output["embedding"] = embedding_all
        eval_output["logits"] = logits_all
        for split_name in ("train", "val", "test"):
            observed = observed_masks[split_name]
            observed_count = int(observed.sum().item())
            eval_output[f"num_observed_{split_name}"] = observed_count
            loss = losses[split_name]
            eval_output[f"loss_{split_name}"] = (
                loss if loss is not None else torch.tensor(float("nan"), device=self.device)
            )
            if observed_count:
                metrics = compute_regression_metrics(
                    self.args.metrics,
                    logits_all[observed],
                    label_all[observed],
                    split_name,
                )
            else:
                metrics = {f"{metric}_{split_name}": float("nan") for metric in self.args.metrics}
            eval_output.update(metrics)

        info = ""
        for key, val in eval_output.items():
            try:
                info += f"\t{key}: {val:.4f}"
            except Exception:
                continue
        prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
        if not mute:
            print(prefix + info)
        return eval_output

    def loss_fn(self, embedding, logits, label, mask, target_mask=None):
        logits = logits.view(-1, self.num_targets)
        label = label.view(-1, self.num_targets).float()
        graph_mask = mask.bool().view(-1, 1).expand_as(label)
        if target_mask is not None:
            graph_mask = graph_mask & target_mask.bool().view_as(label)
        return masked_mse(logits, label, graph_mask)

    @property
    def default_model(self):
        return load_graph_cls_default_model(self.args, input_dim=self.num_feats, output_dim=self.num_targets, client_id=self.client_id)

    @property
    def default_optim(self):
        if self.args.optim == "adam":
            from torch.optim import Adam
            return Adam

    @property
    def num_samples(self):
        target_mask = getattr(self.data, "y_mask", None)
        if target_mask is None:
            return len(self.data)
        if hasattr(self, "train_mask"):
            graph_mask = self.train_mask.detach().cpu().bool()
            return int(target_mask.detach().cpu()[graph_mask].sum().item())
        return int(target_mask.detach().cpu().sum().item())

    @property
    def num_feats(self):
        return self.data[0].x.shape[1]

    @property
    def num_targets(self):
        return int(getattr(self.data, "num_targets", 1))

    @property
    def default_loss_fn(self):
        return nn.MSELoss()

    @property
    def default_train_val_test_split(self):
        return 0.8, 0.1, 0.1

    @property
    def train_val_test_path(self):
        if self.args.train_val_test == "default_split":
            return osp.join(self.data_dir, "graph_reg", "default_split")
        split_dir = f"split_{self.args.train_val_test}"
        return osp.join(self.data_dir, "graph_reg", split_dir)

    def load_train_val_test_split(self):
        if self.client_id is None and len(self.args.dataset) == 1:
            train_mask, val_mask, test_mask = self._load_server_split()
        else:
            train_mask, val_mask, test_mask = self._load_client_split()

        self._attach_graph_cache_ids()
        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)

        self.train_dataloader = DataLoader(self._graphs_from_mask(self.train_mask), batch_size=self.args.batch_size, shuffle=False)
        self.val_dataloader = DataLoader(self._graphs_from_mask(self.val_mask), batch_size=self.args.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self._graphs_from_mask(self.test_mask), batch_size=self.args.batch_size, shuffle=False)

        self.splitted_data = {
            "data": self.data,
            "train_dataloader": self.train_dataloader,
            "val_dataloader": self.val_dataloader,
            "test_dataloader": self.test_dataloader,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask,
        }
        self.processed_data = processing(args=self.args, splitted_data=self.splitted_data, processed_dir=self.data_dir, client_id=self.client_id)

    def _attach_graph_cache_ids(self):
        if not hasattr(self.data, "global_map"):
            return
        for local_id, graph in enumerate(self.data):
            cache_id = self.data.global_map.get(local_id, local_id)
            graph.fedssp_cache_id = torch.tensor([int(cache_id)], dtype=torch.long)

    def _load_server_split(self):
        glb_train, glb_val, glb_test = [], [], []
        for client_id in range(self.args.num_clients):
            for name, bucket in [("train", glb_train), ("val", glb_val), ("test", glb_test)]:
                path = osp.join(self.train_val_test_path, f"glb_{name}_{client_id}.pkl")
                if osp.exists(path):
                    with open(path, "rb") as file:
                        bucket += pickle.load(file)

        if glb_train or glb_val or glb_test:
            max_glb_id = max(glb_train + glb_val + glb_test)
            num_graphs = len(self.data)
            if max_glb_id >= num_graphs:
                return self.local_graph_train_val_test_split(self.data, self.args.train_val_test)
            return (
                idx_to_mask_tensor(glb_train, num_graphs).bool(),
                idx_to_mask_tensor(glb_val, num_graphs).bool(),
                idx_to_mask_tensor(glb_test, num_graphs).bool(),
            )
        return self.local_graph_train_val_test_split(self.data, self.args.train_val_test)

    def _load_client_split(self):
        train_path = osp.join(self.train_val_test_path, f"train_{self.client_id}.pt")
        val_path = osp.join(self.train_val_test_path, f"val_{self.client_id}.pt")
        test_path = osp.join(self.train_val_test_path, f"test_{self.client_id}.pt")
        glb_train_path = osp.join(self.train_val_test_path, f"glb_train_{self.client_id}.pkl")
        glb_val_path = osp.join(self.train_val_test_path, f"glb_val_{self.client_id}.pkl")
        glb_test_path = osp.join(self.train_val_test_path, f"glb_test_{self.client_id}.pkl")

        if osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path):
            return torch.load(train_path), torch.load(val_path), torch.load(test_path)

        train_mask, val_mask, test_mask = self.local_graph_train_val_test_split(self.data, self.args.train_val_test)
        if not osp.exists(self.train_val_test_path):
            os.makedirs(self.train_val_test_path)
        torch.save(train_mask, train_path)
        torch.save(val_mask, val_path)
        torch.save(test_mask, test_path)

        if len(self.args.dataset) == 1:
            self._save_global_split_ids(train_mask, glb_train_path)
            self._save_global_split_ids(val_mask, glb_val_path)
            self._save_global_split_ids(test_mask, glb_test_path)
        return train_mask, val_mask, test_mask

    def _save_global_split_ids(self, mask, path):
        glb_ids = []
        for local_id in mask.nonzero():
            local_id = local_id.item()
            glb_ids.append(self.data.global_map[local_id])
        with open(path, "wb") as file:
            pickle.dump(glb_ids, file)

    def local_graph_train_val_test_split(self, local_graphs, split, shuffle=True):
        num_graphs = len(local_graphs)
        split_values = [getattr(graph, "split", None) for graph in local_graphs]
        if all(value in {"train", "val", "test"} for value in split_values):
            train_idx = [i for i, value in enumerate(split_values) if value == "train"]
            val_idx = [i for i, value in enumerate(split_values) if value == "val"]
            test_idx = [i for i, value in enumerate(split_values) if value == "test"]
            return (
                idx_to_mask_tensor(train_idx, num_graphs).bool(),
                idx_to_mask_tensor(val_idx, num_graphs).bool(),
                idx_to_mask_tensor(test_idx, num_graphs).bool(),
            )

        if split == "default_split":
            train_, val_, test_ = self.default_train_val_test_split
        else:
            train_, val_, test_ = extract_floats(split)

        graph_ids = np.arange(num_graphs)
        if shuffle:
            np.random.shuffle(graph_ids)
        train_end = int(train_ * num_graphs)
        val_end = int((train_ + val_) * num_graphs)
        return (
            idx_to_mask_tensor(graph_ids[:train_end].tolist(), num_graphs).bool(),
            idx_to_mask_tensor(graph_ids[train_end:val_end].tolist(), num_graphs).bool(),
            idx_to_mask_tensor(graph_ids[val_end:int((train_ + val_ + test_) * num_graphs)].tolist(), num_graphs).bool(),
        )
