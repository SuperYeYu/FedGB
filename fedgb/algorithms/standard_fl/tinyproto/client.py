import torch
import torch.nn as nn
from fedgb.training.base import BaseClient
from fedgb.algorithms.standard_fl.tinyproto.tinyproto_config import config
from fedgb.algorithms.standard_fl.tinyproto.utils import (
    compute_class_prototypes,
    prototype_distance_logits,
    sparse_proto_targets,
    sparsify_prototypes,
)


class TinyProtoClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(TinyProtoClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device, personalized=True
        )
        self.num_prototypes = 1 if self.args.task == "graph_reg" else self.task.num_global_classes
        self._local_proto = {}
        self._local_counts = torch.zeros(self.num_prototypes, device=device)

    def execute(self):
        device = self.device
        lam = config["tinyproto_lambda"]
        global_protos = self.message_pool["server"].get("global_protos", {})
        proto_masks = self.message_pool["server"].get("proto_masks", {})
        original_loss_fn = self.task.loss_fn

        if global_protos is not None and len(global_protos) > 0:
            mse = nn.MSELoss()

            def proto_loss_fn(embedding, logits, label, mask):
                base_loss = original_loss_fn(embedding, logits, label, mask)
                if mask.sum() == 0:
                    return base_loss
                emb_masked = embedding[mask]
                if self.args.task == "graph_reg":
                    proto_target = self._regression_prototype_targets(
                        emb_masked.shape[0], global_protos, proto_masks, emb_masked.shape[1]
                    )
                else:
                    y_masked = label[mask]
                    proto_target = sparse_proto_targets(
                        y_masked,
                        global_protos,
                        proto_masks,
                        emb_masked.shape[1],
                        device,
                    )
                proto_loss = mse(proto_target, emb_masked)
                return base_loss + lam * proto_loss

            self.task.loss_fn = proto_loss_fn

        self.task.train()
        self.task.loss_fn = original_loss_fn

        self.update_local_prototypes()
        self.install_proto_evaluator()

    def _regression_prototype_targets(self, count, global_protos, proto_masks, feature_dim):
        from fedgb.algorithms.standard_fl.tinyproto.utils import expand_sparse_prototypes
        full = expand_sparse_prototypes(global_protos, proto_masks, feature_dim, device=self.device)
        return full[0].unsqueeze(0).expand(count, -1)

    def update_local_prototypes(self):
        with torch.no_grad():
            eval_output = self.task.evaluate(mute=True)
            embedding = eval_output["embedding"]
            if self.args.task == "graph_reg":
                selected = embedding[self.task.train_mask]
                count = int(selected.shape[0])
                proto = selected.mean(dim=0).detach()
                if config["tinyproto_simple_scale"]:
                    proto = proto * count
                protos = {0: proto}
                counts = torch.tensor([count], dtype=torch.float32, device=self.device)
            else:
                labels = self.task.data.y.to(self.device)
                protos, counts = compute_class_prototypes(
                    embedding,
                    labels,
                    self.task.train_mask,
                    self.task.num_global_classes,
                    simple_scale=config["tinyproto_simple_scale"],
                )
            proto_masks = self.message_pool["server"].get("proto_masks", {})
            if config["tinyproto_add_cps"]:
                protos = sparsify_prototypes(protos, proto_masks)
            self._local_proto = protos
            self._local_counts = counts.detach().clone()

    def install_proto_evaluator(self):
        if self.args.task == "graph_reg":
            return
        if not config["tinyproto_proto_eval"]:
            return
        global_protos = self.message_pool["server"].get("global_protos", {})
        proto_masks = self.message_pool["server"].get("proto_masks", {})
        if not global_protos:
            return

        def override_evaluate(splitted_data=None, mute=False):
            current_override = self.task.override_evaluate
            self.task.override_evaluate = None
            try:
                result = self.task.evaluate(splitted_data, mute=True)
            finally:
                self.task.override_evaluate = current_override
            logits = prototype_distance_logits(
                result["embedding"],
                global_protos,
                proto_masks,
                self.task.num_global_classes,
                result["embedding"].shape[1],
            )
            if logits is None:
                return result

            data = self.task.splitted_data if splitted_data is None else splitted_data
            labels = self._labels_for_eval(data).to(self.device)
            masks = self._masks_for_eval(data)
            for split, mask in masks.items():
                selected = mask.to(self.device).bool()
                if selected.sum() == 0:
                    result[f"accuracy_{split}"] = 0.0
                else:
                    pred = torch.argmax(logits[selected], dim=1)
                    result[f"accuracy_{split}"] = (
                        (pred == labels[selected]).float().mean().item()
                    )
            if not mute:
                prefix = f"[client {self.client_id}]"
                print(
                    prefix
                    + f"\taccuracy_train: {result.get('accuracy_train', 0):.4f}"
                    + f"\taccuracy_val: {result.get('accuracy_val', 0):.4f}"
                    + f"\taccuracy_test: {result.get('accuracy_test', 0):.4f}"
                )
            return result

        self.task.override_evaluate = override_evaluate

    def _labels_for_eval(self, splitted_data):
        if self.args.task == "node_cls":
            return splitted_data["data"].y
        labels = torch.zeros(len(splitted_data["data"]), device=self.device).long()
        for mask_name, loader_name in [
            ("train_mask", "train_dataloader"),
            ("val_mask", "val_dataloader"),
            ("test_mask", "test_dataloader"),
        ]:
            indices = splitted_data[mask_name].nonzero().squeeze().tolist()
            if isinstance(indices, int):
                indices = [indices]
            pos = 0
            for batch in splitted_data[loader_name]:
                count = batch.num_graphs
                labels[indices[pos : pos + count]] = batch.y.to(self.device)
                pos += count
        return labels

    def _masks_for_eval(self, splitted_data):
        return {
            "train": splitted_data["train_mask"],
            "val": splitted_data["val_mask"],
            "test": splitted_data["test_mask"],
        }

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "local_proto": self._local_proto,
            "local_counts": self._local_counts,
        }
