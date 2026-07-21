import torch
from fedgb.training.base import BaseClient
from fedgb.algorithms.standard_fl.feroma.feroma_config import config
from fedgb.algorithms.standard_fl.feroma.utils import latent_descriptor, regression_latent_descriptor


class FEROMAClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(FEROMAClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self._descriptor = None

    def execute(self):
        with torch.no_grad():
            my_key = f"server_{self.client_id}"
            if my_key in self.message_pool:
                target_weight = self.message_pool[my_key]["weight"]
            else:
                target_weight = self.message_pool["server"]["weight"]

            for local_param, global_param in zip(
                self.task.model.parameters(), target_weight
            ):
                local_param.data.copy_(global_param.to(self.device))

            self._descriptor = self._extract_descriptor()

        self.task.train()

    def send_message(self):
        with torch.no_grad():
            descriptor = self._descriptor
            if descriptor is None:
                descriptor = self._extract_descriptor()

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "descriptor": descriptor,
        }

    def _extract_descriptor(self):
        eval_output = self.task.evaluate(mute=True)
        embedding = eval_output["embedding"]
        if self.args.task == "node_cls":
            labels = self.task.data.y.to(self.device)
            mask = self.task.train_mask
            return latent_descriptor(
                embedding[mask],
                labels[mask],
                self.task.num_global_classes,
                include_std=config["feroma_include_std"],
            ).to(self.device)

        if self.args.task == "graph_cls":
            labels = self.task.data.y.to(self.device)
            mask = self.task.train_mask
            return latent_descriptor(
                embedding[mask],
                labels[mask],
                self.task.num_global_classes,
                include_std=config["feroma_include_std"],
            ).to(self.device)

        if self.args.task == "graph_reg":
            return regression_latent_descriptor(
                embedding[self.task.train_mask],
                include_std=config["feroma_include_std"],
            ).to(self.device)

        raise ValueError(f"FEROMA does not support task {self.args.task}.")
