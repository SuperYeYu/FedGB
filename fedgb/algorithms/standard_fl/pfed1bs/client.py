import torch
from fedgb.training.base import BaseClient
from fedgb.algorithms.standard_fl.pfed1bs.pfed1bs_config import config
from fedgb.algorithms.standard_fl.pfed1bs.utils import (
    alignment_gradients,
    one_bit_random_sketch,
    reset_model_with_seed,
)


class PFed1BSClient(BaseClient):
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(PFed1BSClient, self).__init__(
            args, client_id, data, data_dir, message_pool, device, personalized=True
        )
        self._target_sketch = None
        self._sketch_meta = None
        reset_model_with_seed(self.task.model, config["pfed1bs_seed"])

    def execute(self):
        server_payload = self.message_pool["server"]
        self._target_sketch = server_payload["sketch"].to(self.device)
        self._sketch_meta = server_payload["sketch_meta"]

        previous_step_preprocess = self.task.step_preprocess
        self.task.step_preprocess = self.step_preprocess
        try:
            self.task.train()
        finally:
            self.task.step_preprocess = previous_step_preprocess

    def step_preprocess(self):
        if self._target_sketch is None or self._sketch_meta is None:
            return

        params = [param for param in self.task.model.parameters() if param.grad is not None]
        if not params:
            return

        align_grads = alignment_gradients(
            params,
            self._target_sketch,
            self._sketch_meta,
            rho=config["pfed1bs_rho"],
            use_hadamard=config["pfed1bs_hadamard"],
        )
        with torch.no_grad():
            for param, align_grad in zip(params, align_grads):
                param.grad.data.mul_(1.0 + config["pfed1bs_mu"])
                param.grad.data.add_(
                    align_grad.to(param.grad.device),
                    alpha=config["pfed1bs_sign_loss_weight"],
                )

    def send_message(self):
        sketch, sketch_meta = one_bit_random_sketch(
            list(self.task.model.parameters()),
            compression_ratio=config["pfed1bs_compression_ratio"],
            seed=config["pfed1bs_seed"],
            use_hadamard=config["pfed1bs_hadamard"],
        )
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "sketch": sketch.detach().clone(),
            "sketch_meta": sketch_meta,
        }
