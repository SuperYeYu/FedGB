import torch
from fedgb.training.base import BaseServer
from fedgb.algorithms.standard_fl.pfed1bs.pfed1bs_config import config
from fedgb.algorithms.standard_fl.pfed1bs.utils import (
    aggregate_one_bit_sketches,
    one_bit_random_sketch,
    reset_model_with_seed,
)


class PFed1BSServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(PFed1BSServer, self).__init__(
            args, global_data, data_dir, message_pool, device, personalized=True
        )
        self.global_sketch = None
        self.sketch_meta = None
        reset_model_with_seed(self.task.model, config["pfed1bs_seed"])

    def execute(self):
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            sketches = [
                self.message_pool[f"client_{cid}"]["sketch"].to(self.device)
                for cid in sampled
            ]
            sample_counts = [
                self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled
            ]
            self.global_sketch = aggregate_one_bit_sketches(sketches, sample_counts)
            self.sketch_meta = self.message_pool[f"client_{sampled[0]}"]["sketch_meta"]

    def send_message(self):
        if self.global_sketch is None:
            self.global_sketch, self.sketch_meta = one_bit_random_sketch(
                list(self.task.model.parameters()),
                compression_ratio=config["pfed1bs_compression_ratio"],
                seed=config["pfed1bs_seed"],
                use_hadamard=config["pfed1bs_hadamard"],
            )

        self.message_pool["server"] = {
            "sketch": self.global_sketch.detach().clone(),
            "sketch_meta": self.sketch_meta,
        }
