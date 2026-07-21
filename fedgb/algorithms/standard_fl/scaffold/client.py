import torch
from fedgb.training.base import BaseClient


class ScaffoldClient(BaseClient):
    """
    Client-side SCAFFOLD with optional algorithm-specific control-variate scaling.
    Defaults reproduce the original implementation:
    correction_scale=1.0, control_update_scale=1.0.
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(ScaffoldClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.local_control = [torch.zeros_like(p.data, requires_grad=False) for p in self.task.model.parameters()]

    def execute(self):
        with torch.no_grad():
            for local_param, global_param in zip(self.task.model.parameters(), self.message_pool["server"]["weight"]):
                local_param.data.copy_(global_param)

        previous_step_preprocess = self.task.step_preprocess
        self.task.step_preprocess = self.step_preprocess
        try:
            self.task.train()
        finally:
            self.task.step_preprocess = previous_step_preprocess

        self.update_local_control()

    def step_preprocess(self):
        correction_scale = float(getattr(self.args, "scaffold_correction_scale", 1.0))
        if correction_scale == 0.0:
            return
        for p, local_control, global_control in zip(
            self.task.model.parameters(),
            self.local_control,
            self.message_pool["server"]["global_control"],
        ):
            if p.grad is None:
                continue
            p.grad.data += correction_scale * (global_control - local_control)

    def update_local_control(self):
        control_update_scale = float(getattr(self.args, "scaffold_control_update_scale", 1.0))
        control_clip = float(getattr(self.args, "scaffold_control_clip", 0.0))
        denom = max(float(self.args.num_epochs) * float(self.args.lr), 1e-12)
        with torch.no_grad():
            for it, (local_state, global_state, global_control) in enumerate(
                zip(self.task.model.parameters(), self.message_pool["server"]["weight"], self.message_pool["server"]["global_control"])
            ):
                updated = self.local_control[it].data - global_control.data
                updated = updated + control_update_scale * (global_state.data - local_state.data) / denom
                if control_clip > 0.0:
                    updated = torch.clamp(updated, min=-control_clip, max=control_clip)
                self.local_control[it].data.copy_(updated)

    def send_message(self):
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "local_control": self.local_control,
        }
