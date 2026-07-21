import torch

from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.cufl.cufl_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.cufl.models import build_cufl_model
from fedgb.algorithms.subgraph_fgl.homogeneous.cufl.utils import (
    aggregate_state_dicts,
    build_proxy_data,
    build_similarity_matrix,
    state_dict_to_parameter_list,
)


def _cfg(args, name):
    return getattr(args, name, config[name])


class CUFLServer(BaseServer):
    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(CUFLServer, self).__init__(args, global_data, data_dir, message_pool, device, personalized=True)
        self._apply_cufl_defaults()
        self.task.load_custom_model(build_cufl_model(self.args, self.task))
        self.proxy_data = build_proxy_data(
            num_features=self.task.data.x.size(1),
            num_proxy=_cfg(args, "cufl_proxy_num"),
            num_nodes=_cfg(args, "cufl_proxy_num_nodes"),
            p_in=_cfg(args, "cufl_proxy_p_in"),
            p_out=_cfg(args, "cufl_proxy_p_out"),
            seed=getattr(args, "seed", 0),
            device=device,
        )
        self._personalized = {}

    def _apply_cufl_defaults(self):
        for name, value in config.items():
            if not hasattr(self.args, name):
                setattr(self.args, name, value)

    def _global_state(self):
        return {name: tensor.detach().clone() for name, tensor in self.task.model.state_dict().items()}

    def execute(self):
        with torch.no_grad():
            sampled = self.message_pool["sampled_clients"]
            local_states = [
                self.message_pool[f"client_{cid}"].get("state_dict")
                for cid in sampled
            ]
            if any(state is None for state in local_states):
                local_states = [
                    {
                        name: param.detach().clone()
                        for name, param in zip(
                            self.task.model.state_dict().keys(),
                            self.message_pool[f"client_{cid}"]["weight"],
                        )
                    }
                    for cid in sampled
                ]

            train_sizes = [self.message_pool[f"client_{cid}"]["num_samples"] for cid in sampled]
            ratio = torch.tensor(train_sizes, dtype=torch.float)
            ratio = (ratio / ratio.sum()).cpu().numpy()

            global_state = aggregate_state_dicts(
                local_states,
                ratios=ratio,
                client_id=-1,
                aggregate_classifier=True,
                mask_aggr=_cfg(self.args, "cufl_mask_aggr"),
                l1=_cfg(self.args, "cufl_l1"),
                device=self.device,
            )
            self.task.model.load_state_dict(global_state, strict=False)

            local_recon = [
                self.message_pool[f"client_{cid}"].get("proxy_recon_edge", [1.0])
                for cid in sampled
            ]
            scales = [
                self.message_pool[f"client_{cid}"].get("scale", _cfg(self.args, "cufl_sim_scale"))
                for cid in sampled
            ]
            sim = build_similarity_matrix(
                local_recon,
                scales=scales,
                metric=_cfg(self.args, "cufl_agg_metric"),
                filter_below_mean=_cfg(self.args, "cufl_filter_below_mean"),
            )

            self._personalized = {}
            for row_idx, cid in enumerate(sampled):
                personalized_state = aggregate_state_dicts(
                    local_states,
                    ratios=sim[row_idx, :],
                    client_id=row_idx,
                    aggregate_classifier=_cfg(self.args, "cufl_aggregate_classifier"),
                    mask_aggr=_cfg(self.args, "cufl_mask_aggr"),
                    l1=_cfg(self.args, "cufl_l1"),
                    device=self.device,
                )
                self._personalized[cid] = personalized_state

    def send_message(self):
        global_state = self._global_state()
        msg = {
            "weight": state_dict_to_parameter_list(self.task.model, global_state, self.device),
            "state_dict": global_state,
            "proxy_data": self.proxy_data,
        }
        for cid, personalized_state in self._personalized.items():
            msg[f"personalized_{cid}"] = personalized_state
        self.message_pool["server"] = msg
