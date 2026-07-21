import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import spmm


def scalar_edge_weight(data):
    edge_weight = getattr(data, "edge_weight", None)
    if edge_weight is None:
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_weight = edge_attr
    return edge_weight


class NormalizedPropagation(MessagePassing):
    def __init__(self, add_self_loops=True):
        super(NormalizedPropagation, self).__init__(aggr="add")
        self.add_self_loops = add_self_loops

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                x.size(self.node_dim),
                False,
                self.add_self_loops,
                self.flow,
                dtype=x.dtype,
            )
        else:
            edge_index = gcn_norm(
                edge_index,
                edge_weight,
                x.size(self.node_dim),
                False,
                self.add_self_loops,
                self.flow,
                dtype=x.dtype,
            )
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class GPRPropagation(nn.Module):
    def __init__(self, k=10, alpha=0.1, init_method="PPR", gamma=None):
        super(GPRPropagation, self).__init__()
        self.k = int(k)
        self.propagation = NormalizedPropagation()
        self.gamma = nn.Parameter(self._init_gamma(init_method, alpha, gamma))

    def _init_gamma(self, init_method, alpha, gamma):
        init_method = str(init_method).upper()
        if init_method == "SGC":
            values = torch.zeros(self.k + 1)
            values[min(max(int(alpha), 0), self.k)] = 1.0
        elif init_method == "PPR":
            values = torch.tensor(
                [alpha * (1 - alpha) ** step for step in range(self.k + 1)],
                dtype=torch.float32,
            )
            values[-1] = (1 - alpha) ** self.k
        elif init_method == "NPPR":
            values = torch.tensor([alpha ** step for step in range(self.k + 1)], dtype=torch.float32)
            values = values / values.abs().sum().clamp_min(1e-12)
        elif init_method == "RANDOM":
            bound = math.sqrt(3.0 / (self.k + 1))
            values = torch.empty(self.k + 1).uniform_(-bound, bound)
            values = values / values.abs().sum().clamp_min(1e-12)
        elif init_method == "WS" and gamma is not None:
            values = torch.as_tensor(gamma, dtype=torch.float32)
        else:
            raise ValueError("Unsupported GPRGNN init_method '{}'.".format(init_method))
        return values

    def forward(self, x, edge_index, edge_weight=None):
        propagated = x
        out = self.gamma[0] * propagated
        for step in range(1, self.k + 1):
            propagated = self.propagation(propagated, edge_index, edge_weight)
            out = out + self.gamma[step] * propagated
        return out


class BernPropagation(nn.Module):
    def __init__(self, k=10):
        super(BernPropagation, self).__init__()
        self.k = int(k)
        self.propagation = NormalizedPropagation()
        self.filter_param = nn.Parameter(torch.ones(self.k + 1, 1))
        coeff = [math.comb(self.k, i) / (2 ** self.k) for i in range(self.k + 1)]
        self.register_buffer("bern_coeff", torch.tensor(coeff, dtype=torch.float32).view(-1, 1))

    def _laplacian_step(self, x, edge_index, edge_weight):
        return x - self.propagation(x, edge_index, edge_weight)

    def _poly_step(self, x, edge_index, edge_weight):
        return x + self.propagation(x, edge_index, edge_weight)

    def forward(self, x, edge_index, edge_weight=None):
        first_poly = [x]
        for _ in range(self.k):
            first_poly.append(self._laplacian_step(first_poly[-1], edge_index, edge_weight))

        filter_param = F.relu(self.filter_param)
        coeff = self.bern_coeff.to(device=x.device, dtype=x.dtype)
        out = torch.zeros_like(x)
        for i in range(self.k + 1):
            poly = first_poly[self.k - i]
            for _ in range(i):
                poly = self._poly_step(poly, edge_index, edge_weight)
            out = out + coeff[i] * filter_param[i] * poly
        return out


class ChebIIPropagation(nn.Module):
    def __init__(self, k=10):
        super(ChebIIPropagation, self).__init__()
        self.k = int(k)
        self.propagation = NormalizedPropagation()
        self.filter_param = nn.Parameter(torch.ones(self.k + 1, 1))
        self.register_buffer("chebynodes_vals", self._build_chebynodes_vals())

    def _build_chebynodes_vals(self):
        columns = []
        for j in range(self.k + 1):
            x_j = math.cos((self.k - j + 0.5) * math.pi / (self.k + 1))
            values = []
            for order in range(self.k + 1):
                if order == 0:
                    values.append(1.0)
                elif order == 1:
                    values.append(x_j)
                else:
                    values.append(2 * x_j * values[order - 1] - values[order - 2])
            columns.append(values)
        return torch.tensor(columns, dtype=torch.float32).t().contiguous()

    def _scaled_laplacian_step(self, x, edge_index, edge_weight):
        return -self.propagation(x, edge_index, edge_weight)

    def forward(self, x, edge_index, edge_weight=None):
        filter_param = F.relu(self.filter_param)
        cheb_values = self.chebynodes_vals.to(device=x.device, dtype=x.dtype)
        coeff = cheb_values @ filter_param
        coeff = 2.0 * coeff / (self.k + 1)
        coeff = coeff.clone()
        coeff[0] = coeff[0] / 2.0

        polys = [x]
        if self.k >= 1:
            polys.append(self._scaled_laplacian_step(x, edge_index, edge_weight))
        for order in range(2, self.k + 1):
            polys.append(
                2.0 * self._scaled_laplacian_step(polys[order - 1], edge_index, edge_weight)
                - polys[order - 2]
            )

        out = torch.zeros_like(x)
        for order in range(self.k + 1):
            out = out + coeff[order] * polys[order]
        return out
