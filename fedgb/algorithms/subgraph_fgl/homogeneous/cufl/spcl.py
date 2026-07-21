import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset


class InnerProductDecoder(nn.Module):
    def forward(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)


class SPCL(nn.Module):
    def __init__(self, max_edges, num_edges, device, pd=0.1, beta=1.0):
        super().__init__()
        self.max_edges = max_edges
        self.num_edges = num_edges
        self._s_mask = nn.Parameter(torch.zeros(max_edges, dtype=torch.float))
        self.accumulated_s_mask = torch.zeros(num_edges, dtype=torch.float, device=device)
        self.graph_recon_degree = torch.zeros(num_edges, dtype=torch.float, device=device)
        self.decoder = InnerProductDecoder()
        self.epsilon = 1e-8
        self.num = self.epsilon
        self.pd = pd
        self.beta = beta
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.s_mask)
        reset(self.decoder)

    def get_state(self):
        return {
            "s_mask": self.state_dict(),
            "accumulated_s_mask": self.accumulated_s_mask.detach().clone(),
            "graph_recon_degree": self.graph_recon_degree.detach().clone(),
            "num": self.num,
            "max_edges": self.max_edges,
            "num_edges": self.num_edges,
        }

    def set_state(self, state, device):
        self.load_state_dict(
            {name: value.to(device) for name, value in state["s_mask"].items()},
            strict=False,
        )
        self.accumulated_s_mask = state["accumulated_s_mask"].to(device)
        self.graph_recon_degree = state["graph_recon_degree"].to(device)
        self.num = state["num"]
        self.max_edges = state["max_edges"]
        self.num_edges = state["num_edges"]

    def recon_loss(self, z, edge_index, pd=None, full_adj=None, loss_type="increase", beta=None):
        pd = self.pd if pd is None else pd
        beta = self.beta if beta is None else beta
        if full_adj is None:
            full_adj = torch.ones(edge_index.size(1), device=z.device)
        full_adj = full_adj.to(z.device)
        pred = self.decoder(z, edge_index).view(-1)
        if loss_type == "both":
            return (
                torch.mean(self.s_mask * pred)
                + beta * torch.mean(self.s_mask)
                + pd * torch.mean((self.s_mask - full_adj) ** 2)
            )
        return torch.sum(self.s_mask * (pred - full_adj) ** 2) - pd * torch.sum(self.s_mask)

    def train_step(self, optimizer, z, edge_index, pd=None, full_adj=None, loss_type="increase", beta=None):
        self.train()
        optimizer.zero_grad()
        loss = self.recon_loss(z, edge_index, pd, full_adj, loss_type, beta)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.s_mask.clamp_(min=0, max=1)
        return float(loss.detach().cpu())

    @torch.no_grad()
    def predict(self, edge_index, threshold=0.5, update_accumul=True):
        mask = self.s_mask > threshold
        masked_edge_index = edge_index[:, mask]
        if update_accumul:
            self.graph_recon_degree = self.s_mask.detach().float() + self.graph_recon_degree
            self.accumulated_s_mask = mask.detach().float() + self.accumulated_s_mask
            self.num += 1
        masked_edge_weight = self.accumulated_s_mask[mask] / self.num
        return masked_edge_index, masked_edge_weight

    @property
    def s_mask(self):
        return self._s_mask[:self.num_edges]
