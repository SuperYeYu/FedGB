import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseInputLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SparseInputLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias


class FedIIHDisenConv(nn.Module):
    def __init__(self, num_factors, num_iterations, tau=1.0, edge_chunk_size=200000):
        super(FedIIHDisenConv, self).__init__()
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.tau = tau
        self.edge_chunk_size = int(edge_chunk_size)

    def forward(self, x, edge_index):
        num_nodes, dim = x.shape
        num_edges = edge_index.shape[1]
        factor_dim = dim // self.num_factors
        src, trg = edge_index[0], edge_index[1]
        x = F.normalize(
            x.view(num_nodes, self.num_factors, factor_dim),
            dim=2,
        ).view(num_nodes, dim)
        u = x
        scatter_idx = trg.view(num_edges, 1).expand(num_edges, dim)

        for _ in range(self.num_iterations):
            u = torch.zeros(num_nodes, dim, device=x.device, dtype=x.dtype)
            for start in range(0, num_edges, self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, num_edges)
                src_chunk = src[start:end]
                trg_chunk = trg[start:end]
                z = x[src_chunk].view(end - start, self.num_factors, factor_dim)
                p = (z * u[trg_chunk].view(end - start, self.num_factors, factor_dim)).sum(dim=2)
                p = F.softmax(p / self.tau, dim=1)
                scatter_src = (z * p.view(end - start, self.num_factors, 1)).view(end - start, dim)
                u.scatter_add_(0, scatter_idx[start:end], scatter_src)
            u = F.normalize(
                (u + x).view(num_nodes, self.num_factors, factor_dim),
                dim=2,
            ).view(num_nodes, dim)
        return u


class FedIIHDisentangledGNN(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super(FedIIHDisentangledGNN, self).__init__()
        self.num_factors = getattr(args, "n_factors", 2)
        self.factor_dim = getattr(args, "n_latentdims", getattr(args, "hid_dim", 128))
        self.routing_iterations = getattr(args, "n_routit", 6)
        self.dropout = getattr(args, "dropout", 0.3)
        self.edge_chunk_size = getattr(args, "edge_chunk_size", 200000)
        num_layers = getattr(args, "n_layers", 4)
        if num_layers <= 2:
            self.num_base_layers = 1
            self.num_hvae_layers = 1
        else:
            self.num_base_layers = num_layers - 2
            self.num_hvae_layers = 2

        hidden_dim = self.num_factors * self.factor_dim
        self.pca = SparseInputLinear(input_dim, hidden_dim)
        self.base_gnn_ls = nn.ModuleList(
            [
                FedIIHDisenConv(self.num_factors, self.routing_iterations, edge_chunk_size=self.edge_chunk_size)
                for _ in range(self.num_base_layers)
            ]
        )
        self.gnn_mean = nn.ModuleList(
            [
                FedIIHDisenConv(self.num_factors, self.routing_iterations, edge_chunk_size=self.edge_chunk_size)
                for _ in range(self.num_hvae_layers)
            ]
        )
        self.gnn_logstddev = nn.ModuleList(
            [
                FedIIHDisenConv(self.num_factors, self.routing_iterations, edge_chunk_size=self.edge_chunk_size)
                for _ in range(self.num_hvae_layers)
            ]
        )
        self.clf = nn.Linear(hidden_dim, output_dim)

    def _dropout(self, x):
        return F.dropout(x, self.dropout, training=self.training)

    def encode_embedding(self, data):
        x, edge_index = data.x, data.edge_index
        x = self._dropout(F.leaky_relu(self.pca(x)))
        for conv in self.base_gnn_ls:
            x = self._dropout(F.leaky_relu(conv(x, edge_index)))
        return x

    def forward(self, data):
        embedding = self.encode_embedding(data)
        logits = self.clf(embedding)
        return embedding, logits

    def encode_for_hvae(self, data):
        x = self.encode_embedding(data)

        mean = x
        for conv in self.gnn_mean:
            mean = self._dropout(F.leaky_relu(conv(mean, data.edge_index)))
        z_mu_n, z_mu_e = torch.chunk(mean, chunks=self.num_factors, dim=1)

        logstd = x
        for conv in self.gnn_logstddev:
            logstd = self._dropout(F.leaky_relu(conv(logstd, data.edge_index)))
        z_logvar_n, z_logvar_e = torch.chunk(logstd, chunks=self.num_factors, dim=1)

        return z_mu_n, z_logvar_n, z_mu_e, z_logvar_e

    def encode_for_HVAE(self, data):
        return self.encode_for_hvae(data)


class FedIIHHVAE(nn.Module):
    def __init__(self, hidden_dim, args):
        super(FedIIHHVAE, self).__init__()
        self.latent_factor_dims = hidden_dim
        self.dropout = getattr(args, "dropout", 0.3)
        self.mu_Alpha = nn.Parameter(torch.empty(hidden_dim), requires_grad=True)
        self.mu_Beta = nn.Parameter(torch.empty(hidden_dim), requires_grad=True)
        nn.init.normal_(self.mu_Alpha)
        nn.init.normal_(self.mu_Beta)
        self.edge_bce = nn.BCEWithLogitsLoss()

    def sampling(self, z_mu_n, z_logvar_n, z_mu_e, z_logvar_e):
        noise_n = torch.randn_like(z_mu_n)
        noise_e = torch.randn_like(z_mu_e)
        z_n = z_mu_n + noise_n * torch.exp(0.5 * z_logvar_n)
        z_e = z_mu_e + noise_e * torch.exp(0.5 * z_logvar_e)
        return z_n, z_e

    def decode(self, z_n, z_e):
        z = torch.cat((z_n, z_e), dim=1)
        return torch.matmul(z, z.t())

    def decode_edges(self, z_n, z_e, batch, negative_ratio=1.0, max_edges=200000):
        z = torch.cat((z_n, z_e), dim=1)
        edge_index = torch.unique(batch.edge_index, dim=1)
        num_pos = edge_index.shape[1]
        if num_pos > max_edges:
            pos_perm = torch.randperm(num_pos, device=edge_index.device)[:max_edges]
            edge_index = edge_index[:, pos_perm]
            num_pos = edge_index.shape[1]

        num_neg = min(int(num_pos * negative_ratio), max_edges)
        neg_src = torch.randint(batch.num_nodes, (num_neg,), device=edge_index.device)
        neg_dst = torch.randint(batch.num_nodes, (num_neg,), device=edge_index.device)
        neg_mask = neg_src != neg_dst
        neg_src = neg_src[neg_mask]
        neg_dst = neg_dst[neg_mask]
        neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)

        sampled_edge_index = torch.cat([edge_index, neg_edge_index], dim=1)
        edge_logits = (z[sampled_edge_index[0]] * z[sampled_edge_index[1]]).sum(dim=1)
        edge_labels = torch.cat(
            [
                torch.ones(edge_index.shape[1], device=edge_logits.device, dtype=edge_logits.dtype),
                torch.zeros(neg_edge_index.shape[1], device=edge_logits.device, dtype=edge_logits.dtype),
            ],
            dim=0,
        )
        return edge_logits, edge_labels

    def kld(self, mu, logvar, q_mu, q_logvar):
        return -0.5 * (
            1
            + logvar
            - q_logvar
            - ((mu - q_mu).pow(2) + torch.exp(logvar)) / torch.exp(q_logvar)
        )

    def log_normal(self, x):
        return -0.5 * (math.log(2 * math.pi) + x.pow(2))

    def loss_function(
        self,
        batch,
        z_mu_n,
        z_logvar_n,
        z_mu_e,
        z_logvar_e,
        alpha_mu,
        beta_mu,
        edge_logits,
        edge_labels=None,
    ):
        device = edge_logits.device
        dtype = edge_logits.dtype
        alpha_mu = alpha_mu.to(device=device, dtype=dtype)
        beta_mu = beta_mu.to(device=device, dtype=dtype)
        alpha_logvar = torch.ones_like(alpha_mu) * math.log(0.5 ** 2)
        beta_logvar = torch.ones_like(beta_mu) * math.log(0.5 ** 2)

        if edge_labels is None:
            adj = torch.zeros((batch.num_nodes, batch.num_nodes), dtype=dtype, device=device)
            if batch.edge_index.numel() > 0:
                src, dst = batch.edge_index[0], batch.edge_index[1]
                adj[src, dst] = 1.0
                adj[dst, src] = 1.0
            recon_loss = self.edge_bce(edge_logits, adj)
        else:
            recon_loss = F.binary_cross_entropy_with_logits(
                edge_logits,
                edge_labels.to(device=device, dtype=dtype),
            )
        kl_structure = self.kld(z_mu_e, z_logvar_e, self.mu_Alpha, alpha_logvar).mean()
        kl_semantic = self.kld(z_mu_n, z_logvar_n, self.mu_Beta, beta_logvar).mean()
        prior_alpha = self.log_normal(self.mu_Alpha).mean()
        prior_beta = self.log_normal(self.mu_Beta).mean()
        extra_kl_alpha = self.kld(self.mu_Alpha, alpha_logvar, alpha_mu, alpha_logvar).mean()
        extra_kl_beta = self.kld(self.mu_Beta, beta_logvar, beta_mu, beta_logvar).mean()
        return recon_loss + kl_structure + kl_semantic + extra_kl_alpha + extra_kl_beta + prior_alpha + prior_beta
