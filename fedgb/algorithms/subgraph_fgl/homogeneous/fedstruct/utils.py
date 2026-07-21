import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_undirected


def _fix_eigenvector_signs(eigvecs):
    signs = torch.sign(eigvecs.sum(dim=0))
    signs[signs == 0] = 1.0
    return eigvecs * signs


def _pad_basis(eigvecs, eigvals, k, device):
    take = eigvals.numel()
    eigvecs = eigvecs.float().to(device)
    eigvals = eigvals.float().to(device)
    if take < k:
        pad = k - take
        eigvecs = torch.cat(
            [eigvecs, torch.zeros(eigvecs.size(0), pad, dtype=eigvecs.dtype, device=device)],
            dim=1,
        )
        eigvals = torch.cat(
            [eigvals, torch.ones(pad, dtype=eigvals.dtype, device=device)],
            dim=0,
        )
    return eigvecs, eigvals


def _build_sparse_laplacian_eigenbasis(edge_index, num_nodes, k, device, normalization="sym"):
    try:
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
    except Exception as exc:
        raise RuntimeError("SciPy sparse eigensolver is required for large FedStruct graphs") from exc

    edge_index = edge_index.detach().cpu()
    if edge_index.numel() > 0:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(row.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    adj.data[:] = 1.0

    degree = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32)
    degree = np.maximum(degree, 1.0)
    if normalization == "normal":
        laplacian = sp.diags(degree, dtype=np.float32) - adj
    else:
        inv_sqrt = np.power(degree, -0.5, dtype=np.float32)
        norm_adj = sp.diags(inv_sqrt, dtype=np.float32) @ adj @ sp.diags(inv_sqrt, dtype=np.float32)
        laplacian = sp.eye(num_nodes, dtype=np.float32, format="csr") - norm_adj

    take = min(k, max(1, num_nodes - 2))
    if num_nodes <= 2:
        eigvals = np.zeros(1, dtype=np.float32)
        eigvecs = np.ones((num_nodes, 1), dtype=np.float32)
    else:
        ncv = min(num_nodes - 1, max(2 * take + 1, 96))
        maxiter = 1000 if num_nodes <= 50000 else 300
        try:
            eigvals, eigvecs = eigsh(
                laplacian,
                k=take,
                which="SM",
                tol=1e-2,
                maxiter=maxiter,
                ncv=ncv,
            )
        except Exception as exc:
            eigvals = getattr(exc, "eigenvalues", None)
            eigvecs = getattr(exc, "eigenvectors", None)
            if eigvals is None or eigvecs is None or len(eigvals) == 0:
                raise
    order = np.argsort(eigvals)
    eigvals = torch.from_numpy(eigvals[order].astype(np.float32))
    eigvecs = torch.from_numpy(eigvecs[:, order].astype(np.float32))
    eigvecs = _fix_eigenvector_signs(eigvecs)
    return _pad_basis(eigvecs, eigvals, k, device)


def build_laplacian_eigenbasis(edge_index, num_nodes, k, device, normalization="sym"):
    """Build the spectral basis used by FedStruct's trainable structural features."""
    k = max(1, int(k))
    if num_nodes > 5000:
        try:
            return _build_sparse_laplacian_eigenbasis(
                edge_index,
                num_nodes,
                k,
                device,
                normalization=normalization,
            )
        except Exception:
            random_basis = torch.randn(num_nodes, k, dtype=torch.float32)
            random_basis = torch.linalg.qr(random_basis, mode="reduced").Q
            eigvals = torch.linspace(0.0, 1.0, steps=k, dtype=torch.float32)
            return _pad_basis(random_basis, eigvals, k, device)

    edge_index = edge_index.to(device)
    if edge_index.numel() > 0:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1.0

    degree = adj.sum(dim=1).clamp_min(1.0)
    eye = torch.eye(num_nodes, dtype=torch.float32, device=device)
    if normalization == "normal":
        laplacian = torch.diag(degree) - adj
    else:
        degree_inv_sqrt = degree.pow(-0.5)
        norm_adj = degree_inv_sqrt[:, None] * adj * degree_inv_sqrt[None, :]
        laplacian = eye - norm_adj

    try:
        eigvals, eigvecs = torch.linalg.eigh(laplacian)
    except RuntimeError:
        try:
            cpu_laplacian = laplacian.detach().cpu().double()
            cpu_laplacian = 0.5 * (cpu_laplacian + cpu_laplacian.t())
            eigvals, eigvecs = torch.linalg.eigh(cpu_laplacian)
        except RuntimeError:
            random_basis = torch.randn(num_nodes, k, dtype=torch.float32)
            random_basis = torch.linalg.qr(random_basis, mode="reduced").Q
            eigvals = torch.linspace(0.0, 1.0, steps=k, dtype=torch.float32)
            return _pad_basis(random_basis, eigvals, k, device)

    order = torch.argsort(eigvals)
    take = min(k, num_nodes)
    eigvals = eigvals[order[:take]].float()
    eigvecs = eigvecs[:, order[:take]].float()

    eigvecs = _fix_eigenvector_signs(eigvecs)
    return _pad_basis(eigvecs, eigvals, k, device)


def spectral_sfv_regularizer(coefficients, eigvals, eps=1e-12):
    weighted = eigvals.to(coefficients.device).view(-1, 1) * coefficients
    numerator = torch.sum(coefficients * weighted)
    denominator = torch.sum(coefficients * coefficients).clamp_min(eps)
    return numerator / denominator


def fedstruct_loss(model, logits, labels, mask, regularizer_coef):
    ce_loss = F.cross_entropy(logits[mask], labels[mask])
    return ce_loss + regularizer_coef * model.spectral_regularizer()
