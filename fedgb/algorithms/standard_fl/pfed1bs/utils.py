import math
from dataclasses import dataclass

import torch


@dataclass
class SketchMeta:
    original_dim: int
    padded_dim: int
    compressed_dim: int
    compression_ratio: float
    seed: int
    use_hadamard: bool


def flatten_parameters(parameters):
    return torch.cat([param.reshape(-1) for param in parameters], dim=0)


def reset_model_with_seed(model, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for module in model.modules():
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()


def unflatten_like(flat, parameters):
    tensors = []
    offset = 0
    for param in parameters:
        next_offset = offset + param.numel()
        tensors.append(flat[offset:next_offset].view_as(param))
        offset = next_offset
    return tensors


def _compressed_dim(original_dim, compression_ratio):
    return max(1, int(original_dim * compression_ratio))


def _padded_dim(original_dim):
    return 1 << (original_dim - 1).bit_length()


def _hadamard_transform_1d(x):
    n = x.numel()
    h = 1
    y = x.reshape(1, n)
    while h < n:
        y = (
            y.reshape(-1, h * 2)
            .reshape(-1, 2, h)
            .transpose(0, 1)
            .contiguous()
        )
        y = torch.stack((y[0] + y[1], y[0] - y[1]), dim=1)
        y = y.reshape(-1, h * 2)
        h *= 2
    return y.reshape(-1)


def _hadamard_indices(meta, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(meta.seed))
    return torch.randperm(meta.padded_dim, generator=generator)[: meta.compressed_dim].to(device)


def _gaussian_projection_matrix(meta, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(meta.seed))
    matrix = torch.randn(
        meta.compressed_dim,
        meta.original_dim,
        generator=generator,
        device="cpu",
        dtype=dtype,
    )
    matrix = matrix.to(device)
    return matrix / math.sqrt(meta.compressed_dim)


def project_flat(flat, compression_ratio=0.1, seed=42, use_hadamard=True):
    original_dim = flat.numel()
    compressed_dim = _compressed_dim(original_dim, compression_ratio)
    padded_dim = _padded_dim(original_dim) if use_hadamard else original_dim
    meta = SketchMeta(
        original_dim=original_dim,
        padded_dim=padded_dim,
        compressed_dim=compressed_dim,
        compression_ratio=float(compression_ratio),
        seed=int(seed),
        use_hadamard=bool(use_hadamard),
    )

    if use_hadamard:
        padded = flat
        if padded_dim > original_dim:
            padded = torch.cat(
                [flat, flat.new_zeros(padded_dim - original_dim)],
                dim=0,
            )
        transformed = _hadamard_transform_1d(padded) / math.sqrt(compressed_dim)
        return transformed[_hadamard_indices(meta, flat.device)], meta

    matrix = _gaussian_projection_matrix(meta, flat.device, flat.dtype)
    return torch.matmul(matrix, flat), meta


def inverse_project(residual, meta):
    residual = residual.reshape(-1)
    if meta.use_hadamard:
        full = residual.new_zeros(meta.padded_dim)
        full[_hadamard_indices(meta, residual.device)] = residual
        recovered = _hadamard_transform_1d(full) / math.sqrt(meta.compressed_dim)
        return recovered[: meta.original_dim]

    matrix = _gaussian_projection_matrix(meta, residual.device, residual.dtype)
    return torch.matmul(matrix.t(), residual)


def one_bit_random_sketch(
    parameters,
    compression_ratio=0.1,
    seed=42,
    use_hadamard=True,
):
    flat = flatten_parameters(parameters)
    projection, meta = project_flat(
        flat,
        compression_ratio=compression_ratio,
        seed=seed,
        use_hadamard=use_hadamard,
    )
    return torch.sign(projection), meta


def aggregate_one_bit_sketches(sketches, sample_counts):
    if len(sketches) == 0:
        raise ValueError("pFed1BS needs at least one client sketch to aggregate.")

    total = float(sum(sample_counts))
    if total <= 0:
        weights = [1.0 / len(sketches)] * len(sketches)
    else:
        weights = [float(count) / total for count in sample_counts]

    aggregate = None
    for sketch, weight in zip(sketches, weights):
        value = sketch.detach().float() * weight
        aggregate = value.clone() if aggregate is None else aggregate + value
    return torch.sign(aggregate)


def alignment_gradients(parameters, target_sketch, meta, rho=6e-5, use_hadamard=None):
    if use_hadamard is not None and bool(use_hadamard) != bool(meta.use_hadamard):
        meta = SketchMeta(
            original_dim=meta.original_dim,
            padded_dim=meta.padded_dim,
            compressed_dim=meta.compressed_dim,
            compression_ratio=meta.compression_ratio,
            seed=meta.seed,
            use_hadamard=bool(use_hadamard),
        )

    flat = flatten_parameters(parameters)
    projection, _ = project_flat(
        flat,
        compression_ratio=meta.compression_ratio,
        seed=meta.seed,
        use_hadamard=meta.use_hadamard,
    )
    residual = torch.tanh(projection / rho) - target_sketch.to(projection.device)
    flat_gradient = inverse_project(residual, meta)
    return unflatten_like(flat_gradient, parameters)
