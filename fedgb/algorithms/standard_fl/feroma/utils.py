import numpy as np
import torch


def latent_descriptor(embedding, labels, num_classes, include_std=True):
    embedding = embedding.detach().float().cpu()
    labels = labels.detach().long().cpu()
    parts = [embedding.mean(dim=0)]
    if include_std:
        parts.append(embedding.std(dim=0, unbiased=False))

    for class_id in range(num_classes):
        selected = embedding[labels == class_id]
        if selected.shape[0] > 1:
            parts.append(selected.mean(dim=0))
            if include_std:
                parts.append(selected.std(dim=0, unbiased=False))
        else:
            missing = torch.full((embedding.shape[1],), -1.0)
            parts.append(missing)
            if include_std:
                parts.append(missing.clone())
    return torch.cat(parts)


def regression_latent_descriptor(embedding, include_std=True):
    embedding = embedding.detach().float().cpu()
    parts = [embedding.mean(dim=0)]
    if include_std:
        parts.append(embedding.std(dim=0, unbiased=False))
    return torch.cat(parts)


class GroupwiseMinMaxScaler:
    def __init__(self, group_dims=None, eps=1e-12):
        self.group_dims = group_dims
        self.eps = eps
        self.fitted = False
        self.group_min = []
        self.group_max = []

    def scale(self, descriptors):
        descriptors = np.asarray(descriptors, dtype=np.float64)
        if descriptors.ndim == 1:
            descriptors = descriptors.reshape(1, -1)
        group_dims = self.group_dims or [descriptors.shape[1]]
        if not self.fitted:
            self.group_min = []
            self.group_max = []
            start = 0
            for dim in group_dims:
                end = start + dim
                values = descriptors[:, start:end].reshape(-1)
                self.group_min.append(float(values.min()))
                self.group_max.append(float(values.max()))
                start = end
            self.fitted = True

        scaled = np.zeros_like(descriptors)
        start = 0
        for dim, min_value, max_value in zip(group_dims, self.group_min, self.group_max):
            end = start + dim
            denom = max(max_value - min_value, self.eps)
            scaled[:, start:end] = (descriptors[:, start:end] - min_value) / denom
            start = end
        return scaled


def _cosine_distance(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def profile_distance_weights(current_descriptors, parent_descriptors=None, distance="euclidean"):
    current = np.asarray(current_descriptors, dtype=np.float64)
    if current.ndim == 1:
        current = current.reshape(1, -1)
    if parent_descriptors is None:
        return np.full((current.shape[0], current.shape[0]), 1.0 / current.shape[0], dtype=np.float64)

    parents = np.asarray(parent_descriptors, dtype=np.float64)
    if parents.ndim == 1:
        parents = parents.reshape(1, -1)
    rows = []
    for cur in current:
        distances = []
        for parent in parents:
            if distance == "euclidean":
                dist = float(np.linalg.norm(cur - parent))
            elif distance == "cosine":
                dist = _cosine_distance(cur, parent)
            else:
                raise ValueError(f"Unsupported FEROMA distance: {distance}")
            distances.append(dist)
        inverse = 1.0 / (np.asarray(distances, dtype=np.float64) + 1e-8)
        rows.append(inverse / inverse.sum())
    return np.vstack(rows)


def personalized_weighted_aggregate(client_params, sample_counts, distance_weights, device):
    personalized = []
    sample_counts = np.asarray(sample_counts, dtype=np.float64)
    distance_weights = np.asarray(distance_weights, dtype=np.float64)
    for row in distance_weights:
        effective = row * sample_counts
        denom = effective.sum()
        if denom <= 0:
            effective = np.ones_like(effective) / len(effective)
        else:
            effective = effective / denom
        model_params = []
        for param_idx in range(len(client_params[0])):
            value = None
            for params, weight in zip(client_params, effective):
                local = params[param_idx].detach().to(device)
                value = local * float(weight) if value is None else value + local * float(weight)
            model_params.append(value.detach().clone())
        personalized.append(model_params)
    return personalized


def weighted_average(client_params, sample_counts, device):
    weights = np.ones((1, len(client_params)), dtype=np.float64)
    return personalized_weighted_aggregate(client_params, sample_counts, weights, device)[0]
