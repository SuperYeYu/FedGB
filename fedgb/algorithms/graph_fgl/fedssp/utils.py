import hashlib
import os
from pathlib import Path

import torch
from torch_geometric.utils import add_self_loops, to_dense_adj


_LAPLACIAN_EIGH_CACHE = {}
_LAPLACIAN_EIGH_CACHE_STATS = {"hit": 0, "disk_hit": 0, "miss": 0}


def clear_laplacian_eigh_cache():
    _LAPLACIAN_EIGH_CACHE.clear()
    _LAPLACIAN_EIGH_CACHE_STATS["hit"] = 0
    _LAPLACIAN_EIGH_CACHE_STATS["disk_hit"] = 0
    _LAPLACIAN_EIGH_CACHE_STATS["miss"] = 0


def laplacian_eigh_cache_stats():
    return {
        "hit": _LAPLACIAN_EIGH_CACHE_STATS["hit"],
        "disk_hit": _LAPLACIAN_EIGH_CACHE_STATS["disk_hit"],
        "miss": _LAPLACIAN_EIGH_CACHE_STATS["miss"],
        "size": len(_LAPLACIAN_EIGH_CACHE),
    }


def compute_laplacian_eigh_batch(batch):
    data_list = batch.to_data_list()
    valid_graphs = [data for data in data_list if data.num_nodes > 0]
    if not valid_graphs:
        device = batch.x.device
        return (
            torch.zeros((0, 0), device=device),
            torch.zeros((0, 0, 0), device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )

    device = batch.x.device
    max_nodes = max(data.num_nodes for data in valid_graphs)
    eigenvalues = []
    eigenvectors = []
    lengths = []

    for data in valid_graphs:
        num_nodes = data.num_nodes
        cache_key = _graph_cache_key(data)
        cached = _LAPLACIAN_EIGH_CACHE.get(cache_key) if cache_key is not None else None
        if cached is not None:
            _LAPLACIAN_EIGH_CACHE_STATS["hit"] += 1
            eig_value, eig_vector = (item.to(device) for item in cached)
        else:
            disk_cached = _load_disk_laplacian_eigh(cache_key)
            if disk_cached is not None:
                _LAPLACIAN_EIGH_CACHE_STATS["disk_hit"] += 1
                _LAPLACIAN_EIGH_CACHE[cache_key] = disk_cached
                eig_value, eig_vector = (item.to(device) for item in disk_cached)
            else:
                _LAPLACIAN_EIGH_CACHE_STATS["miss"] += 1
                eig_value, eig_vector = compute_laplacian_eigh_single(data, device=device)
                cpu_cached = (eig_value.detach().cpu(), eig_vector.detach().cpu())
                if cache_key is not None:
                    _LAPLACIAN_EIGH_CACHE[cache_key] = cpu_cached
                    _save_disk_laplacian_eigh(cache_key, cpu_cached)

        padded_values = eig_value.new_zeros(max_nodes)
        padded_vectors = eig_vector.new_zeros(max_nodes, max_nodes)
        padded_values[:num_nodes] = eig_value
        padded_vectors[:num_nodes, :num_nodes] = eig_vector

        eigenvalues.append(padded_values)
        eigenvectors.append(padded_vectors)
        lengths.append(num_nodes)

    return (
        torch.stack(eigenvalues, dim=0),
        torch.stack(eigenvectors, dim=0),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def _graph_cache_key(data):
    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.shape[1])
    key_mode = os.environ.get("FEDSSP_LAPLACIAN_CACHE_KEY", "hash").lower()
    if key_mode == "id" and hasattr(data, "fedssp_cache_id"):
        cache_id = data.fedssp_cache_id
        if torch.is_tensor(cache_id):
            if cache_id.numel() == 0:
                cache_id = None
            else:
                cache_id = int(cache_id.view(-1)[0].item())
        if cache_id is not None:
            return ("id", int(cache_id), num_nodes, num_edges)
    return ("hash", _graph_structure_hash(data), num_nodes, num_edges)


def _graph_structure_hash(data):
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.detach().cpu().to(torch.long).contiguous()
    if edge_index.numel() > 0:
        order_key = edge_index[0] * max(num_nodes, 1) + edge_index[1]
        edge_index = edge_index[:, torch.argsort(order_key)].contiguous()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(num_nodes).encode("utf-8"))
    digest.update(str(int(edge_index.shape[1])).encode("utf-8"))
    digest.update(edge_index.numpy().tobytes())
    return digest.hexdigest()


def configure_laplacian_eigh_disk_cache(cache_dir=None):
    if cache_dir is None:
        cache_dir = os.environ.get("FEDSSP_LAPLACIAN_CACHE_DIR")
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["FEDSSP_LAPLACIAN_CACHE_DIR"] = str(cache_path)
        return cache_path
    return None


def compute_laplacian_eigh_single(data, device=None):
    if device is None:
        device = data.x.device
    num_nodes = int(data.num_nodes)
    edge_index, _ = add_self_loops(data.edge_index.to(device), num_nodes=num_nodes)
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    degree = torch.diag(adj.sum(dim=1))
    laplacian = degree - adj
    return torch.linalg.eigh(laplacian)


def precompute_laplacian_eigh_for_graphs(graphs, cache_dir=None, force=False):
    cache_path = configure_laplacian_eigh_disk_cache(cache_dir)
    if cache_path is None:
        raise ValueError("cache_dir or FEDSSP_LAPLACIAN_CACHE_DIR is required")
    count = {"total": 0, "computed": 0, "skipped": 0, "missing_key": 0}
    for data in graphs:
        count["total"] += 1
        cache_key = _graph_cache_key(data)
        if cache_key is None:
            count["missing_key"] += 1
            continue
        path = _disk_cache_path(cache_key)
        if path is not None and path.exists() and not force:
            count["skipped"] += 1
            continue
        eig_value, eig_vector = compute_laplacian_eigh_single(data, device=torch.device("cpu"))
        cpu_cached = (eig_value.detach().cpu(), eig_vector.detach().cpu())
        _save_disk_laplacian_eigh(cache_key, cpu_cached)
        count["computed"] += 1
    return count


def _disk_cache_path(cache_key):
    cache_dir = os.environ.get("FEDSSP_LAPLACIAN_CACHE_DIR")
    if cache_dir is None or cache_key is None:
        return None
    key_type, key_value, num_nodes, num_edges = cache_key
    if key_type == "hash":
        key_value = str(key_value)
        return Path(cache_dir) / "hash" / key_value[:2] / f"h{key_value}_n{num_nodes}_e{num_edges}.pt"
    return Path(cache_dir) / "id" / f"g{key_value}_n{num_nodes}_e{num_edges}.pt"


def _load_disk_laplacian_eigh(cache_key):
    path = _disk_cache_path(cache_key)
    if path is None or not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["eigenvalues"], payload["eigenvectors"]


def _save_disk_laplacian_eigh(cache_key, cached):
    path = _disk_cache_path(cache_key)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save({"eigenvalues": cached[0], "eigenvectors": cached[1]}, tmp_path)
    os.replace(tmp_path, path)


def fedssp_shared_state_dict(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if _is_fedssp_shared_parameter(name)
    }


def load_fedssp_shared_state(model, shared_state):
    with torch.no_grad():
        named_params = dict(model.named_parameters())
        for name, value in shared_state.items():
            if name in named_params and _is_fedssp_shared_parameter(name):
                named_params[name].data.copy_(value.to(named_params[name].device))


def direct_average_state_dicts(state_dicts, device):
    if not state_dicts:
        return {}
    averaged = {}
    for name in state_dicts[0]:
        total = torch.zeros_like(state_dicts[0][name], device=device)
        for state in state_dicts:
            total = total + state[name].to(device)
        averaged[name] = total / len(state_dicts)
    return averaged


def _is_fedssp_shared_parameter(name):
    return ("eig_encoder" in name or "filter_encoder" in name) and "atom_encoder" not in name
