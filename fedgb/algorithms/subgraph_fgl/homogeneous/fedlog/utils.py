import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree


def clone_state_dict(module):
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in module.state_dict().items())


def load_state_dict_to_module(module, state_dict, device):
    if state_dict is None:
        return
    module.load_state_dict(OrderedDict((key, value.to(device)) for key, value in state_dict.items()))


def average_state_dicts(payloads, device):
    total = sum(weight for _, weight in payloads)
    if total <= 0:
        total = len(payloads)
        payloads = [(state, 1.0) for state, _ in payloads]

    averaged = OrderedDict()
    for state_dict, weight in payloads:
        coeff = weight / total
        for key, value in state_dict.items():
            value = value.to(device)
            if key not in averaged:
                averaged[key] = coeff * value
            else:
                averaged[key] += coeff * value
    return averaged


def euclidean_dist(x, y):
    return torch.pow(x.unsqueeze(1) - y.unsqueeze(0), 2).sum(dim=2)


def build_task_edge_index(num_proto_nodes, num_query_nodes, device=None):
    device = device or torch.device("cpu")
    proto_ids = torch.arange(num_proto_nodes, device=device)
    query_ids = torch.arange(num_query_nodes, device=device) + num_proto_nodes
    row = proto_ids.repeat(num_query_nodes)
    col = query_ids.repeat_interleave(num_proto_nodes)
    return torch.stack([row, col], dim=0)


def neighbor_mean(embeddings, edge_index):
    num_nodes = embeddings.size(0)
    if edge_index.numel() == 0:
        return torch.zeros_like(embeddings)
    src, dst = edge_index.to(embeddings.device)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    out = torch.zeros_like(embeddings)
    out.index_add_(0, dst, embeddings[src])
    deg = torch.zeros(num_nodes, device=embeddings.device, dtype=embeddings.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=embeddings.dtype))
    return out / deg.clamp_min(1.0).unsqueeze(1)


def repeated_class_labels(num_classes, num_proto, device=None):
    return torch.arange(num_classes, device=device).repeat_interleave(num_proto)


def _class_rates(labels, train_mask, num_classes, device):
    labels = labels.to(device)
    train_mask = train_mask.to(device).bool()
    counts = torch.bincount(labels[train_mask], minlength=num_classes).float()
    total = counts.sum().clamp_min(1.0)
    return counts / total


def build_condensed_graph(x_head, x_tail, num_classes, num_proto, local_labels, train_mask):
    device = x_head.device
    y = repeated_class_labels(num_classes, num_proto, device=device)
    edge_index = torch.arange(y.numel(), device=device).repeat(2, 1)
    graph = Data(edge_index=edge_index, y=y)
    graph.x_head = x_head.detach().clone()
    graph.x_tail = x_tail.detach().clone()
    graph.x_head_proto = graph.x_head.reshape(num_classes, num_proto, -1).mean(dim=1)
    graph.x_tail_proto = graph.x_tail.reshape(num_classes, num_proto, -1).mean(dim=1)
    graph.cls_rate = _class_rates(local_labels, train_mask, num_classes, device).detach().clone()
    return graph


def build_global_synthetic_data(condensed_graphs, num_classes, num_proto, use_tail=False):
    if not condensed_graphs:
        return None
    feat_key = "x_tail" if use_tail else "x_head"
    device = getattr(condensed_graphs[0], feat_key).device
    feature_dim = getattr(condensed_graphs[0], feat_key).size(-1)
    stacked = torch.stack(
        [getattr(graph, feat_key).reshape(num_classes, num_proto, feature_dim) for graph in condensed_graphs],
        dim=0,
    )
    rates = torch.stack([graph.cls_rate.to(device) for graph in condensed_graphs], dim=0)
    rates = rates / rates.sum(dim=0, keepdim=True).clamp_min(1e-12)
    weighted = (stacked * rates[:, :, None, None]).sum(dim=0)
    x = weighted.reshape(num_classes * num_proto, feature_dim)
    y = repeated_class_labels(num_classes, num_proto, device=device)
    edge_index = torch.arange(x.size(0), device=device).repeat(2, 1)
    return Data(x=x.detach().clone(), edge_index=edge_index, y=y.detach().clone())


def blend_head_tail_log_probs(head_log_probs, tail_log_probs, node_degree, head_deg_thres):
    alpha = torch.sigmoid((node_degree.to(head_log_probs.device).float() - (head_deg_thres + 1.0))).unsqueeze(1)
    probs = alpha * head_log_probs.exp() + (1.0 - alpha) * tail_log_probs.exp()
    return torch.log(probs.clamp_min(1e-12))


def class_prototype_init(data, num_classes, num_proto, input_dim, train_mask, device):
    labels = data.y.to(device)
    train_mask = train_mask.to(device).bool()
    features = data.x.to(device)
    prototypes = []
    global_mean = features[train_mask].mean(dim=0) if train_mask.any() else torch.zeros(input_dim, device=device)
    for class_id in range(num_classes):
        selected = train_mask & (labels == class_id)
        if selected.any():
            proto = features[selected].mean(dim=0)
        else:
            proto = global_mean
        repeated = proto.unsqueeze(0).repeat(num_proto, 1)
        repeated = repeated + 0.01 * torch.randn_like(repeated)
        prototypes.append(repeated)
    return torch.cat(prototypes, dim=0)


def condensed_graph_loss(
    model,
    head_adapter,
    tail_adapter,
    data,
    train_mask,
    syn_feat_head,
    syn_feat_tail,
    num_classes,
    num_proto,
    head_deg_thres,
):
    embedding, _ = model(data)
    query = embedding[train_mask]
    labels = data.y[train_mask]
    if query.numel() == 0:
        return embedding.sum() * 0.0

    syn_data_head = Data(
        x=syn_feat_head,
        edge_index=torch.arange(syn_feat_head.size(0), device=syn_feat_head.device).repeat(2, 1),
    )
    syn_data_tail = Data(
        x=syn_feat_tail,
        edge_index=torch.arange(syn_feat_tail.size(0), device=syn_feat_tail.device).repeat(2, 1),
    )
    syn_head_embedding, _ = model(syn_data_head)
    syn_tail_embedding, _ = model(syn_data_tail)
    proto_head = syn_head_embedding.reshape(num_classes, num_proto, -1).mean(dim=1)
    proto_tail = syn_tail_embedding.reshape(num_classes, num_proto, -1).mean(dim=1)

    neighbor = neighbor_mean(embedding, data.edge_index)[train_mask]
    edge_index = build_task_edge_index(syn_head_embedding.size(0), query.size(0), device=query.device)
    _, adapted_head = head_adapter(query, neighbor, syn_head_embedding, edge_index)
    _, adapted_tail = tail_adapter(query, neighbor, syn_tail_embedding, edge_index)

    head_log_probs = F.log_softmax(-euclidean_dist(adapted_head, proto_head), dim=1)
    tail_log_probs = F.log_softmax(-euclidean_dist(adapted_tail, proto_tail), dim=1)
    node_degree = degree(data.edge_index[0], num_nodes=data.x.size(0)).to(query.device)[train_mask]
    logits = blend_head_tail_log_probs(head_log_probs, tail_log_probs, node_degree, head_deg_thres)
    return F.nll_loss(logits, labels)


def synthetic_data_loss(model, head_adapter, tail_adapter, global_syn_data, syn_feat_head, syn_feat_tail, num_classes, num_proto):
    if global_syn_data is None or global_syn_data.x.numel() == 0:
        return syn_feat_head.sum() * 0.0

    data = global_syn_data.to(syn_feat_head.device)
    embedding, _ = model(data)
    syn_data_head = Data(
        x=syn_feat_head,
        edge_index=torch.arange(syn_feat_head.size(0), device=syn_feat_head.device).repeat(2, 1),
    )
    syn_data_tail = Data(
        x=syn_feat_tail,
        edge_index=torch.arange(syn_feat_tail.size(0), device=syn_feat_tail.device).repeat(2, 1),
    )
    syn_head_embedding, _ = model(syn_data_head)
    syn_tail_embedding, _ = model(syn_data_tail)
    proto_head = syn_head_embedding.reshape(num_classes, num_proto, -1).mean(dim=1)
    proto_tail = syn_tail_embedding.reshape(num_classes, num_proto, -1).mean(dim=1)
    edge_index = build_task_edge_index(syn_head_embedding.size(0), embedding.size(0), device=embedding.device)
    neighbor = neighbor_mean(embedding, data.edge_index)
    _, adapted_head = head_adapter(embedding, neighbor, syn_head_embedding, edge_index)
    _, adapted_tail = tail_adapter(embedding, neighbor, syn_tail_embedding, edge_index)
    head_log_probs = F.log_softmax(-euclidean_dist(adapted_head, proto_head), dim=1)
    tail_log_probs = F.log_softmax(-euclidean_dist(adapted_tail, proto_tail), dim=1)
    log_probs = torch.log((0.5 * head_log_probs.exp() + 0.5 * tail_log_probs.exp()).clamp_min(1e-12))
    return F.nll_loss(log_probs, data.y)


def fedlog_metric_logits(
    model,
    head_adapter,
    tail_adapter,
    data,
    syn_feat_head,
    syn_feat_tail,
    num_classes,
    num_proto,
    head_deg_thres,
):
    embedding, _ = model(data)
    syn_data_head = Data(
        x=syn_feat_head,
        edge_index=torch.arange(syn_feat_head.size(0), device=syn_feat_head.device).repeat(2, 1),
    )
    syn_data_tail = Data(
        x=syn_feat_tail,
        edge_index=torch.arange(syn_feat_tail.size(0), device=syn_feat_tail.device).repeat(2, 1),
    )
    syn_head_embedding, _ = model(syn_data_head)
    syn_tail_embedding, _ = model(syn_data_tail)
    proto_head = syn_head_embedding.reshape(num_classes, num_proto, -1).mean(dim=1)
    proto_tail = syn_tail_embedding.reshape(num_classes, num_proto, -1).mean(dim=1)
    edge_index = build_task_edge_index(syn_head_embedding.size(0), embedding.size(0), device=embedding.device)
    neighbor = neighbor_mean(embedding, data.edge_index)
    _, adapted_head = head_adapter(embedding, neighbor, syn_head_embedding, edge_index)
    _, adapted_tail = tail_adapter(embedding, neighbor, syn_tail_embedding, edge_index)
    head_log_probs = F.log_softmax(-euclidean_dist(adapted_head, proto_head), dim=1)
    tail_log_probs = F.log_softmax(-euclidean_dist(adapted_tail, proto_tail), dim=1)
    node_degree = degree(data.edge_index[0], num_nodes=data.x.size(0)).to(embedding.device)
    return blend_head_tail_log_probs(head_log_probs, tail_log_probs, node_degree, head_deg_thres)


def parameter_norm_loss(*parameters):
    loss = None
    for param in parameters:
        item = torch.mean(torch.norm(param, dim=1))
        loss = item if loss is None else loss + item
    return loss if loss is not None else torch.tensor(0.0)
