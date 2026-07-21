import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings
try:
    from pynndescent import NNDescent
    pynndescent_available = True
except Exception:
    NNDescent = None
    pynndescent_available = False


ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine'):
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean_old(M, u):
    _, nf = np.unique(u, return_counts=True)
    idx = np.argsort(u)
    M = M[idx, :]
    M = np.vstack((np.zeros((1, M.shape[1])), M))

    np.cumsum(M, axis=0, out=M)
    cnf = np.cumsum(nf)
    nf1 = np.insert(cnf, 0, 0)
    nf1 = nf1[:-1]

    M = M[cnf, :] - M[nf1, :]
    M = M / nf[:, None]
    return M


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=True):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    # Cast input data to float32
    data = data.astype(np.float32)

    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:
        adj, orig_dist = clust_rank(mat, initial_rank, distance)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c


import torch
import torch.nn.functional as F


def proto_align_loss(proto1, proto2, num_classes, temperature=0.5):
    """
    :param proto1: 来自第一个视图的原型，形状为 [num_classes, feature_dim]
    :param proto2: 来自第二个视图的原型，形状为 [num_classes, feature_dim]
    :param num_classes: 类别总数
    :param temperature: 温度参数，用于调整损失函数的敏感性
    """
    # 计算所有正样本对 (proto1[i], proto2[i]) 之间的相似度
    positive_sim = F.cosine_similarity(proto1, proto2, dim=1) / temperature

    # 计算所有可能的负样本对
    negative_sim = torch.mm(proto1, proto2.t()) / temperature
    # 确保正样本对的相似度在负样本矩阵中为无效，防止自比较
    mask = torch.eye(num_classes).bool().to(proto1.device)
    negative_sim.masked_fill_(mask, float('-inf'))

    # 通过softmax来计算每一行的对数概率
    negative_sim_exp = torch.exp(negative_sim)
    positive_sim_exp = torch.exp(positive_sim)
    sum_negatives = negative_sim_exp.sum(dim=1)

    # Info-NCE loss 计算
    loss = -torch.log(positive_sim_exp / (positive_sim_exp + sum_negatives))

    return loss.mean()

def get_proto_norm_weighted(num_classes, embedding, class_label, weight,unique_labels):
    m1= F.one_hot(class_label, num_classes=num_classes)
    m2 = (m1 * weight[:, None]).t()
    m = m2 / (m2.sum(dim=1, keepdim=True)+ 1e-6)
    m = m[unique_labels]
    return torch.mm(m, embedding)


import torch


def prototypes_to_label_dict(prototypes, unique_labels, weights=None):
    result = {}
    labels = unique_labels.detach().cpu().tolist()
    for row, label in enumerate(labels):
        proto = prototypes[row].detach().view(1, -1)
        if weights is None:
            result[int(label)] = proto
        else:
            result[int(label)] = {
                "proto": proto,
                "weight": float(weights[row]),
            }
    return result


def _flatten_global_prototypes(global_protos, device):
    proto_tensors = []
    proto_labels = []
    if not global_protos:
        return None, None

    for label, proto_list in global_protos.items():
        if isinstance(proto_list, torch.Tensor):
            proto_list = [proto_list]
        for proto in proto_list:
            if proto is None:
                continue
            proto = proto.detach().to(device).view(1, -1)
            proto_tensors.append(proto)
            proto_labels.append(int(label))

    if len(proto_tensors) == 0:
        return None, None

    return torch.cat(proto_tensors, dim=0), torch.tensor(proto_labels, device=device, dtype=torch.long)


def global_prototype_logits(
    embedding,
    global_protos,
    num_classes,
    temperature=0.2,
    top_k=None,
    eps=1e-8,
):
    proto_bank, proto_labels = _flatten_global_prototypes(global_protos, embedding.device)
    logits = embedding.new_full((embedding.shape[0], num_classes), -30.0)
    available = torch.zeros(num_classes, dtype=torch.bool, device=embedding.device)
    if proto_bank is None:
        return logits, available

    available[proto_labels.unique()] = True
    similarities = torch.mm(
        F.normalize(embedding, dim=1),
        F.normalize(proto_bank, dim=1).t(),
    ) / max(float(temperature), eps)

    num_proto = similarities.shape[1]
    if top_k is None or top_k <= 0 or top_k >= num_proto:
        weights = torch.softmax(similarities, dim=1)
        selected_labels = proto_labels.unsqueeze(0).expand(embedding.shape[0], -1)
    else:
        top_values, top_indices = torch.topk(similarities, k=min(int(top_k), num_proto), dim=1)
        weights = torch.softmax(top_values, dim=1)
        selected_labels = proto_labels[top_indices]

    scores = embedding.new_zeros((embedding.shape[0], num_classes))
    scores.scatter_add_(1, selected_labels, weights)
    return torch.log(scores.clamp_min(eps)), available


def global_prototype_guidance_loss(
    embedding,
    labels,
    mask,
    global_protos,
    num_classes,
    temperature=0.2,
    mse_weight=0.1,
):
    selected = mask.bool()
    if selected.sum() == 0 or not global_protos:
        return embedding.sum() * 0.0

    train_embedding = embedding[selected]
    train_labels = labels[selected].long()
    proto_bank, proto_labels = _flatten_global_prototypes(global_protos, embedding.device)
    if proto_bank is None:
        return embedding.sum() * 0.0

    has_positive = torch.isin(train_labels, proto_labels.unique())
    if has_positive.sum() == 0:
        return embedding.sum() * 0.0

    train_embedding = train_embedding[has_positive]
    train_labels = train_labels[has_positive]
    similarities = torch.mm(
        F.normalize(train_embedding, dim=1),
        F.normalize(proto_bank, dim=1).t(),
    ) / max(float(temperature), 1e-8)
    positive_mask = train_labels.unsqueeze(1).eq(proto_labels.unsqueeze(0))
    info_loss = -torch.logsumexp(
        similarities.masked_fill(~positive_mask, float("-inf")),
        dim=1,
    ) + torch.logsumexp(similarities, dim=1)

    if mse_weight <= 0:
        return info_loss.mean()

    mean_protos = []
    for label in train_labels.detach().cpu().tolist():
        mean_protos.append(proto_bank[proto_labels == int(label)].mean(dim=0))
    mean_protos = torch.stack(mean_protos, dim=0)
    mse_loss = F.mse_loss(train_embedding, mean_protos)
    return info_loss.mean() + float(mse_weight) * mse_loss


def com_distillation_loss(
    teacher_logits,
    student_logits,
    adj_orig,
    adj_sampled,
    edge_index=None,
    edge_label=None,
    temperature=0.1,
    detach_teacher=True,
):
    teacher_dist = F.softmax(teacher_logits / temperature, dim=-1)
    if detach_teacher:
        teacher_dist = teacher_dist.detach()
    student_dist = F.log_softmax(student_logits / temperature, dim=-1)
    kd_loss = temperature * temperature * F.kl_div(
        student_dist,
        teacher_dist,
        reduction="batchmean",
    )

    if edge_index is not None and edge_label is not None:
        edge_list = edge_index[:, edge_label.to(edge_index.device).bool()].detach()
    else:
        edge_mask = torch.triu((adj_orig > 0).float() * (adj_sampled > 0).float()).detach()
        edge_list = (edge_mask + edge_mask.t()).nonzero(as_tuple=False).t()
    if edge_list.numel() == 0:
        return kd_loss

    teacher_neigh = F.softmax(teacher_logits[edge_list[1]] / temperature, dim=-1)
    if detach_teacher:
        teacher_neigh = teacher_neigh.detach()
    student_neigh = F.log_softmax(student_logits[edge_list[0]] / temperature, dim=-1)
    kd_loss = kd_loss + temperature * temperature * F.kl_div(
        student_neigh,
        teacher_neigh,
        reduction="batchmean",
    )
    return kd_loss

def _unpack_proto_entry(entry, device):
    if isinstance(entry, dict):
        proto = entry["proto"].detach().to(device).view(1, -1)
        weight = float(entry.get("weight", 1.0))
    else:
        proto = entry.detach().to(device).view(1, -1)
        weight = 1.0
    return proto, max(weight, 1e-12)


def aggregate_fggp_prototypes(local_protos_list, device, use_finch=False):
    agg_protos_label = {}
    for local_protos in local_protos_list:
        for label, proto_entry in local_protos.items():
            label = int(label)
            if label not in agg_protos_label:
                agg_protos_label[label] = []
            agg_protos_label[label].append(_unpack_proto_entry(proto_entry, device))

    aggregated = {}
    for label, proto_entries in agg_protos_label.items():
        if len(proto_entries) == 1:
            aggregated[label] = [proto_entries[0][0]]
            continue

        if use_finch:
            proto_array = np.array([
                proto.squeeze(0).detach().cpu().numpy().reshape(-1)
                for proto, _ in proto_entries
            ])
            c, _, _ = FINCH(
                proto_array,
                initial_rank=None,
                req_clust=None,
                distance="cosine",
                ensure_early_exit=False,
                verbose=False,
            )
            cluster_ids = c[:, -1]
            selected = []
            for cluster_id in np.unique(cluster_ids).tolist():
                selected_array = np.where(cluster_ids == cluster_id)[0]
                weights = np.array([proto_entries[idx][1] for idx in selected_array], dtype=np.float32)
                weights = weights / max(float(weights.sum()), 1e-12)
                proto = np.sum(proto_array[selected_array] * weights[:, None], axis=0, keepdims=True)
                selected.append(torch.tensor(proto, device=device))
            aggregated[label] = selected
        else:
            stacked = torch.cat([proto for proto, _ in proto_entries], dim=0)
            weights = torch.tensor(
                [weight for _, weight in proto_entries],
                dtype=stacked.dtype,
                device=device,
            ).view(-1, 1)
            aggregated[label] = [(stacked * weights).sum(dim=0, keepdim=True) / weights.sum().clamp_min(1e-12)]

    return aggregated


def get_norm_and_orig(data):
    # 假设data_loader已经有了原始的邻接矩阵存储在data_loader.adj中
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]

    # 构造邻接矩阵
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1

    # 计算度矩阵D并进行归一化处理
    degree = adj.sum(dim=1, keepdim=True)
    deg_inv_sqrt = degree.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # 计算标准化后的邻接矩阵 \hat{A} = D^{-1/2} * A * D^{-1/2}
    norm_adj = deg_inv_sqrt * adj * deg_inv_sqrt.t()

    # 更新data_loader中的属性
    if data.is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    data.adj_orig = adj.to(device)  # 原始邻接矩阵
    data.norm_adj = norm_adj.to(device)  # 标准化邻接矩阵

    return data


def proto_align_loss(proto1, proto2,  temperature=0.5):
    """
    :param proto1: 来自第一个视图的原型，形状为 [num_classes, feature_dim]
    :param proto2: 来自第二个视图的原型，形状为 [num_classes, feature_dim]
    :param num_classes: 类别总数
    :param temperature: 温度参数，用于调整损失函数的敏感性
    """

    num_classes = proto1.shape[0]
    if num_classes <= 1:
        return proto1.sum() * 0.0
    logits = torch.mm(
        F.normalize(proto1, dim=1),
        F.normalize(proto2, dim=1).t(),
    ) / max(float(temperature), 1e-8)
    target = torch.arange(num_classes, device=proto1.device)
    return F.cross_entropy(logits, target)

