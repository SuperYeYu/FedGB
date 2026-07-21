import copy
import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from sknetwork.clustering import Louvain
import sys
from sklearn.cluster import KMeans
try:
    import pymetis as metis
except ImportError:
    metis = None
import torch_geometric.utils
from tqdm import tqdm
import torch_geometric

def get_subgraph_pyg_data(global_dataset, node_list):
    """
    Extract a subgraph from the global dataset given a list of node indices.

    Args:
        global_dataset (Data): The global graph dataset.
        node_list (list): List of node indices to include in the subgraph.

    Returns:
        Data: The subgraph containing the specified nodes and their edges.
    """
    global_edge_index = global_dataset.edge_index
    node_id_set = set(node_list)
    global_id_to_local_id = {}
    local_id_to_global_id = {}
    local_edge_list = []
    for local_id, global_id in enumerate(node_list):
        global_id_to_local_id[global_id] = local_id
        local_id_to_global_id[local_id] = global_id
        
    for edge_id in tqdm(range(global_edge_index.shape[1]), desc="Processing Edge Mapping"):
        src = global_edge_index[0, edge_id].item()
        tgt = global_edge_index[1, edge_id].item()
        if src in node_id_set and tgt in node_id_set:
            local_id_src = global_id_to_local_id[src]
            local_id_tgt = global_id_to_local_id[tgt]
            local_edge_list.append((local_id_src, local_id_tgt))
            
    local_edge_index = torch.tensor(local_edge_list).T
    
    
    local_subgraph = Data(x=global_dataset.x[node_list], edge_index=local_edge_index, y=global_dataset.y[node_list])
    local_subgraph.global_map = local_id_to_global_id
    
    if hasattr(global_dataset, "num_classes"):
        local_subgraph.num_global_classes = global_dataset.num_classes
    else:
        local_subgraph.num_global_classes = global_dataset.num_global_classes
    return local_subgraph


def _target_node_type(data, args=None):
    if args is not None and hasattr(args, "target_node"):
        return args.target_node
    return data.node_types[0]


def _ensure_hetero_node_features(data):
    fallback_dtype = None
    for node_type in data.node_types:
        if hasattr(data[node_type], "x"):
            fallback_dtype = data[node_type].x.dtype
            break
    if fallback_dtype is None:
        fallback_dtype = torch.float32

    for node_type in data.node_types:
        if not hasattr(data[node_type], "x"):
            data[node_type].x = torch.eye(data[node_type].num_nodes, dtype=fallback_dtype)


def _hetero_global_offsets(data):
    offsets = {}
    cursor = 0
    for node_type in data.node_types:
        offsets[node_type] = cursor
        cursor += data[node_type].num_nodes
    return offsets


def hetero_to_homogeneous_node_data(data, target_node=None, global_node_offsets=None):
    target_node = target_node or data.node_types[0]
    _ensure_hetero_node_features(data)
    global_node_offsets = _hetero_global_offsets(data) if global_node_offsets is None else global_node_offsets

    max_dim = max(data[node_type].x.shape[1] for node_type in data.node_types)
    local_offsets = {}
    xs = []
    ys = []
    node_types = []
    global_map = {}
    cursor = 0

    source_global_map = getattr(data, "global_map", {})
    for type_id, node_type in enumerate(data.node_types):
        x = data[node_type].x.to(torch.float32)
        if x.shape[1] < max_dim:
            pad = torch.zeros(x.shape[0], max_dim - x.shape[1], dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        xs.append(x)

        y = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        if node_type == target_node and hasattr(data[node_type], "y"):
            y = data[node_type].y.long().view(-1)
        ys.append(y)
        node_types.append(torch.full((x.shape[0],), type_id, dtype=torch.long, device=x.device))

        local_offsets[node_type] = cursor
        per_type_global_map = source_global_map.get(node_type, {}) if isinstance(source_global_map, dict) else {}
        for local_id in range(x.shape[0]):
            original_id = per_type_global_map.get(local_id, local_id)
            global_map[cursor + local_id] = global_node_offsets[node_type] + int(original_id)
        cursor += x.shape[0]

    edge_parts = []
    edge_types = []
    for edge_type_id, edge_type in enumerate(data.edge_types):
        if not hasattr(data[edge_type], "edge_index"):
            continue
        edge_index = data[edge_type].edge_index.long()
        if edge_index.numel() == 0:
            continue
        src_type, _, dst_type = edge_type
        shifted = edge_index.clone()
        shifted[0] += local_offsets[src_type]
        shifted[1] += local_offsets[dst_type]
        edge_parts.append(shifted)
        edge_types.append(torch.full((shifted.shape[1],), edge_type_id, dtype=torch.long, device=shifted.device))

    edge_index = torch.cat(edge_parts, dim=1) if edge_parts else torch.empty((2, 0), dtype=torch.long)
    homogeneous = Data(x=torch.cat(xs, dim=0), y=torch.cat(ys, dim=0), edge_index=edge_index.contiguous())
    homogeneous.node_type = torch.cat(node_types, dim=0)
    if edge_types:
        homogeneous.edge_type = torch.cat(edge_types, dim=0)
    homogeneous.target_node_type = target_node
    homogeneous.hetero_node_types = list(data.node_types)
    homogeneous.hetero_edge_types = list(data.edge_types)
    homogeneous.global_map = global_map
    if hasattr(data, "num_global_classes"):
        homogeneous.num_global_classes = int(data.num_global_classes)
    elif hasattr(data[target_node], "y") and data[target_node].y.numel() > 0:
        homogeneous.num_global_classes = int(data[target_node].y.max().item()) + 1
    return homogeneous


def get_heterogeneous_subgraph_pyg_data(global_dataset, node_list, target_node=None):
    global_data = global_dataset[0] if not hasattr(global_dataset, "node_types") else global_dataset
    target_node = target_node or global_data.node_types[0]
    _ensure_hetero_node_features(global_data)

    node_dict = {node_type: set() for node_type in global_data.node_types}
    node_dict[target_node] = set(int(node_id) for node_id in node_list)

    changed = True
    while changed:
        changed = False
        for edge_type in global_data.edge_types:
            src_type, _, dst_type = edge_type
            if dst_type == target_node:
                continue
            src_nodes = node_dict[src_type]
            if not src_nodes:
                continue
            edge_index = global_data[edge_type].edge_index
            selected = torch.isin(edge_index[0].cpu(), torch.tensor(list(src_nodes), dtype=torch.long))
            dst_nodes = set(edge_index[1, selected].cpu().tolist())
            before = len(node_dict[dst_type])
            node_dict[dst_type].update(int(node_id) for node_id in dst_nodes)
            changed = changed or len(node_dict[dst_type]) > before

    node_lists = {node_type: sorted(nodes) for node_type, nodes in node_dict.items()}
    global_to_local = {}
    local_to_global = {}
    for node_type, nodes in node_lists.items():
        global_to_local[node_type] = {global_id: local_id for local_id, global_id in enumerate(nodes)}
        local_to_global[node_type] = {local_id: global_id for local_id, global_id in enumerate(nodes)}

    local_subgraph = HeteroData()
    for node_type, nodes in node_lists.items():
        local_subgraph[node_type].x = global_data[node_type].x[nodes]
        if node_type == target_node and hasattr(global_data[node_type], "y"):
            local_subgraph[node_type].y = global_data[node_type].y[nodes]

    for edge_type in global_data.edge_types:
        src_type, _, dst_type = edge_type
        local_edges = []
        edge_index = global_data[edge_type].edge_index
        src_allowed = global_to_local[src_type]
        dst_allowed = global_to_local[dst_type]
        for edge_id in range(edge_index.shape[1]):
            src = int(edge_index[0, edge_id].item())
            dst = int(edge_index[1, edge_id].item())
            if src in src_allowed and dst in dst_allowed:
                local_edges.append((src_allowed[src], dst_allowed[dst]))
        if local_edges:
            local_subgraph[edge_type].edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
        else:
            local_subgraph[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

    local_subgraph.global_map = local_to_global
    local_subgraph.num_global_classes = int(global_data[target_node].y.max().item()) + 1
    return hetero_to_homogeneous_node_data(
        local_subgraph,
        target_node=target_node,
        global_node_offsets=_hetero_global_offsets(global_data),
    )


def subgraph_fl_heterogeneous_label_skew(args, global_dataset):
    print("Conducting subgraph-fl heterogeneous label skew simulation...")
    global_data = global_dataset[0] if not hasattr(global_dataset, "node_types") else global_dataset
    target_node = _target_node_type(global_data, args)
    node_labels = global_data[target_node].y.cpu().numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(node_labels, return_counts=True)

    print(f"target_node: {target_node}")
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")

    min_size = 0
    K = len(unique_labels)
    try_cnt = 0
    while min_size < args.least_samples:
        if try_cnt > args.dirichlet_try_cnt:
            raise RuntimeError(
                "Client data size does not meet the minimum requirement. "
                "Try larger dirichlet_alpha or lower least_samples."
            )

        client_indices = [[] for _ in range(num_clients)]
        for class_id in unique_labels:
            idx_k = np.where(node_labels == class_id)[0]
            np.random.shuffle(idx_k)
            if len(idx_k) >= num_clients:
                base = idx_k[:num_clients]
                remainder = idx_k[num_clients:]
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.maximum(proportions, 1e-9)
                proportions = proportions / proportions.sum()
                split_points = (np.cumsum(proportions) * len(remainder)).astype(int)[:-1]
                parts = np.split(remainder, split_points)
                for client_id in range(num_clients):
                    client_indices[client_id].append(int(base[client_id]))
                    client_indices[client_id].extend(parts[client_id].tolist())
            else:
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.maximum(proportions, 1e-9)
                proportions = proportions / proportions.sum()
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                parts = np.split(idx_k, split_points)
                client_indices = [cur + part.tolist() for cur, part in zip(client_indices, parts)]
            min_size = min(len(indices) for indices in client_indices)
        try_cnt += 1

    local_data = []
    client_label_counts = [[0] * K for _ in range(num_clients)]
    label_to_position = {label: idx for idx, label in enumerate(unique_labels)}
    for client_id in range(num_clients):
        client_indices[client_id] = sorted(client_indices[client_id])
        for label in unique_labels:
            client_label_counts[client_id][label_to_position[label]] = int((node_labels[client_indices[client_id]] == label).sum())
        local_data.append(get_heterogeneous_subgraph_pyg_data(global_data, client_indices[client_id], target_node))

    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data



def graph_fl_cross_domain(args, global_dataset):
    print("Conducting graph-fl cross domain simulation...")
    local_data = []
    for client_id in range(args.num_clients):
        local_graphs = global_dataset[client_id] # list(InMemoryDataset) -> InMemoryDataset
        local_graphs.num_global_classes = global_dataset[client_id].num_classes
        local_data.append(local_graphs)
    return local_data



def graph_fl_label_skew(args, global_dataset, shuffle=True):
    """
    Simulate cross-domain federated learning for graph data.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (list): List of global graph datasets for each client.

    Returns:
        list: List of local graph datasets for each client.
    """
    print("Conducting graph-fl label skew simulation...")
    
    graph_labels = global_dataset.y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)
    
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = graph_labels.shape[0]

    try_cnt = 0
    while min_size < args.least_samples:
        if try_cnt > args.dirichlet_try_cnt:
            print(f"Client data size does not meet the minimum requirement {args.least_samples}. Try 'args.dirichlet_alpha' larger than {args.dirichlet_alpha} /  'args.try_cnt' larger than {args.try_cnt} / 'args.least_sampes' lower than {args.least_samples}.")
            sys.exit(0)
            
        client_indices = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(graph_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,client_indices)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            client_indices = [idx_j + idx.tolist() for idx_j,idx in zip(client_indices,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in client_indices])
        try_cnt += 1
   
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (graph_labels[client_indices[client_id]] == class_i).sum()
        
        list.sort(client_indices[client_id])
        
        local_id_to_global_id = {}
        for local_id, global_id in enumerate(client_indices[client_id]):
            local_id_to_global_id[local_id] = global_id
        
        local_graphs = global_dataset.copy(client_indices[client_id]) # InMemoryDataset -> deep-copy subset
        local_graphs.num_global_classes = global_dataset.num_classes
        local_graphs.global_map = local_id_to_global_id
        local_data.append(local_graphs)
    
    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data
    
    
def graph_fl_topology_skew(args, global_dataset, shuffle=True):
    """
    Simulate topology skew in federated learning for graph data.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.
        shuffle (bool, optional): If True, shuffle the dataset. Defaults to True.

    Returns:
        list: List of local graph datasets for each client with topology skew.
    """
    print("Conducting graph-fl topology skew simulation...")
    
    graph_labels = global_dataset.y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)
    
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = graph_labels.shape[0]
    client_indices = [[] for _ in range(num_clients)]
    
    d_list = []
    for graph_id, data in enumerate(global_dataset):
        deg = torch_geometric.utils.degree(data.edge_index[1], num_nodes=data.num_nodes)
        
        d_list.append((graph_id, deg.mean()))
    
    d_list.sort(key= lambda x: x[1])
    
    segment_len = len(d_list) // num_clients
    for client_id in range(num_clients):
        left = client_id * segment_len
        right = segment_len * (client_id + 1)
        if client_id == num_clients - 1:
            right = len(d_list)
        
        segment = d_list[left : right]
        client_indices[client_id] = [x[0] for x in segment]
    
    assert sum([len(x) for x in client_indices]) == N
    
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (graph_labels[client_indices[client_id]] == class_i).sum()
        
        list.sort(client_indices[client_id])
        
        local_id_to_global_id = {}
        for local_id, global_id in enumerate(client_indices[client_id]):
            local_id_to_global_id[local_id] = global_id
        
        local_graphs = global_dataset.copy(client_indices[client_id]) # InMemoryDataset -> deep-copy subset
        local_graphs.num_global_classes = global_dataset.num_classes
        local_graphs.global_map = local_id_to_global_id
        local_data.append(local_graphs)
    
    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data
    
    
    
    
    
  
def graph_fl_feature_skew(args, global_dataset, shuffle=True):
    """
    Simulate feature skew in federated learning for graph data.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.
        shuffle (bool, optional): If True, shuffle the dataset. Defaults to True.

    Returns:
        list: List of local graph datasets for each client with feature skew.
    """
    print("Conducting graph-fl feature skew simulation...")
    
    graph_labels = global_dataset.y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(graph_labels, return_counts=True)
    
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = graph_labels.shape[0]
    client_indices = [[] for _ in range(num_clients)]
    
    f_list = []
    for graph_id, data in enumerate(global_dataset):
        avg_feature = data.x.mean()
        f_list.append((graph_id, avg_feature))
        # deg = torch_geometric.utils.degree(data.edge_index[1], num_nodes=data.num_nodes)
        # f_list.append((graph_id, deg.mean()))
    
    f_list.sort(key= lambda x: x[1])
    
    segment_len = len(f_list) // num_clients
    for client_id in range(num_clients):
        left = client_id * segment_len
        right = segment_len * (client_id + 1)
        if client_id == num_clients - 1:
            right = len(f_list)
        
        segment = f_list[left : right]
        client_indices[client_id] = [x[0] for x in segment]
    
    assert sum([len(x) for x in client_indices]) == N
    
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (graph_labels[client_indices[client_id]] == class_i).sum()
        
        list.sort(client_indices[client_id])
        
        local_id_to_global_id = {}
        for local_id, global_id in enumerate(client_indices[client_id]):
            local_id_to_global_id[local_id] = global_id
        
        local_graphs = global_dataset.copy(client_indices[client_id]) # InMemoryDataset -> deep-copy subset
        local_graphs.num_global_classes = global_dataset.num_classes
        local_graphs.global_map = local_id_to_global_id
        local_data.append(local_graphs)
    
    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data 
    
    
    
    
    
def subgraph_fl_label_skew(args, global_dataset, shuffle=True):
    """
    Simulate label skew in federated learning for subgraphs.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (list): List of global graph datasets.
        shuffle (bool, optional): If True, shuffle the dataset. Defaults to True.

    Returns:
        list: List of local subgraph datasets for each client with label skew.
    """
    print("Conducting subgraph-fl label skew simulation...")
    node_labels = global_dataset[0].y.numpy()
    num_clients = args.num_clients
    alpha = args.dirichlet_alpha
    unique_labels, label_counts = np.unique(node_labels, return_counts=True)
    
    print(f"num_classes: {len(unique_labels)}")
    print(f"global label distribution: {label_counts}")
       
    min_size = 0
    K = len(unique_labels)
    N = node_labels.shape[0]

    try_cnt = 0
    while min_size < args.least_samples:
        if try_cnt > args.dirichlet_try_cnt:
            print(f"Client data size does not meet the minimum requirement {args.least_samples}. Try 'args.dirichlet_alpha' larger than {args.dirichlet_alpha} /  'args.try_cnt' larger than {args.try_cnt} / 'args.least_sampes' lower than {args.least_samples}.")
            sys.exit(0)
            
        client_indices = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(node_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,client_indices)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            client_indices = [idx_j + idx.tolist() for idx_j,idx in zip(client_indices,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in client_indices])
        try_cnt += 1
   
    
    local_data = []
    client_label_counts = [[0] * K for _ in range(args.num_clients)]
    for client_id in range(args.num_clients):
        for class_i in range(K):
            client_label_counts[client_id][class_i] = (node_labels[client_indices[client_id]] == class_i).sum()
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)
    print(f"label_counts:\n{np.array(client_label_counts)}")
    return local_data


def subgraph_fl_louvain_plus(args, global_dataset):
    """
    Simulate subgraph federated learning using the Louvain+ method.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.

    Returns:
        list: List of local subgraph datasets for each client using Louvain+ method.
    """
    print("Conducting subgraph-fl louvain+ simulation...")
    louvain = Louvain(modularity='newman', resolution=args.louvain_resolution, return_aggregate=True) # resolution 越大产生的社区越多, 社区粒度越小
    adj_csr = to_scipy_sparse_matrix(global_dataset[0].edge_index)
    fit_result = louvain.fit_predict(adj_csr)
    communities = {}
    for node_id, com_id in enumerate(fit_result):
        if com_id not in communities:
            communities[com_id] = {"nodes":[], "num_nodes":0, "label_distribution":[0] * global_dataset.num_classes}
        communities[com_id]["nodes"].append(node_id)
        
    for com_id in communities.keys():
        communities[com_id]["num_nodes"] = len(communities[com_id]["nodes"])
        for node in communities[com_id]["nodes"]:
            label = copy.deepcopy(global_dataset[0].y[node])
            communities[com_id]["label_distribution"][label] += 1

    num_communities = len(communities)
    clustering_data = np.zeros(shape=(num_communities, global_dataset.num_classes))
    for com_id in communities.keys():
        for class_i in range(global_dataset.num_classes):
            clustering_data[com_id][class_i] = communities[com_id]["label_distribution"][class_i]
        clustering_data[com_id, :] /= clustering_data[com_id, :].sum()

    kmeans = KMeans(n_clusters=args.num_clients)
    kmeans.fit(clustering_data)

    clustering_labels = kmeans.labels_

    client_indices = {client_id: [] for client_id in range(args.num_clients)}
    
    for com_id in range(num_communities):
        client_indices[clustering_labels[com_id]] += communities[com_id]["nodes"]
    
    
      
    local_data = []
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)

    return local_data


def subgraph_fl_metis_plus(args, global_dataset):
    """
    Simulate subgraph federated learning using the Metis+ method.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.

    Returns:
        list: List of local subgraph datasets for each client using Metis+ method.
    """
    print("Conducting subgraph-fl metis+ simulation...")
    if metis is None:
        raise ImportError("pymetis is required for subgraph_fl_metis_plus. Please install pymetis or use Louvain.")
    graph_nx = to_networkx(global_dataset[0], to_undirected=True)
    communities = {com_id: {"nodes":[], "num_nodes":0, "label_distribution":[0] * global_dataset.num_classes} 
                            for com_id in range(args.metis_num_coms)}
    n_cuts, membership = metis.part_graph(args.metis_num_coms, graph_nx)
    for com_id in range(args.metis_num_coms):
        com_indices = np.where(np.array(membership) == com_id)[0]
        com_indices = list(com_indices)
        communities[com_id]["nodes"] = com_indices
        communities[com_id]["num_nodes"] = len(com_indices)
        for node in communities[com_id]["nodes"]:
            label = copy.deepcopy(global_dataset[0].y[node])
            communities[com_id]["label_distribution"][label] += 1
    
    num_communities = len(communities)
    clustering_data = np.zeros(shape=(num_communities, global_dataset.num_classes))
    for com_id in communities.keys():
        for class_i in range(global_dataset.num_classes):
            clustering_data[com_id][class_i] = communities[com_id]["label_distribution"][class_i]
        clustering_data[com_id, :] /= clustering_data[com_id, :].sum()

    kmeans = KMeans(n_clusters=args.num_clients)
    kmeans.fit(clustering_data)

    clustering_labels = kmeans.labels_

    client_indices = {client_id: [] for client_id in range(args.num_clients)}
    
    for com_id in range(num_communities):
        client_indices[clustering_labels[com_id]] += communities[com_id]["nodes"]
    
    local_data = []
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)
    
    return local_data
    
    



    
    
    

def subgraph_fl_metis(args, global_dataset):
    """
    Simulate subgraph federated learning using the Metis method.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.

    Returns:
        list: List of local subgraph datasets for each client using Metis method.
    """
    print("Conducting subgraph-fl metis simulation...")
    if metis is None:
        raise ImportError("pymetis is required for subgraph_fl_metis. Please install pymetis or use Louvain.")
    graph_nx = to_networkx(global_dataset[0], to_undirected=True)
    n_cuts, membership = metis.part_graph(args.num_clients, graph_nx)
    
    client_indices = [None] * args.num_clients
    for client_id in range(args.num_clients):
        client_indices[client_id] = np.where(np.array(membership) == client_id)[0].tolist()
        
    local_data = []
    
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, client_indices[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)
    
    return local_data
    
    

def subgraph_fl_louvain(args, global_dataset):
    """
    Simulate subgraph federated learning using the Louvain method.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        global_dataset (Data): The global graph dataset.

    Returns:
        list: List of local subgraph datasets for each client using Louvain method.
    """
    print("Conducting subgraph-fl louvain simulation...")
    if hasattr(global_dataset[0], "node_types"):
        print("Detected HeteroData; using heterogeneous label-skew target-node split instead of homogeneous Louvain.")
        return subgraph_fl_heterogeneous_label_skew(args, global_dataset)
    louvain = Louvain(modularity='newman', resolution=args.louvain_resolution, return_aggregate=True)
    num_nodes = global_dataset[0].x.shape[0]
    adj_csr = to_scipy_sparse_matrix(global_dataset[0].edge_index)
    fit_result = louvain.fit_predict(adj_csr)
    partition = {}
    for node_id, com_id in enumerate(fit_result):
        partition[node_id] = int(com_id)

    groups = []

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)
    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    group_len_max = num_nodes // args.num_clients - args.louvain_delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    print(groups)

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}

    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {
        k: v
        for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)
    }

    owner_node_ids = {owner_id: [] for owner_id in range(args.num_clients)}

    owner_nodes_len = num_nodes // args.num_clients
    owner_list = [i for i in range(args.num_clients)]
    owner_ind = 0

    give_up = 1000

    for group_i in sort_len_dict.keys():
        while (
            len(owner_list) >= 2
            and len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len
        ):
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        cnt = 0
        while (
            len(owner_node_ids[owner_list[owner_ind]]) +
                len(partition_groups[group_i])
            >= owner_nodes_len + args.louvain_delta
        ):
            owner_ind = (owner_ind + 1) % len(owner_list)
            cnt += 1
            if cnt > give_up:
                cnt = 0
                min_v = 1e15
                for i in range(len(owner_list)):
                    if len(owner_node_ids[owner_list[owner_ind]]) < min_v:
                        min_v = len(owner_node_ids[owner_list[owner_ind]])
                        owner_ind = i
                break

        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]

    local_data = []
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(global_dataset, owner_node_ids[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(local_subgraph.edge_index.view(2,-1))
        local_data.append(local_subgraph)

    return local_data
