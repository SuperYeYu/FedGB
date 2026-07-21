import os
from os import path as osp
from fedgb.data.global_dataset_loader import load_global_dataset
from torch_geometric.data import Data, Dataset
from torch_geometric.data import HeteroData
from torch_geometric.utils import remove_self_loops, to_undirected
import copy
import torch
import json

from fedgb.data.release_loader import normalize_client_payload


class LocalGlobalDataset:
    def __init__(self, data, num_classes):
        self.data = data
        self.num_classes = num_classes


def normalize_global_map(global_map, num_nodes):
    if torch.is_tensor(global_map):
        return global_map.long().cpu()

    if isinstance(global_map, dict):
        keys = list(global_map.keys())
        values = list(global_map.values())

        local_keys = all(isinstance(key, int) and 0 <= key < num_nodes for key in keys)
        local_values = all(isinstance(value, int) and 0 <= value < num_nodes for value in values)

        if local_keys and not local_values:
            ordered = [global_map[local_id] for local_id in range(num_nodes)]
        elif local_values:
            ordered = [None] * num_nodes
            for global_id, local_id in global_map.items():
                ordered[int(local_id)] = int(global_id)
            if any(item is None for item in ordered):
                raise RuntimeError("Cannot normalize global_map dict with missing local ids.")
        else:
            ordered = [global_map[local_id] for local_id in range(num_nodes)]
        return torch.tensor(ordered, dtype=torch.long)

    return torch.tensor(global_map, dtype=torch.long)


def build_global_dataset_from_local_data(local_data):
    xs = {}
    ys = {}
    edge_parts = []
    num_classes = 0

    for data in local_data:
        if not hasattr(data, "global_map"):
            raise RuntimeError("Cannot build global data from local data without 'global_map'.")

        global_map = normalize_global_map(data.global_map, data.x.shape[0])

        for local_id, global_id in enumerate(global_map.tolist()):
            if global_id not in xs:
                xs[global_id] = data.x[local_id].detach().cpu()
                ys[global_id] = data.y[local_id].detach().cpu()

        edge_index = data.edge_index.long().cpu()
        src = global_map[edge_index[0]]
        dst = global_map[edge_index[1]]
        edge_parts.append(torch.stack([src, dst], dim=0))

        if hasattr(data, "num_global_classes"):
            num_classes = max(num_classes, int(data.num_global_classes))
        elif hasattr(data, "y") and data.y.numel() > 0:
            num_classes = max(num_classes, int(data.y.max().item()) + 1)

    if not xs:
        raise RuntimeError("Cannot build global data because local data is empty.")

    num_nodes = max(xs.keys()) + 1
    feature_dim = next(iter(xs.values())).numel()
    x = torch.zeros(num_nodes, feature_dim, dtype=torch.float32)
    y = torch.zeros(num_nodes, dtype=torch.long)

    for global_id, feat in xs.items():
        x[global_id] = feat.float()
        y[global_id] = ys[global_id].long()

    edge_index = torch.cat(edge_parts, dim=1)
    edge_index = torch.unique(edge_index.t(), dim=0).t().contiguous()

    global_data = Data(x=x, y=y, edge_index=edge_index)
    global_data.num_global_classes = num_classes
    return LocalGlobalDataset(global_data, num_classes)


def infer_num_classes(dataset_or_data, target_node=None, global_data=None):
    if hasattr(dataset_or_data, "num_classes"):
        return int(dataset_or_data.num_classes)

    data = global_data if global_data is not None else dataset_or_data
    if isinstance(data, HeteroData) or hasattr(data, "node_types"):
        target_node = target_node or data.node_types[0]
        if hasattr(data[target_node], "y") and data[target_node].y.numel() > 0:
            return int(data[target_node].y.max().item()) + 1

    if hasattr(data, "num_global_classes"):
        return int(data.num_global_classes)

    if hasattr(data, "y") and data.y is not None and data.y.numel() > 0:
        labels = data.y[data.y >= 0]
        if labels.numel() > 0:
            return int(labels.max().item()) + 1

    raise AttributeError("Cannot infer num_classes from dataset or global data.")


def preprocess_loaded_pyg_data(data):
    if hasattr(data, "x"):
        data.x = data.x.to(torch.float32)
    if hasattr(data, "y"):
        data.y = data.y.squeeze() # could be int64 (for classification) / float32 (for regression)

    if not hasattr(data, "edge_index"):
        data._data_list = None
        return data

    edge_index = data.edge_index.to(torch.int64)
    edge_type = getattr(data, "edge_type", None)
    edge_attr = getattr(data, "edge_attr", None)
    has_edge_type = torch.is_tensor(edge_type) and edge_type.numel() == edge_index.size(1)

    if has_edge_type:
        mask = edge_index[0] != edge_index[1]
        data.edge_index = edge_index[:, mask].contiguous()
        data.edge_type = edge_type.to(torch.int64)[mask].contiguous()
        if torch.is_tensor(edge_attr) and edge_attr.size(0) == edge_index.size(1):
            data.edge_attr = edge_attr[mask].contiguous()
    elif torch.is_tensor(edge_attr) and edge_attr.size(0) == edge_index.size(1):
        data.edge_index, data.edge_attr = remove_self_loops(*to_undirected(edge_index, edge_attr))
        if hasattr(data, "edge_type"):
            del data.edge_type
    else:
        data.edge_index = remove_self_loops(to_undirected(edge_index))[0]
        if hasattr(data, "edge_type"):
            del data.edge_type

    data.edge_index = data.edge_index.to(torch.int64)
    # reset cache
    data._data_list = None
    return data


class FGLDataset(Dataset):
    def __init__(self, args, transform=None, pre_transform=None, pre_filter=None):
        """Federated Graph Learning Dataset class.
        This class handles the creation and management of datasets for federated graph learning
        scenarios.

        Args:
            args (Namespace): Arguments specifying the dataset and simulation parameters.
            transform (Optional[Callable]): A function/transform that takes in a Data object
                and returns a transformed version.
            pre_transform (Optional[Callable]): A function/transform that takes in a Data object
                and returns a transformed version before saving to disk.
            pre_filter (Optional[Callable]): A function that takes in a Data object and returns
                a boolean value, indicating whether the data object should be included in the final dataset.
        """
        self.check_args(args)
        self.args = args
        super(FGLDataset, self).__init__(args.root, transform, pre_transform, pre_filter)
        self.load_data()

    
    @property
    def global_root(self) -> str:
        """Get the global root directory for datasets."""
        return osp.join(self.root, "global")
    
    @property
    def distrib_root(self) -> str:
        """Get the distributed root directory for datasets."""
        return osp.join(self.root, "distrib")
    
    
    @property
    def raw_dir(self) -> str:
        """Get the raw directory for datasets."""
        return self.root

    def check_args(self, args):
        """Check the validity of the provided arguments.
        Args:
            args (Namespace): Arguments specifying the dataset and simulation parameters.
        """
        if args.scenario == "graph_fl":
            from fedgb.legacy_config import supported_graph_fl_datasets, supported_graph_fl_simulations, supported_graph_fl_task
            for dataset in args.dataset:
                assert dataset in supported_graph_fl_datasets, f"Invalid graph-fl dataset '{dataset}'."
            assert args.simulation_mode in supported_graph_fl_simulations, f"Invalid graph_fl simulation mode '{args.simulation_mode}'."
            assert args.task in supported_graph_fl_task, f"Invalid graph-fl task '{args.task}'."
            
            
        elif args.scenario == "subgraph_fl":
            from fedgb.legacy_config import supported_subgraph_fl_datasets, supported_subgraph_fl_simulations, supported_subgraph_fl_task
            for dataset in args.dataset:
                assert dataset in supported_subgraph_fl_datasets, f"Invalid subgraph_fl dataset '{dataset}'."
            assert args.simulation_mode in supported_subgraph_fl_simulations, f"Invalid subgraph_fl simulation mode '{args.simulation_mode}'."
            assert args.task in supported_subgraph_fl_task, f"Invalid graph_fl task '{args.task}'."
        
        if args.simulation_mode == "graph_fl_cross_domain":
            assert len(args.dataset) == args.num_clients , f"For graph-fl cross domain simulation, the number of clients must be equal to the number of used datasets (args.num_clients={args.num_clients}; used_datasets: {args.dataset})."
        elif args.simulation_mode == "graph_fl_label_skew":
            assert len(args.dataset) == 1, f"For graph-fl label skew simulation, only single dataset is supported."
        elif args.simulation_mode == "subgraph_fl_label_skew":
            assert len(args.dataset) == 1, f"For subgraph-fl label skew simulation, only single dataset is supported."
        elif args.simulation_mode == "subgraph_fl_louvain_plus":
            assert len(args.dataset) == 1, f"For subgraph-fl louvain clustering simulation, only single dataset is supported."
        elif args.simulation_mode == "subgraph_fl_metis_plus":
            assert len(args.dataset) == 1, f"For subgraph-fl metis clustering simulation, only single dataset is supported."
            
        
    
    @property
    def processed_dir(self) -> str:
        """Get the processed directory for datasets."""
        release_partition = getattr(self.args, "processed_partition", None)
        if release_partition:
            return osp.join(self.distrib_root, release_partition)
        if self.args.simulation_mode in ["subgraph_fl_label_skew", "graph_fl_label_skew"]:
            skew_alpha = getattr(self.args, "skew_alpha", self.args.dirichlet_alpha)
            simulation_name = f"{self.args.simulation_mode}_{skew_alpha:.2f}"
        elif self.args.simulation_mode in ["subgraph_fl_louvain_plus", "subgraph_fl_louvain"]:
            simulation_name = f"{self.args.simulation_mode}_{self.args.louvain_resolution}"
        elif self.args.simulation_mode in ["subgraph_fl_metis_plus"]:
            simulation_name = f"{self.args.simulation_mode}_{self.args.metis_num_coms}"
        else:
            simulation_name = self.args.simulation_mode
            
        fmt_dataset_list = copy.deepcopy(self.args.dataset)
        fmt_dataset_list = sorted(fmt_dataset_list)
           
        
        return osp.join(self.distrib_root,
                        "_".join([simulation_name, "_".join(fmt_dataset_list), f"client_{self.args.num_clients}"]))
        
                            
    @property
    def raw_file_names(self):
        """Get the raw file names for the dataset."""
        return []

    @property
    def processed_file_names(self) -> str:
        """Get the processed file names for the dataset."""
        files_names = ["data_{}.pt".format(i) for i in range(self.args.num_clients)]
        return files_names

    def len(self):
        """Return the number of local client datasets for PyG Dataset compatibility."""
        return self.args.num_clients

    def get(self, idx):
        """Return one local client dataset for PyG Dataset compatibility."""
        return self.local_data[idx]


    def get_client_data(self, client_id):
        """Get the data for a specific client.

        Args:
            client_id (int): The client ID.

        Returns:
            Data: The data object for the client.
        """
        data = torch.load(osp.join(self.processed_dir, "data_{}.pt".format(client_id)), weights_only=False)
        public_scenario = getattr(
            self.args,
            "public_scenario",
            "graph" if self.args.scenario == "graph_fl" else "homo_subgraph",
        )
        data = normalize_client_payload(
            data,
            scenario=public_scenario,
            task=self.args.task,
            target_node=getattr(self.args, "target_node", None),
        )
        return preprocess_loaded_pyg_data(data)

    def save_client_data(self, data, client_id):
        """Save the data for a specific client.

        Args:
            data (Data): The data object to be saved.
            client_id (int): The client ID.
        """
        torch.save(data, osp.join(self.processed_dir, "data_{}.pt".format(client_id)))

    def process(self):
        """Process the dataset according to the specified simulation mode."""
        if len(self.args.dataset) == 1:
            global_dataset = load_global_dataset(self.global_root, scenario=self.args.scenario, dataset=self.args.dataset[0])
        else:
            global_dataset = [load_global_dataset(self.global_root, scenario=self.args.scenario, dataset=dataset_i) for dataset_i in self.args.dataset]

        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if self.args.simulation_mode == "graph_fl_label_skew":
            from fedgb.data.simulation import graph_fl_label_skew
            self.local_data = graph_fl_label_skew(self.args, global_dataset)
        elif self.args.simulation_mode == "graph_fl_cross_domain":
            from fedgb.data.simulation import graph_fl_cross_domain
            self.local_data = graph_fl_cross_domain(self.args, global_dataset)
        elif self.args.simulation_mode == "graph_fl_topology_skew":
            from fedgb.data.simulation import graph_fl_topology_skew
            self.local_data = graph_fl_topology_skew(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_label_skew":
            from fedgb.data.simulation import subgraph_fl_label_skew
            self.local_data = subgraph_fl_label_skew(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_louvain_plus":
            from fedgb.data.simulation import subgraph_fl_louvain_plus
            self.local_data = subgraph_fl_louvain_plus(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_metis_plus":
            from fedgb.data.simulation import subgraph_fl_metis_plus
            self.local_data = subgraph_fl_metis_plus(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_louvain":
            from fedgb.data.simulation import subgraph_fl_louvain
            self.local_data = subgraph_fl_louvain(self.args, global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_metis":
            from fedgb.data.simulation import subgraph_fl_metis
            self.local_data = subgraph_fl_metis(self.args, global_dataset)
        elif self.args.simulation_mode == "graph_fl_feature_skew":
            from fedgb.data.simulation import graph_fl_feature_skew
            self.local_data = graph_fl_feature_skew(self.args, global_dataset)

        
        
        for client_id in range(self.args.num_clients):
            self.save_client_data(self.local_data[client_id], client_id)
            
        self.save_dataset_description()
        
    def save_dataset_description(self):
        """Save the description of the dataset to a file."""
        file_path = os.path.join(self.processed_dir, "description.txt")
        args_str = json.dumps(vars(self.args), indent=4)
        with open(file_path, 'w') as file:
            file.write(args_str)
            print(f"Saved dataset arguments to {file_path}.")


    def load_data(self):
        """Load the data for all clients."""
        self.local_data = [self.get_client_data(client_id) for client_id in range(self.args.num_clients)]
        
        
        if len(self.args.dataset) == 1:
            try:
                global_dataset = load_global_dataset(self.global_root, scenario=self.args.scenario, dataset=self.args.dataset[0])
            except Exception as exc:
                if self.args.scenario != "subgraph_fl":
                    raise
                print(
                    f"Warning: failed to load global dataset from {self.global_root} "
                    f"({exc}). Building global data from local client files instead."
                )
                global_dataset = build_global_dataset_from_local_data(self.local_data)
            if self.args.scenario == "graph_fl":
                self.global_data = global_dataset
            else:
                self.global_data = global_dataset.data
                if isinstance(self.global_data, HeteroData) or hasattr(self.global_data, "node_types"):
                    from fedgb.data.simulation import hetero_to_homogeneous_node_data
                    self.global_data = hetero_to_homogeneous_node_data(
                        self.global_data,
                        getattr(self.args, "target_node", None),
                    )
                self.global_data = preprocess_loaded_pyg_data(self.global_data)
                
            if self.args.task == "graph_reg":
                self.global_data.num_targets = int(getattr(global_dataset, "num_targets", 1))
            else:
                self.global_data.num_global_classes = infer_num_classes(
                    global_dataset,
                    target_node=getattr(self.args, "target_node", None),
                    global_data=self.global_data,
                )
        else:
            self.global_data = None
        
