import torch
import torch.nn as nn
from fedgb.tasks.base import BaseTask
from fedgb.utils.basic_utils import extract_floats, idx_to_mask_tensor, mask_tensor_to_idx
from os import path as osp
from fedgb.utils.metrics import compute_supervised_metrics
import os
import torch
from fedgb.utils.task_utils import load_graph_cls_default_model
import pickle
from torch_geometric.loader import DataLoader
import numpy as np
from fedgb.data.processing import processing



class GraphClsTask(BaseTask):
    """
    Task class for graph classification in a federated learning setup.

    Attributes:
        client_id (int): ID of the client.
        data_dir (str): Directory containing the data.
        args (Namespace): Arguments containing model and training configurations.
        device (torch.device): Device to run the computations on.
        data (object): Data specific to the task.
        model (torch.nn.Module): Model to be trained.
        optim (torch.optim.Optimizer): Optimizer for the model.
        train_mask (torch.Tensor): Mask for the training set.
        val_mask (torch.Tensor): Mask for the validation set.
        test_mask (torch.Tensor): Mask for the test set.
        train_dataloader (DataLoader): DataLoader for the training set.
        val_dataloader (DataLoader): DataLoader for the validation set.
        test_dataloader (DataLoader): DataLoader for the test set.
        splitted_data (dict): Dictionary containing split data and DataLoaders.
        processed_data (object): Processed data for training.
    """
    
    def __init__(self, args, client_id, data, data_dir, device):
        """
        Initialize the GraphClsTask with provided arguments, data, and device.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): ID of the client.
            data (object): Data specific to the task.
            data_dir (str): Directory containing the data.
            device (torch.device): Device to run the computations on.
        """
        super(GraphClsTask, self).__init__(args, client_id, data, data_dir, device)
        
    def _move_batch(self, batch):
        return batch.to(self.device)
        
        
        
    def _graphs_from_mask(self, mask):
        indices = mask.detach().cpu().nonzero(as_tuple=True)[0]
        return self.data[indices]

    def train(self, splitted_data=None):
        """
        Train the model on the provided or processed data.

        Args:
            splitted_data (dict, optional): Dictionary containing split data and DataLoaders. Defaults to None.
        """
        if splitted_data is None:
            splitted_data = self.processed_data # use processed_data to train
        else:
            names = ["data", "train_dataloader", "val_dataloader", "test_dataloader", "train_mask", "val_mask", "test_mask"]
            for name in names:
                assert name in splitted_data
                
        self.model.train()
        for _ in range(self.args.num_epochs):
            for batch in splitted_data["train_dataloader"]:
                batch = self._move_batch(batch)
                self.optim.zero_grad()
                embedding, logits = self.model.forward(batch)
                loss_train = self.loss_fn(embedding, logits, batch.y, torch.ones_like(batch.y).bool())
                loss_train.backward()
                if self.step_preprocess is not None:
                    self.step_preprocess()
                self.optim.step()
            
    def evaluate(self, splitted_data=None, mute=False):
        """
        Evaluate the model on the provided or processed data.

        Args:
            splitted_data (dict, optional): Dictionary containing split data and DataLoaders. Defaults to None.
            mute (bool, optional): If True, suppress the print statements. Defaults to False.

        Returns:
            dict: Dictionary containing evaluation metrics and results.
        """
        if splitted_data is None:
            splitted_data = self.splitted_data # use splitted_data to evaluate
        else:
            names = ["data", "train_dataloader", "val_dataloader", "test_dataloader", "train_mask", "val_mask", "test_mask"]
            for name in names:
                assert name in splitted_data
                
        eval_output = {}
        self.model.eval()
        
        num_samples = len(splitted_data["data"])
        num_global_classes = splitted_data["data"].num_global_classes
        
        embedding_all = torch.zeros((num_samples, self.args.hid_dim)).to(self.device)
        logits_all = torch.zeros((num_samples, num_global_classes)).to(self.device)
        label_all = torch.zeros((num_samples)).to(self.device).long()
        
        train_idx = splitted_data["train_mask"].nonzero().squeeze().tolist()
        if isinstance(train_idx, int):
            train_idx = [train_idx]
        val_idx = splitted_data["val_mask"].nonzero().squeeze().tolist()
        if isinstance(val_idx, int):
            val_idx = [val_idx]
        test_idx = splitted_data["test_mask"].nonzero().squeeze().tolist()
        if isinstance(test_idx, int):
            test_idx = [test_idx]
        
        
        train_cnt = 0
        val_cnt = 0
        test_cnt = 0
        
        with torch.no_grad():
            for batch in splitted_data["train_dataloader"]:
                batch = self._move_batch(batch)
                embedding, logits = self.model.forward(batch)
                embedding_all[train_idx[train_cnt:train_cnt+batch.num_graphs]] = embedding
                logits_all[train_idx[train_cnt:train_cnt+batch.num_graphs]] = logits
                label_all[train_idx[train_cnt:train_cnt+batch.num_graphs]] = batch.y
                train_cnt += batch.num_graphs
            for batch in splitted_data["val_dataloader"]:
                batch = self._move_batch(batch)
                embedding, logits = self.model.forward(batch)
                embedding_all[val_idx[val_cnt:val_cnt+batch.num_graphs]] = embedding
                logits_all[val_idx[val_cnt:val_cnt+batch.num_graphs]] = logits
                label_all[val_idx[val_cnt:val_cnt+batch.num_graphs]] = batch.y
                val_cnt += batch.num_graphs
            for batch in splitted_data["test_dataloader"]:
                batch = self._move_batch(batch)
                embedding, logits = self.model.forward(batch)
                embedding_all[test_idx[test_cnt:test_cnt+batch.num_graphs]] = embedding
                logits_all[test_idx[test_cnt:test_cnt+batch.num_graphs]] = logits
                label_all[test_idx[test_cnt:test_cnt+batch.num_graphs]] = batch.y
                test_cnt += batch.num_graphs

            loss_train = self.loss_fn(embedding_all, logits_all, label_all, splitted_data["train_mask"])
            loss_val = self.loss_fn(embedding_all, logits_all, label_all, splitted_data["val_mask"])
            loss_test = self.loss_fn(embedding_all, logits_all, label_all, splitted_data["test_mask"])

        eval_output["embedding"] = embedding_all
        eval_output["logits"] = logits_all
        eval_output["loss_train"] = loss_train 
        eval_output["loss_val"]   = loss_val
        eval_output["loss_test"]  = loss_test
        
        
        metric_train = compute_supervised_metrics(metrics=self.args.metrics, logits=logits_all[splitted_data["train_mask"]], labels=label_all[splitted_data["train_mask"]], suffix="train")
        metric_val = compute_supervised_metrics(metrics=self.args.metrics, logits=logits_all[splitted_data["val_mask"]], labels=label_all[splitted_data["val_mask"]], suffix="val")
        metric_test = compute_supervised_metrics(metrics=self.args.metrics, logits=logits_all[splitted_data["test_mask"]], labels=label_all[splitted_data["test_mask"]], suffix="test")
        eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}
        
        info = ""
        for key, val in eval_output.items():
            try:
                info += f"\t{key}: {val:.4f}"
            except:
                continue
            
        prefix = f"[client {self.client_id}]" if self.client_id is not None else "[server]"
        if not mute:
            print(prefix+info)
        return eval_output
    

    def loss_fn(self, embedding, logits, label, mask):
        """
        Calculate the loss for the model.

        Args:
            embedding (torch.Tensor): Embeddings from the model.
            logits (torch.Tensor): Logits from the model.
            label (torch.Tensor): Ground truth labels.
            mask (torch.Tensor): Mask to filter the logits and labels.

        Returns:
            torch.Tensor: Calculated loss.
        """
        return self.default_loss_fn(logits[mask], label[mask])
        
    @property
    def default_model(self):   
        """
        Get the default model for graph classification.

        Returns:
            torch.nn.Module: Default model.
        """         
        return load_graph_cls_default_model(self.args, input_dim=self.num_feats, output_dim=self.num_global_classes, client_id=self.client_id)
    
    @property
    def default_optim(self):
        """
        Get the default optimizer for the task.

        Returns:
            torch.optim.Optimizer: Default optimizer.
        """
        if self.args.optim == "adam":
            from torch.optim import Adam
            return Adam
    
    @property
    def num_samples(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)
    
    @property
    def num_feats(self):
        """
        Get the number of features in the dataset.

        Returns:
            int: Number of features.
        """
        return self.data[0].x.shape[1]
    
    @property
    def num_global_classes(self):
        """
        Get the number of global classes in the dataset.

        Returns:
            int: Number of global classes.
        """
        return self.data.num_global_classes
        
    @property
    def default_loss_fn(self):
        """
        Get the default loss function for the task.

        Returns:
            function: Default loss function.
        """
        return nn.CrossEntropyLoss()
    
    @property
    def default_train_val_test_split(self):
        """
        Get the default train/validation/test split.

        Returns:
            tuple: Default train/validation/test split ratios.
        """
        return 0.8, 0.1, 0.1
        
  
    @property
    def train_val_test_path(self):
        """
        Get the path to the train/validation/test split file.

        Returns:
            str: Path to the split file.
        """
        if self.args.train_val_test == "default_split":
            return osp.join(self.data_dir, f"graph_cls", "default_split")
        else:
            split_dir = f"split_{self.args.train_val_test}" 
            return osp.join(self.data_dir, f"graph_cls", split_dir)
    

    def load_train_val_test_split(self):
        if self.client_id is None and len(self.args.dataset) == 1:
            train_mask, val_mask, test_mask = self._load_server_split()
        else:
            train_mask, val_mask, test_mask = self._load_client_split()

        self._attach_graph_cache_ids()
        self.train_mask = train_mask.to(self.device)
        self.val_mask = val_mask.to(self.device)
        self.test_mask = test_mask.to(self.device)

        self.train_dataloader = DataLoader(self._graphs_from_mask(self.train_mask), batch_size=self.args.batch_size, shuffle=False)
        self.val_dataloader = DataLoader(self._graphs_from_mask(self.val_mask), batch_size=self.args.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self._graphs_from_mask(self.test_mask), batch_size=self.args.batch_size, shuffle=False)

        self.splitted_data = {
            "data": self.data,
            "train_dataloader": self.train_dataloader,
            "val_dataloader": self.val_dataloader,
            "test_dataloader": self.test_dataloader,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask,
        }
        self.processed_data = processing(args=self.args, splitted_data=self.splitted_data, processed_dir=self.data_dir, client_id=self.client_id)

    def _load_server_split(self):
        glb_train, glb_val, glb_test = [], [], []
        for client_id in range(self.args.num_clients):
            for name, bucket in [("train", glb_train), ("val", glb_val), ("test", glb_test)]:
                split_path = osp.join(self.train_val_test_path, f"glb_{name}_{client_id}.pkl")
                if osp.exists(split_path):
                    with open(split_path, "rb") as file:
                        bucket += pickle.load(file)
        all_ids = glb_train + glb_val + glb_test
        if all_ids and max(all_ids) < self.num_samples:
            return (
                idx_to_mask_tensor(glb_train, self.num_samples).bool(),
                idx_to_mask_tensor(glb_val, self.num_samples).bool(),
                idx_to_mask_tensor(glb_test, self.num_samples).bool(),
            )
        return self.local_graph_train_val_test_split(self.data, self.args.train_val_test)

    def _load_client_split(self):
        train_path = osp.join(self.train_val_test_path, f"train_{self.client_id}.pt")
        val_path = osp.join(self.train_val_test_path, f"val_{self.client_id}.pt")
        test_path = osp.join(self.train_val_test_path, f"test_{self.client_id}.pt")
        glb_train_path = osp.join(self.train_val_test_path, f"glb_train_{self.client_id}.pkl")
        glb_val_path = osp.join(self.train_val_test_path, f"glb_val_{self.client_id}.pkl")
        glb_test_path = osp.join(self.train_val_test_path, f"glb_test_{self.client_id}.pkl")
        if osp.exists(train_path) and osp.exists(val_path) and osp.exists(test_path):
            return torch.load(train_path), torch.load(val_path), torch.load(test_path)
        train_mask, val_mask, test_mask = self.local_graph_train_val_test_split(self.data, self.args.train_val_test)
        os.makedirs(self.train_val_test_path, exist_ok=True)
        torch.save(train_mask, train_path)
        torch.save(val_mask, val_path)
        torch.save(test_mask, test_path)
        if len(self.args.dataset) == 1:
            for mask, split_path in [
                (train_mask, glb_train_path),
                (val_mask, glb_val_path),
                (test_mask, glb_test_path),
            ]:
                global_ids = [self.data.global_map[idx.item()] for idx in mask.nonzero()]
                with open(split_path, "wb") as file:
                    pickle.dump(global_ids, file)
        return train_mask, val_mask, test_mask

    def _attach_graph_cache_ids(self):
        if not hasattr(self.data, "global_map"):
            return
        for local_id, graph in enumerate(self.data):
            cache_id = self.data.global_map.get(local_id, local_id)
            graph.fedssp_cache_id = torch.tensor([int(cache_id)], dtype=torch.long)
        

    def local_graph_train_val_test_split(self, local_graphs, split, shuffle=True):
        """
        Split the local graphs into train, validation, and test sets.

        Args:
            local_graphs (object): Local graphs to be split.
            split (str or tuple): Split ratios or default split identifier.
            shuffle (bool, optional): If True, shuffle the graphs before splitting. Defaults to True.

        Returns:
            tuple: Masks for the train, validation, and test sets.
        """
        num_graphs = self.num_samples
        
        if split == "default_split":
            train_, val_, test_ = self.default_train_val_test_split
        else:
            train_, val_, test_ = extract_floats(split)
        
        train_mask = idx_to_mask_tensor([], num_graphs)
        val_mask = idx_to_mask_tensor([], num_graphs)
        test_mask = idx_to_mask_tensor([], num_graphs)
        labels = local_graphs.y.view(-1)
        for class_i in range(local_graphs.num_global_classes):
            class_i_graph_mask = labels == class_i
            num_class_i_graphs = class_i_graph_mask.sum()
            class_i_graph_list = mask_tensor_to_idx(class_i_graph_mask)
            if shuffle:
                np.random.shuffle(class_i_graph_list)
            train_mask += idx_to_mask_tensor(class_i_graph_list[:int(train_ * num_class_i_graphs)], num_graphs)
            val_mask += idx_to_mask_tensor(class_i_graph_list[int(train_ * num_class_i_graphs) : int((train_+val_) * num_class_i_graphs)], num_graphs)
            test_mask += idx_to_mask_tensor(class_i_graph_list[int((train_+val_) * num_class_i_graphs): min(num_class_i_graphs, int((train_+val_+test_) * num_class_i_graphs))], num_graphs)
        
        
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
        return train_mask, val_mask, test_mask
