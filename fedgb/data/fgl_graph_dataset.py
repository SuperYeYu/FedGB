import copy
from typing import Iterable, Sequence

import torch


class FGLGraphDataset:
    """Lightweight graph-level dataset for pre-partitioned FGL graph clients."""

    def __init__(
        self,
        graphs,
        num_targets=1,
        global_map=None,
        client_name=None,
        task_type="graph_regression",
        num_global_classes=None,
        target_names=None,
        active_target_names=None,
    ):
        self.graphs = list(graphs)
        self.num_targets = int(num_targets)
        self.target_names = list(target_names or [])
        if self.target_names and len(self.target_names) != self.num_targets:
            raise ValueError("target_names must match num_targets")
        self.active_target_names = list(
            self.target_names if active_target_names is None else active_target_names
        )
        self.client_name = client_name
        self.task_type = task_type
        self.num_global_classes = int(num_global_classes) if num_global_classes is not None else None
        self.global_map = global_map if global_map is not None else {i: i for i in range(len(self.graphs))}
        self.y = self._stack_y()
        self.y_mask = self._stack_y_mask()

    def _is_classification(self):
        return self.task_type in {"graph_cls", "graph_classification", "classification"}

    def _stack_y(self):
        if len(self.graphs) == 0:
            if self._is_classification():
                return torch.empty((0,), dtype=torch.long)
            return torch.empty((0, self.num_targets), dtype=torch.float32)

        if self._is_classification():
            labels = []
            for graph in self.graphs:
                y = graph.y if torch.is_tensor(graph.y) else torch.tensor(graph.y)
                labels.append(int(y.view(-1)[0].item()))
            return torch.tensor(labels, dtype=torch.long)

        labels = []
        for graph in self.graphs:
            y = graph.y if torch.is_tensor(graph.y) else torch.tensor(graph.y)
            labels.append(y.float().view(1, -1))
        return torch.cat(labels, dim=0)

    def _stack_y_mask(self):
        if len(self.graphs) == 0:
            return torch.empty((0, self.num_targets), dtype=torch.bool)

        masks = []
        for graph in self.graphs:
            y = graph.y if torch.is_tensor(graph.y) else torch.tensor(graph.y)
            mask = getattr(graph, "y_mask", None)
            if mask is None:
                mask = torch.ones_like(y, dtype=torch.bool)
            elif not torch.is_tensor(mask):
                mask = torch.tensor(mask)
            mask = mask.bool().view(1, -1)
            if mask.shape[1] != self.num_targets:
                raise ValueError("graph y_mask width must match num_targets")
            masks.append(mask)
        return torch.cat(masks, dim=0)

    def __len__(self):
        return len(self.graphs)

    def __iter__(self):
        return iter(self.graphs)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.graphs[idx]
        if isinstance(idx, slice):
            indices = list(range(len(self.graphs)))[idx]
            return self.copy(indices)
        if torch.is_tensor(idx):
            idx = idx.detach().cpu()
            if idx.dtype == torch.bool:
                indices = idx.nonzero(as_tuple=True)[0].tolist()
            else:
                indices = idx.view(-1).long().tolist()
            return self.copy(indices)
        if isinstance(idx, Sequence):
            if len(idx) > 0 and isinstance(idx[0], bool):
                indices = [i for i, flag in enumerate(idx) if flag]
            else:
                indices = [int(i) for i in idx]
            return self.copy(indices)
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def copy(self, indices=None):
        if indices is None:
            indices = list(range(len(self.graphs)))
        indices = list(indices)
        graphs = [self.graphs[i] for i in indices]
        global_map = {new_i: self.global_map.get(old_i, old_i) for new_i, old_i in enumerate(indices)}
        return FGLGraphDataset(
            graphs=graphs,
            num_targets=self.num_targets,
            global_map=global_map,
            client_name=self.client_name,
            task_type=self.task_type,
            num_global_classes=getattr(self, "num_global_classes", None),
            target_names=getattr(self, "target_names", None),
            active_target_names=getattr(self, "active_target_names", None),
        )

    def to(self, device):
        # Keep graph objects on CPU for large graph-level datasets. Tasks move mini-batches to device.
        return self

    @property
    def num_features(self):
        return self.graphs[0].x.shape[1] if len(self.graphs) > 0 else 0

    @property
    def num_classes(self):
        num_global_classes = getattr(self, "num_global_classes", None)
        if num_global_classes is not None:
            return num_global_classes
        if self.y.numel() == 0:
            return 0
        return int(self.y.max().item()) + 1
