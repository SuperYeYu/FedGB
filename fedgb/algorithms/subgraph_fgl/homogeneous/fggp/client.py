import torch
import torch.nn as nn
from fedgb.training.base import BaseClient
import copy
from torch_geometric.utils import to_torch_csc_tensor
from fedgb.algorithms.subgraph_fgl.homogeneous.fggp.models import FedGCN, MLP
from sklearn.neighbors import kneighbors_graph
from fedgb.algorithms.subgraph_fgl.homogeneous.fggp.fggp_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fggp.utils import (
    com_distillation_loss,
    global_prototype_guidance_loss,
    global_prototype_logits,
    get_norm_and_orig,
    get_proto_norm_weighted,
    prototypes_to_label_dict,
    proto_align_loss,
)
import torch.nn.functional as F
from fedgb.utils.metrics import compute_supervised_metrics
try:
    from pynndescent import NNDescent
except Exception:
    NNDescent = None


DENSE_AUG_NODE_LIMIT = 12000
NEGATIVE_EDGE_RATIO = 1.0
MAX_AUG_EDGES = 400000
ANN_NODE_THRESHOLD = 20000


class FGGPClient(BaseClient):
    """
    FGGPClient is a client-side implementation for the Federated Graph Learning with Generalizable Prototypes 
    (FGGP) framework. This client handles local training, model updates, and prototype generation in a 
    federated learning setting, focusing on overcoming domain shifts across clients.

    Attributes:
        global_model (nn.Module): A copy of the global model used to compute global embeddings.
        personal_project (nn.Module): A projection layer used for personalizing embeddings.
        data2 (torch_geometric.data.Data): A copy of the data with modified edges for use in the FGGP algorithm.
    """
    
    
    
    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        """
        Initializes the FGGPClient.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            client_id (int): The ID of the client.
            data (torch_geometric.data.Data): The graph data specific to the client's task.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between client and server.
            device (torch.device): The device on which computations will be performed (e.g., CPU or GPU).
        """
        super(FGGPClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        self.task.load_custom_model(FedGCN(nfeat=self.task.num_feats, nhid=self.args.hid_dim,
                                           nclass=self.task.num_global_classes, nlayer=self.args.num_layers,
                                           dropout=self.args.dropout))
        self.global_model = copy.deepcopy(self.task.model)
        self.task.splitted_data["data"].adj = self._normalized_adj(self.task.data.edge_index, self.task.data.num_nodes)
        self.task.data.adj = self.task.splitted_data["data"].adj
        self.personal_project = MLP(self.args.hid_dim,self.args.hid_dim,0.5)
        self.global_protos = {}
        self.current_round = 0
        self.task.override_evaluate = self.evaluate

    def _normalized_adj(self, edge_index, num_nodes):
        edge_index = edge_index.to(self.device)
        loop = torch.arange(num_nodes, device=self.device)
        edge_index = torch.cat([edge_index, torch.stack([loop, loop], dim=0)], dim=1)
        values = torch.ones(edge_index.shape[1], device=self.device)
        row, col = edge_index
        degree = values.new_zeros(num_nodes)
        degree.scatter_add_(0, row, values)
        norm_values = values * degree.clamp_min(1e-12).pow(-0.5)[row] * degree.clamp_min(1e-12).pow(-0.5)[col]
        return torch.sparse_coo_tensor(edge_index, norm_values, (num_nodes, num_nodes), device=self.device).coalesce()




    def get_custom_loss_fn(self):
        """
        Returns the custom loss function used during local training. The loss function includes:
        - Cross-entropy loss for classification.
        - Graph augmentation loss for learning on augmented graph structures.
        - Prototype alignment loss to align local and global prototypes.
        """
        def custom_loss_fn(embedding, logits, label, mask):
            loss_ce = torch.nn.functional.cross_entropy(logits[mask], label[mask])
            if self.current_round < int(config["fggp_warmup_rounds"]):
                return loss_ce

            adj_sampled, adj_logits = self.task.model.aug(self.data2)
            self.data2.adj = adj_sampled
            emb_g,logits_g = self.task.model(self.data2)
            ga_loss = loss_ce.new_zeros(())
            if config["ga_weight"] > 0:
                if hasattr(self.data2, "aug_edge_label"):
                    ga_loss = F.binary_cross_entropy_with_logits(
                        adj_logits,
                        self.data2.aug_edge_label.to(adj_logits.device, dtype=adj_logits.dtype),
                        pos_weight=self.data2.pos_weight.to(adj_logits.device),
                    )
                else:
                    ga_loss = self.data2.norm_w * F.binary_cross_entropy_with_logits(
                        adj_logits,
                        self.data2.adj_orig,
                        pos_weight=self.data2.pos_weight,
                    )
            loss_ce2 = loss_ce.new_zeros(())
            if config["aug_ce_weight"] > 0:
                loss_ce2 = F.cross_entropy(logits_g[mask],self.data2.y[mask])
            kd_loss = loss_ce.new_zeros(())
            if config["kd_weight"] > 0:
                kd_loss = com_distillation_loss(
                    logits,
                    logits_g,
                    self.data2.adj_orig if hasattr(self.data2, "adj_orig") else None,
                    adj_sampled,
                    edge_index=getattr(self.data2, "aug_edge_index", None),
                    edge_label=getattr(self.data2, "aug_edge_label", None),
                    temperature=config["kd_temperature"],
                    detach_teacher=config["kd_detach_teacher"],
                )
            pseudo_labels, confidences, proto_mask = self._prototype_assignments(logits, label, mask)
            unique_labels = torch.unique(pseudo_labels[proto_mask])
            loss_pa = loss_ce.new_zeros(())
            if unique_labels.numel() > 1:
                proto = get_proto_norm_weighted(
                    self.task.num_global_classes,
                    embedding[proto_mask],
                    pseudo_labels[proto_mask],
                    confidences[proto_mask],
                    unique_labels,
                )
                proto_global = get_proto_norm_weighted(
                    self.task.num_global_classes,
                    emb_g[proto_mask],
                    pseudo_labels[proto_mask],
                    confidences[proto_mask],
                    unique_labels,
                )
                loss_pa = proto_align_loss(proto_global, proto, temperature=0.5)
            loss_global_proto = global_prototype_guidance_loss(
                embedding,
                label,
                mask,
                self.global_protos,
                self.task.num_global_classes,
                temperature=config["global_proto_temperature"],
                mse_weight=config["global_proto_mse_weight"],
            )

            loss = (
                config["ce_weight"] * loss_ce
                + config["ga_weight"] * ga_loss
                + config["aug_ce_weight"] * loss_ce2
                + config["pa_weight"] * loss_pa
                + config["kd_weight"] * kd_loss
                + config["global_proto_weight"] * loss_global_proto
            )
            return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        return custom_loss_fn




    def execute(self):
        """
        Executes the local training process. This involves:
        - Synchronizing the local and global model parameters with the server.
        - Calculating the k-nearest neighbors graph for global embeddings.
        - Training the model using the custom loss function.
        """
        server_msg = self.message_pool.get("server", {})
        self.current_round = int(self.message_pool.get("round", 0))
        self.global_protos = {
            int(label): [proto.to(self.device) for proto in proto_list]
            for label, proto_list in server_msg.get("global_protos", {}).items()
        }
        with torch.no_grad():
            for (local_param, g_p,global_param) in zip(self.task.model.parameters(), self.global_model.parameters(),server_msg["weight"]):
                local_param.data.copy_(global_param)
                g_p.data.copy_(global_param)


        self.task.loss_fn = self.get_custom_loss_fn()

        self.data2 = self.task.splitted_data["data"].clone()
        if self.current_round < int(config["fggp_warmup_rounds"]):
            union_edge_index = self.task.data.edge_index
        else:
            self.global_model.eval()
            globel_emb, _ = self.global_model(self.task.data)
            self.task.data.global_edge_index = self._global_knn_edge_index(globel_emb)
            del globel_emb, _
            combined_edge_index = torch.cat([self.task.data.edge_index, self.task.data.global_edge_index], dim=1)
            union_edge_index = torch.unique(combined_edge_index, dim=1)

        self.data2.edge_index = union_edge_index
        self.data2.adj = self._normalized_adj(union_edge_index, self.data2.num_nodes)
        if self.current_round >= int(config["fggp_warmup_rounds"]):
            self._prepare_augmented_view()

        self.task.train()

    def _prototype_assignments(self, logits, labels, train_mask):
        probabilities = torch.softmax(logits, dim=1)
        confidences, predicted_labels = probabilities.max(1)
        confidences = confidences.detach().clone()
        pseudo_labels = predicted_labels.detach().clone().type_as(labels)
        train_mask = train_mask.bool()
        if config["fggp_use_pseudo_prototypes"]:
            proto_mask = train_mask | (confidences >= float(config["fggp_pseudo_confidence_threshold"]))
        else:
            proto_mask = train_mask.clone()
        pseudo_labels[train_mask] = labels[train_mask]
        confidences[train_mask] = 1.0
        if proto_mask.sum() == 0:
            proto_mask = train_mask.clone()
        return pseudo_labels, confidences, proto_mask

    def _global_knn_edge_index(self, embedding):
        embedding_cpu = embedding.detach().cpu()
        num_nodes = embedding_cpu.shape[0]
        k = int(config["neibor_num"])
        if NNDescent is not None and num_nodes >= ANN_NODE_THRESHOLD:
            index = NNDescent(
                embedding_cpu.numpy(),
                n_neighbors=k + 1,
                metric="cosine",
                random_state=int(getattr(self.args, "seed", 0)) + self.client_id,
            )
            neighbors, _ = index.neighbor_graph
            row = torch.arange(num_nodes).repeat_interleave(k)
            col = torch.from_numpy(neighbors[:, 1:k + 1].reshape(-1)).long()
            return torch.stack([row, col], dim=0).to(self.device)

        adj = kneighbors_graph(embedding_cpu, k, metric="cosine")
        adj.setdiag(1)
        coo = adj.tocoo()
        return torch.stack(
            [
                torch.from_numpy(coo.row).long(),
                torch.from_numpy(coo.col).long(),
            ],
            dim=0,
        ).to(self.device)

    def _prepare_augmented_view(self):
        if self.data2.num_nodes <= DENSE_AUG_NODE_LIMIT:
            self.data2 = get_norm_and_orig(self.data2)
            adj_orig = self.data2.adj_orig
            norm_w = adj_orig.shape[0] ** 2 / float((adj_orig.shape[0] ** 2 - adj_orig.sum()) * 2)
            pos_weight = torch.FloatTensor([float(adj_orig.shape[0] ** 2 - adj_orig.sum()) / adj_orig.sum()]).to(
                self.device)
            self.data2.norm_w = norm_w
            self.data2.pos_weight = pos_weight
            return

        edge_index = self.data2.edge_index
        positive_edge_index = torch.unique(edge_index, dim=1)
        num_pos = positive_edge_index.shape[1]
        num_neg = min(int(num_pos * NEGATIVE_EDGE_RATIO), MAX_AUG_EDGES)
        if num_pos > MAX_AUG_EDGES:
            pos_perm = torch.randperm(num_pos, device=edge_index.device)[:MAX_AUG_EDGES]
            positive_edge_index = positive_edge_index[:, pos_perm]
            num_pos = positive_edge_index.shape[1]
            num_neg = min(num_neg, MAX_AUG_EDGES)

        neg_src = torch.randint(self.data2.num_nodes, (num_neg,), device=edge_index.device)
        neg_dst = torch.randint(self.data2.num_nodes, (num_neg,), device=edge_index.device)
        neg_mask = neg_src != neg_dst
        neg_src = neg_src[neg_mask]
        neg_dst = neg_dst[neg_mask]
        negative_edge_index = torch.stack([neg_src, neg_dst], dim=0)

        aug_edge_index = torch.cat([positive_edge_index, negative_edge_index], dim=1)
        aug_edge_label = torch.cat(
            [
                torch.ones(positive_edge_index.shape[1], device=edge_index.device),
                torch.zeros(negative_edge_index.shape[1], device=edge_index.device),
            ],
            dim=0,
        )
        self.data2.aug_edge_index = aug_edge_index
        self.data2.aug_edge_label = aug_edge_label
        self.data2.aug_message_mask = torch.arange(aug_edge_index.shape[1], device=edge_index.device) < positive_edge_index.shape[1]
        self.data2.adj_orig_edge_index = positive_edge_index
        pos = float(positive_edge_index.shape[1])
        neg = float(max(negative_edge_index.shape[1], 1))
        self.data2.pos_weight = torch.tensor([neg / max(pos, 1.0)], device=self.device)

    def evaluate(self, splitted_data=None, mute=False):
        if splitted_data is None:
            splitted_data = self.task.splitted_data
        else:
            names = ["data", "train_mask", "val_mask", "test_mask"]
            for name in names:
                assert name in splitted_data

        self.task.model.eval()
        with torch.no_grad():
            embedding, logits = self.task.model.forward(splitted_data["data"])
            if config["use_fcpp_eval"] and self.global_protos:
                proto_logits, available = global_prototype_logits(
                    embedding,
                    self.global_protos,
                    self.task.num_global_classes,
                    temperature=config["fcpp_temperature"],
                    top_k=config["fcpp_top_k"],
                )
                if available.any():
                    unavailable = ~available
                    proto_logits[:, unavailable] = logits[:, unavailable]
                    logits = (
                        config["fcpp_model_logit_weight"] * logits
                        + config["fcpp_proto_logit_weight"] * proto_logits
                    )
            loss_train = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["train_mask"])
            loss_val = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["val_mask"])
            loss_test = self.task.loss_fn(embedding, logits, splitted_data["data"].y, splitted_data["test_mask"])

        eval_output = {
            "embedding": embedding,
            "logits": logits,
            "loss_train": loss_train,
            "loss_val": loss_val,
            "loss_test": loss_test,
        }
        metric_train = compute_supervised_metrics(
            metrics=self.args.metrics,
            logits=logits[splitted_data["train_mask"]],
            labels=splitted_data["data"].y[splitted_data["train_mask"]],
            suffix="train",
        )
        metric_val = compute_supervised_metrics(
            metrics=self.args.metrics,
            logits=logits[splitted_data["val_mask"]],
            labels=splitted_data["data"].y[splitted_data["val_mask"]],
            suffix="val",
        )
        metric_test = compute_supervised_metrics(
            metrics=self.args.metrics,
            logits=logits[splitted_data["test_mask"]],
            labels=splitted_data["data"].y[splitted_data["test_mask"]],
            suffix="test",
        )
        eval_output = {**eval_output, **metric_train, **metric_val, **metric_test}

        info = ""
        for key, val in eval_output.items():
            try:
                info += f"\t{key}: {val:.4f}"
            except Exception:
                continue

        if not mute:
            print(f"[client {self.client_id}]" + info)
        return eval_output

    def send_message(self):
        """
        Sends the client's local model parameters and the computed prototypes to the server.
        """
        self.task.model.eval()
        emb,logits = self.task.model(self.task.splitted_data["data"])
        #feat = self.personal_project(emb)

        pseudo_labels, confidences, proto_mask = self._prototype_assignments(
            logits,
            self.task.splitted_data["data"].y,
            self.task.splitted_data["train_mask"],
        )
        unique_labels = torch.unique(pseudo_labels[proto_mask])
        proto = get_proto_norm_weighted(
            self.task.num_global_classes,
            emb[proto_mask],
            pseudo_labels[proto_mask],
            confidences[proto_mask],
            unique_labels,
        )
        weights = []
        for label in unique_labels.detach().cpu().tolist():
            selected = proto_mask & (pseudo_labels == int(label))
            weights.append(float(confidences[selected].sum().detach().cpu()))
        tensor_dict = prototypes_to_label_dict(
            proto,
            unique_labels,
            weights=weights if config["fggp_weighted_prototype_aggregation"] else None,
        )


        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "protos" : tensor_dict
        }
