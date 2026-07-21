import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import NeighborSampler

from fedgb.algorithms.subgraph_fgl.homogeneous.feddep.localdep import Encoder
from fedgb.algorithms.subgraph_fgl.homogeneous.feddep.dec_cluster.clustering import train_clustering
from fedgb.algorithms.subgraph_fgl.homogeneous.feddep.feddep_config import config


def _as_stacked_tensor(values, device, dtype):
    if torch.is_tensor(values):
        return values.to(device=device, dtype=dtype)
    if isinstance(values, np.ndarray):
        return torch.as_tensor(values, device=device, dtype=dtype)
    if len(values) == 0:
        return torch.empty(0, device=device, dtype=dtype)
    if torch.is_tensor(values[0]):
        return torch.stack([v.to(device=device, dtype=dtype) for v in values], dim=0)
    return torch.as_tensor(np.asarray(values), device=device, dtype=dtype)


def LocalRecLoss(pred_embs, true_embs, pred_missing, true_missing, num_preds):
    device, dtype = pred_embs.device, pred_embs.dtype
    pred_embs = pred_embs.view(pred_embs.shape[0], num_preds, -1)
    true_embs = _as_stacked_tensor(true_embs, device=device, dtype=dtype).view(pred_embs.shape[0], num_preds, -1)

    pred_missing = torch.round(pred_missing.detach().view(-1).cpu()).long().clamp_(0, num_preds).to(device)
    true_missing = true_missing.detach().view(-1).cpu().long().clamp_(0, num_preds).to(device)

    dist = (pred_embs.unsqueeze(2) - true_embs.unsqueeze(1)).pow(2).mean(dim=-1)
    idx = torch.arange(num_preds, device=device)
    valid_true = idx.view(1, 1, num_preds) < true_missing.view(-1, 1, 1)
    dist = dist.masked_fill(~valid_true, float("inf"))
    min_dist = dist.min(dim=2).values
    min_dist = torch.where(torch.isfinite(min_dist), min_dist, torch.zeros_like(min_dist))

    valid_pred = (idx.view(1, num_preds) < pred_missing.view(-1, 1)) & (true_missing.view(-1, 1) > 0)
    loss = torch.where(valid_pred, min_dist, torch.zeros_like(min_dist))
    return loss.mean(dim=1).mean(dim=0).float()


def FedRecLoss(pred_embs, true_embs, pred_missing, num_preds):
    device, dtype = pred_embs.device, pred_embs.dtype
    pred_embs = pred_embs.view(pred_embs.shape[0], num_preds, -1)
    true_embs = _as_stacked_tensor(true_embs, device=device, dtype=dtype).view(pred_embs.shape[0], num_preds, -1)

    pred_missing = pred_missing.detach().view(-1).cpu().long().clamp_(0, num_preds).to(device)
    dist = (pred_embs.unsqueeze(2) - true_embs.unsqueeze(1)).pow(2).mean(dim=-1)
    min_dist = dist.min(dim=2).values
    idx = torch.arange(num_preds, device=device)
    valid_pred = idx.view(1, num_preds) < pred_missing.view(-1, 1)
    loss = torch.where(valid_pred, min_dist, torch.zeros_like(min_dist))
    return loss.mean(dim=1).mean(dim=0).float()

def get_prototypes(emb, K, batch_size, device, ae_pretrained_epochs, ae_finetune_epochs, dec_epochs):
    emb_shape = emb.shape[1]
    proto_idx = train_clustering(
        node_embs=emb, num_prototypes=K,
        batch_size=batch_size, device=device, ae_pretrained_epochs=ae_pretrained_epochs,
        ae_finetune_epochs=ae_finetune_epochs, dec_epochs=dec_epochs).reshape(-1)

    prototypes = np.zeros(shape=(K, emb_shape))
    proto_idx = np.asarray(proto_idx, dtype=np.int32).reshape(-1)
    if emb.device != "cpu":
        emb = emb.cpu()
    emb = emb.numpy()
    for cluster in range(K):
        row_ix = np.where(proto_idx == cluster)
        prototypes[cluster] = emb[row_ix].mean(axis=0)
    return prototypes, proto_idx


def get_emb(data, hid_dim, output_dim, num_layers, device):
    subgraph_sampler = NeighborSampler(
        data.edge_index,
        num_nodes=data.num_nodes,
        node_idx=torch.tensor([i for i in range(data.num_nodes)]),
        sizes=[5] * num_layers,
        batch_size=4096,
        shuffle=False)
    train_idx = torch.where(data.train_mask == True)[0]
    dataloader = {
        "data": data,
        "train": NeighborSampler(
            data.edge_index,
            num_nodes=data.num_nodes,
            node_idx=train_idx,
            sizes=[5] * num_layers,
            batch_size=config["encoder_batch_size"],
            shuffle=True
        ),
        "val": subgraph_sampler,
        "test": subgraph_sampler
    }

    encoder = Encoder(
        input_dim=data.x.shape[1],
        hid_dim=hid_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.5).to(device)

    encoder.train()
    optim = torch.optim.Adam(encoder.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(config["encoder_epochs"]):
        total_loss, total_correct = 0, 0
        for batch_size, n_id, adjs in dataloader["train"]:
            x, y = data.x[n_id].to(device), data.y[n_id[:batch_size]].to(device)
            out = encoder.forward(x=x, adjs=adjs)
            loss = F.cross_entropy(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            total_correct += int(out.argmax(dim=-1).cpu().eq(y.cpu()).sum())

        loss = total_loss / len(dataloader["train"])
        approx_acc = total_correct / int(data.train_mask.sum())
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}")

    encoder.eval()
    all_emb = encoder.get_encoder(dataloader["data"].x.to(device), dataloader["test"]).detach()
    return all_emb


class HideGraph(BaseTransform):
    def __init__(self, encoder_hid_dim, encoder_output_dim, encoder_num_layers, hidden_portion, num_preds, num_protos, device):
        self.encoder_hid_dim = encoder_hid_dim
        self.encoder_output_dim = encoder_output_dim
        self.encoder_num_layers = encoder_num_layers
        self.hidden_portion = hidden_portion
        self.num_preds = num_preds
        self.num_protos = num_protos
        self.device = device
        
        
    def forward(self, data):
        # get prototypes
        emb = get_emb(data=data, hid_dim=self.encoder_hid_dim, output_dim=self.encoder_output_dim, num_layers=self.encoder_num_layers, device=self.device)
        cluster_emb = emb.detach().cpu()
        self.prototypes, self.proto_idx = get_prototypes(
            emb=cluster_emb,
            K=self.num_protos,
            batch_size=config["cluster_batch_size"],
            device=torch.device("cpu"),
            ae_pretrained_epochs=config["ae_pretrained_epochs"],
            ae_finetune_epochs=config["ae_finetune_epochs"],
            dec_epochs=config["dec_epochs"])
        del emb, cluster_emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.emb = np.zeros((len(self.proto_idx), len(self.prototypes[0])))
        for i in range(len(self.emb)):
            self.emb[i] = self.prototypes[self.proto_idx[i]]

        val_ids = torch.where(data.val_mask == True)[0].to("cpu")
        hide_ids = np.random.choice(
            val_ids, int(len(val_ids) * self.hidden_portion), replace=False)
        remaining_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        remaining_mask[hide_ids] = False

        hide_set = set(int(i) for i in np.asarray(hide_ids).reshape(-1))
        ids_missing = [[] for _ in range(data.num_nodes)]
        edge_index_cpu = data.edge_index.detach().cpu()
        for src, dst in edge_index_cpu.t().tolist():
            if src in hide_set and dst not in hide_set:
                ids_missing[dst].append(src)
            if dst in hide_set and src not in hide_set:
                ids_missing[src].append(dst)

        emb_tensor = torch.as_tensor(self.emb, dtype=torch.float32)
        num_missing = torch.zeros((data.num_nodes, 1), dtype=torch.float32)
        x_missing = torch.zeros((data.num_nodes, self.num_preds, self.emb.shape[1]), dtype=torch.float32)
        for node_id, missing_ids in enumerate(ids_missing):
            if not missing_ids:
                continue
            num_missing[node_id] = len(missing_ids)
            selected_ids = missing_ids[:self.num_preds]
            x_missing[node_id, :len(selected_ids)] = emb_tensor[selected_ids]
        self.x_missing = x_missing

        remaining_nodes = torch.where(remaining_mask == True)[0]
        old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long)
        old_to_new[remaining_nodes] = torch.arange(remaining_nodes.numel(), dtype=torch.long)
        src, dst = edge_index_cpu
        edge_mask = remaining_mask[src] & remaining_mask[dst]
        new_edge_index = old_to_new[edge_index_cpu[:, edge_mask]]

        impaired_graph = Data(
            x=data.x[remaining_nodes].detach().cpu(),
            edge_index=new_edge_index,
            y=data.y[remaining_nodes].detach().cpu(),
            train_mask=data.train_mask[remaining_nodes].detach().cpu(),
            val_mask=data.val_mask[remaining_nodes].detach().cpu(),
            test_mask=data.test_mask[remaining_nodes].detach().cpu(),
            global_map=remaining_nodes.clone().detach().cpu(),
            emb=emb_tensor[remaining_nodes],
            num_missing=num_missing[remaining_nodes],
            x_missing=x_missing[remaining_nodes],
        )
        return impaired_graph, self.emb, self.x_missing

    def __call__(self, data):
        return self.forward(data)


@torch.no_grad()
def GraphMender(model, impaired_data, original_data, num_preds):
    pred_missing, pred_feats, _ = model(impaired_data)
    # Mend the original data
    original_data = original_data.detach().cpu()
    new_edge_index = original_data.edge_index.T
    pred_missing = pred_missing.detach().cpu().numpy()

    pred_feats = pred_feats.detach().cpu().reshape((len(pred_missing), num_preds, -1))

    emb_len = pred_feats.shape[-1]
    start_id = original_data.num_nodes
    mend_emb = torch.zeros(size=(start_id, num_preds, emb_len))
    for node in range(len(pred_missing)):
        num_fill_nodes = np.around(pred_missing[node]).astype(np.int32).item()
        if num_fill_nodes > 0:
            org_id = impaired_data.global_map[node]
            mend_emb[org_id][:num_fill_nodes] += pred_feats[node][:num_fill_nodes]

    filled_data = {
        "data": Data(
                x=original_data.x,
                edge_index=new_edge_index.T,
                y=original_data.y,
                # train_idx=torch.where(original_data.train_mask == True)[0],
                # valid_idx=torch.where(original_data.val_mask == True)[0],
                # test_idx=torch.where(original_data.test_mask == True)[0],
                mend_emb=mend_emb),
        "train_mask": original_data.train_mask,
        "val_mask": original_data.val_mask,
        "test_mask": original_data.test_mask
    }
    return filled_data
