import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output + self.bias

class FedGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(FedGCN, self).__init__()
        self.layers = nn.ModuleList()
        if nlayer > 1:
            self.layers.append(GraphConvolution(nfeat, nhid))
            for _ in range(nlayer - 2):
                self.layers.append(GraphConvolution(nhid, nhid))
            self.layers.append(GraphConvolution(nhid, nclass))
        else:
            self.layers.append(GraphConvolution(nfeat, nclass))

        self.dropout = dropout


    def forward(self, data):
        x, adj = data.x, data.adj
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.layers[-1](x, adj)

        return x, logits

    def _normalize_sparse_adj(self, edge_index, values, num_nodes):
        loop_index = torch.arange(num_nodes, device=edge_index.device)
        loop_edge_index = torch.stack([loop_index, loop_index], dim=0)
        edge_index = torch.cat([edge_index, loop_edge_index], dim=1)
        values = torch.cat([values, torch.ones(num_nodes, device=values.device, dtype=values.dtype)], dim=0)
        row, col = edge_index
        degree = values.new_zeros(num_nodes)
        degree.scatter_add_(0, row, values)
        deg_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
        norm_values = values * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(
            edge_index,
            norm_values,
            (num_nodes, num_nodes),
            device=values.device,
        ).coalesce()

    def aug(self, data):
        with torch.no_grad():
            node_features,_ = self.forward(data)

        if hasattr(data, "aug_edge_index") and hasattr(data, "aug_edge_label"):
            edge_index = data.aug_edge_index.to(node_features.device)
            edge_logits = (node_features[edge_index[0]] * node_features[edge_index[1]]).sum(dim=1).clamp(-30, 30)
            if self.training:
                eps = torch.finfo(edge_logits.dtype).eps
                uniform = torch.rand_like(edge_logits).clamp_(eps, 1 - eps)
                gumbel_noise = -torch.log(-torch.log(uniform))
                edge_weight = torch.sigmoid((edge_logits + gumbel_noise) / 0.5)
            else:
                edge_weight = torch.sigmoid(edge_logits)

            message_mask = getattr(data, "aug_message_mask", data.aug_edge_label > 0).to(edge_weight.device)
            message_edges = edge_index[:, message_mask]
            message_weight = edge_weight[message_mask].clamp_min(1e-6)
            if message_edges.numel() == 0:
                loops = torch.arange(data.num_nodes, device=edge_weight.device)
                message_edges = torch.stack([loops, loops], dim=0)
                message_weight = torch.ones(data.num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
            adj_sampled = self._normalize_sparse_adj(message_edges, message_weight, data.num_nodes)
            return adj_sampled, edge_logits

        logits = torch.matmul(node_features, node_features.t())

        adj_sampled = self.gumbel_softmax(logits, tau=0.5)
        return adj_sampled, logits

    def gumbel_softmax(self, logits, tau):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(MLP, self).__init__()
        # 定义第一个线性层
        self.fc1 = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        # 定义第二个线性层
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 输入通过第一个线性层
        x = self.fc1(x)
        # 应用ReLU激活函数
        x = F.relu(x)
        # 应用dropout
        x = self.dropout(x)
        # 输出通过第二个线性层
        x = self.fc2(x)
        return x
