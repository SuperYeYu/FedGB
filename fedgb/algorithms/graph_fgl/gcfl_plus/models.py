import torch
from fedgb.models.gin import GIN as DefaultGIN
from fedgb.models.gin import GINEncoder
from fedgb.models.gine import GINE as DefaultGINE
from fedgb.models.gine import GINEEncoder


def _is_gine(model_name):
    if isinstance(model_name, (list, tuple)):
        model_name = model_name[0] if model_name else "gin"
    return str(model_name).lower() == "gine"


class CrossDomainGIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nlayer, dropout, model_name="gin", edge_dim=None):
        super(CrossDomainGIN, self).__init__()
        encoder_cls = GINEEncoder if _is_gine(model_name) else GINEncoder
        kwargs = {"edge_dim": edge_dim} if encoder_cls is GINEEncoder else {}
        self.encoder = encoder_cls(input_dim=nfeat, hid_dim=nhid, num_layers=nlayer, dropout=dropout, **kwargs)


class GIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, model_name="gin", edge_dim=None):
        super(GIN, self).__init__()
        model_cls = DefaultGINE if _is_gine(model_name) else DefaultGIN
        kwargs = {"edge_dim": edge_dim} if model_cls is DefaultGINE else {}
        model = model_cls(
            input_dim=nfeat,
            hid_dim=nhid,
            output_dim=nclass,
            num_layers=nlayer,
            dropout=dropout,
            **kwargs,
        )
        self.encoder = model.encoder
        self.head = model.head

    def forward(self, data):
        embedding = self.encoder(data)
        logits = self.head(embedding)
        return embedding, logits
