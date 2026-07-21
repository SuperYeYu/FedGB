from argparse import Namespace

from fedgb.models.rgcn import RGCN
from fedgb.utils.task_utils import load_node_edge_level_default_model


def test_node_model_factory_builds_rgcn_with_relation_count():
    args = Namespace(
        model=["rgcn"],
        num_clients=2,
        hid_dim=64,
        num_layers=2,
        dropout=0.5,
        rgcn_num_relations=11,
    )
    model = load_node_edge_level_default_model(args, input_dim=32, output_dim=3, client_id=0)
    assert isinstance(model, RGCN)
    assert model.encoder.num_relations == 11

