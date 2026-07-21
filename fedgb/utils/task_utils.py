from torch_geometric.nn.pool import *




def load_graph_cls_default_model(args, input_dim, output_dim, client_id=None):
    """
    Load the default model for graph classification tasks.

    Args:
        args (Namespace): Arguments containing model configurations.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        client_id (int, optional): ID of the client in federated learning. Defaults to None.

    Returns:
        torch.nn.Module: The initialized model.
    """
    if client_id is None: # server
        model_name = args.model[0]
    else: # client
        if len(args.model) > 1:
            model_id = int(len(args.model) * client_id / args.num_clients)
            model_name = args.model[model_id]
        else:
            model_name = args.model[0]
        
            
    if model_name == "gin":
        from fedgb.models.gin import GIN
        return GIN(
            input_dim=input_dim,
            hid_dim=args.hid_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            graph_pooling_type=getattr(args, "graph_pooling_type", "sum"),
        )
    elif model_name == "gine":
        from fedgb.models.gine import GINE
        return GINE(
            input_dim=input_dim,
            hid_dim=args.hid_dim,
            output_dim=output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            graph_pooling_type=getattr(args, "graph_pooling_type", "sum"),
            edge_dim=getattr(args, "edge_dim", None),
        )
    elif model_name == "global_edge":
        from fedgb.models.global_edge import GlobalEdge
        return GlobalEdge(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "global_pan":
        from fedgb.models.global_pan import GlobalPAN
        return GlobalPAN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "global_sag":
        from fedgb.models.global_sag import GlobalSAG
        return GlobalSAG(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    else:
        raise ValueError(
            "Unsupported graph_cls model '{}'. Use one of: gin, gine, global_edge, global_pan, global_sag.".format(model_name)
        )



def load_node_edge_level_default_model(args, input_dim, output_dim, client_id=None):
    """
    Load the default model for node and edge level tasks.

    Args:
        args (Namespace): Arguments containing model configurations.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        client_id (int, optional): ID of the client in federated learning. Defaults to None.

    Returns:
        torch.nn.Module: The initialized model.
    """
    if client_id is None: # server
        model_name = args.model[0]
    else: # client
        if len(args.model) > 1:
            model_id = int(len(args.model) * client_id / args.num_clients)
            model_name = args.model[model_id]
        else:
            model_name = args.model[0]
    if model_name == "mlp":
        from fedgb.models.mlp import MLP
        return MLP(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gcn":
        from fedgb.models.gcn import GCN
        return GCN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gat":
        from fedgb.models.gat import GAT
        return GAT(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "graphsage":
        from fedgb.models.graphsage import GraphSAGE
        return GraphSAGE(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "sgc":
        from fedgb.models.sgc import SGC
        return SGC(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gcn2":
        from fedgb.models.gcn2 import GCN2
        return GCN2(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "rgcn":
        from fedgb.models.rgcn import RGCN
        return RGCN(
            input_dim=input_dim,
            hid_dim=args.hid_dim,
            output_dim=output_dim,
            num_relations=getattr(args, "rgcn_num_relations", 8),
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    elif model_name == "appnp":
        from fedgb.models.appnp import APPNP
        return APPNP(
            input_dim=input_dim,
            hid_dim=args.hid_dim,
            output_dim=output_dim,
            num_layers=2,
            dropout=args.dropout,
            k=getattr(args, "appnp_k", 10),
            alpha=getattr(args, "appnp_alpha", 0.1),
        )
    elif model_name == "gprgnn":
        from fedgb.models.gprgnn import GPRGNN
        return GPRGNN(
            input_dim=input_dim,
            hid_dim=args.hid_dim,
            output_dim=output_dim,
            num_layers=2,
            dropout=args.dropout,
            k=getattr(args, "gprgnn_k", 10),
            alpha=getattr(args, "gprgnn_alpha", 0.1),
            init_method=getattr(args, "gprgnn_init_method", "PPR"),
            dprate=getattr(args, "gprgnn_dprate", 0.0),
        )
    elif model_name == "chebnet":
        from fedgb.models.chebnet import ChebNet
        return ChebNet(
            input_dim=input_dim,
            hid_dim=args.hid_dim,
            output_dim=output_dim,
            num_layers=2,
            dropout=args.dropout,
            k=getattr(args, "chebnet_k", 2),
        )
    elif model_name == "bernnet":
        from fedgb.models.bernnet import BernNet
        return BernNet(
            input_dim=input_dim,
            hid_dim=args.hid_dim,
            output_dim=output_dim,
            num_layers=2,
            dropout=args.dropout,
            k=getattr(args, "bernnet_k", 10),
            dprate=getattr(args, "bernnet_dprate", 0.0),
        )
    elif model_name == "chebnetii":
        from fedgb.models.chebnetii import ChebNetII
        return ChebNetII(
            input_dim=input_dim,
            hid_dim=args.hid_dim,
            output_dim=output_dim,
            num_layers=2,
            dropout=args.dropout,
            k=getattr(args, "chebnetii_k", 10),
            dprate=getattr(args, "chebnetii_dprate", 0.0),
        )
    else:
        raise ValueError
