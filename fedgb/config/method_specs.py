"""Runtime import metadata for the 42 public FedGB baselines."""

from importlib import import_module


HOMO_MODELS = ("gcn", "gat", "graphsage", "sgc", "gcn2", "mlp")
HETERO_MODELS = ("rgcn",)
GRAPH_MODELS = ("gin", "gine", "global_edge", "global_pan", "global_sag")


def _compatibility(package):
    if ".standard_fl." in package:
        return {
            "family": "standard_fl",
            "scenarios": ("homo_subgraph", "hetero_subgraph", "graph"),
            "tasks": ("node_cls", "graph_cls", "graph_reg"),
            "models": {
                "homo_subgraph": HOMO_MODELS,
                "hetero_subgraph": HETERO_MODELS,
                "graph": GRAPH_MODELS,
            },
        }
    if ".subgraph_fgl.heterogeneous." in package:
        return {
            "family": "subgraph_fgl",
            "scenarios": ("hetero_subgraph",),
            "tasks": ("node_cls",),
            "models": {"hetero_subgraph": HETERO_MODELS},
        }
    if ".subgraph_fgl.homogeneous." in package:
        return {
            "family": "subgraph_fgl",
            "scenarios": ("homo_subgraph",),
            "tasks": ("node_cls",),
            "models": {"homo_subgraph": HOMO_MODELS},
        }
    return {
        "family": "graph_fgl",
        "scenarios": ("graph",),
        "tasks": ("graph_cls", "graph_reg"),
        "models": {"graph": GRAPH_MODELS},
    }


def _spec(package, client_class, server_class):
    return {
        "package": package,
        "client_class": client_class,
        "server_class": server_class,
        **_compatibility(package),
    }


METHOD_SPECS = {
    "fedavg": _spec("fedgb.algorithms.standard_fl.fedavg", "FedAvgClient", "FedAvgServer"),
    "fedprox": _spec("fedgb.algorithms.standard_fl.fedprox", "FedProxClient", "FedProxServer"),
    "scaffold": _spec("fedgb.algorithms.standard_fl.scaffold", "ScaffoldClient", "ScaffoldServer"),
    "moon": _spec("fedgb.algorithms.standard_fl.moon", "MoonClient", "MoonServer"),
    "feddc": _spec("fedgb.algorithms.standard_fl.feddc", "FedDCClient", "FedDCServer"),
    "fedproto": _spec("fedgb.algorithms.standard_fl.fedproto", "FedProtoClient", "FedProtoServer"),
    "fedexp": _spec("fedgb.algorithms.standard_fl.fedexp", "FedExPClient", "FedExPServer"),
    "fedlaw": _spec("fedgb.algorithms.standard_fl.fedlaw", "FedLAWClient", "FedLAWServer"),
    "fedala": _spec("fedgb.algorithms.standard_fl.fedala", "FedALAClient", "FedALAServer"),
    "fedtgp": _spec("fedgb.algorithms.standard_fl.fedtgp", "FedTGPClient", "FedTGPServer"),
    "fedluar": _spec("fedgb.algorithms.standard_fl.fedluar", "FedLUARClient", "FedLUARServer"),
    "feroma": _spec("fedgb.algorithms.standard_fl.feroma", "FEROMAClient", "FEROMAServer"),
    "pfed1bs": _spec("fedgb.algorithms.standard_fl.pfed1bs", "PFed1BSClient", "PFed1BSServer"),
    "tinyproto": _spec("fedgb.algorithms.standard_fl.tinyproto", "TinyProtoClient", "TinyProtoServer"),
    "fedsage_plus": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fedsage_plus", "FedSagePlusClient", "FedSagePlusServer"),
    "fedgta": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fedgta", "FedGTAClient", "FedGTAServer"),
    "fedpub": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fedpub", "FedPubClient", "FedPubServer"),
    "fgssl": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fgssl", "FGSSLClient", "FGSSLServer"),
    "adafgl": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.adafgl", "AdaFGLClient", "AdaFGLServer"),
    "fedppn": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fedppn", "FedPPNClient", "FedPPNServer"),
    "fedtad": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fedtad", "FedTADClient", "FedTADServer"),
    "fedspray": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fedspray", "FedSprayClient", "FedSprayServer"),
    "hifgl": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.hifgl", "HiFGLClient", "HiFGLServer"),
    "feddep": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.feddep", "FedDEPClient", "FedDEPEServer"),
    "fggp": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fggp", "FGGPClient", "FGGPServer"),
    "fediih": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fediih", "FedIIHClient", "FedIIHServer"),
    "s2fgl": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.s2fgl", "S2FGLClient", "S2FGLServer"),
    "fedlog": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fedlog", "FedLoGClient", "FedLoGServer"),
    "fedstruct": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fedstruct", "FedStructClient", "FedStructServer"),
    "cufl": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.cufl", "CUFLClient", "CUFLServer"),
    "spp_fgc": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.spp_fgc", "FedGraphClient", "FedGraphServer"),
    "fedrgl": _spec("fedgb.algorithms.subgraph_fgl.homogeneous.fedrgl", "FedRGLClient", "FedRGLServer"),
    "fedlit": _spec("fedgb.algorithms.subgraph_fgl.heterogeneous.fedlit", "FedLITClient", "FedLITServer"),
    "fedda": _spec("fedgb.algorithms.subgraph_fgl.heterogeneous.fedda", "FedDAClient", "FedDAServer"),
    "fedhgn": _spec("fedgb.algorithms.subgraph_fgl.heterogeneous.fedhgn", "FedHGNClient", "FedHGNServer"),
    "gcfl_plus": _spec("fedgb.algorithms.graph_fgl.gcfl_plus", "GCFLPlusClient", "GCFLPlusServer"),
    "fedstar": _spec("fedgb.algorithms.graph_fgl.fedstar", "FedStarClient", "FedStarServer"),
    "fedssp": _spec("fedgb.algorithms.graph_fgl.fedssp", "FedSSPClient", "FedSSPServer"),
    "optgdba": _spec("fedgb.algorithms.graph_fgl.optgdba", "OptGDBAClient", "OptGDBAServer"),
    "fedgmark": _spec("fedgb.algorithms.graph_fgl.fedgmark", "FedGMarkClient", "FedGMarkServer"),
    "nigdba": _spec("fedgb.algorithms.graph_fgl.nigdba", "NIGDBAClient", "NIGDBAServer"),
    "fedvn": _spec("fedgb.algorithms.graph_fgl.fedvn", "FedVNClient", "FedVNServer"),
}


def resolve_method_class(method, role):
    if method not in METHOD_SPECS:
        raise ValueError(f"Algorithm '{method}' is not a public FedGB baseline.")
    if role not in {"client", "server"}:
        raise ValueError(f"Unknown method role '{role}'.")
    spec = METHOD_SPECS[method]
    module = import_module(f"{spec['package']}.{role}")
    return getattr(module, spec[f"{role}_class"])

