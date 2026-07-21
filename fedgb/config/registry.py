"""Canonical public method registry for the FedGB paper baselines."""

STANDARD_FL_METHODS = frozenset(
    {
        "fedavg",
        "fedprox",
        "scaffold",
        "moon",
        "feddc",
        "fedproto",
        "fedexp",
        "fedlaw",
        "fedala",
        "fedtgp",
        "fedluar",
        "feroma",
        "pfed1bs",
        "tinyproto",
    }
)

SUBGRAPH_FGL_METHODS = frozenset(
    {
        "fedsage_plus",
        "fedgta",
        "fedpub",
        "fgssl",
        "adafgl",
        "fedppn",
        "fedtad",
        "fedspray",
        "hifgl",
        "feddep",
        "fggp",
        "fediih",
        "s2fgl",
        "fedlog",
        "fedstruct",
        "cufl",
        "spp_fgc",
        "fedrgl",
        "fedlit",
        "fedda",
        "fedhgn",
    }
)

GRAPH_FGL_METHODS = frozenset(
    {"gcfl_plus", "fedstar", "fedssp", "optgdba", "fedgmark", "nigdba", "fedvn"}
)

ALL_METHODS = STANDARD_FL_METHODS | SUBGRAPH_FGL_METHODS | GRAPH_FGL_METHODS

