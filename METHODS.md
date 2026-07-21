# FedGB Methods

## Standard FL

`FedAvg`, `FedProx`, `SCAFFOLD`, `MOON`, `FedDC`, `FedProto`, `FedExp`, `FedLAW`, `FedALA`, `FedTGP`, `FedLUAR`, `FEROMA`, `pFed1BS`, and `TinyProto`.

All standard FL methods support homogeneous subgraph node classification, heterogeneous subgraph node classification with RGCN, graph classification, and graph regression.

FedProto, FedTGP, TinyProto, FEROMA, and FedLAW were originally classification-oriented. FedGB preserves their classification paths and adds explicit graph-regression adapters: task-level latent prototypes for prototype methods, regression latent statistics for FEROMA, and validation MSE for FedLAW aggregation. These extensions are isolated from the original classification behavior.

## Homogeneous Subgraph FGL

`FedSAGE+`, `FedGTA`, `FedPUB`, `FGSSL`, `AdaFGL`, `FedPPN`, `FedTAD`, `FedSpray`, `HiFGL`, `FedDEP`, `FGGP`, `FedIIH`, `S2FGL`, `FedLoG`, `FedStruct`, `CUFL`, `SPP-FGC`, and `FedRGL`.

## Heterogeneous Subgraph FGL

`FedLIT`, `FedDA`, and `FedHGN` use PyTorch Geometric and preserve typed relations.

## Graph-Level FGL

`GCFL+`, `FedStar`, `FedSSP`, `Opt-GDBA`, `FedGMark`, `NI-GDBA`, and `FedVN` support the graph-level benchmark pipeline. Classification and regression paths are verified separately.

Each implementation lives in its own directory with separate `client.py`, `server.py`, and algorithm-specific configuration modules. Public names and runtime import classes are defined in `fedgb/config/registry.py` and `fedgb/config/method_specs.py`.
