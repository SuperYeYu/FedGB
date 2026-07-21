# FedGB Reproducibility Guide

## Environment

Use Linux x86_64, an NVIDIA GPU, Python 3.12.13, PyTorch 2.5.1+cu121, and PyTorch Geometric 2.8.0. Install the pinned dependencies from `configs/environment/requirements-linux-cu121.txt` and record the runtime with:

```bash
python scripts/verify/capture_environment.py > environment.json
```

## Data Integrity

Use `FedGB-datasets-v1.0.0.tar.zst` without regenerating graph structures or splits. Confirm its checksum and run:

```bash
PYTHONPATH=. python scripts/verify/validate_datasets.py
```

All reported experiments use the fixed client files and fixed train/validation/test masks distributed in that archive.

## Experiment Protocol

Public defaults are 100 communication rounds, two local epochs for subgraph tasks, one local epoch for graph-level tasks, hidden dimension 64, two GNN layers, dropout 0.5, and full client participation. Each experiment should be repeated with the seeds reported by the paper. The resolved configuration stored under `results/` is the authoritative record for a run.

To reproduce a method, select the matching entry script, set `algorithm`, `dataset`, `model`, `gpuid`, and `seed`, then run the script. Method-specific configuration modules are applied automatically.

## Release Validation

Before using a release for paper results, run:

```bash
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python scripts/verify/release_audit.py
PYTHONPATH=. python scripts/verify/build_smoke_fixtures.py
PYTHONPATH=. python scripts/verify/run_smoke_matrix.py --gpus 0,1
```

The smoke matrix contains 91 one-round cases covering all 42 public methods and their declared node classification, graph classification, and graph regression paths.

