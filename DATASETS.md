# FedGB Datasets

FedGB uses eight data sources and 18 benchmark variants. Sixteen processed variants are distributed through cloud storage; the two EICU variants must be reconstructed locally by credentialed users.

| Level | Variant | Clients | Task | Features | Availability |
|---|---|---:|---|---:|---|
| Heterogeneous subgraph | EICU-Het | 40 | LOS classification, 3 classes | 315 | Credentialed local build |
| Homogeneous subgraph | EICU-Hom | 40 | LOS classification, 3 classes | 315 | Credentialed local build |
| Heterogeneous subgraph | ICIJ-Het | 20 | Node classification | 272 | Download |
| Homogeneous subgraph | ICIJ-Hom | 20 | Node classification | 272 | Download |
| Heterogeneous subgraph | ICIJ-Het-Cross | 20 | Node classification | 272 | Download |
| Homogeneous subgraph | ICIJ-Hom-Cross | 20 | Node classification | 272 | Download |
| Homogeneous subgraph | AML-HI | 29 | Node classification | 9 | Download |
| Homogeneous subgraph | AML-HI-Cross | 29 | Node classification | 9 | Download |
| Homogeneous subgraph | XS-Video | 5 | Node classification | 792 | Download |
| Graph | PubChem | 13 | Graph classification | 19 node / 16 edge | Download |
| Graph | TCGA | 27 | Graph classification, 5 tasks | 64 | Download |
| Graph | TCGA-S1 | 17 | Clinical stage classification | 64 | Download |
| Graph | TCGA-S2 | 4 | Tumor grade classification | 64 | Download |
| Graph | TCGA-S3 | 4 | Progression/recurrence classification | 64 | Download |
| Graph | NOMAD | 6 | Graph regression | 16 | Download |
| Graph | OPTIMADE | 8 | Masked graph regression, 19 target fields | 12 node / 24 edge | Download |
| Graph | OPTIMADE-S1 | 4 | Thermodynamic stability regression | 12 node / 24 edge | Download |
| Graph | OPTIMADE-S2 | 3 | Band-gap regression | 12 node / 24 edge | Download |

## Dataset Statistics

| Variant | Clients | Graphs | Avg. nodes | Tasks |
|---|---:|---:|---:|---:|
| PubChem | 13 | 30,525 | 26.23 | 1 |
| TCGA | 27 | 293,786 | 90.19 | 5 |
| TCGA-S1 | 17 | 160,750 | 94.04 | 1 |
| TCGA-S2 | 4 | 35,719 | 85.62 | 1 |
| TCGA-S3 | 4 | 40,116 | 83.88 | 1 |
| NOMAD | 6 | 90,972 | 10.35 | 1 |
| OPTIMADE | 8 | 104,775 | provider-dependent | 19 target fields across 2 task families |

TCGA contains 293,786 graphs across 27 clients and five tasks. `TCGA-S2` is the four-client tumor-grade subset, while `TCGA-S3` is the four-client progression/recurrence subset. These definitions are enforced by `fedgb/config/paper_dataset_contract.json`.

## EICU Credentialed Build

FedGB does not redistribute raw eICU records, intermediate patient-level files, or generated graph payloads. Authorized PhysioNet users can reproduce both variants with [`scripts/prepare_data/eicu/README.md`](scripts/prepare_data/eicu/README.md).

The validated build has 40 hospital clients, 116,383 ICU stays, 79,294 patients, one three-class length-of-stay task, and 315-dimensional features. Splits are class-stratified at the patient level within each hospital client using 5% train, 15% validation, and 80% test with seed 42. Discharge-related stay features are excluded. EICU-Het contains five node types and eight directed relation types; EICU-Hom contains stay nodes connected through shared treatment and medication concepts.

TCGA and all TCGA single-task variants use PCA-64 node features. PubChem uses 19-dimensional atom features and 16-dimensional edge features. Dataset validation rejects graphs whose node feature dimensions do not match the registered contract.

The full OPTIMADE variant uses eight natural provider clients in this fixed order:

| Client | Provider | Graphs | Avg. nodes | Avg. edges | Active targets |
|---:|---|---:|---:|---:|---:|
| 0 | alexandria_pbe | 30,000 | 10.24 | 72.11 | 5 |
| 1 | alexandria_pbesol | 14,159 | 8.00 | 49.94 | 7 |
| 2 | matterverse | 22,894 | 25.79 | 205.81 | 0 |
| 3 | mp | 8,073 | 30.78 | 242.70 | 3 |
| 4 | mpdd | 7,500 | 8.59 | 58.16 | 4 |
| 5 | nmd | 4,598 | 23.02 | 182.50 | 0 |
| 6 | oqmd | 11,200 | 8.16 | 54.10 | 5 |
| 7 | twodmatpedia | 6,351 | 10.44 | 74.77 | 0 |

Every OPTIMADE graph has `y` and `y_mask` tensors with shape `[1, 19]`. `y_mask` identifies observed targets, so missing labels are never optimized as zero-valued labels. Matterverse, NMD, and 2DMatPedia are unlabeled clients with all-false masks. The dataset uses fixed 80/10/10 graph splits. `OPTIMADE-S1` and `OPTIMADE-S2` provide scalar-task subsets for thermodynamic-stability and band-gap experiments.

The table reports statistics to two decimal places. Direct aggregation from the serialized graph tensors can differ by `0.01` in the last displayed digit for a small number of clients; graph counts, provider order, and target counts are exact.

Each directory contains only the processed global data, client `data_i.pt` files, fixed train/validation/test splits, and `fedgb_manifest.json`. Raw source data, tuning outputs, algorithm caches, and training outputs are excluded.

All public variants use `schema_version: "1.0"`. The authoritative machine-readable mapping is `fedgb/config/dataset_manifest.json`.

## Unified Schemas

- Homogeneous subgraph clients are PyG `Data` objects with `x`, `edge_index`, `y`, and `global_map`.
- Heterogeneous subgraph clients are relation-aware PyG `Data` objects with the homogeneous fields plus `node_type`, `edge_type`, and `target_node_type`.
- Graph-level clients are `FGLGraphDataset` objects whose graphs are PyG `Data` objects with `x`, `edge_index`, `y`, and a fixed split. Masked multi-target regression datasets additionally provide a boolean `y_mask` with the same shape as `y`.
- Every variant uses contiguous `data_0.pt` through `data_{n-1}.pt` files and `<task>/default_split/{train,val,test}_i.pt` masks.

All algorithms consume these formats through the shared loader in `fedgb/data/release_loader.py`; algorithm directories must not implement dataset-specific file parsing.

## Release Archive

The public dataset consists of `FedGB-datasets-v1.0.0.tar.zst` and `SHA256SUMS`. It contains the 16 downloadable variants and excludes EICU-Het and EICU-Hom. Download both files from the [FedGB Google Drive folder](https://drive.google.com/drive/folders/1EoSEs8UYkjR4Ve2RDqYxwbE-6pO9y9K7?usp=sharing), then verify the archive before extraction with `sha256sum -c SHA256SUMS`.

The archive must pass `PYTHONPATH=. python scripts/verify/validate_datasets.py` without modifying any dataset file. The validator reports absent EICU builds as skipped.

Maintainers build and audit the archive with:

```bash
PYTHONPATH=. python scripts/release/build_dataset_archive.py \
  --dataset-root datasets \
  --registry fedgb/config/dataset_manifest.json \
  --paper-contract fedgb/config/paper_dataset_contract.json \
  --output FedGB-datasets-v1.0.0.tar.zst \
  --compression-level 10 \
  --threads -1
```

The command includes only variants marked `availability: "download"`, rejects credentialed or unregistered roots, checks manifests and public text metadata, and writes `SHA256SUMS` after the archive audit passes.

## Public Download And Simulation

Two scripts expose the retained public downloader and simulation pipeline:

```bash
python scripts/prepare_data/prepare_homo_subgraph.py
python scripts/prepare_data/prepare_graph.py
```

Edit their top-level `CONFIG` dictionaries before running. Both support `--dry-run` to print the resolved operation without downloading, and `--force` to replace an existing output directory.

The homogeneous script supports Cora, CiteSeer, PubMed, CS, Physics, Computers, Photo, Chameleon, Squirrel, Tolokers, Actor, Amazon-ratings, Roman-empire, Questions, Minesweeper, Reddit, and Flickr. Partition modes are `louvain`, `louvain_plus`, `metis`, `metis_plus`, and `label_skew`.

The graph script supports the public TUDataset classification sources AIDS, BZR, COLLAB, COX2, DD, DHFR, ENZYMES, IMDB-BINARY, IMDB-MULTI, MUTAG, NCI1, PROTEINS, and PTC_MR. Partition modes are `label_skew`, `topology_skew`, and `feature_skew`.

Every generated variant has its own `fedgb_manifest.json`, so it is discovered automatically even though it is not part of the fixed paper benchmark registry. For example, after generating `Cora-Louvain-10`, use:

```python
CONFIG = {"algorithm": "fedgta", "dataset": "Cora-Louvain-10", "model": "gcn", "num_clients": 10}
```
