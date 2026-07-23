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


## EICU Credentialed Build

FedGB does not redistribute raw eICU records, intermediate patient-level files, or generated graph payloads. Authorized PhysioNet users can reproduce both variants with [`scripts/prepare_data/eicu/README.md`](scripts/prepare_data/eicu/README.md).

The validated build has 40 hospital clients, 116,383 ICU stays, 79,294 patients, one three-class length-of-stay task, and 315-dimensional features. Splits are class-stratified at the patient level within each hospital client using 5% train, 15% validation, and 80% test with seed 42. Discharge-related stay features are excluded. EICU-Het contains five node types and eight directed relation types; EICU-Hom contains stay nodes connected through shared treatment and medication concepts.


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
