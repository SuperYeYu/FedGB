# FedGB Datasets

The processed benchmark archive is distributed separately from the source repository.

Extract the archive into this directory so that paths such as `datasets/AML-HI/`, `datasets/ICIJ-Het/`, `datasets/PubChem/`, and `datasets/TCGA/` exist directly below `datasets/`.

The `FedGB-datasets-v1.0.0.tar.zst` archive contains 16 downloadable processed variants. EICU-Het and EICU-Hom are not redistributed; credentialed users reconstruct them by following [`../scripts/prepare_data/eicu/README.md`](../scripts/prepare_data/eicu/README.md). See [`../DATASETS.md`](../DATASETS.md) for client counts, tasks, statistics, feature dimensions, and availability. TCGA and all TCGA subsets use 64-dimensional PCA features.

After extraction, run:

```bash
PYTHONPATH=. python scripts/verify/validate_datasets.py
```

The download URL and checksum instructions are provided in the root [`README.md`](../README.md#download-and-installation).
