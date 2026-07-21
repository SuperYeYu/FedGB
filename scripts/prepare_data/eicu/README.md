# Reconstructing FedGB EICU-Het and EICU-Hom

This directory contains the complete reproducible pipeline from licensed eICU Collaborative Research Database v2.0 CSV tables to the FedGB PyG release format.

## Data access

eICU-CRD is credentialed data and is not redistributed by FedGB. Obtain authorized access from PhysioNet, accept the applicable data use agreement, and download the decompressed v2.0 CSV tables. Do not upload raw eICU tables, intermediate patient-level files, or generated `.pt` files to a public repository unless your data use agreement explicitly permits it.

## Prediction task

Only ICU length-of-stay three-class node classification is retained:

- `0`: ICU LOS below 3 days
- `1`: ICU LOS from 3 days (inclusive) to 7 days (exclusive)
- `2`: ICU LOS of at least 7 days

`patient.csv:unitdischargeoffset` is used only to calculate the label. It is never a model feature.

## Leakage control

The pipeline uses a strict feature allowlist. The following fields are explicitly forbidden from stay features: `dischargeweight`, `hospitaldischargeyear`, `hospitaldischargetime24`, `hospitaldischargeoffset`, `hospitaldischargelocation`, `hospitaldischargestatus`, `unitdischargetime24`, `unitdischargeoffset`, `unitdischargelocation`, `unitdischargestatus`, and `hospitaladmitoffset`. APACHE APS variables are retained. Lab, vital, past-history, and allergy features are restricted to ICU hours 0 through 24 where an event offset is available.

Diagnosis, treatment, and medication relations intentionally use the complete ICU stay. This matches the benchmark design requested by the authors, but it means graph topology may contain post-admission information. Researchers evaluating a strict admission-time prediction setting should additionally time-restrict these relations.

## Split protocol

The 40 hospital clients are shared by EICU-Het and EICU-Hom and ordered by descending number of stays. Each hospital is split independently at the patient level, so repeated stays from one patient within the same hospital cannot cross sets. The deterministic class-stratified targets are 5% train, 15% validation, and 80% test with seed 42.

## Two-stage construction

The raw-data server needs only Python's standard library:

```bash
python build_intermediate.py \
  --raw-root /path/to/eicu-collaborative-research-database-2.0 \
  --output-root /path/to/eicu2_intermediate
```

Run the PyG export in an environment containing the packages from `requirements.txt`:

```bash
python export_fedgb.py \
  --source-root /path/to/eicu2_intermediate \
  --output-root /path/to/EICU_2
```

Validate the complete result against an existing FedGB checkout:

```bash
python validate_dataset.py \
  --intermediate-root /path/to/eicu2_intermediate \
  --output-root /path/to/EICU_2 \
  --fedgb-root /path/to/FedGB \
  --expected-contract expected_contract.json \
  --report /path/to/EICU_2/validation_report.json
```

When raw data and PyG are available on the same machine, `run_all.py` runs all stages.

## Outputs

`het/` and `hom/` follow the same schema used by the released FedGB datasets. Each contains `fedgb_manifest.json`, one global PyG payload, 40 client payloads, local boolean masks, and global split-ID caches. FedGB can load them through `fedgb.data.release_loader` without a custom dataset class.

The validated reference build contains 116,383 ICU stays from 79,294 patients. Both variants use 315-dimensional node features and one three-class task. EICU-Het has five node types and eight directed relation types; EICU-Hom contains only stay nodes. Aggregate validation targets are recorded in `expected_contract.json`.

The generated directories are named `het/` and `hom/`. To use the standard FedGB paths, link or move them to `datasets/EICU-Het/` and `datasets/EICU-Hom/`, or set `dataset_root` directly to the appropriate generated directory.

EICU-Het uses relation-aware PyG `Data` with node and edge type tensors. EICU-Hom projects treatment and medication incidence into symmetric stay-to-stay edges and retains pairs sharing more than three distinct concepts. The global homogeneous payload intentionally has no projected edges because the full projection is prohibitively large; client graphs contain the benchmark edges.

## Tests

Standard-library tests:

```bash
python -m unittest tests.test_eicu2_core tests.test_build_intermediate -v
```

PyTorch/PyG export test:

```bash
python -m unittest tests.test_export_fedgb -v
```
