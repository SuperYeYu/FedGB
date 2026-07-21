import importlib.util
from pathlib import Path

import pytest

from fedgb.config.datasets import dataset_registry
from fedgb.config.runtime import build_run_config


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "prepare_data" / "eicu"


def test_registry_marks_exactly_two_credentialed_build_variants():
    registry = dataset_registry()
    credentialed = {name for name, item in registry.items() if item["availability"] == "credentialed_build"}
    downloadable = {name for name, item in registry.items() if item["availability"] == "download"}
    assert credentialed == {"EICU-Het", "EICU-Hom"}
    assert len(downloadable) == 16


def test_eicu_builder_contains_code_but_no_generated_or_raw_data():
    assert (BUILDER / "README.md").is_file()
    assert (BUILDER / "run_all.py").is_file()
    forbidden_suffixes = {".pt", ".csv", ".pkl", ".pickle", ".parquet", ".pyc"}
    forbidden_names = {"BUILD_SUMMARY.json", "VALIDATION_REPORT.json"}
    leaked = [
        path.relative_to(BUILDER).as_posix()
        for path in BUILDER.rglob("*")
        if "__pycache__" not in path.parts
        and (path.name in forbidden_names or (path.is_file() and path.suffix.lower() in forbidden_suffixes))
    ]
    assert not leaked


def test_missing_eicu_build_has_actionable_runtime_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="credentialed.*scripts/prepare_data/eicu/README.md"):
        build_run_config(
            family="standard_fl",
            scenario="hetero_subgraph",
            algorithm="fedavg",
            dataset="EICU-Het",
            model="rgcn",
            dataset_root=tmp_path / "EICU-Het",
        )


def test_validation_skips_absent_credentialed_builds(tmp_path):
    path = ROOT / "scripts" / "verify" / "validate_datasets.py"
    spec = importlib.util.spec_from_file_location("validate_datasets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registry = {
        "EICU-Het": {
            "name": "EICU-Het",
            "availability": "credentialed_build",
        }
    }
    result = module.validate_registered_datasets(tmp_path, registry=registry)
    assert result == {"validated": [], "skipped": ["EICU-Het"], "errors": []}
