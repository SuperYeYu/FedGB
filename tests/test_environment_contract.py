import json
from pathlib import Path

import pytest

from fedgb.algorithms.subgraph_fgl.homogeneous.adafgl._utils import validate_native_runtime


ROOT = Path(__file__).resolve().parents[1]


def test_linux_cuda_requirements_pin_verified_core_versions():
    text = (ROOT / "configs" / "environment" / "requirements-linux-cu121.txt").read_text(
        encoding="utf-8"
    )
    assert "torch==2.5.1+cu121" in text
    assert "torch-geometric==2.8.0" in text
    assert "numpy==2.4.4" in text
    assert "scipy==1.18.0" in text
    assert "scikit-learn==1.9.0" in text


def test_project_metadata_limits_the_public_release_to_python_312():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.13"' in text


def test_environment_metadata_declares_supported_platform():
    payload = json.loads(
        (ROOT / "configs" / "environment" / "verified-environment.json").read_text(encoding="utf-8")
    )
    assert payload["python"] == "3.12.13"
    assert payload["cuda_runtime"] == "12.1"
    assert payload["platform"] == "linux-x86_64"
    assert payload["gpu"] == "NVIDIA GeForce RTX 4090"


def test_adafgl_native_runtime_reports_unsupported_platform(tmp_path):
    with pytest.raises(RuntimeError, match="Linux x86_64"):
        validate_native_runtime(cuda=False, platform_name="Windows", machine="AMD64", library_dir=tmp_path)


def test_adafgl_native_runtime_reports_missing_library(tmp_path):
    with pytest.raises(RuntimeError, match="libmatmul.so"):
        validate_native_runtime(cuda=False, platform_name="Linux", machine="x86_64", library_dir=tmp_path)
