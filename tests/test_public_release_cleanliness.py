import re
from pathlib import Path

import fedgb


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "DATASETS.md",
    ROOT / "datasets" / "README.md",
    ROOT / "REPRODUCIBILITY.md",
)


def test_public_docs_describe_the_current_dataset_contract():
    data_guide = (ROOT / "DATASETS.md").read_text(encoding="utf-8")
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)

    assert "## Dataset Statistics" in data_guide
    assert "TCGA contains 293,786 graphs across 27 clients and five tasks." in data_guide
    assert "FedGB-datasets-v1.0.0.tar.zst" in combined


def test_software_and_dataset_artifacts_use_version_1_0_0():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert fedgb.__version__ == "1.0.0"
    assert re.search(r'^version = "1\.0\.0"$', pyproject, flags=re.MULTILINE)
    assert re.search(r"^version: 1\.0\.0$", citation, flags=re.MULTILINE)
    assert "FedGB-datasets-v1.0.0.tar.zst" in readme
