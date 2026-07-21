import re
from pathlib import Path

from fedgb.config.registry import ALL_METHODS


ROOT = Path(__file__).resolve().parents[1]


def readme_text():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def github_anchor(heading):
    anchor = heading.strip().lower()
    anchor = re.sub(r"[^a-z0-9 -]", "", anchor)
    return anchor.replace(" ", "-")


def test_readme_lists_every_public_algorithm_identifier():
    text = readme_text().lower()
    missing = sorted(method for method in ALL_METHODS if f"`{method}`" not in text)
    assert not missing, f"README is missing public algorithms: {missing}"


def test_readme_documents_all_public_entrypoints_and_data_preparation_scripts():
    text = readme_text()
    paths = [
        "examples/run_fgl_homo_subgraph.py",
        "examples/run_fgl_hetero_subgraph.py",
        "examples/run_fgl_graph.py",
        "examples/run_fl_homo_subgraph.py",
        "examples/run_fl_hetero_subgraph.py",
        "examples/run_fl_graph.py",
        "scripts/prepare_data/prepare_homo_subgraph.py",
        "scripts/prepare_data/prepare_graph.py",
    ]
    missing = [path for path in paths if path not in text]
    assert not missing, f"README is missing public scripts: {missing}"


def test_readme_contents_match_all_second_level_sections():
    text = readme_text()
    contents = text.split("## Contents", 1)[1].split("\n## ", 1)[0]
    links = re.findall(r"^- \[[^]]+\]\(#([^)]+)\)$", contents, flags=re.MULTILINE)
    headings = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    expected = [github_anchor(heading) for heading in headings if heading != "Contents"]

    assert links == expected


def test_readme_documents_core_environment_data_and_publication_information():
    text = readme_text()
    required = [
        "Linux x86_64",
        "Python 3.12",
        "CUDA 12.1",
        "FedGB-datasets-v1.0.0.tar.zst",
        "scripts/verify/validate_datasets.py",
        'schema_version: "1.0"',
        "METHODS.md",
        "DATASETS.md",
        "CITATION.cff",
        "LICENSE",
        "OpenFGL",
    ]
    missing = [item for item in required if item not in text]
    assert not missing, f"README is missing core information: {missing}"


def test_readme_has_no_broken_local_links_and_remains_english_ascii():
    text = readme_text()
    local_links = re.findall(r"\[[^]]+\]\((?!https?://|#)([^)#]+)(?:#[^)]+)?\)", text)
    missing = [link for link in local_links if not (ROOT / link).exists()]

    assert not missing, f"README has broken local links: {missing}"
    assert text.isascii(), "README must remain English-only ASCII text."


def test_dataset_overview_image_is_tracked_and_precedes_contents():
    text = readme_text()
    image_path = "assets/images/fgl_our_datasets.png"

    assert (ROOT / image_path).is_file()
    assert f'src="{image_path}"' in text
    assert 'alt="Overview of the FedGB datasets"' in text
    assert text.index(image_path) < text.index("## Contents")


def test_brand_image_and_title_share_the_first_heading():
    text = readme_text()
    image_path = "assets/images/fedgb_picture.png"
    title = "A Real-World Federated Graph Benchmark from Simulated Partitions to Natural Client Scenarios"
    heading = re.search(r"<h1[^>]*>(.*?)</h1>", text, flags=re.DOTALL)

    assert (ROOT / image_path).is_file()
    assert heading is not None
    assert f'src="{image_path}"' in heading.group(1)
    assert title in heading.group(1)
    assert heading.group(1).index(image_path) < heading.group(1).index(title)
    assert "FedGB:" not in heading.group(1)


def test_released_dataset_setup_is_complete_and_actionable():
    text = readme_text()
    section = text.split("## Released Dataset Setup", 1)[1].split("\n## ", 1)[0]
    required = [
        "https://drive.google.com/drive/folders/1EoSEs8UYkjR4Ve2RDqYxwbE-6pO9y9K7?usp=sharing",
        "repository root",
        "FedGB-datasets-v1.0.0.tar.zst",
        "SHA256SUMS",
        "sudo apt-get install zstd",
        "sha256sum -c SHA256SUMS",
        "tar --use-compress-program=unzstd",
        "datasets/AML-HI/",
        "16 downloadable",
        "EICU credentials",
        "scripts/prepare_data/eicu/README.md",
        "PYTHONPATH=. python scripts/verify/validate_datasets.py",
        "python examples/run_fgl_homo_subgraph.py --dry-run",
    ]
    missing = [item for item in required if item not in section]
    assert not missing, f"Dataset setup is missing required steps: {missing}"


def test_dataset_overview_and_schema_are_merged_into_released_setup():
    text = readme_text()
    section = text.split("## Released Dataset Setup", 1)[1].split("\n## ", 1)[0]
    required = [
        "### Dataset Overview and Unified Schemas",
        "EICU-Het",
        "AML-HI",
        "PubChem",
        "OPTIMADE-S2",
        "schema_version",
        "fedgb/data/release_loader.py",
        "### Download and Installation",
    ]
    missing = [item for item in required if item not in section]

    assert not missing, f"Merged dataset section is missing: {missing}"
    assert "## Datasets and Unified Schemas" not in text
    assert "[Datasets and Unified Schemas](#datasets-and-unified-schemas)" not in text
