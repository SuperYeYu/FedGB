from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_release_layout():
    assert (ROOT / "fedgb" / "__init__.py").is_file()
    assert not (ROOT / "openfgl").exists()

    expected_examples = {
        "run_fgl_homo_subgraph.py",
        "run_fgl_hetero_subgraph.py",
        "run_fgl_graph.py",
        "run_fl_homo_subgraph.py",
        "run_fl_hetero_subgraph.py",
        "run_fl_graph.py",
    }
    assert expected_examples == {path.name for path in (ROOT / "examples").glob("*.py")}

    for filename in [
        "README.md",
        "DATASETS.md",
        "METHODS.md",
        "REPRODUCIBILITY.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "NOTICE",
        "LICENSE",
    ]:
        assert (ROOT / filename).is_file(), filename


def test_runtime_results_are_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/*" in gitignore
    assert "!results/.gitkeep" in gitignore
    assert ".smoke_fixtures/" in gitignore
    assert ".smoke_results/" in gitignore
    assert "dataset_validation_report.json" in gitignore

