import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_do_not_import_openfgl():
    offenders = []
    for path in list((ROOT / "fedgb").rglob("*.py")) + list((ROOT / "examples").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "openfgl" or name.startswith("openfgl.") for name in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []

