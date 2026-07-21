import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_package_has_no_dgl_imports():
    offenders = []
    for path in (ROOT / "fedgb").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "dgl" or name.startswith("dgl.") for name in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
