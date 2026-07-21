#!/usr/bin/env python3
"""Static publication audit for the FedGB repository."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

from fedgb.config.registry import ALL_METHODS


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EXAMPLES = {
    "run_fgl_homo_subgraph.py",
    "run_fgl_hetero_subgraph.py",
    "run_fgl_graph.py",
    "run_fl_homo_subgraph.py",
    "run_fl_hetero_subgraph.py",
    "run_fl_graph.py",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".toml", ".yml", ".yaml", ".cff"}
INTERNAL_PATH = re.compile(r"/(?:opt/data/private|data/zfzhu_nas)/yyy(?:/|\b)")
PLACEHOLDER = re.compile(r"\b(?:TBD|TODO|FIXME)(?:_[A-Z0-9_]+)?\b")
EICU_FORBIDDEN_SUFFIXES = {".pt", ".csv", ".pkl", ".pickle", ".parquet", ".pyc"}
EICU_FORBIDDEN_NAMES = {"BUILD_SUMMARY.json", "VALIDATION_REPORT.json"}
EICU_FORBIDDEN_ARCHIVE_SUFFIXES = {
    ".7z",
    ".bz2",
    ".gz",
    ".rar",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
    ".zst",
}
EICU_PUBLIC_SUFFIXES = {".json", ".md", ".py", ".txt"}


def audit_text_files(root):
    root = Path(root)
    errors = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache", "tests"} for part in relative.parts):
            continue
        if relative in {
            Path("scripts/verify/release_audit.py"),
            Path("scripts/release/build_dataset_archive.py"),
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if INTERNAL_PATH.search(text):
            errors.append(f"{relative}: internal absolute path")
        if PLACEHOLDER.search(text):
            errors.append(f"{relative}: release placeholder")
    return errors


def executable_openfgl_imports():
    offenders = []
    for base in [ROOT / "fedgb", ROOT / "examples", ROOT / "scripts"]:
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(module == "openfgl" or module.startswith("openfgl.") for module in modules):
                    offenders.append(str(path.relative_to(ROOT)))
    return sorted(set(offenders))


def find_eicu_public_leaks(root):
    root = Path(root)
    leaks = []
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        if (
            path.name in EICU_FORBIDDEN_NAMES
            or path.suffix.lower() in EICU_FORBIDDEN_SUFFIXES | EICU_FORBIDDEN_ARCHIVE_SUFFIXES
            or path.suffix.lower() not in EICU_PUBLIC_SUFFIXES
        ):
            leaks.append(str(path.relative_to(root)))
    return sorted(leaks)


def main():
    errors = []
    examples = {path.name for path in (ROOT / "examples").glob("*.py")}
    if examples != EXPECTED_EXAMPLES:
        errors.append(f"Unexpected examples: {sorted(examples ^ EXPECTED_EXAMPLES)}")
    if len(ALL_METHODS) != 42:
        errors.append(f"Expected 42 methods, found {len(ALL_METHODS)}")
    manifest = json.loads((ROOT / "scripts" / "prepare_data" / "dataset_sources.json").read_text())
    if len(manifest["variants"]) != 18:
        errors.append(f"Expected 18 dataset variants, found {len(manifest['variants'])}")
    credentialed = {
        item["name"] for item in manifest["variants"] if item.get("availability") == "credentialed_build"
    }
    if credentialed != {"EICU-Het", "EICU-Hom"}:
        errors.append(f"Unexpected credentialed dataset variants: {sorted(credentialed)}")
    if not (ROOT / "datasets" / "README.md").is_file():
        errors.append("Missing datasets/README.md placeholder directory.")
    imports = executable_openfgl_imports()
    if imports:
        errors.append(f"Executable OpenFGL imports: {imports}")
    forbidden = []
    for path in ROOT.rglob("*"):
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.name.endswith((".orig", ".bak", ".remote.py", ".fedgb.tmp")):
            forbidden.append(str(path.relative_to(ROOT)))
    if forbidden:
        errors.append(f"Forbidden temporary files: {forbidden}")
    eicu_root = ROOT / "scripts" / "prepare_data" / "eicu"
    eicu_leaks = find_eicu_public_leaks(eicu_root)
    if eicu_leaks:
        errors.append(f"EICU data or generated reports are public: {eicu_leaks}")
    errors.extend(audit_text_files(ROOT))
    if errors:
        for error in errors:
            print("FAIL", error)
        raise SystemExit(1)
    print("FedGB static release audit passed.")


if __name__ == "__main__":
    main()
