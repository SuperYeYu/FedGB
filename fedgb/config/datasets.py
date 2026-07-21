"""Dataset metadata loaded from the FedGB release manifest."""

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from fedgb.data.schema import SCHEMA_VERSION


@lru_cache(maxsize=1)
def dataset_manifest():
    path = files("fedgb.config").joinpath("dataset_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def dataset_registry():
    payload = dataset_manifest()
    return {item["name"]: item for item in payload["variants"]}


def get_dataset_spec(public_name, dataset_root=None):
    registry = dataset_registry()
    if public_name in registry:
        return registry[public_name]
    root = Path(dataset_root) if dataset_root else Path(__file__).resolve().parents[2] / "datasets" / public_name
    manifest_path = root / "fedgb_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("name") != public_name:
            raise ValueError(f"Dataset manifest name {manifest.get('name')!r} does not match '{public_name}'.")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Dataset '{public_name}' uses schema {manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}."
            )
        return manifest
    choices = ", ".join(sorted(registry))
    raise ValueError(
        f"Unknown FedGB dataset '{public_name}'. Supported release datasets: {choices}. "
        "Generated datasets must contain fedgb_manifest.json."
    )

