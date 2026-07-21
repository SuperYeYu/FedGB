#!/usr/bin/env python3
"""Validate all released FedGB processed datasets without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fedgb.config.datasets import dataset_registry
from fedgb.data.validation import validate_dataset_variant


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts" / "prepare_data" / "dataset_sources.json"


def validate_registered_datasets(dataset_root, registry=None, validator=validate_dataset_variant):
    dataset_root = Path(dataset_root)
    registry = dataset_registry() if registry is None else registry
    result = {"validated": [], "skipped": [], "errors": []}
    for name, item in registry.items():
        root = dataset_root / name
        if not root.is_dir() and item.get("availability") == "credentialed_build":
            result["skipped"].append(name)
            continue
        try:
            result["validated"].append(validator(root, item))
        except Exception as exc:
            result["errors"].append(f"{name}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--report", type=Path, default=ROOT / "dataset_validation_report.json")
    args = parser.parse_args()
    result = validate_registered_datasets(args.dataset_root)
    for item in result["validated"]:
        print(
            f"validated {item['dataset']}: {item['clients']} clients, "
            f"feature_dims={item['feature_dims']}",
            flush=True,
        )
    for name in result["skipped"]:
        print(f"skipped {name}: credentialed local build not present", flush=True)

    output = args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["errors"]:
        raise SystemExit("\n".join(result["errors"]))
    print(
        f"validated {len(result['validated'])} dataset variants; "
        f"skipped {len(result['skipped'])}; report: {output}"
    )


if __name__ == "__main__":
    main()
