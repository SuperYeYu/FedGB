#!/usr/bin/env python3
"""Regenerate fixed splits from an existing eICU_2 intermediate directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from eicu2_core import split_records_by_hospital


def rebuild_splits(root: Path, train: float = 0.05, val: float = 0.15, seed: int = 42) -> dict:
    root = Path(root)
    with (root / "stays.csv").open(newline="", encoding="utf-8") as handle:
        stays = list(csv.DictReader(handle))
    assignments = split_records_by_hospital(stays, train, val, seed)
    split_counts = Counter()
    class_split_counts = defaultdict(Counter)
    with (root / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("hospitalid", "stay_nid", "split"))
        writer.writeheader()
        for stay in stays:
            split = assignments[(stay["hospitalid"], stay["uniquepid"])]
            writer.writerow({"hospitalid": stay["hospitalid"], "stay_nid": stay["stay_nid"], "split": split})
            split_counts[split] += 1
            class_split_counts[str(stay["icu_los_3class"])][split] += 1
    schema_path = root / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["split"] = {
        "unit": "patient within hospital", "seed": seed, "train": train, "val": val,
        "test": 1.0 - train - val, "method": "deterministic joint grouped class-stratified allocation",
    }
    schema["counts"]["split_counts"] = dict(split_counts)
    schema["counts"]["class_split_counts"] = {
        label: dict(counts) for label, counts in class_split_counts.items()
    }
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"split_counts": dict(split_counts), "class_split_counts": schema["counts"]["class_split_counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--train", type=float, default=0.05)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(rebuild_splits(args.root, args.train, args.val, args.seed), indent=2))


if __name__ == "__main__":
    main()
