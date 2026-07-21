#!/usr/bin/env python3
"""Run the complete eICU_2 build in an environment with raw data and PyG."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_intermediate import BuildConfig, build_intermediate
from export_fedgb import export_fedgb
from validate_dataset import validate_expected_contract, validate_fedgb, validate_intermediate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--fedgb-root", type=Path)
    args = parser.parse_args()
    intermediate = args.work_root / "intermediate"
    build_intermediate(args.raw_root, intermediate, BuildConfig())
    validate_intermediate(intermediate)
    export_fedgb(intermediate, args.output_root, hom_threshold=3)
    report = validate_fedgb(args.output_root, args.fedgb_root)
    validate_expected_contract(report, Path(__file__).with_name("expected_contract.json"))


if __name__ == "__main__":
    main()
