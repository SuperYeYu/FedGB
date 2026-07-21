#!/usr/bin/env python3
"""Capture the runtime versions used for a FedGB release validation."""

import importlib.metadata
import json
import platform
import subprocess

import torch


PACKAGES = [
    "torch",
    "torch-geometric",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "networkx",
    "ogb",
    "fvcore",
]


def main():
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        gpu = None
    payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "packages": {},
    }
    for package in PACKAGES:
        try:
            payload["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            payload["packages"][package] = None
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

