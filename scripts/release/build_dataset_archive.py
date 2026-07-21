#!/usr/bin/env python3
"""Build and audit the downloadable FedGB dataset archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import zstandard


INTERNAL_PATH_MARKERS = (b"/opt/data/private/yyy", b"/data/zfzhu_nas/yyy")
FIXED_MTIME = 0
DEFAULT_PAPER_CONTRACT = Path(__file__).resolve().parents[2] / "fedgb" / "config" / "paper_dataset_contract.json"


def _registry(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["variants"]


def _download_specs(path):
    return sorted(
        (item for item in _registry(path) if item.get("availability") == "download"),
        key=lambda item: item["name"],
    )


def _tar_info(path: Path, archive_name: str):
    info = tarfile.TarInfo(archive_name)
    stat = path.stat()
    info.size = stat.st_size if path.is_file() else 0
    info.mode = stat.st_mode & 0o777
    info.mtime = FIXED_MTIME
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.type = tarfile.REGTYPE if path.is_file() else tarfile.DIRTYPE
    return info


def _add_tree(archive, source: Path, archive_root: PurePosixPath):
    archive.addfile(_tar_info(source, archive_root.as_posix()))
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        relative = PurePosixPath(path.relative_to(source).as_posix())
        info = _tar_info(path, (archive_root / relative).as_posix())
        if path.is_file():
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        elif path.is_dir():
            archive.addfile(info)


def _stream_contains_internal_path(stream):
    overlap = b""
    boundary = max(len(marker) for marker in INTERNAL_PATH_MARKERS)
    while chunk := stream.read(16 * 1024 * 1024):
        content = overlap + chunk
        if any(marker in content for marker in INTERNAL_PATH_MARKERS):
            return True
        overlap = content[-boundary:]
    return False


def build_archive(dataset_root, registry_path, output, compression_level=10, threads=0):
    dataset_root = Path(dataset_root)
    output = Path(output)
    specs = _download_specs(registry_path)
    if len(specs) != 16 and output.name.startswith("FedGB-datasets-v1.0.0"):
        raise ValueError(f"FedGB v1.0.0 requires 16 downloadable variants, found {len(specs)}")
    missing = [item["name"] for item in specs if not (dataset_root / item["name"]).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing downloadable dataset roots: {missing}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    compressor = zstandard.ZstdCompressor(level=compression_level, threads=threads)
    try:
        with temporary.open("wb") as raw, compressor.stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                readme = dataset_root / "README.md"
                if readme.is_file():
                    info = _tar_info(readme, "datasets/README.md")
                    with readme.open("rb") as stream:
                        archive.addfile(info, stream)
                for spec in specs:
                    _add_tree(archive, dataset_root / spec["name"], PurePosixPath("datasets") / spec["name"])
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def audit_archive(archive_path, registry_path):
    specs = _download_specs(registry_path)
    expected = {item["name"]: item for item in specs}
    credentialed = sorted(
        item["name"] for item in _registry(registry_path) if item.get("availability") == "credentialed_build"
    )
    found = set()
    manifests = {}
    offenders = []
    archive_path = Path(archive_path)
    with archive_path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as decompressed:
        with tarfile.open(fileobj=decompressed, mode="r|") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if len(parts) >= 2 and parts[0] == "datasets" and parts[1] != "README.md":
                    found.add(parts[1])
                if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "fedgb_manifest.json":
                    stream = archive.extractfile(member)
                    manifests[parts[1]] = json.loads(stream.read().decode("utf-8"))
                    continue
                if member.isfile():
                    stream = archive.extractfile(member)
                    if _stream_contains_internal_path(stream):
                        offenders.append(member.name)
    forbidden = sorted(set(credentialed) & found)
    if forbidden:
        raise ValueError(f"Credentialed variants are present in the archive: {forbidden}")
    if found != set(expected):
        raise ValueError(f"Archive variants differ: expected {sorted(expected)}, found {sorted(found)}")
    if set(manifests) != set(expected):
        raise ValueError(f"Missing archive manifests: {sorted(set(expected) - set(manifests))}")
    for name, manifest in manifests.items():
        spec = expected[name]
        if manifest.get("name") != name or int(manifest.get("num_clients", -1)) != int(spec["num_clients"]):
            raise ValueError(f"Archive manifest mismatch for {name}")
    if offenders:
        raise ValueError(f"Archive contains internal absolute paths: {offenders}")
    return {
        "archive": archive_path.name,
        "variants": sorted(found),
        "credentialed_variants": credentialed,
        "internal_path_offenders": offenders,
        "manifests": manifests,
    }


def validate_paper_contract(dataset_root, paper_contract_path):
    import torch

    dataset_root = Path(dataset_root)
    contract = json.loads(Path(paper_contract_path).read_text(encoding="utf-8"))
    checked = []
    task_aliases = {
        "clinical stage": "clinical_stage_high_vs_low",
        "tumor grade": "clinical_grade_high_vs_low",
        "progression/recurrence": "progression_or_recurrence_vs_free",
        "thermodynamic stability": "thermodynamic_stability",
        "band gap": "electronic_band_gap",
    }
    for item in contract["variants"]:
        if item["availability"] != "download" or item["level"] != "graph":
            continue
        root = dataset_root / item["name"]
        manifest = json.loads((root / "fedgb_manifest.json").read_text(encoding="utf-8"))
        partition = root / "distrib" / manifest["processed_partition"]
        num_graphs = 0
        num_nodes = 0
        for client_id in range(int(item["num_clients"])):
            payload = torch.load(partition / f"data_{client_id}.pt", map_location="cpu", weights_only=False)
            num_graphs += len(payload.graphs)
            num_nodes += sum(int(graph.num_nodes) for graph in payload.graphs)
        if "num_graphs" in item and num_graphs != int(item["num_graphs"]):
            raise ValueError(f"Paper contract graph count mismatch for {item['name']}")
        if "avg_nodes" in item and abs(num_nodes / num_graphs - float(item["avg_nodes"])) > 0.011:
            raise ValueError(f"Paper contract average node mismatch for {item['name']}")
        if "task_name" in item:
            description = (partition / "description.txt").read_text(encoding="utf-8")
            expected = task_aliases.get(item["task_name"], item["task_name"])
            if expected not in description:
                raise ValueError(f"Paper contract task mismatch for {item['name']}")
        checked.append(item["name"])
    return checked


def write_checksum(archive_path, output=None):
    archive_path = Path(archive_path)
    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    output = Path(output) if output else archive_path.parent / "SHA256SUMS"
    output.write_text(f"{digest.hexdigest()}  {archive_path.name}\n", encoding="ascii")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--paper-contract", type=Path, default=DEFAULT_PAPER_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compression-level", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    checked_contracts = validate_paper_contract(args.dataset_root, args.paper_contract)
    build_archive(
        args.dataset_root,
        args.registry,
        args.output,
        compression_level=args.compression_level,
        threads=args.threads,
    )
    report = audit_archive(args.output, args.registry)
    checksum = write_checksum(args.output)
    print(json.dumps(
        {**report, "paper_contract_variants": checked_contracts, "checksum_file": str(checksum)},
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
