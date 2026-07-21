#!/usr/bin/env python3
"""Run or resume the FedGB smoke matrix with bounded subprocesses."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def classify(stderr):
    lowered = stderr.lower()
    if "out of memory" in lowered:
        return "oom"
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return "import"
    if "nan" in lowered or "inf" in lowered:
        return "numerical"
    if "filenotfounderror" in lowered or "no such file" in lowered:
        return "data"
    if "attributeerror" in lowered or "typeerror" in lowered or "valueerror" in lowered:
        return "contract"
    return "algorithm"


def assign_gpus(cases, gpus):
    if not gpus:
        raise ValueError("At least one GPU id is required unless --cpu is used.")
    return [(case, gpus[index % len(gpus)]) for index, case in enumerate(cases)]


def run_case(case, gpu, opts):
    case_dir = opts.output / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "verify" / "run_smoke_case.py"),
        "--case-id", case["id"],
        "--output", str(case_dir),
    ]
    if opts.cpu:
        command.append("--cpu")
    else:
        command += ["--gpuid", str(gpu)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    started = time.time()
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=opts.timeout,
        )
        (case_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (case_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
        passed = process.returncode == 0 and (case_dir / "result.json").exists()
        return {
            "status": "passed" if passed else "failed",
            "returncode": process.returncode,
            "duration_sec": round(time.time() - started, 3),
            "category": None if passed else classify(process.stderr),
            "gpu": None if opts.cpu else gpu,
        }
    except subprocess.TimeoutExpired as exc:
        (case_dir / "stdout.log").write_text(exc.stdout or "", encoding="utf-8")
        (case_dir / "stderr.log").write_text(exc.stderr or "", encoding="utf-8")
        return {
            "status": "failed",
            "returncode": None,
            "duration_sec": round(time.time() - started, 3),
            "category": "timeout",
            "gpu": None if opts.cpu else gpu,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / ".smoke_results")
    parser.add_argument("--rerun-failed", action="store_true")
    opts = parser.parse_args()

    matrix = json.loads((ROOT / "scripts" / "verify" / "smoke_matrix.json").read_text())
    wanted = set(opts.case_ids or [case["id"] for case in matrix["cases"]])
    cases = [case for case in matrix["cases"] if case["id"] in wanted]
    opts.output.mkdir(parents=True, exist_ok=True)
    status_path = opts.output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    gpus = [int(item) for item in opts.gpus.split(",") if item.strip()]

    runnable = []
    for case in cases:
        previous = status.get(case["id"], {})
        if previous.get("status") == "passed":
            continue
        if previous.get("status") == "failed" and not opts.rerun_failed:
            continue
        runnable.append(case)

    assignments = [(case, None) for case in runnable] if opts.cpu else assign_gpus(runnable, gpus)
    groups = {None: []} if opts.cpu else {gpu: [] for gpu in gpus}
    for case, gpu in assignments:
        groups[gpu].append(case)
    lock = threading.Lock()

    def run_group(gpu, group):
        for case in group:
            result = run_case(case, gpu, opts)
            with lock:
                status[case["id"]] = result
                temporary = status_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
                temporary.replace(status_path)
                print(case["id"], result, flush=True)

    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [executor.submit(run_group, gpu, group) for gpu, group in groups.items()]
        for future in futures:
            future.result()

    summary = {}
    for item in status.values():
        key = item["status"] if item["status"] == "passed" else f"failed:{item['category']}"
        summary[key] = summary.get(key, 0) + 1
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

