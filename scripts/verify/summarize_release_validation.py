#!/usr/bin/env python3
"""Collect compact FedGB release validation artifacts."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from fedgb.data.validation import INTERNAL_PATH_PATTERN


ROOT = Path(__file__).resolve().parents[2]


def sanitize_report_text(text):
    return INTERNAL_PATH_PATTERN.sub("<private-source>/", text)


def summarize_smoke_status(status):
    failures = Counter(
        item.get("category") or "unknown"
        for item in status.values()
        if item.get("status") != "passed"
    )
    passed = sum(item.get("status") == "passed" for item in status.values())
    return {
        "total": len(status),
        "passed": passed,
        "failed": len(status) - passed,
        "failure_categories": dict(sorted(failures.items())),
        "duration_sec": round(sum(float(item.get("duration_sec", 0)) for item in status.values()), 3),
    }


def render_markdown(summary):
    lines = [
        "# FedGB Smoke Matrix Summary",
        "",
        f"- Total cases: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Aggregate duration: {summary['duration_sec']:.3f} seconds",
    ]
    if summary["failure_categories"]:
        lines.extend(["", "## Failure Categories", ""])
        lines.extend(
            f"- {category}: {count}"
            for category, count in summary["failure_categories"].items()
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-status", type=Path, default=ROOT / ".smoke_results" / "status.json")
    parser.add_argument("--dataset-report", type=Path, default=ROOT / "dataset_validation_report.json")
    parser.add_argument("--pytest-report", type=Path, default=ROOT / "pytest-report.txt")
    parser.add_argument("--environment", type=Path, default=ROOT / "environment.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "release-validation")
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    status = json.loads(opts.smoke_status.read_text(encoding="utf-8"))
    summary = summarize_smoke_status(status)
    (opts.output / "smoke-status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (opts.output / "smoke-summary.md").write_text(render_markdown(summary), encoding="utf-8")
    shutil.copy2(opts.dataset_report, opts.output / "dataset-validation.json")
    shutil.copy2(opts.environment, opts.output / "environment.json")
    pytest_text = opts.pytest_report.read_text(encoding="utf-8")
    (opts.output / "pytest-report.txt").write_text(
        sanitize_report_text(pytest_text), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
