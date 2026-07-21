import json

from scripts.verify.summarize_release_validation import sanitize_report_text, summarize_smoke_status


def test_smoke_summary_counts_passes_and_failure_categories():
    status = {
        "a": {"status": "passed", "duration_sec": 1.2, "category": None},
        "b": {"status": "failed", "duration_sec": 2.0, "category": "contract"},
        "c": {"status": "failed", "duration_sec": 3.0, "category": "timeout"},
    }
    summary = summarize_smoke_status(status)
    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 2
    assert summary["failure_categories"] == {"contract": 1, "timeout": 1}
    assert summary["duration_sec"] == 6.2


def test_report_text_sanitizes_private_server_paths():
    text = "warning at /opt/data/private/yyy/yyy_env/lib/python3.12/site-packages/module.py"
    sanitized = sanitize_report_text(text)
    assert "/opt/data/private" not in sanitized
    assert "<private-source>/" in sanitized
