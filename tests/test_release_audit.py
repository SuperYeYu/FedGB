from pathlib import Path

from scripts.verify.release_audit import (
    EICU_FORBIDDEN_ARCHIVE_SUFFIXES,
    audit_text_files,
    find_eicu_public_leaks,
)


def test_text_audit_rejects_internal_paths_and_placeholders(tmp_path):
    (tmp_path / "public.md").write_text(
        "source=/opt/data/private/yyy/data\nlink=TBD_PUBLIC_LINK\n", encoding="utf-8"
    )
    errors = audit_text_files(tmp_path)
    assert any("internal absolute path" in error for error in errors)
    assert any("release placeholder" in error for error in errors)


def test_text_audit_accepts_public_release_metadata(tmp_path):
    (tmp_path / "README.md").write_text(
        "Download FedGB-datasets-v1.0.0.tar.zst from the public dataset folder.\n",
        encoding="utf-8",
    )
    assert audit_text_files(tmp_path) == []


def test_text_audit_ignores_intentional_negative_test_fixtures(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "fixture.py").write_text(
        "bad = '/opt/data/private/yyy/source'\nplaceholder = 'TBD_PUBLIC_LINK'\n",
        encoding="utf-8",
    )
    assert audit_text_files(tmp_path) == []


def test_eicu_privacy_audit_blocks_common_raw_archive_formats():
    assert {".gz", ".zip", ".tar", ".zst"} <= EICU_FORBIDDEN_ARCHIVE_SUFFIXES


def test_eicu_privacy_audit_rejects_compressed_raw_tables(tmp_path):
    (tmp_path / "patient.csv.gz").write_bytes(b"credentialed data")
    (tmp_path / "README.md").write_text("public instructions\n", encoding="utf-8")
    assert find_eicu_public_leaks(tmp_path) == ["patient.csv.gz"]
