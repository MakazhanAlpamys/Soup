"""The small record check works offline and distinguishes raw-data verification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "flash_liger_record", _ROOT / "benchmarks" / "harness" / "flash_liger_record.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.fixture
def record(tmp_path):
    source = _ROOT / "benchmarks" / "evidence"
    manifest_path = source / "flash-liger-378-h200" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["retained_files"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    target = tmp_path / "flash-liger-378-h200"
    shutil.copyfile(manifest_path, target / "manifest.json")
    return target


def test_summary_only_passes_without_bulk_and_says_what_was_checked(record):
    result = _MODULE.check_record(record)
    assert result == {"validation_level": "summary_only", "cells": 32,
                      "retained_files_verified": 6, "raw_files_verified": 0}
    assert not (record / "timings").exists()
    assert not (record / "probes").exists()


def test_summary_only_cli_explicitly_reports_absent_raw_data(record, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["flash_liger_record.py", "--record", str(record)])
    _MODULE.main()
    assert "raw files not supplied or checked" in capsys.readouterr().out


def test_modified_retained_summary_fails_its_checksum(record):
    (record / "summary.json").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _MODULE.check_record(record)


@pytest.mark.parametrize("change", ["ratio", "missing_cell", "oom_ratio"])
def test_summary_arithmetic_is_checked_even_with_an_updated_digest(record, change):
    path = record / "summary.json"
    summary = json.loads(path.read_text())
    if change == "ratio":
        summary["rows"][0]["throughput_ratio_to_sdpa"] = 2.0
    elif change == "missing_cell":
        summary["rows"].pop()
    else:
        next(row for row in summary["rows"] if not row["completed_repetitions"])[
            "throughput_ratio_to_sdpa"
        ] = 1.0
    data = (json.dumps(summary) + "\n").encode()
    path.write_bytes(data)
    manifest_path = record / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["retained_files"]["flash-liger-378-h200/summary.json"] = {
        "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
    }
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _MODULE.check_record(record)


def test_explicit_raw_verification_does_not_silently_fall_back_to_summary(record):
    with pytest.raises(FileNotFoundError):
        _MODULE.check_record(record, record.parent)


def test_manifest_cannot_read_outside_the_evidence_tree(record):
    path = record / "manifest.json"
    manifest = json.loads(path.read_text())
    metadata = manifest["retained_files"].pop("flash-liger-378-h200/summary.json")
    manifest["retained_files"]["flash-liger-378-h200/../../outside"] = metadata
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="member path"):
        _MODULE.check_record(record)
