"""Check the retained H200 record; optionally verify the separately archived raw runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import runpy
from pathlib import Path
from urllib.parse import urlsplit

from rich.console import Console

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DEFAULT_RECORD = Path(__file__).parents[1] / "evidence" / "flash-liger-378-h200"


def _file(root: Path, name: str) -> Path:
    parts = name.split("/")
    if (len(parts) < 2 or parts[0] not in (
        "flash-liger-378-h200", "flash-liger-378-preflight",
    ) or any(part in ("", ".", "..") or "\\" in part for part in parts)):
        raise ValueError("Invalid evidence member path")
    base = os.path.realpath(root)
    path = os.path.realpath(os.path.join(base, *parts))
    if os.path.commonpath([base, path]) != base:
        raise ValueError("Evidence member escapes its root")
    return Path(path)


def _verify_files(root: Path, inventory: dict) -> None:
    for name, expected in inventory.items():
        data = _file(root, name).read_bytes()
        if (len(data) != expected["size_bytes"]
                or hashlib.sha256(data).hexdigest() != expected["sha256"]):
            raise ValueError(f"Evidence checksum mismatch: {name}")


def _validate_summary(summary: dict, protocol: dict) -> None:
    expected = {(sequence, checkpoint, arm) for sequence in protocol["sequences"]
                for checkpoint in protocol["checkpoints"] for arm in protocol["arms"]}
    rows = summary["rows"]
    keys = {(row["sequence"], row["checkpoint"], row["arm"]) for row in rows}
    if (not summary["complete_grid"] or summary["missing"] or keys != expected
            or len(rows) != len(expected) or len(expected) != 32):
        raise ValueError("Summary must account for the complete 32-cell grid")
    indexed = {(row["sequence"], row["checkpoint"], row["arm"]): row for row in rows}
    for row in rows:
        completed = row["completed_repetitions"]
        if not completed:
            if (row["activation_probe_status"] != "oom"
                    or row["failures"] != ["not_run_probe_oom"] * protocol["repeats"]
                    or row["throughput_ratio_to_sdpa"] is not None):
                raise ValueError("An unmeasured OOM cell cannot claim a throughput ratio")
            continue
        if (completed != protocol["repeats"] or row["failures"]
                or row["activation_probe_status"] != "ok"):
            raise ValueError("Successful cells must contain every planned repetition")
        throughput = row["median_tokens_per_second"]
        if (not all(math.isfinite(row[key]) and row[key] > 0 for key in (
            "min_tokens_per_second", "median_tokens_per_second", "max_tokens_per_second",
            "median_peak_allocated_bytes", "max_peak_allocated_bytes",
        )) or not row["min_tokens_per_second"] <= throughput <= row["max_tokens_per_second"]
                or row["median_peak_allocated_bytes"] > row["max_peak_allocated_bytes"]
                or row["sm_clock_min_mhz"] > row["sm_clock_max_mhz"]):
            raise ValueError("Invalid summary range")
        baseline = indexed[(row["sequence"], row["checkpoint"], "sdpa")]
        ratio = throughput / baseline["median_tokens_per_second"]
        if not math.isclose(row["throughput_ratio_to_sdpa"], ratio, rel_tol=1e-12):
            raise ValueError("Summary ratio disagrees with its SDPA baseline")


def check_record(record: Path, raw_root: Path | None = None) -> dict:
    manifest = json.loads((record / "manifest.json").read_text())
    archive = manifest["archive"]
    url = urlsplit(archive["url"])
    if (manifest["schema_version"] != 1 or not _SHA256.fullmatch(archive["sha256"])
            or archive["files"] != len(manifest["raw_files"])
            or url.scheme != "https" or url.netloc != "github.com"
            or "/releases/download/" not in url.path or url.query or url.fragment):
        raise ValueError("A versioned archive URL and SHA-256 manifest are required")
    for name, metadata in manifest["raw_files"].items():
        _file(record.parent, name)
        if not _SHA256.fullmatch(metadata["sha256"]) or metadata["size_bytes"] < 0:
            raise ValueError("Invalid raw-file digest or size")
    _verify_files(record.parent, manifest["retained_files"])
    protocol = json.loads((record / "protocol.json").read_text())
    summary = json.loads((record / "summary.json").read_text())
    _validate_summary(summary, protocol)
    result = {"validation_level": "summary_only", "cells": len(summary["rows"]),
              "retained_files_verified": len(manifest["retained_files"]),
              "raw_files_verified": 0}
    if raw_root is not None:
        _verify_files(raw_root, manifest["raw_files"])
        summarize = runpy.run_path(str(Path(__file__).with_name("issue378_sweep.py")))[
            "summarize"
        ]
        regenerated = summarize(raw_root / "flash-liger-378-h200", protocol)
        if regenerated != summary:
            raise ValueError("Recomputed raw summary differs from the retained summary")
        result.update(validation_level="raw_verified", raw_files_verified=archive["files"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=_DEFAULT_RECORD)
    parser.add_argument("--raw-root", type=Path, help="Extracted archive root; never downloaded")
    args = parser.parse_args()
    console = Console()
    try:
        result = check_record(args.record, args.raw_root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        console.print(f"FAIL: {exc}", markup=False)
        raise SystemExit(1) from exc
    console.print(result)
    if args.raw_root is None:
        console.print("PASS: retained summary only; raw files not supplied or checked.")
    else:
        console.print("PASS: raw checksums and recomputed summary verified.")


if __name__ == "__main__":
    main()
