"""Bounded Linux direct-read calibration; never a training/performance verdict."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mmap
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

BLOCK_BYTES = 4 << 20
FILE_BYTES = 128 << 20
TOLERANCE_BYTES = 4 << 20


def assess(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Missing, reset, cached or contaminated counters cannot pass calibration."""
    failures = []
    if len(samples) != 2:
        failures.append("exactly two calibration reads required")
    for index, sample in enumerate(samples):
        for field in ("application_bytes", "device_bytes", "process_bytes"):
            value = sample.get(field)
            if type(value) is not int or value < 0:
                failures.append(f"sample {index}: invalid {field}")
        if sample.get("application_bytes") != FILE_BYTES:
            failures.append(f"sample {index}: incomplete read")
        for field in ("device_bytes", "process_bytes"):
            value = sample.get(field)
            if type(value) is int and abs(value - FILE_BYTES) > TOLERANCE_BYTES:
                failures.append(f"sample {index}: {field} outside fixed calibration tolerance")
        if sample.get("content_sha256") != sample.get("expected_sha256") or not re.fullmatch(
            r"[0-9a-f]{64}", str(sample.get("content_sha256", ""))
        ):
            failures.append(f"sample {index}: content hash mismatch or absent")
    return {
        "status": "PASS_COUNTER_CALIBRATION" if not failures else "BLOCKED",
        "failures": failures,
        "performance_verdict": "NO_VERDICT",
        "physical_byte_scope": (
            "whole Linux block device; not NAND traffic or exclusive attribution"
        ),
    }


def process_read_bytes() -> int:
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            value = int(line.split()[1])
            if value >= 0:
                return value
    raise RuntimeError("read_bytes unavailable")


def snapshot(path: Path) -> dict[str, Any]:
    from benchmarks.harness.arm_validity import disk_device, disk_reads

    device = disk_device(str(path))
    if not device or not re.fullmatch(r"nvme\d+n\d+", device):
        raise RuntimeError("single NVMe mapping required")
    physical = disk_reads(device)
    if physical is None:
        raise RuntimeError("physical counter unavailable")
    return {
        "device": device,
        "device_bytes": physical,
        "process_bytes": process_read_bytes(),
        "monotonic_s": time.monotonic(),
    }


def difference(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    if before["device"] != after["device"]:
        raise RuntimeError("physical device changed")
    values = {field: after[field] - before[field] for field in ("device_bytes", "process_bytes")}
    if any(value < 0 for value in values.values()):
        raise RuntimeError("counter reset")
    return values


def direct_read(path: Path, deadline: float) -> dict[str, Any]:
    """Read only the owned regular file; no buffered fallback or raw-device access."""
    if sys.platform != "linux" or not hasattr(os, "O_DIRECT"):
        raise RuntimeError("Linux O_DIRECT is required")
    if path.is_symlink() or not path.is_file() or path.stat().st_size != FILE_BYTES:
        raise ValueError("expected owned regular calibration file")
    import stat

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    total = 0
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("not a regular file")
        with io.FileIO(descriptor, "rb", closefd=False) as handle:
            with mmap.mmap(-1, BLOCK_BYTES) as buffer:
                while total < FILE_BYTES:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("calibration deadline expired")
                    got = handle.readinto(buffer)
                    if got != BLOCK_BYTES:
                        raise OSError(f"short direct read: {got}")
                    digest.update(buffer)
                    total += got
    finally:
        os.close(descriptor)
    return {"application_bytes": total, "content_sha256": digest.hexdigest()}


def run(output: Path) -> dict[str, Any]:
    from benchmarks.harness.arm_validity import disk_device, disk_reads

    if sys.platform != "linux":
        raise RuntimeError("hardware calibration requires Linux")
    output.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "status": "RUNNING",
        "samples": [],
        "performance_verdict": "NO_VERDICT",
        "pid": os.getpid(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "file_bytes": FILE_BYTES,
        "tolerance_bytes": TOLERANCE_BYTES,
    }
    deadline = time.monotonic() + 120
    try:
        device = disk_device(str(output))
        if not device or not re.fullmatch(r"nvme\d+n\d+", device):
            raise RuntimeError(f"single physical NVMe mapping required: {device!r}")
        result["device"] = device
        path = output / "owned_calibration.bin"
        # Real non-sparse bytes. Only this fresh output directory is written.
        digest = hashlib.sha256()
        with path.open("xb") as handle:
            for _ in range(FILE_BYTES // BLOCK_BYTES):
                if time.monotonic() >= deadline:
                    raise TimeoutError("calibration creation deadline expired")
                block = os.urandom(BLOCK_BYTES)
                handle.write(block)
                digest.update(block)
            handle.flush()
            os.fsync(handle.fileno())
        result["expected_sha256"] = digest.hexdigest()
        for _ in range(2):
            before_disk, before_process = disk_reads(device), process_read_bytes()
            row = direct_read(path, deadline)
            after_process, after_disk = process_read_bytes(), disk_reads(device)
            row.update(
                expected_sha256=digest.hexdigest(),
                device_bytes=None
                if before_disk is None or after_disk is None
                else after_disk - before_disk,
                process_bytes=after_process - before_process,
                raw_device_start=before_disk,
                raw_device_end=after_disk,
                raw_process_start=before_process,
                raw_process_end=after_process,
            )
            result["samples"].append(row)
        result.update(assess(result["samples"]))
    except Exception as error:
        result.update(status="BLOCKED", error=repr(error))
    finally:
        (output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    from rich.console import Console

    Console().print(result["status"])
    return 0 if result["status"] == "PASS_COUNTER_CALIBRATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
