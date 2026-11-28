"""Inject owned direct reads outside an observer's process tree to test foreign-read rejection."""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.harness.streaming_controls.io_calibration import FILE_BYTES, direct_read


def observe(path: Path, ready_fd: int) -> None:
    from benchmarks.harness.arm_validity import (
        DiskReadings,
        disk_device,
        disk_reads,
        foreign_reads_check_result,
        other_disk_reads,
        process_reads,
    )

    device = disk_device(str(path))
    start = DiskReadings.take(device, disk_reads, other_disk_reads, process_reads)
    os.write(ready_fd, b"READY\n")
    os.close(ready_fd)
    if not select.select([sys.stdin], [], [], 120)[0] or sys.stdin.readline() != "DONE\n":
        raise TimeoutError("foreign-reader handshake expired")
    end = DiskReadings.take(device, disk_reads, other_disk_reads, process_reads)
    receipt = foreign_reads_check_result(str(path), device, None, start, end)
    receipt.update(observer_pid=os.getpid(), reader_pid=os.getppid())
    sys.stdout.write(json.dumps(receipt) + "\n")


def assess(validity: dict[str, Any], completed_bytes: int, hashes_match: bool) -> dict[str, Any]:
    passed = (
        completed_bytes == FILE_BYTES * 16
        and hashes_match is True
        and validity.get("outcome") == "void"
    )
    return {
        "status": "PASS_FOREIGN_READER_DETECTED" if passed else "BLOCKED",
        "performance_verdict": "NO_VERDICT",
        "validity": validity,
        "completed_bytes": completed_bytes,
        "hashes_match": hashes_match,
    }


def run(calibration: Path, output: Path) -> dict[str, Any]:
    if sys.platform != "linux":
        raise RuntimeError("Linux control required")
    previous = json.loads((calibration / "receipt.json").read_text())
    if (
        previous["status"] != "PASS_COUNTER_CALIBRATION"
        or previous["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ):
        raise RuntimeError("same-boot calibration required")
    output.mkdir(parents=True, exist_ok=False)
    path = calibration / "owned_calibration.bin"
    result: dict[str, Any] = {"status": "BLOCKED", "performance_verdict": "NO_VERDICT"}
    read_fd, write_fd = os.pipe()
    process = None
    deadline = time.monotonic() + 120
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.harness.streaming_controls.foreign_read_control",
                "--observe",
                str(path),
                "--ready-fd",
                str(write_fd),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            pass_fds=(write_fd,),
        )
        os.close(write_fd)
        write_fd = -1
        if not select.select([read_fd], [], [], 15)[0] or os.read(read_fd, 6) != b"READY\n":
            raise RuntimeError("observer did not establish counters")
        # The parent reads while its child observes. Parent is outside child's own tree.
        total = 0
        hashes_match = True
        for _ in range(16):
            row = direct_read(path, deadline)
            total += row["application_bytes"]
            hashes_match &= row["content_sha256"] == previous["expected_sha256"]
        stdout, stderr = process.communicate(
            "DONE\n", timeout=max(0.1, deadline - time.monotonic())
        )
        (output / "observer_stdout.txt").write_text(stdout)
        (output / "observer_stderr.txt").write_text(stderr)
        if process.returncode != 0:
            raise RuntimeError(f"observer failed: {process.returncode}")
        result = assess(json.loads(stdout), total, hashes_match)
    except Exception as error:
        result.update(status="BLOCKED", error=repr(error))
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate()
            if not (output / "observer_stdout.txt").exists():
                (output / "observer_stdout.txt").write_text(stdout)
                (output / "observer_stderr.txt").write_text(stderr)
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observe", type=Path)
    parser.add_argument("--ready-fd", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.observe is not None:
        if args.ready_fd is None:
            parser.error("--ready-fd required")
        observe(args.observe, args.ready_fd)
        return 0
    if args.calibration is None or args.output is None:
        parser.error("--calibration and --output required")
    result = run(args.calibration, args.output)
    from rich.console import Console

    Console().print(result["status"])
    return 0 if result["status"] == "PASS_FOREIGN_READER_DETECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
