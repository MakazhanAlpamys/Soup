"""Prospective bounded POSIX setup supervisor; independent of SSH output pipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def group_leader(script: str, exit_path: str) -> None:
    """Keep the owned session leader alive until its supervisor cleans up the group."""
    code = subprocess.call(["/bin/bash", script], stdin=subprocess.DEVNULL)
    path = Path(exit_path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"returncode": code}) + "\n")
    temporary.replace(path)
    while True:
        signal.pause()


def validate_bounds(max_seconds: float, deadline_unix_s: float, *, purpose: str = "setup") -> None:
    """Validate every planned phase before launch; setup keeps its original bound."""
    bounds = {"setup": 1800, "measurement": 5400}
    if purpose not in bounds:
        raise ValueError("Unknown supervisor purpose")
    limit = bounds[purpose]
    if (
        not math.isfinite(max_seconds)
        or not 0 < max_seconds <= limit
        or not math.isfinite(deadline_unix_s)
        or deadline_unix_s <= time.time()
    ):
        raise ValueError(f"Unexpired deadline and {purpose} bound <={limit}s required")


def supervise(
    script: Path,
    expected_sha256: str,
    output: Path,
    max_seconds: float,
    deadline_unix_s: float,
    *,
    purpose: str = "setup",
) -> dict:
    """Run a pinned setup script once; retain logs/exit status without stdout piping."""
    if os.name != "posix":
        raise ValueError("POSIX process groups are required")
    validate_bounds(max_seconds, deadline_unix_s, purpose=purpose)
    if script.is_symlink() or not script.is_file():
        raise ValueError("Setup script must be a regular, non-symlink file")
    source = script.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("Setup script SHA mismatch")
    output = Path(os.path.realpath(output))
    output.mkdir(parents=True, exist_ok=False)
    # Execute the verified copy so changes to the caller's file cannot race launch.
    frozen = output / "setup.sh"
    frozen.write_bytes(source)
    started = time.monotonic()
    receipt = {
        "status": "STARTING",
        "script_sha256": expected_sha256,
        "deadline_unix_s": deadline_unix_s,
        "max_seconds": max_seconds,
        "purpose": purpose,
        "performance_verdict": "NO_VERDICT",
    }

    def save() -> None:
        temporary = output / "result.tmp"
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "result.json")

    save()
    try:
        with (output / "stdout_stderr.log").open("xb") as log:
            child_exit = output / "child_exit.json"
            process = subprocess.Popen(
                [
                    sys.executable,
                    os.path.realpath(__file__),
                    "--group-leader",
                    str(frozen),
                    str(child_exit),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                receipt.update(status="RUNNING", pid=process.pid, process_group=process.pid)
                save()
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Setup group leader exited unexpectedly")
                    remaining = min(
                        max_seconds - (time.monotonic() - started), deadline_unix_s - time.time()
                    )
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(process.args, max_seconds)
                    if child_exit.exists():
                        code = json.loads(child_exit.read_text())["returncode"]
                        receipt.update(status="PASS" if code == 0 else "FAILED", returncode=code)
                        break
                    time.sleep(min(0.05, remaining))
            except BaseException as error:
                receipt.update(
                    status="TIMEOUT" if isinstance(error, subprocess.TimeoutExpired) else "FAILED",
                    error=repr(error),
                )
                if not isinstance(error, subprocess.TimeoutExpired):
                    raise
            finally:
                # The retained live leader pins the group identity even after setup exits.
                receipt["group_cleanup_confirmed"] = False
                if process.poll() is None and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2)
                    receipt["group_cleanup_confirmed"] = True
                receipt["leader_returncode"] = process.returncode
                if "returncode" not in receipt:
                    receipt["returncode"] = process.returncode
                if not receipt["group_cleanup_confirmed"]:
                    receipt["status"] = "FAILED"
    except BaseException as error:
        receipt.update(status="FAILED", error=repr(error))
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        save()
    return receipt


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--group-leader":
        group_leader(sys.argv[2], sys.argv[3])
        return 1  # The supervisor must terminate the retained leader.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--deadline-unix-s", type=float, required=True)
    args = parser.parse_args()
    # Launch with setsid/nohup as documented; shell disconnection must not interrupt setup.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def interrupted(_signum: int, _frame: object) -> None:
        raise InterruptedError("Setup supervisor termination requested")

    signal.signal(signal.SIGTERM, interrupted)
    result = supervise(
        args.script, args.sha256, args.output, args.max_seconds, args.deadline_unix_s
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
