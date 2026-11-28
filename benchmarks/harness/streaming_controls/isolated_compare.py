"""Fresh-process streaming numerical comparison. Physical performance acceptance stays separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.harness.arm_validity import CHECKS, arm_outcome
from benchmarks.harness.streaming_controls.compare import save
from benchmarks.harness.streaming_controls.setup_supervisor import supervise


def compare_results(results: list[dict[str, Any]], order: list[str]) -> dict[str, Any]:
    failures = []
    if not results or len(results) != len(order) or set(order) != {"A", "B"}:
        failures.append("incomplete paired order")
    reference = results[0] if results else {}
    for position, item in enumerate(results):
        if (
            item.get("status") != "ARM_COMPLETE"
            or item.get("order") != order[position : position + 1]
        ):
            failures.append(f"arm {position}: incomplete or wrong order")
            continue
        if item.get("weight_sha256") != reference.get("weight_sha256"):
            failures.append(f"arm {position}: weights differ")
        for field in (
            "config",
            "steps_per_arm",
            "warmup_steps",
            "mode",
            "torch",
            "workload_sha256",
            "effective_tokens_per_step",
        ):
            if item.get(field) != reference.get(field):
                failures.append(f"arm {position}: {field} differs")
        rows, ref_rows = item.get("records", []), reference.get("records", [])
        if len(rows) != 1 or len(ref_rows) != 1:
            failures.append(f"arm {position}: single record required")
            continue
        row, ref = rows[0], ref_rows[0]
        if row.get("status") != "COMPLETED" or row.get("changed_adapter") is not True:
            failures.append(f"arm {position}: no completed update")
        for field in ("initial_adapter_sha256", "final_adapter_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get(field, ""))) or row.get(
                field
            ) != ref.get(field):
                failures.append(f"arm {position}: {field} mismatch")
        updates = row.get("updates", [])
        if len(updates) != item.get("steps_per_arm") or not updates:
            failures.append(f"arm {position}: missing steps")
        for step, update in enumerate(updates):
            loss = update.get("loss")
            if (
                type(loss) not in (int, float)
                or not math.isfinite(loss)
                or update.get("step") != step
                or not re.fullmatch(r"[0-9a-f]{64}", str(update.get("gradient_sha256", "")))
            ):
                failures.append(f"arm {position}: invalid loss, step or gradient evidence")
        if [(x.get("loss"), x.get("gradient_sha256")) for x in updates] != [
            (x.get("loss"), x.get("gradient_sha256")) for x in ref.get("updates", [])
        ]:
            failures.append(f"arm {position}: numerical mismatch")
    return {
        "status": "NUMERICAL_PASS" if not failures else "BLOCKED",
        "failures": failures,
        "performance_verdict": "NO_VERDICT",
    }


CLEANUP_RESERVE_SECONDS = 40


def require_valid_cuda_arm(item: dict[str, Any]) -> None:
    """Stop before spending on another arm once a frozen validity gate fails."""
    rows = item.get("records", [])
    if len(rows) != 1:
        raise RuntimeError("CUDA arm is missing its validity record; no retry")
    validity = rows[0].get("validity", {})
    checks = validity.get("checks", {})
    if (
        set(validity.get("required", [])) != set(CHECKS)
        or set(checks) != set(CHECKS)
        or arm_outcome(checks, CHECKS) != "ok"
    ):
        raise RuntimeError("CUDA arm failed frozen arm validity; no retry")
    clock = validity.get("gpu_clock", {})
    if (
        clock.get("outcome") != "ok"
        or clock.get("samples", 0) < 2
        or clock.get("unknown_samples") != 0
    ):
        raise RuntimeError("CUDA arm failed complete clock observation; no retry")


def run(
    output: Path,
    mode: str,
    deadline_unix_s: float | None = None,
    prepared: Path | None = None,
    *,
    abba_blocks: int = 5,
) -> dict[str, Any]:
    if mode not in ("cpu-smoke", "cuda-pilot", "cuda-representative"):
        raise ValueError("unknown mode")
    if type(abba_blocks) is not int or abba_blocks not in (5, 20):
        raise ValueError("abba_blocks must be the frozen pilot (5) or confirmation (20)")
    if abba_blocks != 5 and mode != "cuda-representative":
        raise ValueError("20 ABBA blocks require cuda-representative mode")
    if (
        mode == "cuda-representative"
        and prepared is None
        or mode == "cuda-pilot"
        and prepared is not None
    ):
        raise ValueError("representative mode needs prepared inputs; synthetic pilot forbids them")
    if os.name != "posix":
        raise RuntimeError("POSIX supervisor required")
    now = time.time()
    cuda = mode != "cpu-smoke"
    bound = (
        (5400 if abba_blocks == 20 else 3600)
        if mode == "cuda-representative"
        else (1800 if cuda else 180)
    )
    if deadline_unix_s is None:
        if cuda:
            raise ValueError("CUDA requires the outer supervisor's absolute deadline")
        deadline_unix_s = now + bound
    if (
        type(deadline_unix_s) not in (int, float)
        or not math.isfinite(deadline_unix_s)
        or deadline_unix_s <= now + CLEANUP_RESERVE_SECONDS
    ):
        raise ValueError("Unexpired finite deadline with 40s cleanup reserve required")
    deadline = min(deadline_unix_s, now + bound) - CLEANUP_RESERVE_SECONDS
    output = Path(os.path.realpath(output))
    output.mkdir(parents=True, exist_ok=False)
    order = list("ABBA") * abba_blocks if cuda else list("AB")
    result: dict[str, Any] = {
        "status": "RUNNING",
        "order": order,
        "arms": [],
        "planned_abba_blocks": abba_blocks if cuda else None,
        "outer_deadline_unix_s": deadline_unix_s,
        "work_deadline_unix_s": deadline,
        "cleanup_reserve_seconds": CLEANUP_RESERVE_SECONDS,
        "performance_verdict": "NO_VERDICT",
    }
    try:
        if prepared is not None:
            from benchmarks.harness.streaming_controls.representative_workload import (
                file_sha,
                load_prepared,
            )

            load_prepared(prepared, cuda=cuda)
            prepared_sha = file_sha(prepared)
            result["prepared_sha256"] = prepared_sha
        if cuda:
            from benchmarks.harness.streaming_controls.foreign_read_control import (
                run as foreign_control,
            )
            from benchmarks.harness.streaming_controls.io_calibration import run as calibrate

            calibration = calibrate(output / "io_calibration")
            if calibration["status"] != "PASS_COUNTER_CALIBRATION":
                raise RuntimeError("physical read calibration blocked; no training arms")
            control = foreign_control(output / "io_calibration", output / "foreign_read_control")
            if control["status"] != "PASS_FOREIGN_READER_DETECTED":
                raise RuntimeError("foreign-reader rejection not proven; no training arms")
        for index, arm in enumerate(order):
            if time.time() >= deadline:
                raise TimeoutError("shared work deadline expired; no new arm")
            directory = output / f"arm_{index:02d}_{arm}"
            command = [
                sys.executable,
                "-m",
                "benchmarks.harness.streaming_controls.isolated_arm",
                "--output",
                str(directory),
                "--mode",
                mode,
                "--arm",
                arm,
            ]
            if prepared is not None:
                if file_sha(prepared) != prepared_sha:
                    raise RuntimeError("prepared workload changed during series")
                command.extend(["--prepared", str(prepared.absolute())])
            script = output / f"arm_{index:02d}.sh"
            script.write_text(
                "#!/bin/sh\nset -eu\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8\nexec "
                + shlex.join(command)
                + "\n"
            )
            supervision = supervise(
                script,
                hashlib.sha256(script.read_bytes()).hexdigest(),
                output / f"supervisor_{index:02d}",
                300,
                deadline,
            )
            if supervision["status"] != "PASS" or not supervision["group_cleanup_confirmed"]:
                raise RuntimeError(f"arm {index} process failed; no retry")
            item = json.loads((directory / "result.json").read_text())
            result["arms"].append(item)
            if item.get("status") != "ARM_COMPLETE":
                raise RuntimeError(f"arm {index} incomplete")
            if prepared is not None and item.get("workload_sha256") != prepared_sha:
                raise RuntimeError("arm did not use the frozen prepared workload")
            if cuda:
                require_valid_cuda_arm(item)
            if index and compare_results(result["arms"], order[: index + 1])["status"] != (
                "NUMERICAL_PASS"
            ):
                raise RuntimeError(f"arm {index} numerical mismatch; stop series")
            save(output / "result.json", result)
        result.update(compare_results(result["arms"], order))
    except Exception as error:
        result.update(status="BLOCKED", error=repr(error))
    finally:
        save(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["cpu-smoke", "cuda-pilot", "cuda-representative"], default="cpu-smoke"
    )
    parser.add_argument("--prepared", type=Path)
    parser.add_argument(
        "--abba-blocks",
        type=int,
        choices=(5, 20),
        default=5,
        help="Frozen schedule: 5 pilot or 20 representative confirmation blocks",
    )
    parser.add_argument(
        "--deadline-unix-s",
        type=float,
        help="Same absolute deadline as outer supervisor; required for CUDA",
    )
    args = parser.parse_args()
    result = run(
        args.output, args.mode, args.deadline_unix_s, args.prepared, abba_blocks=args.abba_blocks
    )
    from rich.console import Console

    Console().print(result["status"])
    return 0 if result["status"] == "NUMERICAL_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
