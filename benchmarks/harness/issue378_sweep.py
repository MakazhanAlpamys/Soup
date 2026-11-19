"""Freeze, execute/resume and summarize the #378 four-arm H200 sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

ARMS = ("sdpa", "fa2", "liger", "fa2_liger")
SEQUENCES = (1024, 2048, 4096, 8192)
CHECKPOINTS = ("off", "on")
console = Console()


def validate_timing(record: dict, protocol: dict, probe: dict) -> None:
    if record["probe"]:
        raise ValueError("A counter/profiler probe is not a timing repetition")
    for key, expected in (("warmup_steps", protocol["warmup"]),
                          ("measured_steps", protocol["steps"]),
                          ("harness_sha256", protocol["worker_sha256"]),
                          ("source_commit", protocol["source_commit"])):
        if record[key] != expected:
            raise ValueError(f"Incomparable {key}")
    if probe["harness_sha256"] != protocol["worker_sha256"]:
        raise ValueError("Activation probe used a different harness")
    if probe["source_commit"] != protocol["source_commit"]:
        raise ValueError("Activation probe used a different source commit")
    if record["realized_batch_shape"] != [4, record["sequence"]]:
        raise ValueError("Wrong realized training shape")
    for key in ("first_batch_sha256", "runtime", "attention_implementation", "module_counts"):
        if record[key] != probe[key]:
            raise ValueError(f"Timing differs from its activation probe: {key}")
    counts = probe["activation_counts"]
    if record["arm"] in ("fa2", "fa2_liger"):
        if counts.get("fa2", 0) <= 0 or counts.get("sdpa", 0) != 0:
            raise ValueError("FA2 activation was not verified")
    elif counts.get("sdpa", 0) <= 0 or counts.get("fa2", 0) != 0:
        raise ValueError("SDPA activation was not verified")
    if record["activation_counts"] is not None or record["profile_top_events"]:
        raise ValueError("Timing runs must have counters and profiler disabled")


def summarize(root: Path, protocol: dict) -> dict:
    rows = []
    missing = []
    for sequence in SEQUENCES:
        for checkpoint in CHECKPOINTS:
            workload_hashes = set()
            for arm in ARMS:
                probe_path = root / "probes" / f"s{sequence}-{checkpoint}-{arm}" / "result.json"
                if not probe_path.exists():
                    missing.append(str(probe_path))
                    continue
                probe = json.loads(probe_path.read_text())
                successes, failures = [], []
                for repeat in range(protocol["repeats"]):
                    path = root / "timings" / f"s{sequence}-{checkpoint}-r{repeat + 1}-{arm}" / (
                        "result.json"
                    )
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    record = json.loads(path.read_text())
                    if any(record[key] != value for key, value in (
                        ("arm", arm), ("sequence", sequence), ("checkpoint", checkpoint),
                    )):
                        raise ValueError("A record is stored in the wrong matrix cell")
                    if record["status"] != "ok":
                        failures.append(record["status"])
                        continue
                    if probe["status"] != "ok":
                        raise ValueError("Successful timing lacks a successful activation probe")
                    validate_timing(record, protocol, probe)
                    successes.append(record)
                    workload_hashes.add(record["tokenized_data_sha256"])
                row = {"sequence": sequence, "checkpoint": checkpoint, "arm": arm,
                       "completed_repetitions": len(successes), "failures": failures,
                       "activation_probe_status": probe["status"]}
                if successes:
                    throughputs = [r["summary"]["input_tokens_per_second"] for r in successes]
                    peaks = [r["summary"]["peak_allocated_bytes"] for r in successes]
                    clocks = [int(sample["clocks.sm"]) for r in successes
                              for sample in r["telemetry"]]
                    row.update({
                        "median_tokens_per_second": statistics.median(throughputs),
                        "min_tokens_per_second": min(throughputs),
                        "max_tokens_per_second": max(throughputs),
                        "median_peak_allocated_bytes": statistics.median(peaks),
                        "max_peak_allocated_bytes": max(peaks),
                        "sm_clock_min_mhz": min(clocks), "sm_clock_max_mhz": max(clocks),
                    })
                rows.append(row)
            if len(workload_hashes) > 1:
                raise ValueError(
                    f"Arms used different actual tokens/labels: S{sequence}/{checkpoint}"
                )
    for row in rows:
        baseline = next((r for r in rows if r["sequence"] == row["sequence"]
                         and r["checkpoint"] == row["checkpoint"] and r["arm"] == "sdpa"), None)
        row["throughput_ratio_to_sdpa"] = None
        if baseline and (
            row["completed_repetitions"] == baseline["completed_repetitions"] == protocol["repeats"]
        ):
            row["throughput_ratio_to_sdpa"] = (row["median_tokens_per_second"]
                                                / baseline["median_tokens_per_second"])
    return {"complete_grid": not missing, "missing": missing, "rows": rows,
            "scope": "training-step timing on one shared H200; no quality or dedicated-card claim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "probes", "timings"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--math-evidence", type=Path, required=True)
    args = parser.parse_args()
    worker = Path(__file__).with_name("issue378_flash_liger.py")
    root = args.output
    math_evidence = json.loads(args.math_evidence.read_text())
    if math_evidence["status"] != "pass":
        raise ValueError("Verify the installed FA2 forward/backward before benchmarking")
    recipe = {
        "model": str(Path(args.model).absolute()), "warmup": args.warmup,
        "steps": args.steps, "repeats": args.repeats, "sequences": list(SEQUENCES),
        "checkpoints": list(CHECKPOINTS), "arms": list(ARMS),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu_selection": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "interleaving": "cyclic arm rotation by repetition, fixed before timing",
        "probe_steps": 8,
        "math_evidence_sha256": hashlib.sha256(args.math_evidence.read_bytes()).hexdigest(),
    }
    if min(args.steps, args.repeats) < 1 or args.warmup < 0:
        raise ValueError("Positive steps/repeats and nonnegative warmup required")
    if args.resume:
        protocol = json.loads((root / "protocol.json").read_text())
        for key, value in recipe.items():
            if protocol[key] != value:
                raise ValueError(f"Cannot resume a changed protocol: {key}")
    else:
        root.mkdir(parents=True, exist_ok=False)
        protocol = {**recipe, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (root / "logs").mkdir(exist_ok=True)
    started = time.perf_counter()

    def execute(sequence, checkpoint, arm, *, probe, repeat=0):
        phase = "probes" if probe else "timings"
        name = f"s{sequence}-{checkpoint}-{arm}" if probe else (
            f"s{sequence}-{checkpoint}-r{repeat + 1}-{arm}"
        )
        directory = root / phase / name
        if (directory / "result.json").exists():
            return json.loads((directory / "result.json").read_text())
        progress = {"phase": phase, "case": name,
                    "elapsed_seconds_this_invocation": time.perf_counter() - started,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        (root / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        console.print(progress)
        command = [sys.executable, str(worker), "--model", args.model, "--arm", arm,
                   "--sequence", str(sequence), "--checkpoint", checkpoint,
                   "--output", str(directory), "--warmup", "0" if probe else str(args.warmup),
                   "--steps", "8" if probe else str(args.steps)]
        if probe:
            command.append("--probe")
            if sequence == 1024 and checkpoint == "off":
                command.append("--profile")
        with (root / "logs" / f"{name}.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if not (directory / "result.json").exists():
            raise RuntimeError(f"Worker exited {completed.returncode} without a result: {name}")
        result = json.loads((directory / "result.json").read_text())
        if result["status"] == "error":
            raise RuntimeError(f"Fix the activation/runtime failure before timing: {name}")
        console.print({"case": name, "status": result["status"], "summary": result.get("summary")})
        return result

    if args.phase in ("all", "probes"):
        for sequence in SEQUENCES:
            for checkpoint in CHECKPOINTS:
                for arm in ARMS:
                    execute(sequence, checkpoint, arm, probe=True)
    if args.phase in ("all", "timings"):
        for sequence in SEQUENCES:
            for checkpoint in CHECKPOINTS:
                for repeat in range(args.repeats):
                    order = ARMS[repeat % 4:] + ARMS[:repeat % 4]
                    for arm in order:
                        probe_path = root / "probes" / f"s{sequence}-{checkpoint}-{arm}" / (
                            "result.json"
                        )
                        probe = json.loads(probe_path.read_text())
                        if probe["status"] == "oom":
                            directory = root / "timings" / (
                                f"s{sequence}-{checkpoint}-r{repeat + 1}-{arm}"
                            )
                            directory.mkdir(parents=True, exist_ok=True)
                            (directory / "result.json").write_text(json.dumps({
                                "status": "not_run_probe_oom", "arm": arm, "sequence": sequence,
                                "checkpoint": checkpoint,
                                "reason": "Same-shape 8-step activation probe exhausted headroom",
                            }, indent=2) + "\n")
                        else:
                            execute(sequence, checkpoint, arm, probe=False, repeat=repeat)
    result = summarize(root, protocol)
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "progress.json").write_text(json.dumps({
        "phase": "finished_" + args.phase, "complete_grid": result["complete_grid"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    console.print({"complete_grid": result["complete_grid"], "rows": len(result["rows"])})


if __name__ == "__main__":
    main()
