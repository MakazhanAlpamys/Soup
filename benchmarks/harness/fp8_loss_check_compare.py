"""Verdict of ``benchmarks/probe-rtx5070-moonlight-fp8-source.md`` from the two arms' folders.

CPU only, read-only (it writes only ``--json``). The limits below are the record's section 2,
committed before either arm ran; this script applies them and nothing else.

    python benchmarks/harness/fp8_loss_check_compare.py \
        --arm-a D:/soup-bench/fp8-loss-check/bf16 --arm-b D:/soup-bench/fp8-loss-check/fp8 \
        --json <verdict.json>

Per arm it reads ``output/rewind.jsonl`` (each step's row and its full-precision loss),
``experiments.db`` (the run's tracker, ``SOUP_DB_PATH``: per-step timestamps, the evaluation
losses, the GPU peak string), ``output/adapter_model.safetensors``, ``train.log`` (the exit
line the run script appends) and ``watchdog.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import statistics
import sys
from datetime import datetime

# Section 2 of the record. Losses in nats.
T1_FIRST_STEP = 0.40
T2_MAX_STEP = 1.6
T3_FINAL_EVAL = 0.30
T4_EPOCH3_MEAN = 0.15
T5_GPU_PEAK_RATIO = 1.10
T6_STEP_TIME_RATIO = 1.10
STEPS = 27
ROWS_PER_EPOCH = 9
EVALS = 3
SECTION_41_STEP1_LOSS = 4.712586402893066  # 4.1's rewind.jsonl, step 1 (row 1)


def _rewind(folder: str) -> dict:
    header, batches = None, {}
    with open(os.path.join(folder, "output", "rewind.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("kind") == "header":
                header = entry
            elif entry.get("kind") == "batch":
                loss = entry["row_loss"][0]
                batches[int(entry["step"])] = {
                    "row": entry["rows"][0],
                    "loss": None if loss is None else float(loss),
                    "tokens": entry["row_tokens"][0],
                }
    return {"header": header, "batches": batches}


def _tracker(folder: str) -> dict:
    conn = sqlite3.connect(f"file:{os.path.join(folder, 'experiments.db')}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    runs = [dict(row) for row in conn.execute("SELECT * FROM runs ORDER BY rowid")]
    if len(runs) != 1:
        conn.close()
        # The run script refuses an arm folder that already holds a run; two runs in one
        # tracker means a folder was reused, and their metrics would mix.
        raise ValueError(f"experiments.db holds {len(runs)} runs; expected exactly 1")
    metrics = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM metrics WHERE run_id = ? ORDER BY id", (runs[0]["run_id"],)
        )
    ]
    conn.close()

    train_times: dict = {}
    evals: dict = {}
    peaks = []
    for row in metrics:
        if row["gpu_mem"]:
            match = re.match(r"\s*([0-9.]+)\s*/", row["gpu_mem"])
            if match:
                peaks.append(float(match.group(1)))
        if row["val_loss"] is not None:
            evals[int(row["step"])] = float(row["val_loss"])
        elif row["grad_norm"] is not None and int(row["step"]) not in train_times:
            train_times[int(row["step"])] = datetime.fromisoformat(row["timestamp"])
    steps = sorted(train_times)
    intervals = [
        (train_times[b] - train_times[a]).total_seconds()
        for a, b in zip(steps, steps[1:])
        if b == a + 1
    ]
    return {
        "runs": runs,
        "evals": evals,
        "gpu_peak_gib": max(peaks) if peaks else None,
        "logged_steps": steps,
        "step_seconds_median": statistics.median(intervals) if intervals else None,
        "step_intervals": len(intervals),
    }


def _adapter(folder: str) -> dict:
    path = os.path.join(folder, "output", "adapter_model.safetensors")
    if not os.path.isfile(path):
        return {"written": False}
    from safetensors.torch import load_file

    tensors = load_file(path)
    finite = all(bool(t.float().isfinite().all()) for t in tensors.values())
    b_nonzero = any(
        bool(t.float().abs().max() > 0) for name, t in tensors.items() if "lora_B" in name
    )
    return {
        "written": True,
        "bytes": os.path.getsize(path),
        "tensors": len(tensors),
        "finite": finite,
        "lora_b_nonzero": b_nonzero,
    }


def _log(folder: str) -> dict:
    path = os.path.join(folder, "train.log")
    if not os.path.isfile(path):
        return {"present": False}
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    exits = re.findall(r"^exit=(-?\d+)\s*$", text, flags=re.MULTILINE)
    plan = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"(?i)\btier\b|read.ahead|layer buffers|pinned|VRAM plan", line)
    ]
    return {
        "present": True,
        "exit_code": int(exits[-1]) if exits else None,
        "stream_plan_lines": plan[:40],
        "resharded": "Re-sharding layer cache" in text,
        "fp8_notice": "FP8 block-quantised base" in text,
    }


def _watchdog(folder: str) -> dict:
    path = os.path.join(folder, "watchdog.json")
    if not os.path.isfile(path):
        return {"present": False}
    with open(path, encoding="utf-8") as handle:
        return {"present": True, **json.load(handle)}


def _arm(folder: str) -> dict:
    arm = {"folder": folder}
    for name, reader in (
        ("rewind", _rewind),
        ("tracker", _tracker),
        ("adapter", _adapter),
        ("log", _log),
        ("watchdog", _watchdog),
    ):
        try:
            arm[name] = reader(folder)
        except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
            arm[name] = {"error": f"{type(exc).__name__}: {exc}"}
    batches = arm["rewind"].get("batches", {})
    evals = arm["tracker"].get("evals", {})
    losses = [batches.get(step, {}).get("loss") for step in range(1, STEPS + 1)]
    arm["completes"] = {
        "exit_0": arm["log"].get("exit_code") == 0,
        "27_finite_step_losses": all(
            value is not None and math.isfinite(value) for value in losses
        ),
        "3_finite_evals": len(evals) == EVALS
        and all(math.isfinite(value) for value in evals.values()),
        "adapter_written_finite_trained": bool(
            arm["adapter"].get("written")
            and arm["adapter"].get("finite")
            and arm["adapter"].get("lora_b_nonzero")
        ),
    }
    arm["COMPLETES"] = all(arm["completes"].values())
    return arm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arm-a", required=True)
    parser.add_argument("--arm-b", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args(argv)

    a, b = _arm(args.arm_a), _arm(args.arm_b)
    report: dict = {"A": a, "B": b}

    inconclusive = []
    for name, arm in (("A", a), ("B", b)):
        dog = arm["watchdog"]
        if not dog.get("present"):
            inconclusive.append(f"{name}: no watchdog summary")
            continue
        if dog.get("fired"):
            inconclusive.append(f"{name}: RAM watchdog fired")
        if dog.get("gaps"):
            inconclusive.append(f"{name}: sampling gap (sleep?) {dog['gaps']}")
        if dog.get("off_ac_samples"):
            inconclusive.append(f"{name}: {dog['off_ac_samples']} samples off AC")
        if dog.get("foreign_compute_pids"):
            inconclusive.append(f"{name}: other GPU processes {dog['foreign_compute_pids']}")
    head_a, head_b = a["rewind"].get("header") or {}, b["rewind"].get("header") or {}
    rows_a = [a["rewind"].get("batches", {}).get(s, {}).get("row") for s in range(1, STEPS + 1)]
    rows_b = [b["rewind"].get("batches", {}).get(s, {}).get("row") for s in range(1, STEPS + 1)]
    if head_a.get("dataset_fingerprint") != head_b.get("dataset_fingerprint"):
        inconclusive.append("dataset fingerprints differ")
    if rows_a != rows_b:
        inconclusive.append("row order differs between arms")
    report["inconclusive"] = inconclusive

    checks: dict = {}
    if a["COMPLETES"] and b["COMPLETES"]:
        la = [a["rewind"]["batches"][s]["loss"] for s in range(1, STEPS + 1)]
        lb = [b["rewind"]["batches"][s]["loss"] for s in range(1, STEPS + 1)]
        deltas = [y - x for x, y in zip(la, lb)]
        epoch3 = slice(2 * ROWS_PER_EPOCH, 3 * ROWS_PER_EPOCH)
        eval_a, eval_b = a["tracker"]["evals"], b["tracker"]["evals"]
        final = max(eval_a)
        peak_a, peak_b = a["tracker"]["gpu_peak_gib"], b["tracker"]["gpu_peak_gib"]
        sec_a, sec_b = a["tracker"]["step_seconds_median"], b["tracker"]["step_seconds_median"]
        checks = {
            "T1_first_step": {"value": abs(deltas[0]), "limit": T1_FIRST_STEP},
            "T2_max_step": {
                "value": max(abs(d) for d in deltas),
                "at_step": 1 + max(range(STEPS), key=lambda i: abs(deltas[i])),
                "limit": T2_MAX_STEP,
            },
            "T3_final_eval": {
                "value": abs(eval_b[final] - eval_a[final]),
                "a": eval_a[final],
                "b": eval_b[final],
                "limit": T3_FINAL_EVAL,
            },
            "T4_epoch3_mean": {
                "value": abs(statistics.mean(lb[epoch3]) - statistics.mean(la[epoch3])),
                "limit": T4_EPOCH3_MEAN,
            },
            "T5_gpu_peak": {
                "value": None if not (peak_a and peak_b) else peak_b / peak_a,
                "a_gib": peak_a,
                "b_gib": peak_b,
                "limit": T5_GPU_PEAK_RATIO,
            },
            "T6_step_time": {
                "value": None if not (sec_a and sec_b) else sec_b / sec_a,
                "a_s": sec_a,
                "b_s": sec_b,
                "limit": T6_STEP_TIME_RATIO,
            },
        }
        for check in checks.values():
            check["pass"] = check["value"] is not None and check["value"] <= check["limit"]
        report["per_step"] = [
            {"step": s + 1, "row": rows_a[s], "a": la[s], "b": lb[s], "delta": deltas[s]}
            for s in range(STEPS)
        ]
        report["epoch_means"] = [
            {
                "epoch": e + 1,
                "a": statistics.mean(la[e * ROWS_PER_EPOCH:(e + 1) * ROWS_PER_EPOCH]),
                "b": statistics.mean(lb[e * ROWS_PER_EPOCH:(e + 1) * ROWS_PER_EPOCH]),
            }
            for e in range(3)
        ]
        report["evals"] = {"a": eval_a, "b": eval_b}
        report["a_step1_minus_section_41"] = la[0] - SECTION_41_STEP1_LOSS
    report["checks"] = checks

    if inconclusive:
        verdict = "INCONCLUSIVE"
    elif not a["COMPLETES"]:
        verdict = "INCONCLUSIVE (arm A did not complete)"
    elif not b["COMPLETES"]:
        # Section 2: FAIL when B's error comes from the code under test (sharding, the FP8
        # dequant, streaming, training); INCONCLUSIVE when it is the box (driver reset, disk
        # full, another process's memory). The traceback in train.log decides; not this script.
        verdict = "B DID NOT COMPLETE: FAIL or INCONCLUSIVE by the cause in train.log"
    elif all(check["pass"] for check in checks.values()):
        verdict = "PASS"
    else:
        verdict = "FAIL (" + ", ".join(k for k, c in checks.items() if not c["pass"]) + ")"
    report["verdict"] = verdict

    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps({"verdict": verdict, "inconclusive": inconclusive, "checks": checks,
                      "completes": {"A": a["completes"], "B": b["completes"]}},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
