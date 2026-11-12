"""Shared, torch-free definitions for the Soup-vs-Unsloth TCO (protocol.md).

Everything the protocol fixes in advance lives here, in ONE place, so that the driver, the meter
and summarize.py cannot drift apart: the case table, the matched hyperparameters, the window
rules, the comparability thresholds and the cost arithmetic.  protocol.md quotes these values;
if the two ever disagree, the protocol text is the pre-registration and this file is the bug.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from typing import Iterable, Optional

RESULT_SCHEMA = "tco-result/1"
FRAMEWORKS = ("soup", "hf", "unsloth")

# --------------------------------------------------------------------------------------------
# Matched hyperparameters (protocol.md section 4).  Every arm gets exactly these.
# --------------------------------------------------------------------------------------------
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
HYPERS = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,          # Unsloth's fast path needs 0; Soup's default is 0.05
    "lora_targets": LORA_TARGETS,
    "lr": 2.0e-4,
    "lr_scheduler": "constant",   # constant, no warmup: no schedule-shape confounder
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "optimizer": "adamw_torch",
    "quantization": "nf4 (bitsandbytes, double-quant on, on-the-fly from the bf16 checkpoint)",
    "max_length": 512,
    "grad_accum": 1,
    "gradient_checkpointing": True,   # policy ON for every arm, each framework's own implementation
    "packing": False,
    "padding_free": False,
    "assistant_only_loss": False,    # loss on every non-pad token, all arms
    "precision": "bf16 where the card has it, else fp16",
}

# --------------------------------------------------------------------------------------------
# Cases.  "window" runs are a fixed step window; "epoch" is the standard job of case 1.
# --------------------------------------------------------------------------------------------
CASES = {
    "smoke": {
        "label": "Pilot (Qwen2.5-1.5B): checks the pipeline and the comparability rule; not a result",
        "model": "Qwen/Qwen2.5-1.5B-Instruct", "batch": 4, "stream": False,
        "warmup_steps": 3, "window_steps": 40, "epoch_rows": None,
        "repeats": 1, "seeds": [3407], "frameworks": ["soup", "hf", "unsloth"],
        "max_wall_s": 900, "max_usd": 0.60,
        "download_gb": 3.1, "disk_gb": 40, "min_ram_gb": 32,
    },
    "1": {
        "label": "Case 1 - the model fits the card (Qwen2.5-7B-Instruct NF4, 24 GB card)",
        "model": "Qwen/Qwen2.5-7B-Instruct", "batch": 8, "stream": False,
        "warmup_steps": 5, "window_steps": 100, "epoch_rows": 10000,
        "repeats": 3, "seeds": [3407, 3408, 3409], "frameworks": ["soup", "hf", "unsloth"],
        "max_wall_s": 2400, "max_usd": 1.00,
        "epoch_max_wall_s": 6000, "epoch_max_usd": 2.00,
        "download_gb": 15.2, "disk_gb": 80, "min_ram_gb": 48,
    },
    "2a": {
        "label": "Case 2a - 32B class (Qwen2.5-32B-Instruct NF4): Soup streamed vs Unsloth",
        "model": "Qwen/Qwen2.5-32B-Instruct", "batch": 1, "stream": True,
        "warmup_steps": 3, "window_steps": 20, "epoch_rows": None,
        "repeats": 3, "seeds": [3407, 3408, 3409], "frameworks": ["soup", "unsloth"],
        "max_wall_s": 1500, "max_usd": 1.00,
        "download_gb": 65.5, "disk_gb": 200, "min_ram_gb": 64,
    },
    "2b": {
        "label": "Case 2b - 70B class (Llama-3.1-70B-Instruct NF4): Soup streamed vs Unsloth",
        "model": "NousResearch/Meta-Llama-3.1-70B-Instruct", "batch": 1, "stream": True,
        "warmup_steps": 3, "window_steps": 20, "epoch_rows": None,
        "repeats": 3, "seeds": [3407, 3408, 3409], "frameworks": ["soup", "unsloth"],
        "max_wall_s": 2400, "max_usd": 1.50,
        "download_gb": 141.1, "disk_gb": 400, "min_ram_gb": 96,
    },
}

DATASET = {
    "name": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "license": "mit",
    "rule": "first N rows, in stored order, whose chat-templated length under the case's own "
            "tokenizer is >= max_length tokens AND whose first user turn (with the generation prompt) "
            "is <= max_length - 128 tokens (Soup refuses rows whose answer is truncated away); "
            "every framework truncates on the right to "
            "max_length, so every training row is EXACTLY max_length tokens and tokens/step is "
            "batch x max_length with no padding in any arm",
    "rows_full": 10000,
}

# --------------------------------------------------------------------------------------------
# Window, comparability and validity thresholds (protocol.md sections 3 and 7).
# --------------------------------------------------------------------------------------------
FINAL_LOSS_STEPS = 16          # final-window loss = mean of the last 16 logged steps
FIRST_LOSS_STEPS = 1           # step-1 loss: LoRA B = 0, so it depends only on base+data
TOL_FIRST_LOSS_REL = 0.01      # step-1 loss must agree across frameworks within 1 %
TOL_FINAL_LOSS_REL_FLOOR = 0.03   # final-window loss: 3 % ... (estimate: the one matched pair seen
                                  # on Kaggle HF-vs-Unsloth differed by 1.2 %)
TOL_FINAL_LOSS_SPREAD_MULT = 2.0  # ... or 2x the worst within-framework repeat spread, if larger
TOL_TOKENS_REL = 0.01          # counted tokens must agree across frameworks within 1 %
TOL_TOKENS_VS_EXPECTED_REL = 0.005   # and with steps x batch x max_length within 0.5 %
SPREAD_VALID_MAX_REL = 0.15    # a repeat set with (max-min)/median > 15 % is reported as noisy
MIN_VALID_REPEATS = 2          # fewer valid repeats -> reported, flagged "not enough repeats"
STEADY_BAND = 0.10            # second-half / first-half tok/s must lie within 1 +- 0.10, else "not steady"


def median_or_none(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.median(vals) if vals else None


def usd_per_million_tokens(tok_per_s: Optional[float], usd_per_hour: Optional[float]) -> Optional[float]:
    """$ per 1,000,000 trained tokens at a steady rate: price per hour / (tokens per hour / 1e6)."""
    if not tok_per_s or tok_per_s <= 0 or usd_per_hour is None:
        return None
    return usd_per_hour / (tok_per_s * 3600.0 / 1e6)


def job_hours(tokens: float, tok_per_s: Optional[float]) -> Optional[float]:
    if not tok_per_s or tok_per_s <= 0:
        return None
    return tokens / tok_per_s / 3600.0


def wall_cap_seconds(max_wall_s: float, max_usd: Optional[float], usd_per_hour: Optional[float]) -> float:
    """One stop mechanism for both caps: the $ cap becomes a wall cap at the given hourly price."""
    cap = float(max_wall_s)
    if max_usd is not None and usd_per_hour and usd_per_hour > 0:
        cap = min(cap, max_usd / usd_per_hour * 3600.0)
    return cap


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------
# Throughput from the append-only step log.  The SAME function builds a finished result and a
# partial one, so a run cut short by a stop or a dead session is summarised by the same rule.
# --------------------------------------------------------------------------------------------
def read_jsonl(path: str) -> list:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a half-written last line after a hard stop: keep everything before it
    return rows


def throughput_from_steps(steps: list, warmup: int) -> dict:
    """steps: [{step, t_begin, t_end, tokens, ...}] in order.

    tok_s_wall = tokens of steps AFTER the warm-up / (t_end[last] - t_end[last warm-up step]).
                 It includes the data loader, logging and callbacks between steps: what a user waits.
    tok_s_step = the same tokens / sum of (t_end - t_begin): fwd + bwd + optimizer only.
    """
    out = {"warmup_steps": warmup, "steps_total": len(steps), "steps_measured": 0,
           "tokens_measured": 0, "tokens_total": sum(int(s.get("tokens", 0)) for s in steps),
           "tok_s_wall": None, "tok_s_step": None, "median_step_s": None, "wall_s_measured": None,
           "second_half_over_first_half": None}
    if len(steps) <= warmup:
        return out
    anchor = steps[warmup - 1]["t_end"] if warmup >= 1 else steps[0]["t_begin"]
    window = steps[warmup:]
    tokens = sum(int(s["tokens"]) for s in window)
    wall = window[-1]["t_end"] - anchor
    inner = [s["t_end"] - s["t_begin"] for s in window]
    half = len(window) // 2
    steady = None
    if half >= 3:
        first, second = window[:half], window[half:]
        t_first = first[-1]["t_end"] - anchor
        t_second = second[-1]["t_end"] - first[-1]["t_end"]
        r1 = sum(int(x["tokens"]) for x in first) / t_first if t_first > 0 else None
        r2 = sum(int(x["tokens"]) for x in second) / t_second if t_second > 0 else None
        steady = (r2 / r1) if (r1 and r2) else None
    out["second_half_over_first_half"] = steady
    out.update({
        "steps_measured": len(window), "tokens_measured": tokens, "wall_s_measured": wall,
        "tok_s_wall": tokens / wall if wall > 0 else None,
        "tok_s_step": tokens / sum(inner) if sum(inner) > 0 else None,
        "median_step_s": statistics.median(inner),
    })
    return out


def loss_summary(logs: list) -> dict:
    """logs: [{step, loss}] one per optimizer step (logging_steps=1)."""
    pts = [(int(r["step"]), float(r["loss"])) for r in logs if r.get("loss") is not None]
    out = {"n_logged": len(pts), "first_step": None, "final_window_mean": None,
           "final_window_n": 0, "curve": pts}
    if not pts:
        return out
    out["first_step"] = pts[0][1]
    tail = [v for _, v in pts[-FINAL_LOSS_STEPS:]]
    out["final_window_mean"] = sum(tail) / len(tail)
    out["final_window_n"] = len(tail)
    return out


def rel_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    base = (abs(a) + abs(b)) / 2.0
    return abs(a - b) / base if base else 0.0


def spread(values: list) -> Optional[float]:
    """(max - min) / median over repeats; None below two values."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    med = statistics.median(vals)
    return (max(vals) - min(vals)) / med if med else None


# --------------------------------------------------------------------------------------------
# Rebuilding result.json from the files a run leaves behind.  The driver calls this at the end
# of a run; summarize.py calls it again for any directory that has run_meta.json but no (or a
# stale) result.json, which is exactly the "session died, pull what is there" case.
# --------------------------------------------------------------------------------------------
def _load_json(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "out of memory", "CUDA error: out of memory")


def classify_failure(log_tail: str, rc: Optional[int]) -> str:
    tail = log_tail or ""
    if any(m in tail for m in OOM_MARKERS):
        return "oom"
    if "page-lock" in tail.lower() or "pinned" in tail.lower() and "refus" in tail.lower():
        return "pin_refused"
    if rc is None:
        return "no_exit_code"
    return f"exit_{rc}"


def rebuild_result(out_dir: str) -> dict:
    meta = _load_json(os.path.join(out_dir, "run_meta.json")) or {}
    steps = read_jsonl(os.path.join(out_dir, "steps.jsonl"))
    logs = read_jsonl(os.path.join(out_dir, "log.jsonl"))
    begin = _load_json(os.path.join(out_dir, "train_begin.json")) or {}
    end = _load_json(os.path.join(out_dir, "train_end.json")) or {}
    capped = _load_json(os.path.join(out_dir, "capped.json"))
    eff = _load_json(os.path.join(out_dir, "effective_config.json"))
    gpu = _load_json(os.path.join(out_dir, "gpu.json")) or {}
    meter_error = _load_json(os.path.join(out_dir, "meter_error.json"))
    warmup = int(meta.get("warmup_steps", 0))
    want = int(meta.get("window_steps", 0))
    tp = throughput_from_steps(steps, warmup)
    loss = loss_summary(logs)
    expected_per_step = int(meta.get("expected_tokens_per_step") or 0)
    tokens_vs_expected = None
    if expected_per_step and steps:
        tokens_vs_expected = (tp["tokens_total"] - expected_per_step * len(steps)) / (
            expected_per_step * len(steps))
    rc = meta.get("worker_rc")
    full = len(steps) >= warmup + want and want > 0
    if full and (rc in (0, None)):
        status = "complete"
    elif capped:
        status = "capped"
    elif steps:
        status = "partial"
    else:
        status = "failed"
    reason = None
    if status in ("failed", "partial") and meta.get("log_tail"):
        reason = classify_failure(meta["log_tail"], rc)
    if status == "failed" and reason is None:
        reason = classify_failure("", rc)

    peak_alloc = max((s.get("alloc_peak_mib") or 0 for s in steps), default=None) or end.get("alloc_peak_mib")
    peak_res = max((s.get("reserved_peak_mib") or 0 for s in steps), default=None) or end.get("reserved_peak_mib")
    t0 = meta.get("t_worker_start_unix")
    pre_train = (begin["unix"] - t0) if (begin.get("unix") and t0) else None
    post_train = (meta["t_worker_end_unix"] - end["unix"]) if (end.get("unix") and meta.get("t_worker_end_unix")) else None
    train_wall = (steps[-1]["t_end"] - steps[0]["t_begin"]) if steps else None
    price = meta.get("price") or {}
    usd_h = price.get("usd_per_hour")
    result = {
        "schema": RESULT_SCHEMA,
        "case": meta.get("case"), "framework": meta.get("framework"),
        "repeat": meta.get("repeat"), "seed": meta.get("seed"), "mode": meta.get("mode", "window"),
        "status": status, "reason": reason, "worker_rc": rc,
        "partial_note": ("rebuilt from append-only files; the run did not finish"
                         if status != "complete" else None),
        "model": meta.get("model"), "batch": meta.get("batch"), "max_length": meta.get("max_length"),
        "window": {"warmup_steps": warmup, "planned_measured_steps": want,
                   "measured_steps": tp["steps_measured"]},
        "tokens": {"total": tp["tokens_total"], "measured": tp["tokens_measured"],
                   "expected_per_step": expected_per_step,
                   "total_vs_expected_rel": tokens_vs_expected},
        "throughput": {k: tp[k] for k in ("tok_s_wall", "tok_s_step", "median_step_s",
                                           "wall_s_measured", "second_half_over_first_half")},
        "usd_per_million_tokens": usd_per_million_tokens(tp["tok_s_wall"], usd_h),
        "memory": {"peak_alloc_mib": peak_alloc, "peak_reserved_mib": peak_res,
                   "nvidia_smi": gpu.get("memory"), "gpu_total_mib": meta.get("gpu_total_mib")},
        "loss": {"first_step": loss["first_step"], "final_window_mean": loss["final_window_mean"],
                 "final_window_n": loss["final_window_n"], "n_logged": loss["n_logged"],
                 "curve": loss["curve"]},
        "order_fingerprint": [s.get("fp") for s in steps],
        "time_s": {"pre_train": pre_train, "train_wall": train_wall, "post_train": post_train,
                   "worker_total": (meta["t_worker_end_unix"] - t0)
                   if (t0 and meta.get("t_worker_end_unix")) else None,
                   "download": meta.get("download_s"), "setup": meta.get("setup_s"),
                   "data_prep": meta.get("data_prep_s")},
        "cold_cache": meta.get("cold_cache"),
        "capped": capped, "meter_error": meter_error,
        "effective_config": eff, "requested": meta.get("requested"),
        "versions": meta.get("versions"), "version_drift": meta.get("version_drift"),
        "hardware": meta.get("hardware"), "price": price, "caps": meta.get("caps"),
        "stream": meta.get("stream"), "gpu_samples": gpu.get("clock_power"),
        "host_memory": meta.get("host_memory"),
        "data": meta.get("data"),
        "fake": bool(meta.get("fake")),
    }
    return result
