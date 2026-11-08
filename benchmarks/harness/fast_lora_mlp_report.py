#!/usr/bin/env python3
"""Normalize a raw #837 harness JSON file into the committed evidence layout."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

ARMS = ("fast", "plain", "liger")
BF16_COMMON_BASE_BYTES = 352_321_536
NF4_FAST_PACKED_AND_STATE_BYTES = 90_879_180


def _max_relative_l2(parity: dict[str, dict[str, float]]) -> float:
    return max((float(value["relative_l2"]) for value in parity.values()), default=0.0)


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert the harness's list-of-results schema without losing raw metrics."""
    rows = raw["results"]
    settings = raw["settings"]
    if len(rows) != 2 or {row["dtype"] for row in rows} != {"bf16", "nf4"}:
        raise ValueError("expected one BF16 and one NF4 result from the #837 fixture")
    if settings.get("tokens") != 2048 or settings.get("rank") != 64:
        raise ValueError("normalization constants apply only to tokens=2048, rank=64")

    results: dict[str, Any] = {}
    for row in rows:
        dtype = row["dtype"]
        key = "bf16_emulated_sm75" if dtype == "bf16" else "nf4"
        raw_saved = row["saved_tensor_bytes"]
        if dtype == "bf16":
            activation_saved = {
                arm: raw_saved[arm] - BF16_COMMON_BASE_BYTES for arm in ARMS
            }
        else:
            activation_saved = dict(raw_saved)
            activation_saved["fast"] -= NF4_FAST_PACKED_AND_STATE_BYTES

        results[key] = {
            "saved_tensor_bytes": raw_saved,
            "activation_only_saved_tensor_bytes_derived": activation_saved,
            "median_forward_backward_ms": {
                arm: row["timing_ms"][arm]["median"] for arm in ARMS
            },
            "p25_p75_ms": {
                arm: [row["timing_ms"][arm]["p25"], row["timing_ms"][arm]["p75"]]
                for arm in ARMS
            },
            "peak_memory_allocated_bytes": {
                arm: row["peak_memory_bytes"][arm]["absolute"] for arm in ARMS
            },
            "max_gradient_relative_l2_vs_plain": {
                arm: _max_relative_l2(row["parity_vs_plain"][arm])
                for arm in ("fast", "liger")
            },
            "output_equal_to_plain": {
                arm: row["parity_vs_plain"][arm]["output"]["relative_l2"] == 0
                for arm in ("fast", "liger")
            },
            "path_proof": row["path_proof"],
            "sm_clock_mhz": {
                "min": row["clock_mhz_during_measurement"]["min_mhz"],
                "median": row["clock_mhz_during_measurement"]["median_mhz"],
                "max": row["clock_mhz_during_measurement"]["max_mhz"],
            },
        }

    protocol = raw["protocol"]
    return {
        "timestamp_utc": raw["timestamp_utc"],
        "git_sha": raw["git_sha"],
        "runtime": {
            name: raw[name]
            for name in (
                "python",
                "platform",
                "torch",
                "torch_cuda",
                "cuda_device",
                "compute_capability",
                "gpu_memory_bytes",
                "nvidia_smi",
                "packages",
            )
        },
        "protocol": {
            "shape": rows[0]["shape"],
            "warmup_per_arm": protocol["warmup_per_arm"],
            "measured_repeats_per_arm": protocol["forward_backward_repeats_per_arm"],
            "timer": "CUDA events, forward plus backward",
            "arms_interleaved": protocol["interleaved_arms"],
            "dtype_order": [row["dtype"] for row in rows],
            "dtype_process_isolation": protocol["dtype_process_isolation"],
            "saved_tensor_bytes": (
                "unique saved tensor views, deduplicated by storage and view metadata"
            ),
            "activation_only_saved_tensor_bytes": (
                "derived arithmetic; excludes common BF16 base weights or Fast-LoRA NF4 "
                "packed weights and quant state"
            ),
            "peak_memory": "torch.cuda.max_memory_allocated absolute peak",
            "noise_rule": (
                "No threshold was specified before the timed run; medians and "
                "within-process IQRs are descriptive only, with no significance claim."
            ),
            "scope": (
                "single synthetic SwiGLU MLP block; no model download; "
                "not end-to-end training"
            ),
            "settings": settings,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    normalized = normalize(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
