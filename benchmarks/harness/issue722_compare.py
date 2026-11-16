"""Validate paired #722 runs and summarize timing, VRAM and numerical agreement."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rich.console import Console


def validate_pair(dense: dict, chunked: dict) -> None:
    if dense["arm"] != "dense" or chunked["arm"] != "chunked":
        raise ValueError("Each pair must contain one dense and one chunked arm")
    for key in (
        "source_commit", "harness_sha256", "student_snapshot", "teacher_snapshot",
        "tokenized_data_sha256", "initial_adapter_sha256", "shape", "runtime", "dtype",
        "warmup_steps", "measured_steps", "timing_scope",
    ):
        if dense[key] != chunked[key]:
            raise ValueError(f"Incomparable runs: {key} differs")
    configs = []
    for record in (dense, chunked):
        config = json.loads(json.dumps(record["config"]))
        config.pop("output")
        config["training"].pop("distill_chunk_size")
        config["training"].pop("distill_checkpoint")
        configs.append(config)
    if configs[0] != configs[1]:
        raise ValueError("Incomparable runs: training config differs beyond the tested flags")
    if dense["config"]["training"]["distill_chunk_size"] is not None or (
        dense["config"]["training"]["distill_checkpoint"]
    ):
        raise ValueError("The dense arm must exercise the shipped defaults")
    if not chunked["config"]["training"]["distill_chunk_size"] or not (
        chunked["config"]["training"]["distill_checkpoint"]
    ):
        raise ValueError("The chunked arm must exercise both opt-in controls")


def array_difference(left_path: Path, right_path: Path) -> dict:
    import numpy as np

    with np.load(left_path) as left, np.load(right_path) as right:
        if set(left.files) != set(right.files) or not left.files:
            raise ValueError("Parameter/gradient names must match and be nonempty")
        max_abs = diff_sq = ref_sq = 0.0
        elements = 0
        for key in left.files:
            a, b = left[key].astype(np.float64), right[key].astype(np.float64)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Non-finite values or incompatible parameter: {key}")
            delta = a - b
            max_abs = max(max_abs, float(np.abs(delta).max()))
            diff_sq += float(np.sum(delta * delta))
            ref_sq += float(np.sum(a * a))
            elements += a.size
        return {"max_absolute_error": max_abs, "relative_l2_error": (
            (diff_sq / ref_sq) ** 0.5 if ref_sq else (0.0 if diff_sq == 0 else float("inf"))
        ), "elements": elements}


def compare_pairs(paths: list[Path]) -> dict:
    if len(paths) < 2 or len(paths) % 2:
        raise ValueError("Supply complete adjacent pairs in their actual run order")
    records = [json.loads((path / "result.json").read_text()) for path in paths]
    pairs = []
    for start in range(0, len(paths), 2):
        indices = sorted((start, start + 1), key=lambda i: records[i]["arm"] == "chunked")
        a, b = indices
        dense, chunked = records[a], records[b]
        validate_pair(dense, chunked)
        grad = array_difference(paths[a] / "first-gradients.npz", paths[b] / "first-gradients.npz")
        final = array_difference(paths[a] / "final-adapter.npz", paths[b] / "final-adapter.npz")
        losses = list(zip(dense["losses"], chunked["losses"], strict=True))
        loss_relative = max(abs(x - y) / max(abs(x), 1e-12) for x, y in losses)
        pairs.append({
            "dense": paths[a].name, "chunked": paths[b].name,
            "time_ratio_chunked_over_dense": (
                chunked["summary"]["seconds_per_step"] / dense["summary"]["seconds_per_step"]
            ),
            "peak_ratio_chunked_over_dense": (
                chunked["summary"]["peak_allocated_bytes"]
                / dense["summary"]["peak_allocated_bytes"]
            ),
            "max_relative_loss_difference": loss_relative,
            "first_step_relative_loss_difference": (
                abs(losses[0][0] - losses[0][1]) / max(abs(losses[0][0]), 1e-12)
            ),
            "first_step_gradients": grad, "final_adapter": final,
            "numerical_gate": (
                abs(losses[0][0] - losses[0][1]) / max(abs(losses[0][0]), 1e-12) <= 1e-5
                and loss_relative <= 1e-2 and grad["relative_l2_error"] <= 1e-2
            ),
        })
    summary = {}
    for arm in ("dense", "chunked"):
        rows = [r for r in records if r["arm"] == arm]
        summary[arm] = {
            key: {"median": statistics.median(values), "min": min(values), "max": max(values)}
            for key in ("seconds_per_step", "input_tokens_per_second", "peak_allocated_bytes")
            if (values := [row["summary"][key] for row in rows])
        }
    return {"run_order": [r["arm"] for r in records], "pairs": pairs, "summary": summary,
            "all_numerical_gates_passed": all(p["numerical_gate"] for p in pairs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_pairs(args.runs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    Console().print(result)
    if not result["all_numerical_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
