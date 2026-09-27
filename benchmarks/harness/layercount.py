#!/usr/bin/env python3
"""Reconstruct STEP 6's layer-count control sweep.

The record reports exact streamed/resident NF4 gradients for multiple synthetic
layer counts. This harness accepts a measurement record and requires a real
multi-point sweep; one successful point is never a reproduction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class MeasurementInvalidError(RuntimeError):
    """The layer-count sweep is invalid."""


def layercount_verdict(samples: list[dict[str, Any]]) -> bool:
    if len(samples) < 2:
        return False
    counts = set()
    sizes = set()
    for sample in samples:
        layers = sample.get("layers")
        exact = sample.get("exact")
        total = sample.get("total")
        if sample.get("quant") != "nf4":
            raise MeasurementInvalidError("layer-count sweep must measure NF4")
        if not isinstance(layers, int) or layers <= 0:
            raise MeasurementInvalidError("layer counts must be positive integers")
        if not isinstance(exact, int) or not isinstance(total, int):
            raise MeasurementInvalidError("exact and total must be integers")
        if total <= 0 or exact < 0 or exact > total:
            raise MeasurementInvalidError("invalid exact gradient count")
        value = float(sample.get("layer_mib", 0.0))
        if not math.isfinite(value) or value <= 0:
            raise MeasurementInvalidError("layer_mib must be finite and positive")
        sizes.add(round(value, 6))
        counts.add(layers)
    return (
        len(counts) >= 2
        and len(sizes) == 1
        and all(sample["exact"] == sample["total"] for sample in samples)
    )


def measure_layer_sweep(*, layer_counts: list[int], measure) -> list[dict[str, Any]]:
    """Run a controlled NF4 synthetic measurement at each requested depth."""
    if len(set(layer_counts)) < 2:
        raise MeasurementInvalidError("layer sweep must contain distinct counts")
    rows = []
    for layers in layer_counts:
        row = dict(measure(layers=layers, quant="nf4"))
        row["layers"] = layers
        row["quant"] = "nf4"
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the STEP 6 layer-count sweep.")
    parser.add_argument("--records", type=Path, help="JSON array of measured sweep rows")
    parser.add_argument("--measure", action="store_true", help="run the synthetic CUDA sweep")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> int:
    assert layercount_verdict([
        {"layers": 32, "quant": "nf4", "layer_mib": 150.0, "exact": 128, "total": 128},
        {"layers": 64, "quant": "nf4", "layer_mib": 150.0, "exact": 128, "total": 128},
    ])
    assert not layercount_verdict([
        {"layers": 32, "quant": "nf4", "layer_mib": 150.0, "exact": 128, "total": 128},
    ])
    print("PASS: layer-count verdict requires a multi-point exact sweep")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.measure:
        print("ERROR: synthetic CUDA sweep requires the historical torch/bitsandbytes stack")
        return 2
    if args.records is None:
        print("ERROR: --records is required; no CUDA measurement was performed")
        return 2
    try:
        samples = json.loads(args.records.read_text(encoding="utf-8"))
        if not isinstance(samples, list):
            raise MeasurementInvalidError("records must be a JSON array")
        if not layercount_verdict(samples):
            print("ERROR: layer-count control relationship not reproduced")
            return 1
        print("RESULT: layer-count control relationship reproduced")
        return 0
    except MeasurementInvalidError as exc:
        print(f"ERROR: invalid measurement: {exc}")
        return 3
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"ERROR: invalid records: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
