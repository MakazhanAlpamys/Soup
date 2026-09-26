#!/usr/bin/env python3
"""Evaluate the STEP 2b forward/backward/curve determinism record.

This is a historical reconstruction. The record's meaningful pattern is an
exact forward, a deterministic resident backward, and a non-deterministic
streamed backward/curve. The JSON input mode keeps replay separate from a
claim that this machine performed CUDA measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class MeasurementInvalidError(RuntimeError):
    """The supplied determinism record cannot support a verdict."""


def _finite(values: list[float], label: str) -> None:
    if not values or any(not math.isfinite(value) for value in values):
        raise MeasurementInvalidError(f"{label} must be non-empty and finite")


def determinism_verdict(record: dict[str, Any]) -> bool:
    """Require the recorded streamed-vs-resident determinism asymmetry."""
    required = (
        "streamed_forward_equal",
        "resident_forward_equal",
        "streamed_gradient_equal",
        "resident_gradient_equal",
        "streamed_curve_equal",
        "resident_curve_equal",
    )
    if any(key not in record for key in required):
        raise MeasurementInvalidError("determinism record is missing fields")
    for label in ("streamed_losses", "resident_losses"):
        _finite(list(record.get(label, [])), label)
    return (
        record["streamed_forward_equal"]
        and record["resident_forward_equal"]
        and not record["streamed_gradient_equal"]
        and record["resident_gradient_equal"]
        and not record["streamed_curve_equal"]
        and record["resident_curve_equal"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the STEP 2b determinism reconstruction record."
    )
    parser.add_argument(
        "--records",
        type=Path,
        help="JSON record produced by an external CUDA protocol run",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> int:
    record = {
        "streamed_forward_equal": True,
        "resident_forward_equal": True,
        "streamed_gradient_equal": False,
        "resident_gradient_equal": True,
        "streamed_curve_equal": False,
        "resident_curve_equal": True,
        "streamed_losses": [1.0, 0.9],
        "resident_losses": [1.0, 0.9],
    }
    assert determinism_verdict(record)
    record["streamed_gradient_equal"] = True
    assert not determinism_verdict(record)
    print("PASS: determinism verdict distinguishes streamed instability from control")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.records is None:
        print("ERROR: --records is required; no CUDA measurement was performed")
        return 2
    try:
        record = json.loads(args.records.read_text(encoding="utf-8"))
        if not determinism_verdict(record):
            print("ERROR: historical determinism pattern not reproduced")
            return 1
        print("RESULT: historical determinism pattern reproduced")
        return 0
    except MeasurementInvalidError as exc:
        print(f"ERROR: invalid measurement: {exc}")
        return 3
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"ERROR: invalid records: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
