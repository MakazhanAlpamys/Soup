#!/usr/bin/env python3
"""Check STEP 6's NF4 per-layer-byte threshold reconstruction.

The record brackets the transition between 163.8 and 171.5 MiB per NF4 layer:
below the bracket is exact, above it is wrong, while bf16 controls remain
exact. Values in this file are a verdict boundary from the record, not newly
measured results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

NF4_EXACT_MAX_MIB = 163.8
NF4_WRONG_MIN_MIB = 171.5


class MeasurementInvalidError(RuntimeError):
    """The depth/bytes data is not a valid measurement."""


def depth_vs_bytes_verdict(samples: list[dict[str, Any]]) -> bool:
    if len(samples) < 3:
        return False
    below = False
    above = False
    bf16_control = False
    for sample in samples:
        quant = sample.get("quant")
        mib = sample.get("layer_mib")
        exact = sample.get("exact")
        total = sample.get("total")
        if quant not in {"nf4", "bf16"}:
            raise MeasurementInvalidError("quant must be nf4 or bf16")
        if not isinstance(mib, (int, float)) or not math.isfinite(mib) or mib <= 0:
            raise MeasurementInvalidError("layer_mib must be finite and positive")
        if not isinstance(exact, int) or not isinstance(total, int) or total <= 0:
            raise MeasurementInvalidError("exact/total must be non-empty integers")
        if exact < 0 or exact > total:
            raise MeasurementInvalidError("exact count is out of range")
        if quant == "nf4" and mib <= NF4_EXACT_MAX_MIB and exact == total:
            below = True
        if quant == "nf4" and mib >= NF4_WRONG_MIN_MIB and exact < total:
            above = True
        if quant == "bf16" and exact == total:
            bf16_control = True
    return below and above and bf16_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the STEP 6 depth-vs-bytes sweep.")
    parser.add_argument("--records", type=Path, help="JSON array of measured sweep rows")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> int:
    assert depth_vs_bytes_verdict([
        {"quant": "nf4", "layer_mib": 163.8, "exact": 192, "total": 192},
        {"quant": "nf4", "layer_mib": 171.5, "exact": 8, "total": 192},
        {"quant": "bf16", "layer_mib": 480.0, "exact": 128, "total": 128},
    ])
    print("PASS: depth-vs-bytes verdict requires both NF4 sides and bf16 control")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.records is None:
        print("ERROR: --records is required; no CUDA measurement was performed")
        return 2
    try:
        samples = json.loads(args.records.read_text(encoding="utf-8"))
        if not isinstance(samples, list):
            raise MeasurementInvalidError("records must be a JSON array")
        if not depth_vs_bytes_verdict(samples):
            print("ERROR: historical depth-vs-bytes relationship not reproduced")
            return 1
        print("RESULT: historical depth-vs-bytes relationship reproduced")
        return 0
    except MeasurementInvalidError as exc:
        print(f"ERROR: invalid measurement: {exc}")
        return 3
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"ERROR: invalid records: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
