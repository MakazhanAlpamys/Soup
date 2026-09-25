#!/usr/bin/env python3
"""Reconstruct STEP 2b's repeated-backward GRADDIFF measurement.

The recorded 32B NF4 run compared repeated streamed backwards with one
resident NF4 reference. The first streamed backward was exact; subsequent
backwards disagreed while the loss stayed equal. This file is a historical
reconstruction, not a claim that the current repaired runtime reproduces it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import bitexact

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))
from mechanism_cost import (  # noqa: E402
    build_streamed_arm,
    copy_lora,
    gradient_snapshot,
    load_resident_reference,
    make_input,
    make_non_vacuous_lora,
    prepare_shards,
    run_backward,
)

DEFAULT_REPEATS = 5
DEFAULT_SEQ = 128
DEFAULT_BATCH = 1
DEFAULT_BUFFERS = 2
DEVICE = "cuda"


class MeasurementInvalidError(RuntimeError):
    """Measurement data cannot support a verdict."""


def repeated_backward_verdict(
    exact_counts: list[int],
    total: int,
) -> bool:
    """Require exact first backward and corruption on later backwards."""
    if total <= 0 or len(exact_counts) < 2:
        return False
    if any(count < 0 or count > total for count in exact_counts):
        raise MeasurementInvalidError("gradient exact counts are out of range")
    return exact_counts[0] == total and any(
        count < total for count in exact_counts[1:]
    )


def _finite_gradients(gradients: dict[str, Any]) -> None:
    import torch

    if not gradients:
        raise MeasurementInvalidError("empty gradient comparison")
    for name, gradient in gradients.items():
        if not torch.isfinite(gradient).all():
            raise MeasurementInvalidError(f"non-finite gradient: {name}")


def compare_gradient_maps(
    actual: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[int, int, float]:
    """Compare non-empty canonical gradient maps against one reference."""
    import torch

    _finite_gradients(actual)
    _finite_gradients(reference)
    if set(actual) != set(reference):
        raise MeasurementInvalidError("gradient parameter sets differ")
    exact = 0
    worst = 0.0
    for name, left in actual.items():
        right = reference[name]
        if left.shape != right.shape:
            raise MeasurementInvalidError(f"gradient shape differs: {name}")
        diff = (left - right).abs()
        worst = max(worst, float(diff.max()))
        exact += int(torch.equal(left, right))
    return exact, len(actual), worst


def cuda_available() -> bool:
    return bitexact.cuda_available()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct repeated streamed backwards from STEP 2b."
    )
    parser.add_argument("--weights", help="local Qwen2.5-32B checkpoint")
    parser.add_argument("--shards", help="local Soup shard directory")
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--buffers", type=int, default=DEFAULT_BUFFERS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> int:
    cases = [3, 1, 1, 1, 1]
    assert repeated_backward_verdict(cases, 3)
    assert not repeated_backward_verdict([3, 3, 3], 3)
    print("PASS: repeated-backward verdict checks first exact and later corruption")
    return 0


def run_measurement(args: argparse.Namespace) -> int:
    import torch

    if not args.weights or not args.shards:
        print("ERROR: --weights and --shards are required")
        return 2
    if min(args.seq, args.batch, args.repeats) <= 0 or args.buffers < 2:
        print("ERROR: seq, batch, repeats must be positive and buffers >= 2")
        return 2
    if not cuda_available():
        print("SKIP: CUDA validation not performed; CUDA is required")
        return 0

    weights = str(Path(args.weights).expanduser().resolve())
    shards = str(Path(args.shards).expanduser().resolve())
    index, arch = prepare_shards(weights, shards)
    streamed, runtime, restore = build_streamed_arm(
        weights, shards, index, arch, "control", args.buffers
    )
    reference = None
    try:
        reference = load_resident_reference(weights)
        make_non_vacuous_lora(streamed)
        copy_lora(streamed, reference)
        input_ids = make_input(reference, seq=args.seq, batch=args.batch)
        reference.zero_grad(set_to_none=True)
        run_backward(reference, input_ids)
        reference_grads = gradient_snapshot(reference)
        exact_counts: list[int] = []
        total = 0
        for repetition in range(args.repeats):
            streamed.zero_grad(set_to_none=True)
            run_backward(streamed, input_ids)
            exact, total, worst = compare_gradient_maps(
                gradient_snapshot(streamed), reference_grads
            )
            exact_counts.append(exact)
            print(f"rep {repetition + 1}: {exact}/{total} exact worst_abs={worst:.6e}")
        if repeated_backward_verdict(exact_counts, total):
            print("RESULT: historical repeated-backward corruption reproduced")
            return 0
        print("ERROR: historical repeated-backward corruption not reproduced")
        return 1
    except MeasurementInvalidError as exc:
        print(f"ERROR: invalid measurement: {exc}")
        return 3
    finally:
        if reference is not None:
            del reference
        runtime.close()
        restore()
        del streamed
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    try:
        return run_self_test() if args.self_test else run_measurement(args)
    except MeasurementInvalidError as exc:
        print(f"ERROR: invalid measurement: {exc}")
        return 3
    except Exception as exc:
        print(f"ERROR: repeat_backward.py failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
