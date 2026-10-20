#!/usr/bin/env python3
"""Measure the STEP 2b forward/backward/curve determinism protocol.

This is a historical reconstruction. The record's meaningful pattern is an
exact forward, a deterministic resident backward, and a non-deterministic
streamed backward/curve. The JSON input mode keeps replay separate from a
claim that this machine performed CUDA measurement; the normal ``--weights``
path performs the measurement in-process.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))


class MeasurementInvalidError(RuntimeError):
    """The supplied determinism record cannot support a verdict."""


def curve_diff(first: list[float], second: list[float]) -> tuple[bool, float, float]:
    if len(first) != len(second) or not first:
        raise MeasurementInvalidError("loss curves must be non-empty and equal length")
    max_abs = 0.0
    max_rel = 0.0
    for left, right in zip(first, second, strict=True):
        if not math.isfinite(left) or not math.isfinite(right):
            raise MeasurementInvalidError("loss curve contains non-finite values")
        difference = abs(left - right)
        max_abs = max(max_abs, difference)
        max_rel = max(max_rel, difference / max(abs(right), 1.0e-30))
    return first == second, max_abs, max_rel


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
    values = [record[key] for key in required]
    if any(not isinstance(value, bool) for value in values):
        raise MeasurementInvalidError("determinism flags must be booleans")
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
        description="Measure the STEP 2b determinism reconstruction."
    )
    parser.add_argument(
        "--records",
        type=Path,
        help="replay a previously measured JSON record instead of measuring",
    )
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--shards", type=Path, default=Path("./soup-shards"))
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _gradient_map(model: Any) -> dict[str, Any]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.grad is not None
    }


def _finite_tensors(values: dict[str, Any], label: str) -> None:
    import torch

    if not values or any(not torch.isfinite(value).all() for value in values.values()):
        raise MeasurementInvalidError(f"{label} contains no finite gradients")


def measure_models(
    streamed: Any,
    resident: Any,
    input_ids: Any,
    curve_batches: list[Any],
) -> dict[str, Any]:
    """Run forward, repeated backward, and curve measurements on both models."""
    import bitexact
    import torch

    from soup_cli.utils.layer_stream_runtime import assert_canonical_parameters_intersect

    assert_canonical_parameters_intersect(streamed, resident)

    def forward(model: Any) -> Any:
        return model(input_ids=input_ids, labels=input_ids).logits.detach().clone()

    streamed_forward = torch.equal(forward(streamed), forward(streamed))
    resident_forward = torch.equal(forward(resident), forward(resident))
    gradients: dict[str, list[dict[str, Any]]] = {"streamed": [], "resident": []}
    losses: dict[str, list[float]] = {"streamed": [], "resident": []}
    for name, model in (("streamed", streamed), ("resident", resident)):
        for _ in range(3):
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, labels=input_ids).loss
            if not torch.isfinite(loss).all():
                raise MeasurementInvalidError(f"{name} loss is non-finite")
            losses[name].append(float(loss.detach()))
            loss.backward()
            current = _gradient_map(model)
            _finite_tensors(current, f"{name} gradients")
            gradients[name].append(current)

    def equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
        if set(left) != set(right):
            raise MeasurementInvalidError("gradient parameter sets differ")
        return all(torch.equal(left[key], right[key]) for key in left)

    def curve_pair(model: Any) -> tuple[list[float], list[float]]:
        state = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        }
        if not state:
            raise MeasurementInvalidError("curve model has no LoRA parameters")

        def restore() -> None:
            with torch.no_grad():
                parameters = dict(model.named_parameters())
                for name, value in state.items():
                    parameters[name].copy_(value)

        try:
            first = bitexact.train_curve(model, curve_batches)
            restore()
            second = bitexact.train_curve(model, curve_batches)
            curve_diff(first, second)
            return first, second
        finally:
            restore()

    streamed_curve = curve_pair(streamed)
    resident_curve = curve_pair(resident)
    return {
        "streamed_forward_equal": streamed_forward,
        "resident_forward_equal": resident_forward,
        "streamed_gradient_equal": equal(
            gradients["streamed"][0], gradients["streamed"][2]
        ),
        "resident_gradient_equal": equal(
            gradients["resident"][0], gradients["resident"][2]
        ),
        "streamed_curve_equal": curve_diff(*streamed_curve)[0],
        "resident_curve_equal": curve_diff(*resident_curve)[0],
        "streamed_losses": losses["streamed"],
        "resident_losses": losses["resident"],
    }


def run_measurement(args: argparse.Namespace) -> int:
    if args.weights is None:
        print("ERROR: --weights is required for an in-process measurement")
        return 2
    import bitexact
    import torch
    from mechanism_cost import build_streamed_arm, prepare_shards

    from soup_cli.utils.spectrum_scan import resolve_model_weights

    if not torch.cuda.is_available():
        print("SKIP: CUDA is required for the measurement path")
        return 0
    weights = resolve_model_weights(str(args.weights))
    shards = str(args.shards.expanduser().resolve())
    index, arch = prepare_shards(weights, shards)
    streamed, runtime, restore = build_streamed_arm(
        weights, shards, index, arch, "control", args.buffers
    )
    resident = None
    try:
        resident = bitexact.load_resident_reference(weights, "nf4")
        bitexact.make_non_vacuous_lora(streamed)
        bitexact.copy_lora(streamed, resident)
        generator = torch.Generator(device="cuda").manual_seed(17)
        inputs = torch.randint(
            0, resident.config.vocab_size,
            (1, args.seq),
            generator=generator,
            device="cuda",
        )
        batches = [inputs.clone() for _ in range(5)]
        record = measure_models(streamed, resident, inputs, batches)
        print(json.dumps(record, indent=2, sort_keys=True))
        ok = determinism_verdict(record)
        print(
            "RESULT: historical determinism pattern reproduced"
            if ok
            else "ERROR: historical determinism pattern not reproduced"
        )
        return 0 if ok else 1
    finally:
        if resident is not None:
            del resident
        runtime.close()
        restore()


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
    try:
        if args.self_test:
            return run_self_test()
        if args.records is None:
            return run_measurement(args)
        record = json.loads(args.records.read_text(encoding="utf-8"))
        if not determinism_verdict(record):
            print("ERROR: historical determinism pattern not reproduced")
            return 1
        print("RESULT: historical determinism pattern reproduced")
        return 0
    except (MeasurementInvalidError, ValueError) as exc:
        print(f"ERROR: invalid measurement: {exc}")
        return 3
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"ERROR: invalid records: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
