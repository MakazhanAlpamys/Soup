#!/usr/bin/env python3
"""Reconstruct STEP 6's layer-count control sweep.

The record reports exact streamed/resident gradients across the 48-to-64 layer
boundary while layers remain small, plus a wrong 48-layer NF4 control above the
per-layer-size bracket. The live mode creates those synthetic checkpoints.
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
    if len(samples) < 4:
        return False
    nf4_depths = set()
    sweep_sizes: dict[str, set[float]] = {"nf4": set(), "bf16": set()}
    bf16_sweep = False
    positive_control = False
    for sample in samples:
        layers = sample.get("layers")
        quant = sample.get("quant")
        role = sample.get("role")
        exact_counts = sample.get("exact_counts")
        total = sample.get("total")
        if quant not in {"nf4", "bf16"}:
            raise MeasurementInvalidError("quant must be nf4 or bf16")
        if role not in {"sweep", "control"}:
            raise MeasurementInvalidError("role must be sweep or control")
        if not isinstance(layers, int) or layers <= 0:
            raise MeasurementInvalidError("layer counts must be positive integers")
        if total != 4 * layers:
            raise MeasurementInvalidError("total must equal four LoRA gradients per layer")
        if not isinstance(exact_counts, list) or len(exact_counts) != 3:
            raise MeasurementInvalidError("exact_counts must contain three backwards")
        if any(not isinstance(exact, int) or exact < 0 or exact > total for exact in exact_counts):
            raise MeasurementInvalidError("invalid exact gradient count")
        value = float(sample.get("layer_mib", 0.0))
        if not math.isfinite(value) or value <= 0:
            raise MeasurementInvalidError("layer_mib must be finite and positive")
        all_exact = all(exact == total for exact in exact_counts)

        if role == "sweep":
            if value > 163.8 or not all_exact:
                return False
            sweep_sizes[quant].add(round(value, 6))
            if quant == "nf4":
                nf4_depths.add(layers)
            else:
                bf16_sweep = True
        else:
            if quant != "nf4" or layers != 48 or value < 171.5:
                return False
            if not any(exact < total for exact in exact_counts[1:]):
                return False
            positive_control = True

    spans_boundary = any(depth <= 48 for depth in nf4_depths) and any(
        depth >= 64 for depth in nf4_depths
    )
    constant_sizes = all(len(sizes) <= 1 for sizes in sweep_sizes.values())
    return spans_boundary and bf16_sweep and positive_control and constant_sizes


def measure_layer_sweep(*, layer_counts: list[int], measure) -> list[dict[str, Any]]:
    """Measure both small-layer sweeps and the above-bracket NF4 control."""
    if len(set(layer_counts)) < 2:
        raise MeasurementInvalidError("layer sweep must contain distinct counts")
    rows = []
    for layers in layer_counts:
        for quant in ("nf4", "bf16"):
            row = dict(
                measure(
                    layers=layers,
                    hidden=64,
                    intermediate=128,
                    quant=quant,
                    role="sweep",
                )
            )
            row.update(layers=layers, quant=quant, role="sweep")
            rows.append(row)

    control = dict(
        measure(
            layers=48,
            hidden=5120,
            intermediate=20480,
            quant="nf4",
            role="control",
        )
    )
    control.update(layers=48, quant="nf4", role="control")
    rows.append(control)
    return rows


def _measure_checkpoint(weights: str, shards: str, quant: str) -> dict[str, Any]:
    """Run three streamed backwards against one resident reference."""
    import bitexact
    import torch
    from mechanism_cost import (
        build_streamed_arm,
        gradient_snapshot,
        make_input,
        prepare_shards,
        run_backward,
    )

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        assert_canonical_parameters_intersect,
        build_meta_skeleton,
        build_streamed_model,
    )

    def restore() -> None:
        return None

    if quant == "nf4":
        index, arch = prepare_shards(weights, shards)
        streamed, runtime, restore = build_streamed_arm(weights, shards, index, arch, "control", 2)
    else:
        runtime_quant = "none"
        probe = build_meta_skeleton(weights, dtype="bfloat16", quant=runtime_quant)
        arch = bitexact.model_arch_name(probe)
        del probe
        index = shard_checkpoint(
            weights,
            shards,
            dtype="bfloat16",
            arch=arch,
            quant=runtime_quant,
            quant_suffixes=(),
            double_quant=True,
            quant_device="cuda",
        )
        streamed, runtime = build_streamed_model(
            model_id=weights,
            shard_dir=shards,
            index=index,
            lora_config=bitexact.lora_config(),
            device="cuda",
            dtype="bfloat16",
            buffers=2,
            pin=True,
            seed=3,
            quant=runtime_quant,
            double_quant=True,
            tier="ram",
        )

    reference = None
    try:
        reference = bitexact.load_resident_reference(weights, quant)
        bitexact.make_non_vacuous_lora(streamed)
        bitexact.copy_lora(streamed, reference)
        assert_canonical_parameters_intersect(streamed, reference)
        input_ids = make_input(reference, seq=128, batch=1)
        reference_loss = run_backward(reference, input_ids)
        reference_grads = gradient_snapshot(reference)
        if (
            not math.isfinite(reference_loss)
            or not reference_grads
            or any(not torch.isfinite(value).all() for value in reference_grads.values())
        ):
            raise MeasurementInvalidError("resident measurement is empty or non-finite")

        exact_counts = []
        for _ in range(3):
            loss = run_backward(streamed, input_ids)
            current = gradient_snapshot(streamed)
            if not math.isfinite(loss) or loss != reference_loss:
                raise MeasurementInvalidError("streamed loss differs from resident")
            if set(current) != set(reference_grads):
                raise MeasurementInvalidError("gradient parameter sets differ")
            if any(not torch.isfinite(value).all() for value in current.values()):
                raise MeasurementInvalidError("streamed gradients are non-finite")
            exact_counts.append(
                sum(torch.equal(current[name], reference_grads[name]) for name in current)
            )

        store_bytes = getattr(getattr(runtime, "source", None), "nbytes", 0)
        layers = getattr(index, "n_layers", 0)
        if store_bytes <= 0 or layers <= 0:
            raise MeasurementInvalidError("streaming store size is unavailable")
        return {
            "layer_mib": store_bytes / layers / (1024 * 1024),
            "exact_counts": exact_counts,
            "total": len(reference_grads),
        }
    finally:
        if reference is not None:
            del reference
        try:
            runtime.close()
        finally:
            restore()
            del streamed
            torch.cuda.empty_cache()


def run_live_sweep(work_dir: Path) -> list[dict[str, Any]]:
    """Create and measure the recorded depth sweep and positive control."""
    import subprocess

    script = Path(__file__).with_name("synth_checkpoint.py")

    def measure(*, layers: int, hidden: int, intermediate: int, quant: str, role: str):
        checkpoint = work_dir / f"llama-{role}-{layers}-{hidden}-{intermediate}"
        shards = work_dir / f"shards-{role}-{layers}-{hidden}-{intermediate}-{quant}"
        if not checkpoint.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--out",
                    str(checkpoint),
                    "--shape",
                    "qwen2.5-14b",
                    "--arch",
                    "llama",
                    "--layers",
                    str(layers),
                    "--hidden",
                    str(hidden),
                    "--intermediate",
                    str(intermediate),
                    "--heads",
                    "4" if hidden == 64 else "40",
                    "--kv-heads",
                    "2" if hidden == 64 else "10",
                    "--vocab",
                    "128" if hidden == 64 else "1024",
                ],
                check=True,
            )
        shards.mkdir(parents=True, exist_ok=True)
        return _measure_checkpoint(str(checkpoint), str(shards), quant)

    return measure_layer_sweep(
        layer_counts=[32, 48, 56, 60, 64, 80],
        measure=measure,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the STEP 6 layer-count sweep.")
    parser.add_argument("--records", type=Path, help="JSON array of measured sweep rows")
    parser.add_argument("--measure", action="store_true", help="run the synthetic CUDA sweep")
    parser.add_argument("--work-dir", type=Path, default=Path("./layercount-work"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> int:
    assert layercount_verdict(
        [
            {
                "layers": 48,
                "quant": "nf4",
                "role": "sweep",
                "layer_mib": 0.01,
                "exact_counts": [192] * 3,
                "total": 192,
            },
            {
                "layers": 64,
                "quant": "nf4",
                "role": "sweep",
                "layer_mib": 0.01,
                "exact_counts": [256] * 3,
                "total": 256,
            },
            {
                "layers": 64,
                "quant": "bf16",
                "role": "sweep",
                "layer_mib": 0.05,
                "exact_counts": [256] * 3,
                "total": 256,
            },
            {
                "layers": 48,
                "quant": "nf4",
                "role": "control",
                "layer_mib": 187.0,
                "exact_counts": [192, 8, 8],
                "total": 192,
            },
        ]
    )
    print("PASS: layer-count verdict requires the sweep and positive control")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    if args.measure:
        import torch

        if not torch.cuda.is_available():
            print("SKIP: CUDA is required for the synthetic measurement")
            return 0
        try:
            samples = run_live_sweep(args.work_dir.expanduser().resolve())
            print(json.dumps(samples, indent=2, sort_keys=True))
            if layercount_verdict(samples):
                print("RESULT: layer-count control relationship reproduced")
                return 0
            print("ERROR: layer-count control relationship not reproduced")
            return 1
        except (MeasurementInvalidError, RuntimeError, ValueError) as exc:
            print(f"ERROR: invalid measurement: {exc}")
            return 3
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
