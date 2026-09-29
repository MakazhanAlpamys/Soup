#!/usr/bin/env python3
"""Check STEP 6's NF4 per-layer-byte threshold reconstruction.

The record brackets the transition between 163.8 and 171.5 MiB per NF4 layer:
below the bracket is exact, above it is wrong, while bf16 controls remain
exact. The live mode reconstructs the recorded synthetic checkpoints; the
constants in the verdict remain boundaries from the historical record.
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
        exact_counts = sample.get("exact_counts")
        total = sample.get("total")
        depth = sample.get("depth")
        if not isinstance(depth, int) or depth <= 0:
            raise MeasurementInvalidError("depth must be a positive integer")
        if quant not in {"nf4", "bf16"}:
            raise MeasurementInvalidError("quant must be nf4 or bf16")
        if not isinstance(mib, (int, float)) or not math.isfinite(mib) or mib <= 0:
            raise MeasurementInvalidError("layer_mib must be finite and positive")
        if not isinstance(total, int) or total <= 0:
            raise MeasurementInvalidError("total must be a positive integer")
        if not isinstance(exact_counts, list) or len(exact_counts) != 3:
            raise MeasurementInvalidError("exact_counts must contain three reads")
        if any(not isinstance(exact, int) or exact < 0 or exact > total for exact in exact_counts):
            raise MeasurementInvalidError("exact count is out of range")
        all_exact = all(exact == total for exact in exact_counts)
        later_wrong = any(exact < total for exact in exact_counts[1:])
        if quant == "nf4" and mib <= NF4_EXACT_MAX_MIB:
            below = True
            if not all_exact:
                return False
        if quant == "nf4" and mib >= NF4_WRONG_MIN_MIB:
            above = True
            if not later_wrong:
                return False
        if quant == "bf16":
            if not all_exact:
                return False
            if mib >= NF4_WRONG_MIN_MIB:
                bf16_control = True
    return below and above and bf16_control


def measure_synthetic_sweep(*, intermediate_sizes: list[int], measure) -> list[dict[str, Any]]:
    """Measure the fixed-depth NF4 sweep and one bf16 control."""
    if len(set(intermediate_sizes)) < 2:
        raise MeasurementInvalidError("the sweep needs at least two layer sizes")
    rows = []
    for intermediate_size in intermediate_sizes:
        row = dict(
            measure(
                depth=48,
                intermediate_size=intermediate_size,
                quant="nf4",
            )
        )
        row.update(depth=48, intermediate_size=intermediate_size, quant="nf4")
        rows.append(row)

    control_size = max(intermediate_sizes)
    control = dict(
        measure(
            depth=32,
            intermediate_size=control_size,
            quant="bf16",
        )
    )
    control.update(depth=32, intermediate_size=control_size, quant="bf16")
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
        probe = build_meta_skeleton(weights, dtype="bfloat16", quant="bf16")
        arch = bitexact.model_arch_name(probe)
        del probe
        index = shard_checkpoint(
            weights,
            shards,
            dtype="bfloat16",
            arch=arch,
            quant="bf16",
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
            quant="bf16",
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
    """Create the recorded synthetic shapes and measure both numeric paths."""
    import subprocess

    intermediate_sizes = [13824, 15360, 17408, 18432, 19456, 20480, 27648]
    script = Path(__file__).with_name("synth_checkpoint.py")

    def measure(*, depth: int, intermediate_size: int, quant: str):
        checkpoint = work_dir / f"llama-{depth}-{intermediate_size}"
        shards = work_dir / f"shards-{depth}-{intermediate_size}-{quant}"
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
                    str(depth),
                    "--intermediate",
                    str(intermediate_size),
                    "--vocab",
                    "1024",
                ],
                check=True,
            )
        shards.mkdir(parents=True, exist_ok=True)
        return _measure_checkpoint(str(checkpoint), str(shards), quant)

    return measure_synthetic_sweep(
        intermediate_sizes=intermediate_sizes,
        measure=measure,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the STEP 6 depth-vs-bytes sweep.")
    parser.add_argument("--records", type=Path, help="JSON array of measured sweep rows")
    parser.add_argument("--measure", action="store_true", help="run the synthetic CUDA sweep")
    parser.add_argument("--work-dir", type=Path, default=Path("./depth-vs-bytes-work"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> int:
    assert depth_vs_bytes_verdict(
        [
            {
                "depth": 48,
                "quant": "nf4",
                "layer_mib": 163.8,
                "exact_counts": [192] * 3,
                "total": 192,
            },
            {
                "depth": 48,
                "quant": "nf4",
                "layer_mib": 171.5,
                "exact_counts": [192, 8, 8],
                "total": 192,
            },
            {
                "depth": 32,
                "quant": "bf16",
                "layer_mib": 935.0,
                "exact_counts": [192] * 3,
                "total": 192,
            },
        ]
    )
    print("PASS: depth-vs-bytes verdict requires both NF4 sides and bf16 control")
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
            if depth_vs_bytes_verdict(samples):
                print("RESULT: historical depth-vs-bytes relationship reproduced")
                return 0
            print("ERROR: historical depth-vs-bytes relationship not reproduced")
            return 1
        except MeasurementInvalidError as exc:
            print(f"ERROR: invalid measurement: {exc}")
            return 3
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
