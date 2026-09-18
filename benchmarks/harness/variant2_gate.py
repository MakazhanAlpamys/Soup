#!/usr/bin/env python3
"""Reproduce the STEP 14 streamed-NF4 repair gate from the H100 record.

The protocol builds two streamed arms in one process against one resident NF4
reference:

* ``control`` disables Soup's ``install_dequant_forward`` repair while the
  streamed model is built. It must reproduce at least one wrong LoRA gradient.
* ``repaired`` uses the shipped path. Every compared LoRA gradient must be
  bit-exact against the same resident reference.

The control is load-bearing. A repaired arm that passes while the control also
passes does not demonstrate that the defect-triggering path was exercised.
Likewise, an empty parameter or gradient intersection is a hard failure rather
than a vacuous ``0 == 0`` success.

Requirements
------------
- CUDA-capable GPU with enough memory for one streamed model plus one resident
  NF4 reference. The historical 32B and 72B runs used H100 80 GB cards.
- PyTorch, transformers, peft, safetensors, and bitsandbytes.
- A locally available checkpoint and writable layer-shard directory.

Typical invocation
------------------
    python benchmarks/harness/variant2_gate.py \
        --weights Qwen/Qwen2.5-32B-Instruct \
        --shards /path/to/qwen32b_nf4 \
        --json /path/to/variant2_gate_32b.json

The historical protocol uses batch 1, sequence length 128, two pinned stream
buffers, and five repetitions. Flags expose the recorded configuration
variations, but changing them changes the protocol. In particular, seq 32 uses
a different bitsandbytes kernel path and is expected to fail this script's
bit-exact STEP 14 verdict. This script publishes no new number by itself; it
reports the tree, runtime versions, parameters, and raw per-repeat results. A
machine without CUDA is an intentional skip and exits zero.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import bitexact as shared

DEFAULT_BATCH = 1
DEFAULT_BUFFERS = 2
DEFAULT_REPEATS = 5
DEFAULT_SEQ = 128
DEFAULT_SEED = 3
INPUT_SEED = 17
DTYPE = "bfloat16"


@dataclass(frozen=True)
class GradientSummary:
    compared: int
    exact: int
    wrong_layers: int
    worst_abs: float


def cuda_available() -> bool:
    return shared.cuda_available()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Soup's STEP 14 streamed-NF4 repair gate."
    )
    parser.add_argument("--weights", required=True, help="checkpoint path or model id")
    parser.add_argument("--shards", required=True, help="layer-shard cache directory")
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--buffers", type=int, default=DEFAULT_BUFFERS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--json", help="optional path for machine-readable evidence")
    return parser.parse_args()


def _canonical_parameters(model: Any) -> dict[str, Any]:
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    return dict(canonical_named_parameters(model))


def shared_lora_gradient_names(left: Any, right: Any) -> frozenset[str]:
    """Return shared LoRA names with gradients on both sides, never an empty set."""
    from soup_cli.utils.layer_stream_runtime import assert_canonical_parameters_intersect

    shared_names = assert_canonical_parameters_intersect(left, right)
    left_params = _canonical_parameters(left)
    right_params = _canonical_parameters(right)
    gradient_names = frozenset(
        name
        for name in shared_names
        if "lora_" in name
        and left_params[name].grad is not None
        and right_params[name].grad is not None
    )
    if not gradient_names:
        raise ValueError(
            "no shared LoRA gradient is available; a 0/0 gradient comparison is not a gate"
        )
    return gradient_names


def _layer_key(name: str) -> str:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return match.group(1) if match else name


def compare_lora_gradients(left: Any, right: Any) -> GradientSummary:
    """Compare the canonical shared LoRA gradient set exactly."""
    import torch

    names = shared_lora_gradient_names(left, right)
    left_params = _canonical_parameters(left)
    right_params = _canonical_parameters(right)
    exact = 0
    worst_abs = 0.0
    wrong_layers: set[str] = set()

    for name in sorted(names):
        left_grad = left_params[name].grad.detach()
        right_grad = right_params[name].grad.detach()
        if left_grad.shape != right_grad.shape:
            worst_abs = float("inf")
            wrong_layers.add(_layer_key(name))
            continue
        difference = float((left_grad - right_grad).abs().max())
        worst_abs = max(worst_abs, difference)
        if torch.equal(left_grad, right_grad):
            exact += 1
        else:
            wrong_layers.add(_layer_key(name))

    return GradientSummary(
        compared=len(names),
        exact=exact,
        wrong_layers=len(wrong_layers),
        worst_abs=worst_abs,
    )


def gate_passes(*, control: GradientSummary, repaired: GradientSummary) -> bool:
    """The repaired arm must pass and the defect-present control must fail."""
    return (
        control.compared > 0
        and control.exact < control.compared
        and repaired.compared > 0
        and repaired.exact == repaired.compared
        and repaired.wrong_layers == 0
    )


def build_streamed_arm(
    *,
    repair_enabled: bool,
    runtime_module: ModuleType | Any,
    build_model: Callable[[], Any],
) -> Any:
    """Build one arm, disabling the repair only for the control constructor."""
    if repair_enabled:
        return build_model()

    original = runtime_module.install_dequant_forward
    runtime_module.install_dequant_forward = lambda _module: 0
    try:
        return build_model()
    finally:
        runtime_module.install_dequant_forward = original


def _rewired_modules(model: Any) -> int:
    return sum(int(getattr(module, "n_dequant_forward", 0)) for module in model.modules())


def _source_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = result.stdout.strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else "unknown"


def _versions() -> dict[str, str]:
    import bitsandbytes
    import peft
    import torch
    import transformers

    return {
        "commit": _source_sha(),
        "torch": str(torch.__version__),
        "bitsandbytes": str(bitsandbytes.__version__),
        "transformers": str(transformers.__version__),
        "peft": str(peft.__version__),
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def _build_arm(
    *,
    repair_enabled: bool,
    weights_dir: str,
    shard_dir: Path,
    index: Any,
    buffers: int,
) -> tuple[Any, Any]:
    import soup_cli.utils.layer_stream_runtime as runtime_module

    def build_model():
        return runtime_module.build_streamed_model(
            model_id=weights_dir,
            shard_dir=str(shard_dir),
            index=index,
            lora_config=shared.lora_config(),
            device="cuda",
            dtype=DTYPE,
            buffers=buffers,
            pin=True,
            require_pin=True,
            seed=DEFAULT_SEED,
            quant="nf4",
            double_quant=True,
            tier="ram",
        )

    model, runtime = build_streamed_arm(
        repair_enabled=repair_enabled,
        runtime_module=runtime_module,
        build_model=build_model,
    )
    if not runtime.pinned:
        runtime.close()
        raise RuntimeError("STEP 14 requires a pinned RAM store; the runtime is pageable")
    return model, runtime


def _run_arm(
    *,
    label: str,
    model: Any,
    reference: Any,
    input_ids: Any,
    repeats: int,
) -> dict[str, Any]:
    import torch

    model.train()
    reference.train()
    summaries: list[GradientSummary] = []
    loss_differences: list[float] = []
    rates: list[float] = []
    peaks: list[float] = []

    for repetition in range(1, repeats + 1):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        model_loss = model(input_ids=input_ids, labels=input_ids).loss
        model_loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peaks.append(float(torch.cuda.max_memory_allocated()) / (1024 * 1024))
        rates.append(float(input_ids.numel()) / elapsed)

        reference_loss = reference(input_ids=input_ids, labels=input_ids).loss
        reference_loss.backward()
        loss_difference = float((model_loss - reference_loss).detach().abs())
        summary = compare_lora_gradients(model, reference)
        summaries.append(summary)
        loss_differences.append(loss_difference)
        print(
            f"{label} repetition {repetition}: gradients {summary.exact}/{summary.compared}, "
            f"wrong_layers={summary.wrong_layers}, worst_abs={summary.worst_abs:.6e}, "
            f"loss_abs={loss_difference:.6e}, tok/s={rates[-1]:.1f}"
        )

    model.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)
    return {
        "gradients": [asdict(summary) for summary in summaries],
        "loss_abs_differences": loss_differences,
        "losses_exact": all(value == 0.0 for value in loss_differences),
        "median_tokens_per_second": statistics.median(rates),
        "peak_memory_mib": max(peaks),
        "rewired_modules": _rewired_modules(model),
    }


def _arm_gate_summary(result: dict[str, Any], *, expect_exact: bool) -> bool:
    summaries = [GradientSummary(**item) for item in result["gradients"]]
    if not summaries or not result["losses_exact"]:
        return False
    if expect_exact:
        return all(
            summary.exact == summary.compared and summary.compared > 0 for summary in summaries
        )
    return all(summary.exact < summary.compared and summary.compared > 0 for summary in summaries)


def _paired_gradient_gate(control: dict[str, Any], repaired: dict[str, Any]) -> bool:
    control_summaries = [GradientSummary(**item) for item in control["gradients"]]
    repaired_summaries = [GradientSummary(**item) for item in repaired["gradients"]]
    return (
        bool(control_summaries)
        and len(control_summaries) == len(repaired_summaries)
        and all(
            gate_passes(control=control_summary, repaired=repaired_summary)
            for control_summary, repaired_summary in zip(
                control_summaries, repaired_summaries, strict=True
            )
        )
    )


def main() -> int:
    args = parse_args()
    if args.seq <= 0:
        print("ERROR: --seq must be positive")
        return 2
    if args.batch <= 0:
        print("ERROR: --batch must be positive")
        return 2
    if args.repeats <= 0:
        print("ERROR: --repeats must be positive")
        return 2
    if args.buffers < 2 or args.buffers > 8:
        print("ERROR: --buffers must be between 2 and 8")
        return 2
    if not cuda_available():
        print("SKIP: intentional skip — CUDA is required for variant2_gate.py")
        return 0

    import torch

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton, quantised_layer_suffixes
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    reference = None
    control = None
    control_runtime = None
    repaired = None
    repaired_runtime = None
    try:
        weights_dir = resolve_model_weights(args.weights)
        shard_dir = Path(args.shards)
        shard_dir.mkdir(parents=True, exist_ok=True)

        probe = build_meta_skeleton(weights_dir, dtype=DTYPE, quant="nf4")
        arch = shared.model_arch_name(probe)
        quant_suffixes = quantised_layer_suffixes(probe)
        del probe

        index = shard_checkpoint(
            weights_dir,
            str(shard_dir),
            dtype=DTYPE,
            arch=arch,
            quant="nf4",
            quant_suffixes=quant_suffixes,
            double_quant=True,
            quant_device="cuda",
        )
        reference = shared.load_resident_reference(weights_dir, "nf4")
        shared.make_non_vacuous_lora(reference)

        generator = torch.Generator(device="cuda").manual_seed(INPUT_SEED)
        input_ids = torch.randint(
            0,
            int(reference.config.vocab_size),
            (args.batch, args.seq),
            generator=generator,
            device="cuda",
        )

        print("RUN: STEP 14 streamed-NF4 control-versus-repair gate")
        print(f"source commit  {_source_sha()}")
        print(f"gpu            {torch.cuda.get_device_name(0)}")
        print(f"weights        {args.weights}")
        print(f"seq            {args.seq}")
        print(f"batch          {args.batch}")
        print(f"buffers        {args.buffers}")
        print(f"repeats        {args.repeats}")

        control, control_runtime = _build_arm(
            repair_enabled=False,
            weights_dir=weights_dir,
            shard_dir=shard_dir,
            index=index,
            buffers=args.buffers,
        )
        shared.copy_lora(reference, control)
        control_result = _run_arm(
            label="control",
            model=control,
            reference=reference,
            input_ids=input_ids,
            repeats=args.repeats,
        )
        control_runtime.close()
        control_runtime = None
        control = None
        gc.collect()
        torch.cuda.empty_cache()

        repaired, repaired_runtime = _build_arm(
            repair_enabled=True,
            weights_dir=weights_dir,
            shard_dir=shard_dir,
            index=index,
            buffers=args.buffers,
        )
        shared.copy_lora(reference, repaired)
        repaired_result = _run_arm(
            label="repaired",
            model=repaired,
            reference=reference,
            input_ids=input_ids,
            repeats=args.repeats,
        )

        control_ok = _arm_gate_summary(control_result, expect_exact=False)
        repaired_ok = _arm_gate_summary(repaired_result, expect_exact=True)
        gradients_ok = _paired_gradient_gate(control_result, repaired_result)
        losses_ok = control_result["losses_exact"] and repaired_result["losses_exact"]
        wiring_ok = (
            control_result["rewired_modules"] == 0 and repaired_result["rewired_modules"] > 0
        )
        passed = gradients_ok and losses_ok and wiring_ok
        report = {
            "protocol": "step14-variant2-gate",
            "versions": _versions(),
            "gpu": torch.cuda.get_device_name(0),
            "parameters": {
                "weights": args.weights,
                "seq": args.seq,
                "batch": args.batch,
                "buffers": args.buffers,
                "repeats": args.repeats,
                "pin": True,
                "quant": "nf4",
            },
            "control": control_result,
            "repaired": repaired_result,
            "passed": passed,
        }
        if args.json:
            _write_json(args.json, report)

        print(f"control_broke   {control_ok}")
        print(f"repair_exact    {repaired_ok}")
        print(f"losses_exact    {losses_ok}")
        print(f"wiring_distinct {wiring_ok}")
        print(f"RESULT: {'passed' if passed else 'failed'}")
        return 0 if passed else 1
    except Exception as exc:
        print(f"ERROR: variant2_gate.py failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if control_runtime is not None:
            control_runtime.close()
        if repaired_runtime is not None:
            repaired_runtime.close()
        control_runtime = repaired_runtime = None
        control = repaired = reference = None
        gc.collect()
        if cuda_available():
            import torch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
