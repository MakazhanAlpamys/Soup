#!/usr/bin/env python3
"""Reconstruct the historical STEP 2b GRADDIFF investigation.

This harness is a reconstruction of the published GRADDIFF slice from issue
#379. It does not replace the historical protocol with a newer repeated-
backward variant. The documented H100 observation was:

- Qwen2.5-32B-Instruct NF4, bf16 compute
- first backward: streamed vs resident LoRA gradients were 256/256 exact
- streamed self-curve: not identical on the second run
- resident self-curve identical: a deliberate reconstruction/control condition
  used by this harness, not a claim about the recorded measurement

The historical record is the source of truth for the expected pattern. The
reconstruction choices below intentionally mirror the original experiment's
structure and keep the default behavior aligned with that historical target:

- sequence length: 128 tokens per sample
- training curve: 5 batches, each used twice for the streamed and resident
  self-checks
- optimizer: AdamW with learning rate 1.0e-3
- one backward comparison followed by two self-curves for each model replica

The later STEP 2b measurements showed that the streamed backward becomes non-
reproducible from the second backward onward while the resident backward remains
deterministic. This harness therefore succeeds only when the historical
GRADDIFF pattern is observed on the default path with the shipped repair
disabled. On repaired main, where that historical divergence is absent, it
reports that the historical result was not reproduced and exits non-zero.

Requirements
------------
- CUDA-capable GPU
- PyTorch, transformers, peft, safetensors, bitsandbytes
- A locally available Qwen2.5-32B-Instruct checkpoint
- Enough GPU memory for a resident NF4 reference plus the streamed model
- No model download is performed

A machine without CUDA exits 0 with an explicit skip.

Exit codes
----------
0 historical GRADDIFF pattern reproduced (or CUDA skip)
1 historical pattern not reproduced
2 invalid CLI/input
3 invalid or non-finite measurement
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path
from typing import Any

import bitexact
from mechanism_cost import install_historical_control

DTYPE = "bfloat16"
DEFAULT_SEQ = 128
DEFAULT_BATCH = 1
DEFAULT_BUFFERS = 2
DEFAULT_CURVE_STEPS = 5
DEFAULT_SEED = 3
INPUT_SEED = 17
DEVICE = "cuda"


class MeasurementInvalidError(RuntimeError):
    """The run cannot support a numerical verdict."""


def cuda_available() -> bool:
    return bitexact.cuda_available()


def runtime_versions() -> tuple[str, str, str]:
    """Return benchmark dependency versions after the CUDA boundary."""

    import bitsandbytes as bnb
    import peft
    import transformers

    return bnb.__version__, peft.__version__, transformers.__version__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the STEP 2b one-backward gradient comparison and "
            "two 5-step self-curves."
        )
    )
    parser.add_argument(
        "--weights",
        help="local Qwen2.5-32B-Instruct checkpoint directory",
    )
    parser.add_argument(
        "--shards",
        help="directory for Soup layer shards",
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=DEFAULT_SEQ,
        help=f"sequence length (default: {DEFAULT_SEQ})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"batch size (default: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--buffers",
        type=int,
        default=DEFAULT_BUFFERS,
        help=f"stream buffer count (default: {DEFAULT_BUFFERS})",
    )
    parser.add_argument(
        "--curve-steps",
        type=int,
        default=DEFAULT_CURVE_STEPS,
        help=f"training steps in each self-curve (default: {DEFAULT_CURVE_STEPS})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the CPU-only acceptance self-test and exit",
    )
    return parser.parse_args()


def canonical_parameters(model: Any) -> dict[str, Any]:
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    return dict(canonical_named_parameters(model))


def capture_lora_state(model: Any) -> dict[str, Any]:
    """Snapshot LoRA tensors to CPU for deterministic curve restarts."""

    state: dict[str, Any] = {}

    for name, parameter in canonical_parameters(model).items():
        if "lora_" in name:
            state[name] = parameter.detach().cpu().clone()

    if not state:
        raise RuntimeError("model produced no LoRA tensors")

    return state


def restore_lora_state(model: Any, state: dict[str, Any]) -> None:
    """Restore a CPU LoRA snapshot into a model."""

    with __import__("torch").no_grad():
        target = canonical_parameters(model)

        for name, value in state.items():
            parameter = target.get(name)

            if parameter is None:
                raise RuntimeError(
                    f"could not match LoRA parameter {name!r}"
                )

            parameter.copy_(
                value.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


def backward_once(model: Any, input_ids: Any) -> tuple[Any, dict[str, Any]]:
    """Run one loss/backward pass and return loss plus canonical LoRA grads."""

    import torch

    torch.manual_seed(INPUT_SEED)

    model.train()
    model.zero_grad(set_to_none=True)

    output = model(
        input_ids=input_ids,
        labels=input_ids,
    )
    loss = output.loss
    loss.backward()

    gradients = bitexact.collect_lora_grads(model)

    if not gradients:
        raise RuntimeError("model produced no LoRA gradients")

    return loss.detach().clone(), gradients


def compare_gradients(
    streamed: Any,
    reference: Any,
    streamed_grads: dict[str, Any],
    reference_grads: dict[str, Any],
) -> tuple[int, int, float, float]:
    """Compare canonical LoRA gradients and reject an empty intersection."""

    import torch

    from soup_cli.utils.layer_stream_runtime import (
        assert_canonical_parameters_intersect,
    )

    assert_canonical_parameters_intersect(
        streamed,
        reference,
    )

    if set(streamed_grads) != set(reference_grads):
        raise RuntimeError(
            "streamed/reference LoRA gradient sets differ"
        )

    exact = 0
    max_abs = 0.0
    max_rel = 0.0

    for name, streamed_gradient in streamed_grads.items():
        reference_gradient = reference_grads[name]

        if streamed_gradient.shape != reference_gradient.shape:
            raise RuntimeError(
                f"gradient shape differs for {name!r}: "
                f"{tuple(streamed_gradient.shape)} != "
                f"{tuple(reference_gradient.shape)}"
            )

        if not torch.isfinite(streamed_gradient).all():
            raise MeasurementInvalidError(
                f"streamed gradient contains non-finite values: {name}"
            )

        if not torch.isfinite(reference_gradient).all():
            raise MeasurementInvalidError(
                f"reference gradient contains non-finite values: {name}"
            )

        diff = (
            streamed_gradient - reference_gradient
        ).abs()

        abs_diff = float(diff.max())
        max_abs = max(max_abs, abs_diff)

        denominator = reference_gradient.abs().clamp_min(1.0e-30)
        rel_diff = float((diff / denominator).max())
        max_rel = max(max_rel, rel_diff)

        if torch.equal(streamed_gradient, reference_gradient):
            exact += 1

    return exact, len(streamed_grads), max_abs, max_rel


def train_curve_once(
    model: Any,
    batches: list[Any],
    initial_lora: dict[str, Any],
) -> list[float]:
    """Run one fresh optimizer curve from the same adapter initialization."""

    import torch

    torch.manual_seed(INPUT_SEED)
    restore_lora_state(model, initial_lora)

    model.train()
    model.zero_grad(set_to_none=True)

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError("model has no trainable parameters")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=1.0e-3,
    )

    losses: list[float] = []

    for input_ids in batches:
        optimizer.zero_grad(set_to_none=True)

        loss = model(
            input_ids=input_ids,
            labels=input_ids,
        ).loss

        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()

    return losses


def curve_diff(
    first: list[float],
    second: list[float],
) -> tuple[bool, float, float]:
    """Return exact equality plus max absolute and relative loss differences."""

    if len(first) != len(second):
        raise RuntimeError("loss curves have different lengths")

    max_abs = 0.0
    max_rel = 0.0

    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        if not math.isfinite(left) or not math.isfinite(right):
            raise MeasurementInvalidError(
                "loss curve contains non-finite values: "
                f"left[{index}]={left!r}, right[{index}]={right!r}"
            )

        difference = abs(left - right)
        max_abs = max(max_abs, difference)
        max_rel = max(
            max_rel,
            difference / max(abs(right), 1.0e-30),
        )

    return first == second, max_abs, max_rel


def historical_pattern_reproduced(
    exact: int,
    total: int,
    streamed_equal: bool,
    resident_equal: bool,
) -> bool:
    """Return the historical GRADDIFF verdict for the recorded control pattern."""

    return (
        exact == total
        and total > 0
        and not streamed_equal
        and resident_equal
    )


def historical_pattern_exit_code(historical_pattern: bool) -> int:
    """Return the CLI exit code for the historical GRADDIFF verdict."""
    return 0 if historical_pattern else 1


def run_self_test() -> int:
    """Exercise the production canonical-overlap guard without CUDA."""

    import torch

    class Tiny(torch.nn.Module):
        def __init__(self, parameter_name: str) -> None:
            super().__init__()
            self.register_parameter(
                parameter_name,
                torch.nn.Parameter(torch.ones(1)),
            )

    left = Tiny("lora_A")
    right = Tiny("lora_B")

    try:
        compare_gradients(
            left,
            right,
            {},
            {},
        )
    except ValueError as exc:
        print(
            "PASS: empty canonical parameter intersection was rejected "
            f"({type(exc).__name__}: {exc})"
        )
        return 0
    except MeasurementInvalidError as exc:
        print(f"ERROR: invalid measurement: {exc}")
        return 3
    except Exception as exc:
        raise RuntimeError(
            "negative acceptance test failed: unexpected exception "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    raise RuntimeError(
        "negative acceptance test failed: empty canonical parameter "
        "intersection was accepted"
    )


def run_measurement(args: argparse.Namespace) -> int:
    import torch

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )

    if not args.weights:
        print("ERROR: --weights is required unless --self-test is used")
        return 2

    if not args.shards:
        print("ERROR: --shards is required unless --self-test is used")
        return 2

    if args.seq <= 0:
        print("ERROR: --seq must be positive")
        return 2

    if args.batch <= 0:
        print("ERROR: --batch must be positive")
        return 2

    if args.buffers < 2 or args.buffers > 8:
        print("ERROR: --buffers must be between 2 and 8")
        return 2

    if args.curve_steps <= 0:
        print("ERROR: --curve-steps must be positive")
        return 2

    if not cuda_available():
        print("SKIP: CUDA is required for graddiff.py")
        return 0

    weights = Path(args.weights).expanduser().resolve()

    if not weights.is_dir():
        print(
            f"ERROR: --weights must be a local checkpoint directory: {weights}"
        )
        return 2

    shard_dir = Path(args.shards).expanduser().resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

    probe = build_meta_skeleton(
        str(weights),
        dtype=DTYPE,
        quant="nf4",
    )

    arch = bitexact.model_arch_name(probe)
    vocab_size = int(probe.config.vocab_size)
    quant_suffixes = quantised_layer_suffixes(probe)

    del probe

    print("RUN: STEP 2b GRADDIFF")
    print(f"torch         {torch.__version__}")
    print(f"gpu           {torch.cuda.get_device_name(0)}")

    bnb_version, peft_version, transformers_version = runtime_versions()
    print(f"bitsandbytes  {bnb_version}")
    print(f"transformers  {transformers_version}")
    print(f"peft          {peft_version}")
    print(f"weights       {weights}")
    print(f"seq           {args.seq}")
    print(f"batch         {args.batch}")
    print(f"buffers       {args.buffers}")
    print(f"curve_steps   {args.curve_steps}")
    print()

    index = shard_checkpoint(
        str(weights),
        str(shard_dir),
        dtype=DTYPE,
        arch=arch,
        quant="nf4",
        quant_suffixes=quant_suffixes,
        double_quant=True,
        quant_device=DEVICE,
    )

    lora_config = bitexact.lora_config()

    import soup_cli.utils.layer_stream_runtime as layer_runtime

    original_install_dequant_forward = layer_runtime.install_dequant_forward
    streamed = None
    runtime = None
    reference = None

    try:
        install_historical_control(layer_runtime)

        streamed, runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=str(shard_dir),
            index=index,
            lora_config=lora_config,
            device=DEVICE,
            dtype=DTYPE,
            buffers=args.buffers,
            pin=True,
            seed=DEFAULT_SEED,
            quant="nf4",
            double_quant=True,
            tier="ram",
        )

        actual_pinned = getattr(runtime, "pinned", None)

        if actual_pinned is not True:
            raise RuntimeError(
                f"requested pin=True, but runtime reports pinned={actual_pinned!r}"
            )

        source = getattr(runtime, "source", None)
        store_bytes = getattr(source, "nbytes", None)

        if not store_bytes or store_bytes <= 0:
            raise RuntimeError("streaming store is empty")

        meta_params = sum(
            1
            for parameter in streamed.parameters()
            if getattr(parameter, "is_meta", False)
        )

        if meta_params <= 0:
            raise RuntimeError(
                "streamed model has no remaining meta parameters; "
                "the streaming path was not exercised"
            )

        reference = bitexact.load_resident_reference(
            str(weights),
            "nf4",
        )

        bitexact.make_non_vacuous_lora(streamed)

        initial_lora = capture_lora_state(streamed)
        copied = bitexact.copy_lora(streamed, reference)

        if copied <= 0:
            raise RuntimeError("no LoRA tensors were copied")

        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(INPUT_SEED)

        input_ids = torch.randint(
            0,
            vocab_size,
            (args.batch, args.seq),
            generator=generator,
            device=DEVICE,
        )

        print("backward      comparing one streamed/reference backward")

        streamed_loss, streamed_grads = backward_once(
            streamed,
            input_ids,
        )

        reference_loss, reference_grads = backward_once(
            reference,
            input_ids,
        )

        if not torch.isfinite(streamed_loss) or not torch.isfinite(reference_loss):
            raise MeasurementInvalidError(
                "initial streamed/reference loss is non-finite"
            )

        if not torch.equal(streamed_loss, reference_loss):
            diff = float(
                (streamed_loss - reference_loss).abs()
            )

            raise RuntimeError(
                "initial streamed/reference losses differ "
                f"(abs_diff={diff:.6e})"
            )

        exact, total, max_abs, max_rel = compare_gradients(
            streamed,
            reference,
            streamed_grads,
            reference_grads,
        )

        print(
            f"A grads: {exact}/{total} bit-exact, "
            f"worst abs {max_abs:.6e} rel {max_rel:.6e}"
        )

        curve_generator = torch.Generator(
            device=DEVICE
        ).manual_seed(INPUT_SEED)

        batches = [
            torch.randint(
                0,
                vocab_size,
                (args.batch, args.seq),
                generator=curve_generator,
                device=DEVICE,
            )
            for _ in range(args.curve_steps)
        ]

        print(
            f"curve         {args.curve_steps}-step streamed self-check twice"
        )

        streamed_curve_1 = train_curve_once(
            streamed,
            batches,
            initial_lora,
        )

        streamed_curve_2 = train_curve_once(
            streamed,
            batches,
            initial_lora,
        )

        streamed_equal, streamed_abs, streamed_rel = curve_diff(
            streamed_curve_1,
            streamed_curve_2,
        )

        print(
            f"B streamed self-identical: {streamed_equal} "
            f"max_abs={streamed_abs:.6e} max_rel={streamed_rel:.6e}"
        )

        print(
            f"curve         {args.curve_steps}-step resident self-check twice"
        )

        resident_curve_1 = train_curve_once(
            reference,
            batches,
            initial_lora,
        )

        resident_curve_2 = train_curve_once(
            reference,
            batches,
            initial_lora,
        )

        resident_equal, resident_abs, resident_rel = curve_diff(
            resident_curve_1,
            resident_curve_2,
        )

        print(
            f"C resident self-identical: {resident_equal} "
            f"max_abs={resident_abs:.6e} max_rel={resident_rel:.6e}"
        )

        historical_pattern = historical_pattern_reproduced(
            exact,
            total,
            streamed_equal,
            resident_equal,
        )

        print()

        if historical_pattern:
            print(
                "RESULT: historical GRADDIFF pattern reproduced"
            )
            return historical_pattern_exit_code(True)

        print(
            "ERROR: historical GRADDIFF pattern was not reproduced"
        )
        print(
            "Expected: all initial gradients exact, "
            "streamed self-curve non-identical, "
            "resident self-curve identical"
        )
        return historical_pattern_exit_code(False)

    finally:
        if reference is not None:
            del reference

        layer_runtime.install_dequant_forward = original_install_dequant_forward
        if runtime is not None:
            runtime.close()
        if streamed is not None:
            del streamed

        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()

    try:
        if args.self_test:
            return run_self_test()

        return run_measurement(args)
    except Exception as exc:
        print(
            f"ERROR: graddiff.py failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
