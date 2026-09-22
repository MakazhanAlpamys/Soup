#!/usr/bin/env python3
"""Measure the throughput cost of pageable versus pinned NF4 streaming.

This is the published STEP 8 / #379 protocol reconstructed from
``benchmarks/gate-h100-validation.md``.

Protocol
--------
- Qwen2.5-32B-Instruct, NF4
- sequence length 256, batch size 1
- 2 stream buffers
- 5 warm-up forward+backward steps
- 20 timed forward+backward steps per repeat
- 3 repeats per arm
- CUDA synchronization around every timed step
- correctness checked against a resident NF4 reference in the same process
  before timing either arm

Historical reference:
- pin=True: 425.07 tok/s
- pin=False: 64.79 tok/s
- pinned/pageable ratio: 6.56x

Those values are historical reference values only. This harness reports new
measurements when run on a suitable CUDA machine.

No model download is required when ``--weights`` points to a local checkpoint
or an already-resolved local cache.

A machine without CUDA exits 0 with an explicit skip.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import bitexact

DTYPE = "bfloat16"
DEFAULT_SEQ = 256
DEFAULT_BATCH = 1
DEFAULT_BUFFERS = 2
DEFAULT_WARMUP = 5
DEFAULT_STEPS = 20
DEFAULT_REPEATS = 3
DEFAULT_SEED = 3
INPUT_SEED = 17
DEVICE = "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure pinned versus pageable NF4 streaming cost from "
            "Soup's STEP 8/#379 protocol."
        )
    )
    parser.add_argument(
        "--weights",
        required=False,
        help="checkpoint path or model id resolvable by Soup's weight resolver",
    )
    parser.add_argument(
        "--shards",
        required=False,
        help="directory in which Soup should create or reuse layer shards",
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
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"discarded warm-up steps per arm (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"timed steps per repeat (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"timing repeats per arm (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the CPU-only acceptance self-test and exit",
    )
    return parser.parse_args()


def cuda_available() -> bool:
    return bitexact.cuda_available()


def capture_lora_state(model: Any) -> dict[str, Any]:
    """Copy LoRA tensors to CPU so timed arms do not retain the reference."""
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    state: dict[str, Any] = {}

    for name, parameter in canonical_named_parameters(model):
        if "lora_" in name:
            state[name] = parameter.detach().cpu().clone()

    if not state:
        raise RuntimeError("reference model produced no LoRA tensors")

    return state


def restore_lora_state(model: Any, state: dict[str, Any]) -> None:
    """Restore a CPU LoRA snapshot into a streamed model."""
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    target = dict(canonical_named_parameters(model))

    with torch.no_grad():
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


def run_correctness(
    streamed: Any,
    reference: Any,
    input_ids: Any,
    *,
    pin: bool,
) -> None:
    """Require canonical overlap, exact loss, and LoRA gradients before timing."""
    from soup_cli.utils.layer_stream_runtime import (
        assert_canonical_parameters_intersect,
    )

    assert_canonical_parameters_intersect(
        streamed,
        reference,
    )

    streamed.zero_grad(set_to_none=True)
    reference.zero_grad(set_to_none=True)

    streamed.train()
    reference.train()

    streamed_loss = streamed(
        input_ids=input_ids,
        labels=input_ids,
    ).loss

    reference_loss = reference(
        input_ids=input_ids,
        labels=input_ids,
    ).loss

    if not __import__("torch").equal(
        streamed_loss,
        reference_loss,
    ):
        diff = (
            streamed_loss - reference_loss
        ).detach().abs().item()

        raise RuntimeError(
            f"pin={pin} loss is not bit-exact before timing "
            f"(abs_diff={diff:.6e})"
        )

    streamed_loss.backward()
    reference_loss.backward()

    ok, max_diff, layer0_sum = bitexact.compare_initial_gradients(
        streamed,
        reference,
    )

    if not ok:
        raise RuntimeError(
            f"pin={pin} failed gradient correctness before timing "
            f"(max_abs_diff={max_diff:.6e})"
        )

    if layer0_sum <= 0.0:
        raise RuntimeError(
            f"pin={pin} produced zero LoRA gradients"
        )


def time_arm(
    streamed: Any,
    *,
    input_ids: Any,
    warmup: int,
    steps: int,
) -> tuple[float, float]:
    """Warm up, then time forward+backward; return tok/s and median step time."""
    import torch

    streamed.train()

    def step() -> None:
        streamed.zero_grad(set_to_none=True)

        loss = streamed(
            input_ids=input_ids,
            labels=input_ids,
        ).loss

        loss.backward()

    for _ in range(warmup):
        step()

    torch.cuda.synchronize()

    durations: list[float] = []

    for _ in range(steps):
        torch.cuda.synchronize()

        started = time.perf_counter()

        step()

        torch.cuda.synchronize()

        durations.append(
            time.perf_counter() - started
        )

    median_step = median(durations)
    tokens = input_ids.numel()
    tok_per_s = tokens / median_step

    return float(tok_per_s), float(median_step)


def run_self_test() -> int:
    """Exercise the production correctness guard without CUDA."""
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
        # This calls the same production correctness path used before timing.
        run_correctness(
            left,
            right,
            None,
            pin=False,
        )
    except ValueError as exc:
        print(
            "PASS: empty canonical parameter intersection was rejected "
            f"({type(exc).__name__}: {exc})"
        )
        return 0
    except Exception as exc:
        raise RuntimeError(
            "negative acceptance test failed: unexpected exception "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    raise RuntimeError(
        "negative acceptance test failed: empty canonical parameter "
        "intersection was accepted"
    )


def build_shards_and_input(
    args: argparse.Namespace,
):
    """Resolve weights, shard once, and create the fixed timing input."""
    import torch

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        quantised_layer_suffixes,
    )
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    weights_dir = resolve_model_weights(args.weights)

    shard_dir = Path(args.shards)
    shard_dir.mkdir(parents=True, exist_ok=True)

    probe = build_meta_skeleton(
        weights_dir,
        dtype=DTYPE,
        quant="nf4",
    )

    arch = bitexact.model_arch_name(probe)
    vocab_size = int(probe.config.vocab_size)
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
        quant_device=DEVICE,
    )

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

    return weights_dir, str(shard_dir), index, input_ids


def run_measurement(args: argparse.Namespace) -> int:
    import torch
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_stream_runtime import build_streamed_model

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

    if args.warmup <= 0:
        print("ERROR: --warmup must be positive")
        return 2

    if args.steps <= 0:
        print("ERROR: --steps must be positive")
        return 2

    if args.repeats <= 0:
        print("ERROR: --repeats must be positive")
        return 2

    if not cuda_available():
        print("SKIP: CUDA is required for pincost.py")
        return 0

    (
        weights_dir,
        shard_dir,
        index,
        input_ids,
    ) = build_shards_and_input(args)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    print("RUN: STEP 8 pinned-versus-pageable NF4 cost")
    print(f"torch         {torch.__version__}")
    print(f"gpu           {torch.cuda.get_device_name(0)}")
    print(f"weights       {args.weights}")
    print("quant         nf4")
    print(f"seq           {args.seq}")
    print(f"batch         {args.batch}")
    print(f"buffers       {args.buffers}")
    print(f"warmup        {args.warmup}")
    print(f"steps         {args.steps}")
    print(f"repeats       {args.repeats}")
    print()

    pinned_rates: list[float] = []
    pageable_rates: list[float] = []

    for repetition in range(args.repeats):
        print(
            f"repeat        {repetition + 1}/{args.repeats}"
        )

        lora_state = None

        for pin in (True, False):
            reference = None
            streamed = None
            runtime = None

            try:
                reference = bitexact.load_resident_reference(
                    weights_dir,
                    "nf4",
                )

                bitexact.make_non_vacuous_lora(reference)

                if lora_state is None:
                    lora_state = capture_lora_state(reference)
                else:
                    restore_lora_state(reference, lora_state)

                print(f"correctness   pin={pin}")

                streamed, runtime = build_streamed_model(
                    model_id=weights_dir,
                    shard_dir=shard_dir,
                    index=index,
                    lora_config=lora_config,
                    device=DEVICE,
                    dtype=DTYPE,
                    buffers=args.buffers,
                    pin=pin,
                    seed=DEFAULT_SEED,
                    quant="nf4",
                    double_quant=True,
                    tier="ram",
                )

                actual_pinned = getattr(
                    runtime,
                    "pinned",
                    None,
                )

                if actual_pinned is not pin:
                    raise RuntimeError(
                        f"requested pin={pin}, but runtime reports "
                        f"pinned={actual_pinned!r}"
                    )

                source = getattr(
                    runtime,
                    "source",
                    None,
                )
                store_bytes = getattr(
                    source,
                    "nbytes",
                    None,
                )

                if not store_bytes or store_bytes <= 0:
                    raise RuntimeError(
                        f"pin={pin} reported an empty streaming store"
                    )

                restore_lora_state(
                    streamed,
                    lora_state,
                )

                run_correctness(
                    streamed,
                    reference,
                    input_ids,
                    pin=pin,
                )

                # Correctness and timing use the same streamed instance.
                # Remove the resident reference before timing so it does not
                # inflate the timed arm's memory footprint.
                del reference
                reference = None

                gc.collect()
                torch.cuda.empty_cache()

                rate, median_step = time_arm(
                    streamed,
                    input_ids=input_ids,
                    warmup=args.warmup,
                    steps=args.steps,
                )

                if pin:
                    pinned_rates.append(rate)
                else:
                    pageable_rates.append(rate)

                print(
                    f"timing        pin={pin} "
                    f"{rate:.2f} tok/s, "
                    f"median_step={median_step:.4f} s"
                )

            finally:
                if reference is not None:
                    del reference

                if runtime is not None:
                    runtime.close()

                if streamed is not None:
                    del streamed

                gc.collect()
                torch.cuda.empty_cache()

    pinned_median = float(
        median(pinned_rates)
    )
    pageable_median = float(
        median(pageable_rates)
    )

    direct_ratio = (
        pinned_median / pageable_median
    )

    inverse_ratio = (
        pageable_median / pinned_median
    )

    print()
    print(
        f"median pin=True          "
        f"{pinned_median:.2f} tok/s"
    )
    print(
        f"median pin=False         "
        f"{pageable_median:.2f} tok/s"
    )
    print(
        f"pin=True/pin=False ratio "
        f"{direct_ratio:.2f}x"
    )
    print(
        f"pin=False/pin=True ratio "
        f"{inverse_ratio:.2f}x"
    )
    print(
        "Historical reference:    "
        "425.07 tok/s vs 64.79 tok/s, 6.56x"
    )
    print("RESULT: pinning cost measured")

    return 0


def main() -> int:
    args = parse_args()

    try:
        if args.self_test:
            return run_self_test()

        return run_measurement(args)
    except Exception as exc:
        print(
            f"ERROR: pincost.py failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
