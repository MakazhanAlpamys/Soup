#!/usr/bin/env python3
"""Measure the cost of the #331 NF4 de-aliasing mechanism.

This is a repository-published reconstruction of the cost protocol recorded in
``benchmarks/gate-h100-validation.md``. The original mechanism_cost.py lived in
the H100 session scratchpad and was never committed, so this file does NOT claim
to be the original scratchpad implementation.

Recorded protocol
-----------------
Real Qwen2.5-32B-Instruct NF4, sequence length 256, batch 1:

- five timing repeats
- four warm-up steps per repeat
- fifteen timed forward+backward steps per repeat
- CUDA synchronization around timed steps
- peak memory from ``torch.cuda.max_memory_allocated`` after
  ``reset_peak_memory_stats``

Historical arms
---------------
- control: historical NF4 ``MatMul4Bit`` path, with the shipped repair disabled
- clone_fwd_quant: historical ``MatMul4Bit`` path, but only the tensors
  captured by bitsandbytes are cloned on the forward
- clone: historical path with every substituted layer tensor cloned

The historical record reports:

    control          416.04 tok/s   4,220 MiB   WRONG 8/256
    clone_fwd_quant  405.09 tok/s  19,720 MiB   exact 256/256
    clone            389.41 tok/s  19,720 MiB   exact 256/256

The numbers above are historical reference values, not asserted measurements
from this harness.

Requirements
------------
- CUDA-capable GPU
- PyTorch
- transformers
- peft
- safetensors
- bitsandbytes
- A locally available Qwen2.5-32B-Instruct checkpoint
- Enough GPU memory for the resident NF4 reference plus the measured arm

No source files under ``src/`` are modified. The harness temporarily
monkeypatches the runtime in-process to recreate the historical mechanism.

A machine without CUDA exits 0 with an explicit skip.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

DTYPE = "bfloat16"
DEFAULT_SEQ = 256
DEFAULT_BATCH = 1
DEFAULT_REPEATS = 5
DEFAULT_WARMUP = 4
DEFAULT_STEPS = 15
DEFAULT_BUFFERS = 2
DEFAULT_SEED = 3
INPUT_SEED = 17
CORRECTNESS_REPEATS = 3
DEVICE = "cuda"


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the recorded #331 NF4 de-aliasing cost on a local "
            "Qwen2.5-32B-Instruct checkpoint."
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
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"timing repeats (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help=f"warm-up steps per repeat (default: {DEFAULT_WARMUP})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"timed steps per repeat (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--buffers",
        type=int,
        default=DEFAULT_BUFFERS,
        help=f"stream buffer count (default: {DEFAULT_BUFFERS})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the CPU-only acceptance self-test and exit",
    )
    return parser.parse_args()


def lora_config():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )


def canonical_parameters(model: Any) -> dict[str, Any]:
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    return dict(canonical_named_parameters(model))


def copy_lora(source: Any, target: Any) -> int:
    """Copy the source adapter tensors into the target model."""

    source_params = canonical_parameters(source)
    target_params = canonical_parameters(target)

    copied = 0

    import torch

    with torch.no_grad():
        for name, target_param in target_params.items():
            if "lora_" not in name:
                continue

            source_param = source_params.get(name)
            if source_param is None:
                raise RuntimeError(
                    f"could not match LoRA parameter {name!r}"
                )

            target_param.copy_(source_param)
            copied += 1

    if copied == 0:
        raise RuntimeError("no LoRA tensors were copied")

    return copied


def make_non_vacuous_lora(model: Any) -> None:
    """Make LoRA-B non-zero so gradient checks cannot pass vacuously."""

    import torch

    generator = torch.Generator(device=DEVICE).manual_seed(23)

    with torch.no_grad():
        for name, parameter in canonical_parameters(model).items():
            if "lora_B" in name:
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    * 0.02
                )


def clone_quant_state(state: Any) -> Any:
    """Clone only tensor state captured by bitsandbytes.

    The NF4 code tables are constant shared tensors and are deliberately not
    cloned. The mutable captured pieces are absmax, offset and nested absmax.
    """

    from bitsandbytes.functional import QuantState

    state2 = None

    if state.nested:
        state2 = QuantState(
            absmax=state.state2.absmax.clone(),
            shape=state.state2.shape,
            code=state.state2.code,
            blocksize=state.state2.blocksize,
            quant_type=state.state2.quant_type,
            dtype=state.state2.dtype,
        )

    offset = (
        state.offset.clone()
        if state.offset is not None
        else None
    )

    return QuantState(
        absmax=state.absmax.clone(),
        shape=state.shape,
        code=state.code,
        blocksize=state.blocksize,
        quant_type=state.quant_type,
        dtype=state.dtype,
        offset=offset,
        state2=state2,
    )


def clone_params4bit(param: Any, spec: Any) -> Any:
    """Clone the packed NF4 bytes and bnb-captured quantization tensors."""

    import bitsandbytes as bnb

    return bnb.nn.Params4bit(
        data=param.data.clone(),
        requires_grad=False,
        quant_state=clone_quant_state(param.quant_state),
        blocksize=spec.blocksize,
        compress_statistics=spec.nested,
        quant_type=spec.quant_type,
        bnb_quantized=True,
    )


def install_historical_control(runtime: Any) -> None:
    """Disable the current #331 repair inside this process only."""

    runtime.install_dequant_forward = lambda _module: 0


def install_clone_fwd_quant(runtime: Any, original_rebuild: Callable) -> None:
    """Clone exactly the tensors captured by historical MatMul4Bit."""

    def cloned_rebuild(key: str, buffers: Any, spec: Any, codes: Any):
        original = original_rebuild(key, buffers, spec, codes)
        return clone_params4bit(original, spec)

    runtime.rebuild_params4bit = cloned_rebuild


def install_full_clone(runtime: Any, original_rebuild: Callable) -> None:
    """Clone every substituted layer tensor on every call.

    ``install_clone_fwd_quant`` already makes a private copy of the tensors
    captured by bitsandbytes. The additional clone here is only for the
    non-quantized layer tensors, matching the recorded "everything, every call"
    arm without cloning the NF4 payload twice.
    """

    install_clone_fwd_quant(runtime, original_rebuild)

    layer_cls = runtime._streamed_layer_class()
    original_substituted = layer_cls._substituted_weights

    def cloned_substituted(self, buffers):
        weights = original_substituted(self, buffers)

        import bitsandbytes as bnb

        for meta_name in self.name_map:
            value = weights[meta_name]

            if isinstance(value, bnb.nn.Params4bit):
                continue

            weights[meta_name] = value.clone()

        return weights

    layer_cls._substituted_weights = cloned_substituted


def prepare_shards(weights: str, shards: str):
    """Create/reuse the same layer shards for every measurement arm."""

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        quantised_layer_suffixes,
    )

    probe = build_meta_skeleton(
        weights,
        dtype=DTYPE,
        quant="nf4",
        double_quant=True,
    )

    config = getattr(probe, "config", None)
    arch = getattr(config, "model_type", None)

    if not arch:
        raise RuntimeError(
            "checkpoint config does not expose model_type"
        )

    quant_suffixes = quantised_layer_suffixes(probe)

    del probe
    gc.collect()

    Path(shards).mkdir(parents=True, exist_ok=True)

    index = shard_checkpoint(
        weights,
        shards,
        dtype=DTYPE,
        arch=str(arch),
        quant="nf4",
        quant_suffixes=quant_suffixes,
        double_quant=True,
        quant_device=DEVICE,
    )

    return index, str(arch)


def load_resident_reference(weights: str):
    """Load the resident NF4 reference before measuring incremental peak VRAM."""

    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        weights,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map={"": "cuda:0"},
    )

    model = get_peft_model(model, lora_config())
    model.config.use_cache = False

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    return model


def build_streamed_arm(
    weights: str,
    shards: str,
    index: Any,
    arch: str,
    arm: str,
    buffers: int,
):
    """Build one historical arm and return its restoration callback.

    The monkeypatch MUST remain active while the model is actually executed:
    StreamedDecoderLayer resolves these globals at forward/constructor time.
    """

    import soup_cli.utils.layer_stream_runtime as runtime

    original_install = runtime.install_dequant_forward
    original_rebuild = runtime.rebuild_params4bit

    layer_cls = runtime._streamed_layer_class()
    original_substituted = layer_cls._substituted_weights

    restored = False

    def restore() -> None:
        nonlocal restored

        if restored:
            return

        runtime.install_dequant_forward = original_install
        runtime.rebuild_params4bit = original_rebuild
        layer_cls._substituted_weights = original_substituted
        restored = True

    try:
        install_historical_control(runtime)

        if arm == "clone_fwd_quant":
            install_clone_fwd_quant(
                runtime,
                original_rebuild,
            )
        elif arm == "clone":
            install_full_clone(
                runtime,
                original_rebuild,
            )
        elif arm != "control":
            raise ValueError(f"unsupported arm: {arm!r}")

        streamed, stream_runtime = runtime.build_streamed_model(
            model_id=weights,
            shard_dir=shards,
            index=index,
            lora_config=lora_config(),
            device=DEVICE,
            dtype=DTYPE,
            buffers=buffers,
            pin=True,
            seed=DEFAULT_SEED,
            quant="nf4",
            double_quant=True,
            tier="ram",
        )

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

        return streamed, stream_runtime, restore

    except Exception:
        restore()
        raise


def run_backward(
    model: Any,
    input_ids: Any,
) -> float:
    """One forward+backward step, returning the loss."""

    model.train()
    model.zero_grad(set_to_none=True)

    output = model(
        input_ids=input_ids,
        labels=input_ids,
    )

    loss = output.loss
    value = float(loss.detach())

    loss.backward()

    return value


def gradient_snapshot(model: Any) -> dict[str, Any]:
    """Collect LoRA gradients using canonical parameter names."""

    params = canonical_parameters(model)

    gradients = {}

    for name, parameter in params.items():
        if "lora_" not in name:
            continue

        if parameter.grad is None:
            continue

        gradients[name] = parameter.grad.detach().float().clone()

    if not gradients:
        raise RuntimeError(
            "model produced no LoRA gradients"
        )

    return gradients


def assert_gradient_parity(
    streamed: Any,
    reference: Any,
) -> tuple[int, int, float]:
    """Require canonical model overlap, then compare only LoRA gradients."""

    import torch

    from soup_cli.utils.layer_stream_runtime import (
        assert_canonical_parameters_intersect,
    )

    names = assert_canonical_parameters_intersect(
        streamed,
        reference,
    )

    if not names:
        raise RuntimeError(
            "canonical parameter intersection is empty"
        )

    streamed_grads = gradient_snapshot(streamed)
    reference_grads = gradient_snapshot(reference)

    if set(streamed_grads) != set(reference_grads):
        raise RuntimeError(
            "streamed/reference LoRA gradient sets differ"
        )

    exact = 0
    worst = 0.0

    for name in streamed_grads:
        left = streamed_grads[name]
        right = reference_grads[name]

        if left.shape != right.shape:
            raise RuntimeError(
                f"gradient shape differs for {name!r}: "
                f"{tuple(left.shape)} != {tuple(right.shape)}"
            )

        diff = (left - right).abs().max().item()
        worst = max(worst, float(diff))

        if torch.equal(left, right):
            exact += 1

    return exact, len(streamed_grads), worst


def correctness_gate(
    streamed: Any,
    reference: Any,
    *,
    repetitions: int,
    arm: str,
    seq: int,
    batch: int,
) -> list[int]:
    """Run the three-backward mechanism gate recorded before timing."""

    import torch

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    vocab_size = int(streamed.config.vocab_size)
    generator = torch.Generator(device=DEVICE).manual_seed(INPUT_SEED)

    input_ids = torch.randint(
        0,
        vocab_size,
        (batch, seq),
        generator=generator,
        device=DEVICE,
    )

    exact_counts: list[int] = []

    for repetition in range(repetitions):
        streamed.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)

        run_backward(streamed, input_ids)

        run_backward(reference, input_ids)

        exact, total, worst = assert_gradient_parity(
            streamed,
            reference,
        )

        exact_counts.append(exact)

        print(
            f"  correctness {repetition + 1}/{repetitions}: "
            f"{exact}/{total} exact, "
            f"worst_abs={worst:.6e}"
        )

    if arm == "control":
        if all(count == total for count in exact_counts):
            raise RuntimeError(
                "control never reproduced a gradient mismatch; the historical "
                "defect is bracketed at 163.8-171.5 MiB per NF4 layer"
            )

        if exact_counts[-1] == total:
            raise RuntimeError(
                "control ended exact; expected the historical stale-buffer "
                "mismatch after repeated backwards"
            )
    else:
        if any(count != total for count in exact_counts):
            raise RuntimeError(
                f"{arm} did not reproduce exact gradients across "
                f"{repetitions} repetitions"
            )

    return exact_counts


def make_input(
    model: Any,
    *,
    seq: int,
    batch: int,
) -> Any:
    import torch

    vocab_size = int(model.config.vocab_size)
    generator = torch.Generator(device=DEVICE).manual_seed(INPUT_SEED)

    return torch.randint(
        0,
        vocab_size,
        (batch, seq),
        generator=generator,
        device=DEVICE,
    )


def timed_repeat(
    model: Any,
    input_ids: Any,
    *,
    warmup: int,
    steps: int,
    batch: int,
    seq: int,
) -> tuple[float, int]:
    """Return tok/s and incremental peak allocated bytes."""

    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        run_backward(model, input_ids)
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)

    total_seconds = 0.0

    for _ in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()

        run_backward(model, input_ids)

        torch.cuda.synchronize()
        total_seconds += time.perf_counter() - started

        model.zero_grad(set_to_none=True)

    peak_bytes = int(torch.cuda.max_memory_allocated())

    tokens = steps * batch * seq
    tok_per_sec = tokens / total_seconds

    return tok_per_sec, peak_bytes


def run_arm(
    weights: str,
    shards: str,
    index: Any,
    arch: str,
    *,
    arm: str,
    seq: int,
    batch: int,
    repeats: int,
    warmup: int,
    steps: int,
    buffers: int,
) -> tuple[float, int]:
    print()
    print(f"ARM: {arm}")

    import torch

    streamed, stream_runtime, restore = build_streamed_arm(
        weights,
        shards,
        index,
        arch,
        arm,
        buffers,
    )

    reference = None
    input_ids = None

    try:
        reference = load_resident_reference(weights)

        make_non_vacuous_lora(streamed)
        copied = copy_lora(streamed, reference)

        print(f"  adapter_tensors_copied {copied}")

        correctness_gate(
            streamed,
            reference,
            repetitions=CORRECTNESS_REPEATS,
            arm=arm,
            seq=seq,
            batch=batch,
        )

        # The recorded peak values are for the measured streamed arm itself,
        # not for the streamed model plus the resident comparison model.
        del reference
        reference = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        input_ids = make_input(
            streamed,
            seq=seq,
            batch=batch,
        )

        rates: list[float] = []
        peaks: list[int] = []

        for repetition in range(repeats):
            tok_per_sec, peak_bytes = timed_repeat(
                streamed,
                input_ids,
                warmup=warmup,
                steps=steps,
                batch=batch,
                seq=seq,
            )

            rates.append(tok_per_sec)
            peaks.append(peak_bytes)

            print(
                f"  timing {repetition + 1}/{repeats}: "
                f"{tok_per_sec:.2f} tok/s, "
                f"peak={peak_bytes / 2**20:.0f} MiB"
            )

        median_tok_per_sec = float(median(rates))
        max_peak = max(peaks)

        print(
            f"  median_tok_per_sec {median_tok_per_sec:.2f}"
        )
        print(
            f"  max_peak {max_peak / 2**20:.0f} MiB"
        )

        return median_tok_per_sec, max_peak

    finally:
        if input_ids is not None:
            del input_ids

        if reference is not None:
            del reference

        stream_runtime.close()
        restore()

        del streamed
        gc.collect()
        torch.cuda.empty_cache()


def run_self_test() -> int:
    """Exercise the negative empty-intersection acceptance case without CUDA."""

    import torch

    class Tiny(torch.nn.Module):
        def __init__(self, parameter_name: str) -> None:
            super().__init__()
            self.register_parameter(
                parameter_name,
                torch.nn.Parameter(torch.ones(1)),
            )

    # Use LoRA-shaped parameter names so that, if the canonical-intersection
    # assertion is removed, the comparison proceeds far enough to hit the
    # separate gradient requirement instead of accidentally passing.
    left = Tiny("lora_A")
    right = Tiny("lora_B")

    try:
        assert_gradient_parity(left, right)
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


def main() -> int:
    args = parse_args()

    if args.self_test:
        return run_self_test()

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

    if args.repeats <= 0:
        print("ERROR: --repeats must be positive")
        return 2

    if args.warmup <= 0:
        print("ERROR: --warmup must be positive")
        return 2

    if args.steps <= 0:
        print("ERROR: --steps must be positive")
        return 2

    if args.buffers != DEFAULT_BUFFERS:
        print(
            "ERROR: mechanism_cost.py currently records the historical "
            f"two-buffer protocol only; got --buffers {args.buffers}"
        )
        return 2

    if not cuda_available():
        print("SKIP: CUDA is required for mechanism_cost.py")
        return 0

    weights = Path(args.weights).expanduser().resolve()

    if not weights.is_dir():
        print(
            f"ERROR: --weights must be a local checkpoint directory: {weights}"
        )
        return 2

    import bitsandbytes as bnb
    import torch

    print("RUN: NF4 mechanism cost reconstruction")
    print(f"torch         {torch.__version__}")
    print(f"bitsandbytes  {bnb.__version__}")
    print(f"gpu           {torch.cuda.get_device_name(0)}")
    print(f"weights       {weights}")
    print(f"seq           {args.seq}")
    print(f"batch         {args.batch}")
    print(f"buffers       {args.buffers}")
    print(f"repeats       {args.repeats}")
    print(f"warmup        {args.warmup}")
    print(f"timed_steps   {args.steps}")
    print()
    print("Historical reference:")
    print("  control          416.04 tok/s   4,220 MiB   WRONG 8/256")
    print("  clone_fwd_quant  405.09 tok/s  19,720 MiB   exact 256/256")
    print("  clone            389.41 tok/s  19,720 MiB   exact 256/256")
    print()
    print(
        "NOTE: the original scratchpad implementation was not committed; "
        "this harness reconstructs the recorded arms and protocol."
    )

    runtimes: list[tuple[str, float, int]] = []

    try:
        print()
        print("Preparing shards...")
        index, arch = prepare_shards(
            str(weights),
            args.shards,
        )

        print(f"arch          {arch}")

        for arm in ("control", "clone_fwd_quant", "clone"):
            tok_per_sec, peak = run_arm(
                str(weights),
                args.shards,
                index,
                arch,
                arm=arm,
                seq=args.seq,
                batch=args.batch,
                repeats=args.repeats,
                warmup=args.warmup,
                steps=args.steps,
                buffers=args.buffers,
            )
            runtimes.append((arm, tok_per_sec, peak))

        control_rate = runtimes[0][1]
        clone_rate = runtimes[1][1]
        full_clone_rate = runtimes[2][1]

        print()
        print("RESULT SUMMARY")
        print(f"{'arm':>18}  {'median tok/s':>14}  {'peak MiB':>12}")
        print("-" * 50)

        for arm, rate, peak in runtimes:
            print(
                f"{arm:>18}  "
                f"{rate:>14.2f}  "
                f"{peak / 2**20:>12.0f}"
            )

        if control_rate <= 0.0:
            print("ERROR: invalid control throughput result")
            return 1

        print()
        clone_fwd_ratio = clone_rate / control_rate
        clone_ratio = full_clone_rate / control_rate

        print(
            "clone_fwd_quant/control throughput ratio: "
            f"{clone_fwd_ratio:.3f}x"
        )
        print(
            "control/clone_fwd_quant ratio (record convention): "
            f"{1.0 / clone_fwd_ratio:.3f}x"
        )

        print(
            "clone/control throughput ratio: "
            f"{clone_ratio:.3f}x"
        )
        print(
            "control/clone ratio (record convention): "
            f"{1.0 / clone_ratio:.3f}x"
        )

        print(
            "RESULT: mechanism-cost protocol completed. "
            "Historical numbers above are references; current measurements "
            "are the rows produced by this reconstruction."
        )

        return 0

    except Exception as exc:
        print(
            f"ERROR: mechanism_cost.py failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1

    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
