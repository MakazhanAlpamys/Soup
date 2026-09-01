#!/usr/bin/env python3
"""Reproduce the layer-streaming bit-exactness measurement.

This is the published ``bitexact.py`` harness described in
``benchmarks/gate-h100-validation.md``:

    shard -> stream -> compare logits/gradients/loss curve
    against a resident reference of matching numerics

Requirements
------------
- CUDA-capable GPU.
- PyTorch, transformers, peft, safetensors, and bitsandbytes for NF4.
- A locally available checkpoint, either as a path or through Soup's
  weight cache/resolver.
- Enough GPU memory for the resident reference as well as the streamed model.

Typical invocation
------------------
    python benchmarks/harness/bitexact.py \
        --weights Qwen/Qwen2.5-0.5B-Instruct \
        --shards /tmp/qwen05b_nf4 \
        --quant nf4 \
        --seq 64

The historical measurement used CUDA bf16 and two stream buffers. The resident
reference always uses the same numerics as the streamed arm: NF4 is compared
with resident NF4, never resident bf16.

No benchmark methodology is introduced here. The harness makes the original
correctness checks executable from the repository and exits non-zero whenever
one of the required parity checks fails.

A machine without CUDA is an intentional skip and exits 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DTYPE = "bfloat16"
DEFAULT_BUFFERS = 2
DEFAULT_STEPS = 100
DEFAULT_BATCH = 1
DEFAULT_SEED = 3
INPUT_SEED = 17


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Soup's layer-streaming bit-exactness gate."
    )
    parser.add_argument(
        "--weights",
        required=True,
        help=(
            "checkpoint path or model id resolvable by Soup's "
            "weight resolver"
        ),
    )
    parser.add_argument(
        "--shards",
        required=True,
        help="directory in which Soup should create or reuse layer shards",
    )
    parser.add_argument(
        "--quant",
        choices=("none", "nf4"),
        default="none",
        help="streamed/base-weight representation (default: none)",
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=512,
        help="sequence length (default: 512)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=(
            "training steps for the loss-curve comparison "
            f"(default: {DEFAULT_STEPS})"
        ),
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
        "--no-pin",
        action="store_true",
        help="use pageable host storage instead of pinned host storage",
    )
    parser.add_argument(
        "--skip-lora-sync",
        action="store_true",
        help=(
            "negative control: do not copy LoRA tensors into the resident "
            "reference; the parity check must then fail."
        ),
    )
    return parser.parse_args()


def model_arch_name(model) -> str:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "")
    if not model_type:
        raise RuntimeError("checkpoint config does not expose model_type")
    return str(model_type)


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


def copy_lora(source, target) -> int:
    """Copy exact adapter tensors from one model into the other."""
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    source_params = dict(canonical_named_parameters(source))
    target_params = dict(canonical_named_parameters(target))

    copied = 0

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


def make_non_vacuous_lora(model) -> None:
    """Give LoRA-B non-zero values so parity cannot pass vacuously."""
    import torch

    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    generator = torch.Generator(device="cuda").manual_seed(23)

    with torch.no_grad():
        for name, parameter in canonical_named_parameters(model):
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


def collect_lora_grads(model):
    from soup_cli.utils.layer_stream_runtime import canonical_named_parameters

    return {
        name: parameter.grad.detach().clone()
        for name, parameter in canonical_named_parameters(model)
        if "lora_" in name and parameter.grad is not None
    }


def enable_resident_checkpointing(model) -> None:
    """Match the historical checkpointed training protocol where supported."""
    model.config.use_cache = False

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()


def load_resident_reference(weights_dir: str, quant: str):
    """Load the resident reference using the same base numerics as streaming."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }

    if quant == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": "cuda:0"}

    model = AutoModelForCausalLM.from_pretrained(
        weights_dir,
        **kwargs,
    )

    if quant != "nf4":
        model = model.to("cuda")

    model = get_peft_model(model, lora_config())
    enable_resident_checkpointing(model)
    return model


def compare_initial_gradients(
    streamed,
    reference,
) -> tuple[bool, float, float]:
    """Compare one backward pass.

    Returns:
        (ok, max_abs_gradient_difference, layer0_gradient_sum)
    """
    import torch

    streamed_grads = collect_lora_grads(streamed)
    reference_grads = collect_lora_grads(reference)

    if not streamed_grads:
        print("ERROR: streamed model produced no LoRA gradients")
        return False, float("inf"), 0.0

    if set(streamed_grads) != set(reference_grads):
        print("ERROR: streamed/reference LoRA gradient sets differ")
        print(
            "streamed-only:",
            sorted(set(streamed_grads) - set(reference_grads)),
        )
        print(
            "reference-only:",
            sorted(set(reference_grads) - set(streamed_grads)),
        )
        return False, float("inf"), 0.0

    max_diff = 0.0

    for name, gradient in streamed_grads.items():
        reference_gradient = reference_grads[name]

        if gradient.shape != reference_gradient.shape:
            print(f"ERROR: gradient shape differs for {name}")
            return False, float("inf"), 0.0

        diff = (gradient - reference_gradient).abs().max().item()
        max_diff = max(max_diff, diff)

        if not torch.equal(gradient, reference_gradient):
            print(
                f"ERROR: LoRA gradient mismatch for {name} "
                f"(max_abs_diff={diff:.6e})"
            )
            return False, max_diff, 0.0

    layer0_sum = sum(
        float(gradient.abs().sum())
        for name, gradient in streamed_grads.items()
        if ".layers.0." in name
    )

    if layer0_sum <= 0.0:
        print("ERROR: layer-0 LoRA gradient is zero")
        return False, max_diff, layer0_sum

    return True, max_diff, layer0_sum


def train_curve(model, batches) -> list[float]:
    model.train()
    model.zero_grad(set_to_none=True)

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = __import__("torch").optim.AdamW(
        trainable,
        lr=1.0e-3,
    )

    losses: list[float] = []

    for input_ids in batches:
        output = model(
            input_ids=input_ids,
            labels=input_ids,
        )

        loss = output.loss
        losses.append(float(loss.detach()))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return losses


def relative_difference(left: float, right: float) -> float:
    denominator = max(abs(right), 1.0e-30)
    return abs(left - right) / denominator


def main() -> int:
    args = parse_args()

    if args.seq <= 0:
        print("ERROR: --seq must be positive")
        return 2

    if args.steps <= 0:
        print("ERROR: --steps must be positive")
        return 2

    if args.batch <= 0:
        print("ERROR: --batch must be positive")
        return 2

    if args.buffers < 2 or args.buffers > 8:
        print("ERROR: --buffers must be between 2 and 8")
        return 2

    if not cuda_available():
        print("SKIP: CUDA is required for bitexact.py")
        return 0

    import torch

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    print("RUN: layer-streaming bit-exactness gate")
    print(f"torch         {torch.__version__}")
    print(f"gpu           {torch.cuda.get_device_name(0)}")
    print(f"weights       {args.weights}")
    print(f"quant         {args.quant}")
    print(f"dtype         {DTYPE}")
    print(f"seq           {args.seq}")
    print(f"batch         {args.batch}")
    print(f"steps         {args.steps}")
    print(f"buffers       {args.buffers}")
    print(f"pin           {not args.no_pin}")

    if args.skip_lora_sync:
        print("mode          NEGATIVE CONTROL: LoRA sync skipped")

    try:
        weights_dir = resolve_model_weights(args.weights)
        shard_dir = Path(args.shards)
        shard_dir.mkdir(parents=True, exist_ok=True)

        probe = build_meta_skeleton(
            weights_dir,
            dtype=DTYPE,
            quant=args.quant,
        )
        arch = model_arch_name(probe)

        quant_suffixes = ()
        if args.quant == "nf4":
            quant_suffixes = quantised_layer_suffixes(probe)

        del probe

        print(f"arch          {arch}")
        print("sharding      preparing layer shards...")

        index = shard_checkpoint(
            weights_dir,
            str(shard_dir),
            dtype=DTYPE,
            arch=arch,
            quant=args.quant,
            quant_suffixes=quant_suffixes,
            double_quant=True,
            quant_device="cuda",
        )

        print(f"layers        {index.n_layers}")

        print("stream        building streamed model...")

        streamed, runtime = build_streamed_model(
            model_id=weights_dir,
            shard_dir=str(shard_dir),
            index=index,
            lora_config=lora_config(),
            device="cuda",
            dtype=DTYPE,
            buffers=args.buffers,
            pin=not args.no_pin,
            seed=DEFAULT_SEED,
            quant=args.quant,
            double_quant=True,
            tier="ram",
        )

        print("reference     loading resident model...")

        reference = load_resident_reference(
            weights_dir,
            args.quant,
        )

        make_non_vacuous_lora(streamed)

        if args.skip_lora_sync:
            copied = 0
        else:
            copied = copy_lora(streamed, reference)

        print(f"adapter_tensors_copied {copied}")

        streamed.eval()
        reference.eval()

        vocab_size = int(streamed.config.vocab_size)

        generator = torch.Generator(device="cuda").manual_seed(INPUT_SEED)

        input_ids = torch.randint(
            0,
            vocab_size,
            (args.batch, args.seq),
            generator=generator,
            device="cuda",
        )

        # Forward parity
        print("forward       comparing logits...")

        with torch.no_grad():
            streamed_logits = streamed(
                input_ids=input_ids
            ).logits
            reference_logits = reference(
                input_ids=input_ids
            ).logits

        max_abs_logit_diff = (
            streamed_logits - reference_logits
        ).abs().max().item()

        if not torch.equal(streamed_logits, reference_logits):
            print(
                "ERROR: streamed/resident logits are not bit-exact "
                f"(max_abs_diff={max_abs_logit_diff:.6e})"
            )
            return 1

        print(
            f"max_abs_logit_diff  {max_abs_logit_diff:.1f} "
            "bit_exact  true"
        )

        # One backward for the gradient gate
        print("backward      comparing LoRA gradients...")

        streamed.train()
        reference.train()

        streamed.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)

        streamed_loss = streamed(
            input_ids=input_ids,
            labels=input_ids,
        ).loss

        reference_loss = reference(
            input_ids=input_ids,
            labels=input_ids,
        ).loss

        if not torch.equal(streamed_loss, reference_loss):
            loss_diff = (
                streamed_loss - reference_loss
            ).detach().abs().item()

            print(
                "ERROR: initial loss is not bit-exact "
                f"(abs_diff={loss_diff:.6e})"
            )
            return 1

        streamed_loss.backward()
        reference_loss.backward()

        gradients_ok, max_grad_diff, layer0_sum = (
            compare_initial_gradients(
                streamed,
                reference,
            )
        )

        if not gradients_ok:
            return 1

        streamed.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)

        print(f"layer0_lora_grad      {layer0_sum:.6e}")
        print(
            "gradients              exact"
            f"  max_abs_diff={max_grad_diff:.1f}"
        )

        # Loss-curve parity
        print(
            f"curve         comparing {args.steps} training steps..."
        )

        curve_generator = torch.Generator(
            device="cuda"
        ).manual_seed(INPUT_SEED)

        batches = [
            torch.randint(
                0,
                vocab_size,
                (args.batch, args.seq),
                generator=curve_generator,
                device="cuda",
            )
            for _ in range(args.steps)
        ]

        streamed_losses = train_curve(
            streamed,
            batches,
        )

        reference_losses = train_curve(
            reference,
            batches,
        )

        curve_max_rel = 0.0

        for index, (got, want) in enumerate(
            zip(
                streamed_losses,
                reference_losses,
                strict=True,
            )
        ):
            curve_max_rel = max(
                curve_max_rel,
                relative_difference(got, want),
            )

            if got != want:
                print(
                    f"ERROR: loss curve diverged at step {index + 1}: "
                    f"streamed={got:.9f} resident={want:.9f}"
                )
                return 1

        print(
            f"curves_equal          true"
            f"  curve_max_rel={curve_max_rel:.1e}"
        )

        # Ensure this really is the streaming model.
        meta_params = sum(
            1
            for parameter in streamed.parameters()
            if getattr(parameter, "is_meta", False)
        )

        if meta_params <= 0:
            print(
                "ERROR: streamed model has no remaining meta parameters; "
                "the streaming path was not exercised"
            )
            return 1

        source = getattr(runtime, "source", None)
        store_bytes = getattr(source, "nbytes", None)

        store_gb = (
            f"{store_bytes / 1e9:.2f} GB"
            if store_bytes is not None
            else "unknown"
        )

        tier = getattr(runtime, "tier", "unknown")
        pinned = getattr(runtime, "pinned", "unknown")

        print(f"meta_params           {meta_params}")
        print(f"store                 {store_gb}")
        print(f"pinned                {pinned}")
        print(f"tier                  {tier}")
        print("RESULT: bit-exact gate passed")

        return 0

    except Exception as exc:
        print(
            f"ERROR: bitexact.py failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
