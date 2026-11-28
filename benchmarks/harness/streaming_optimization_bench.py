"""Explicit streaming compile/FlashAttention study; never provisions hardware.

The original pretrained capture/comparison and synchronized timing harnesses are
reused unchanged. Each invocation must run in its own fresh Python process.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

ARMS = (
    "baseline", "compile", "flash", "compile+flash", "compile_lora", "compile_lora_unfused",
    "compile_fx_aten",
)


def study_config(config: Any, arm: str) -> Any:
    """Change only the investigated attention switch; reject unrelated methods."""
    from soup_cli.config.schema import SoupConfig

    if arm not in ARMS:
        raise ValueError(f"unknown study arm {arm!r}")
    if config.training.use_liger or config.training.fast_lora:
        raise ValueError("study requires use_liger=false and fast_lora=false")
    if not config.training.stream_layers or config.training.quantization != "none":
        raise ValueError("study requires dense layer streaming")
    payload = config.model_dump()
    payload["training"]["use_flash_attn"] = "flash" in arm
    return SoupConfig.model_validate(payload)


def compiler_counters() -> dict:
    from torch._dynamo.utils import counters

    return {group: dict(counters[group]) for group in ("stats", "frames", "graph_break")}


@contextlib.contextmanager
def study_wrapper(arm: str, provenance: dict, warmup: int | None = None):
    """Install only in this dedicated process and restore the wrapper afterward."""
    import torch
    from transformers import TrainerCallback

    import soup_cli.trainer.sft as sft

    original = sft.SFTTrainerWrapper

    class StudyWrapper(original):
        def setup(self, *args, **kwargs):
            result = super().setup(*args, **kwargs)
            base = self.model.get_base_model()
            selected = base.config._attn_implementation
            provenance["attention_backend"] = selected
            if "flash" in arm and selected not in {"flash_attention_2", "flash_attention_3"}:
                raise ValueError("requested external FlashAttention did not activate")
            if not self._stream_runtime or self._streaming_liger_patches:
                raise ValueError("not an ordinary streamed PEFT model")
            provenance["compiled_decoders"] = 0
            provenance["compiled_lora_linears"] = 0
            provenance["preserve_eager_ops"] = arm == "compile_lora_unfused"
            if arm in {"compile_lora", "compile_lora_unfused"}:
                from benchmarks.harness.streaming_compile_candidate import (
                    compile_streamed_lora_linears,
                )

                count = compile_streamed_lora_linears(
                    self.model, preserve_eager_ops=arm == "compile_lora_unfused",
                )
                provenance["compiled_lora_linears"] = count
                if count != base.config.num_hidden_layers * 7:
                    raise ValueError("compiler did not reach all seven LoRA projections per layer")
            elif "compile" in arm:
                from benchmarks.harness.streaming_compile_candidate import compile_streamed_decoders

                backend = "eager" if arm == "compile_fx_aten" else "inductor"
                provenance["compiler_backend"] = backend
                provenance["compiled_decoders"] = compile_streamed_decoders(
                    self.model, backend=backend,
                )
                if provenance["compiled_decoders"] != base.config.num_hidden_layers:
                    raise ValueError("compiler did not reach all decoder layers")
            provenance["counters_before_train"] = compiler_counters()
            if arm in {"compile_lora", "compile_lora_unfused"}:
                provenance["compiler_recompile_limit"] = torch._dynamo.config.recompile_limit
            if warmup is not None and "compile" in arm:
                class CompileWindow(TrainerCallback):
                    def on_step_begin(self, _args, state, _control, **_kwargs):
                        if state.global_step == warmup:
                            provenance["counters_after_warmup"] = compiler_counters()

                    def on_step_end(self, _args, state, _control, **_kwargs):
                        if state.global_step > warmup:
                            current = compiler_counters()
                            if current != provenance["counters_after_warmup"]:
                                raise ValueError("compilation occurred in the measured window")

                self.trainer.add_callback(CompileWindow())
            provenance["torch"] = torch.__version__
            provenance["matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
            if "compile" in arm:
                import torch._inductor.config as inductor_config

                provenance["inductor_force_same_precision"] = getattr(
                    inductor_config, "force_same_precision", None
                )
                provenance["inductor_emulate_precision_casts"] = (
                    inductor_config.emulate_precision_casts
                )
            return result

    sft.SFTTrainerWrapper = StudyWrapper
    try:
        yield
        provenance["counters_after_train"] = compiler_counters()
        if "compile" in arm:
            before = provenance["counters_before_train"]["stats"].get("unique_graphs", 0)
            after = provenance["counters_after_train"]["stats"].get("unique_graphs", 0)
            if after <= before:
                raise ValueError("compiler arm executed no new compiled graph")
    finally:
        sft.SFTTrainerWrapper = original


def run(config_path: str, arm: str, output: Path, *, capture: bool, warmup: int, steps: int):
    import yaml

    from benchmarks.harness import liger_streaming_bench, liger_streaming_parity
    from soup_cli.config.loader import load_config

    config = study_config(load_config(config_path), arm)
    output.parent.mkdir(parents=True, exist_ok=True)
    frozen_config = output.parent / (output.name + "-input.yaml")
    if frozen_config.exists():
        raise ValueError("refuse to overwrite a previous study input")
    frozen_config.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    provenance = {"arm": arm, "status": "STARTED", "capture": capture}
    try:
        with study_wrapper(arm, provenance, None if capture else warmup):
            if capture:
                liger_streaming_parity.capture(str(frozen_config), "baseline", output)
            else:
                liger_streaming_bench.run(
                    config, arm="baseline", warmup=warmup, steps=steps, output=output
                )
        provenance["status"] = "COMPLETED_NOT_A_PARITY_OR_PERFORMANCE_VERDICT"
    except BaseException as exc:
        provenance.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        output.mkdir(parents=True, exist_ok=True)
        (output / "study-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )


def attention_semantics(config_path: str, arm: str, output: Path) -> None:
    """Check causal/padding isolation at fixed shape; retain training tensors.

    Left padding places masked keys before valid queries, so a missing padding
    mask cannot accidentally pass because of causality alone. Cross-shape output
    differences are recorded separately: SDPA may change numerical kernels there.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import set_seed

    from benchmarks.harness.liger_streaming_bench import build_config
    from benchmarks.harness.liger_streaming_parity import configure_determinism, initialize_adapters
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.utils.seeding import resolve_training_seed

    configure_determinism()
    output.mkdir(parents=True, exist_ok=False)
    config = build_config(
        study_config(load_config(config_path), arm), arm="baseline", output=str(output / "adapter")
    )
    set_seed(resolve_training_seed(config.training))
    provenance, tensors = {"arm": arm, "status": "STARTED"}, {}
    wrapper = None
    try:
        with study_wrapper(arm, provenance):
            from soup_cli.trainer.sft import SFTTrainerWrapper

            wrapper = SFTTrainerWrapper(config, device="cuda", report_to="none")
            wrapper.setup(load_dataset(config.data))
            adapters = initialize_adapters(wrapper.model)
            for name, value in adapters.items():
                tensors["initial/" + name] = value.detach().cpu().clone()
            inputs = (torch.arange(32, device="cuda").reshape(2, 16) % 30) + 3
            mask = torch.ones_like(inputs)
            mask[1, :7] = 0
            inputs[1, :7] = wrapper.tokenizer.pad_token_id
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            labels = inputs.clone().masked_fill(mask == 0, -100)
            batch = {
                "input_ids": inputs, "attention_mask": mask,
                "position_ids": positions, "labels": labels,
            }
            for name, value in batch.items():
                tensors["batch/" + name] = value.cpu()
            wrapper.model.train()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
                ]
            ) as profile:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = wrapper.model(**batch)
                result.loss.backward()
                torch.cuda.synchronize()
            provenance["instrumented_semantics_not_timing"] = True
            provenance["attention_and_compiler_events"] = sorted({
                event.name for event in profile.events()
                if any(marker in event.name.lower() for marker in ("flash", "triton", "compiled"))
            })
            if "flash" in arm and not any(
                marker in name.lower()
                for name in provenance["attention_and_compiler_events"]
                for marker in ("flashattn", "flash_attn", "flash::")
            ):
                raise ValueError("external FlashAttention execution not observed by profiler")
            for name, value in adapters.items():
                if value.grad is None:
                    raise ValueError("missing semantic-check adapter gradient")
                tensors["gradient/" + name] = value.grad.detach().cpu().clone()
            tensors["logits/valid"] = result.logits.detach()[mask.bool()].cpu().clone()
            tensors["loss"] = result.loss.detach().cpu().double().reshape(1)
            expected = result.logits.detach().clone()
            wrapper.model.zero_grad(set_to_none=True)
            changed = inputs.clone()
            changed[0, 12:] = (changed[0, 12:] + 11) % 30 + 3
            changed_padding = inputs.clone()
            changed_padding[1, :7] = torch.arange(7, device="cuda") + 19
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                future = wrapper.model(
                    input_ids=changed, attention_mask=mask, position_ids=positions
                ).logits
                padded = wrapper.model(
                    input_ids=changed_padding, attention_mask=mask, position_ids=positions
                ).logits
                single = wrapper.model(
                    input_ids=inputs[1:2, 7:], attention_mask=mask[1:2, 7:],
                    position_ids=positions[1:2, 7:],
                ).logits
            if not torch.equal(expected[0, :12], future[0, :12]):
                raise ValueError("future tokens changed a causal prefix")
            if not torch.equal(expected[mask.bool()], padded[mask.bool()]):
                raise ValueError("masked padding tokens changed valid-token logits")
            tensors["diagnostic/unpadded_logits"] = single.detach().cpu().clone()
            provenance["cross_shape_diagnostic"] = {
                "passed_same_tolerance": torch.allclose(
                    expected[1, 7:], single[0], rtol=1e-2, atol=1e-5
                ),
                "max_abs": float((expected[1, 7:] - single[0]).abs().max()),
                "admissibility": "diagnostic only; dispatcher can change with shape",
            }
            if not all(torch.isfinite(value).all() for value in tensors.values()):
                raise ValueError("non-finite semantic-check tensor")
            provenance.update(causal_prefix_exact=True, masked_padding_isolation_exact=True)
        provenance["status"] = "CAPTURED"
    except BaseException as exc:
        provenance.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if wrapper is not None:
            wrapper._close_stream_runtime()
        save_file({name: value.contiguous() for name, value in tensors.items()},
                  str(output / "trace.safetensors"))
        (output / "study-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--semantics", action="store_true")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.semantics:
        attention_semantics(args.config, args.arm, args.output)
    else:
        run(
            args.config, args.arm, args.output, capture=args.capture,
            warmup=args.warmup, steps=args.steps,
        )


if __name__ == "__main__":
    main()
