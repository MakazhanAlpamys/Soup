"""Soup streaming integration; CPU substitutes test wiring, never Triton numerics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig


def config_payload() -> dict:
    return {
        "base": "local-llama",
        "task": "sft",
        "backend": "transformers",
        "data": {"train": "train.jsonl", "chat_template": "chatml"},
        "training": {
            "fast_lora": True,
            "quantization": "none",
            "seed": 792,
            "lora": {"r": 4, "alpha": 4, "dropout": 0},
        },
    }


def _tiny_streaming_base(tmp_path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from tests.test_issue691_sft_packing import _write_tiny_tokenizer

    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
            max_position_embeddings=128,
            use_cache=False,
        )
    )
    base = tmp_path / "model"
    model.save_pretrained(base)
    _write_tiny_tokenizer(str(base))
    return str(base)


def _stream_config(base, output, *, enabled, quantization, source="ram"):
    payload = config_payload()
    payload.update(base=base, output=str(output))
    payload["data"]["max_length"] = 64
    payload["training"].update(
        fast_lora=enabled,
        stream_layers=True,
        stream_buffers=2,
        stream_source=source,
        stream_pin=False,
        stream_disk_kind="nvme" if source == "disk" else None,
        quantization=quantization,
        batch_size=1,
        epochs=1,
        logging_steps=1,
        save_steps=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        warmup_ratio=0,
    )
    payload["training"]["lora"]["target_modules"] = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]
    return SoupConfig.model_validate(payload)


def _payload() -> dict:
    payload = config_payload()
    payload["training"].update(fast_lora=False, use_liger=True, stream_layers=True, batch_size=1)
    return payload


@pytest.mark.parametrize("source", ["ram", "disk"])
@pytest.mark.parametrize("quantization", ["none", "4bit"])
@pytest.mark.parametrize("components", [None, ["rms_norm"], ["swiglu"], ["rms_norm", "swiglu"]])
def test_streaming_liger_config_round_trip(source, quantization, components):
    payload = _payload()
    payload["training"].update(
        stream_source=source, quantization=quantization, liger_components=components
    )
    config = SoupConfig.model_validate(payload)
    restored = SoupConfig.model_validate_json(config.model_dump_json())
    assert restored.training.liger_components == components


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"liger_components": []}, "liger_components"),
        ({"liger_components": ["rope"]}, "liger_components"),
        ({"liger_components": ["cross_entropy"]}, "liger_components"),
        ({"liger_components": ["fused_linear_cross_entropy"]}, "liger_components"),
        ({"liger_components": ["rms_norm", "rms_norm"]}, "duplicate"),
        ({"fast_lora": True}, "fast_lora"),
        ({"use_cut_ce": True}, "use_cut_ce"),
        ({"gradient_checkpointing": "selective"}, "checkpointing"),
    ],
)
def test_streaming_liger_rejects_unsupported_options(updates, match):
    payload = _payload()
    payload["training"].update(updates)
    with pytest.raises(ValidationError, match=match):
        SoupConfig.model_validate(payload)


@pytest.mark.parametrize("use_liger,stream_layers", [(False, True), (True, False)])
def test_component_selection_requires_streaming_liger(use_liger, stream_layers):
    payload = _payload()
    payload["training"].update(
        use_liger=use_liger, stream_layers=stream_layers, liger_components=["rms_norm"]
    )
    with pytest.raises(ValidationError, match="liger_components"):
        SoupConfig.model_validate(payload)


def test_streaming_liger_requires_sft():
    payload = _payload()
    payload["task"] = "dpo"
    with pytest.raises(ValidationError, match="use_liger"):
        SoupConfig.model_validate(payload)


def test_streaming_liger_cpu_refusal_is_explicit():
    from soup_cli.utils.liger_streaming import validate_streaming_liger_environment

    with pytest.raises(ValueError, match="CUDA"):
        validate_streaming_liger_environment("cpu")


def _install_cpu_substitutes(monkeypatch) -> None:
    """Only stand in for the two autograd APIs; use real Soup/PEFT/streaming."""
    import torch

    from soup_cli.utils import liger_streaming

    class ReferenceRMS(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, eps, offset, casting_mode, in_place):
            assert weight is None and offset == 0 and casting_mode == "llama"
            assert in_place is False
            rstd = (hidden.float().square().mean(-1, keepdim=True) + eps).rsqrt()
            ctx.save_for_backward(hidden, rstd)
            return (hidden.float() * rstd).to(hidden.dtype)

        @staticmethod
        def backward(ctx, gradient):
            hidden, rstd = ctx.saved_tensors
            normalized = hidden * rstd
            scaled = gradient
            dx = (scaled - normalized * (scaled * normalized).mean(-1, keepdim=True)) * rstd
            return dx, None, None, None, None, None

    class ReferenceSwiGLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate, up):
            ctx.save_for_backward(gate, up)
            return torch.nn.functional.silu(gate) * up

        @staticmethod
        def backward(ctx, gradient):
            gate, up = ctx.saved_tensors
            return (
                torch.ops.aten.silu_backward(gradient * up, gate),
                gradient * torch.nn.functional.silu(gate),
            )

    monkeypatch.setattr(
        liger_streaming, "validate_streaming_liger_environment", lambda _device: None
    )
    monkeypatch.setattr(liger_streaming, "_load_liger_ops", lambda: (ReferenceRMS, ReferenceSwiGLU))


def _config(base, output, *, enabled, quantization="none", source="ram", components=None):
    payload = _stream_config(
        base, output, enabled=False, quantization=quantization, source=source
    ).model_dump()
    payload["training"].update(use_liger=enabled, liger_components=components if enabled else None)
    return SoupConfig.model_validate(payload)


def _nonuniform_norm_base(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    base = _tiny_streaming_base(tmp_path)
    model = AutoModelForCausalLM.from_pretrained(base)
    with torch.no_grad():
        for idx, layer in enumerate(model.model.layers):
            # Unit norms would hide wrong-layer weight reuse. Distinct weights
            # make recomputing against a recycled pool slot numerically visible.
            layer.input_layernorm.weight.fill_(0.7 + idx * 0.1)
            layer.post_attention_layernorm.weight.fill_(1.3 - idx * 0.1)
        model.model.norm.weight.fill_(1.2)
    model.save_pretrained(base)
    return base


def _save_evidence(precision, source, quantization, components, arm, result, stats):
    """Opt-in raw tensors, saved before cross-arm assertions, including failures."""
    if not os.environ.get("SOUP_LIGER_STREAM_EVIDENCE"):
        return
    from importlib.metadata import PackageNotFoundError, version

    import torch
    from safetensors.torch import save_file

    destination = Path(os.environ["SOUP_LIGER_STREAM_EVIDENCE"])
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{precision or 'cpu-substitute'}-{source}-{quantization}-{'-'.join(components)}-{arm}"
    tensors = {}
    for group, position in (("initial", 0), ("final", 2)):
        for name, value in result[position].items():
            tensors[f"{group}/{name}"] = value.contiguous().clone()
    for step, values in enumerate(result[1]):
        for name, value in values.items():
            tensors[f"gradient/{step}/{name}"] = value.contiguous().clone()
    for step, value in enumerate(result[4]):
        tensors[f"logits/{step}"] = value.contiguous().clone()
    tensors["loss"] = torch.tensor(result[3], dtype=torch.float64)
    save_file(tensors, str(destination / f"{stem}.safetensors"))
    versions = {}
    for package in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "liger-kernel",
        "triton",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    (destination / f"{stem}.json").write_text(
        json.dumps(
            {
                "status": "ARM_CAPTURED_NOT_A_PARITY_VERDICT",
                "precision": precision or "cpu-substitute",
                "source": source,
                "quantization": quantization,
                "components": components,
                "arm": arm,
                "versions": versions,
                "runtime_stats": stats,
                "rtol": 1e-5 if precision is None else 1e-2,
                "atol": 1e-7 if precision is None else 1e-5,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("components", [["rms_norm"], ["swiglu"], ["rms_norm", "swiglu"]])
@pytest.mark.parametrize("source", ["ram", "disk"])
@pytest.mark.parametrize("quantization", ["none", "4bit"])
@pytest.mark.parametrize(
    "precision",
    [
        None,
        pytest.param("fp16", marks=pytest.mark.gpu),
        pytest.param("bf16", marks=pytest.mark.gpu),
    ],
)
def test_soup_liger_streaming_training_and_adapter_round_trip(
    tmp_path, monkeypatch, components, source, quantization, precision
):
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, TrainerCallback

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import layer_shard, layer_stream

    device = "cuda" if precision else "cpu"
    if precision is None:
        _install_cpu_substitutes(monkeypatch)
    else:
        if precision == "bf16":
            assert torch.cuda.is_bf16_supported(including_emulation=False)
        # Exercise native FP16 and BF16 on the same Ampere+ device. Actual
        # kernels and capability probes are never mocked in the GPU arm.
        monkeypatch.setattr(
            layer_stream,
            "resolve_stream_dtype",
            lambda _target: {"fp16": "float16", "bf16": "bfloat16"}[precision],
        )
        monkeypatch.setattr(
            SFTTrainerWrapper,
            "_resolve_mixed_precision",
            lambda *_args: (precision == "bf16", precision == "fp16"),
        )
    if quantization == "4bit":
        pytest.importorskip("bitsandbytes")
    torch.set_num_threads(1)
    base = _nonuniform_norm_base(tmp_path)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda _base: str(tmp_path / "shards"))
    rows = [
        {
            "messages": [
                {"role": "user", "content": "the cat sat on the mat"},
                {"role": "assistant", "content": "hello world"},
            ]
        }
    ] * 4
    results = []
    for enabled in (False, True):
        output = tmp_path / ("liger" if enabled else "ordinary")
        wrapper = SFTTrainerWrapper(
            _config(
                base,
                output,
                enabled=enabled,
                source=source,
                quantization=quantization,
                components=components,
            ),
            device=device,
        )
        try:
            wrapper.setup({"train": rows})
            runtime = wrapper._stream_runtime
            layers = wrapper.model.get_base_model().model.layers
            assert len(layers) == 4 and runtime.pool.n == 2 and runtime.tier == source
            assert all(layer.use_checkpoint for layer in layers)
            assert not getattr(wrapper.trainer.args, "use_liger_kernel", False)
            assert not getattr(wrapper, "_liger_applied", False)
            assert layers[0].inner.input_layernorm.weight.is_meta
            assert layers[0].inner.mlp.gate_proj.get_base_layer().weight.is_meta
            counts = {"rms_norm": 9, "swiglu": 4}
            assert wrapper._streaming_liger_patches == (
                {name: counts[name] for name in components} if enabled else {}
            )
            generator = torch.Generator().manual_seed(123)
            adapters = {
                name: param for name, param in wrapper.model.named_parameters() if "lora_" in name
            }
            with torch.no_grad():
                for name, param in adapters.items():
                    if "lora_B" in name:
                        param.copy_(torch.randn(param.shape, generator=generator) * 0.03)
            initial = {name: param.detach().cpu().clone() for name, param in adapters.items()}
            gradients, logits = [], []
            backward_calls = set()

            def observe(_module, _inputs, result):
                logits.append(result.logits.detach().cpu().clone())
                # Observe the actual autograd nodes without recording a whole
                # profiler trace (Kineto can exhaust memory on Windows).
                pending, visited = [result.logits.grad_fn], set()
                while pending:
                    node = pending.pop()
                    if node is None or node in visited:
                        continue
                    visited.add(node)
                    name = node.name()
                    if any(marker in name for marker in ("Liger", "Reference")):
                        node.register_hook(lambda *_args, name=name: backward_calls.add(name))
                    pending.extend(child for child, _index in node.next_functions)

            handle = wrapper.model.register_forward_hook(observe)

            class Capture(TrainerCallback):
                def on_pre_optimizer_step(self, _args, _state, _control, model=None, **_kwargs):
                    gradients.append(
                        {
                            name: param.grad.detach().cpu().clone()
                            for name, param in model.named_parameters()
                            if "lora_" in name
                        }
                    )
                    assert all(
                        param.grad is None
                        for name, param in model.named_parameters()
                        if "lora_" not in name
                    )

            wrapper.trainer.add_callback(Capture())
            wrapper.trainer.args.report_to = []
            wrapper.trainer.args.disable_tqdm = True
            wrapper.train()
            handle.remove()
            for component, marker in (
                ("rms_norm", "RMS"),
                ("swiglu", "SiLUMul" if precision else "SwiGLU"),
            ):
                assert any(marker in event and "Backward" in event for event in backward_calls) == (
                    enabled and component in components
                )
            assert runtime.stats()["layer_loads"] > 4 * len(rows)
            assert len(gradients) == 2 and len(logits) == 4
            assert all(torch.isfinite(value).all() for step in gradients for value in step.values())
            assert all(torch.isfinite(value).all() for value in logits)
            assert any(
                value.abs().sum() > 0 for name, value in gradients[0].items() if "lora_A" in name
            )
            final = {name: param.detach().cpu().clone() for name, param in adapters.items()}
            assert any(not torch.equal(value, initial[name]) for name, value in final.items())
            for directory in (output, output / "checkpoint-1", output / "checkpoint-2"):
                saved = load_file(str(directory / "adapter_model.safetensors"))
                assert len(saved) == 56 and all(".inner." not in name for name in saved)
            loaded = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(base), str(output), is_trainable=True
            )
            reloaded = {
                name: param.detach().cpu()
                for name, param in loaded.named_parameters()
                if "lora_" in name
            }
            assert set(reloaded) == set(final)
            assert all(torch.equal(value, reloaded[name]) for name, value in final.items())
            losses = [row["loss"] for row in wrapper.trainer.state.log_history if "loss" in row]
            results.append((initial, gradients, final, losses, logits))
            _save_evidence(
                precision,
                source,
                quantization,
                components,
                "liger" if enabled else "ordinary",
                results[-1],
                runtime.stats(),
            )
        finally:
            wrapper._close_stream_runtime()
    assert all(torch.equal(value, results[1][0][name]) for name, value in results[0][0].items())
    rtol, atol = (1e-5, 1e-7) if precision is None else (1e-2, 1e-5)
    for first, second in zip(results[0][1], results[1][1], strict=True):
        for name, value in first.items():
            torch.testing.assert_close(value, second[name], rtol=rtol, atol=atol)
    for name, value in results[0][2].items():
        torch.testing.assert_close(value, results[1][2][name], rtol=rtol, atol=atol)
    for first, second in zip(results[0][4], results[1][4], strict=True):
        torch.testing.assert_close(first, second, rtol=rtol, atol=max(atol, 1e-6))
    assert len(results[0][3]) == len(results[1][3]) == 2
    assert all(
        f"{first:.3f}" == f"{second:.3f}"
        for first, second in zip(results[0][3], results[1][3], strict=True)
    )


def test_streamed_patches_are_local_idempotent_and_preserve_parameters(tmp_path, monkeypatch):
    from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import layer_shard, liger_streaming

    _install_cpu_substitutes(monkeypatch)
    base = _tiny_streaming_base(tmp_path)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda _base: str(tmp_path / "shards"))
    wrapper = SFTTrainerWrapper(_config(base, tmp_path / "output", enabled=False), device="cpu")
    try:
        wrapper._setup_streaming_transformers(wrapper.config, wrapper.config.training)
        model = wrapper.model
        parameters = {name: id(param) for name, param in model.named_parameters()}
        keys = set(model.state_dict())
        expected = {"rms_norm": 9, "swiglu": 4}
        assert liger_streaming.apply_streaming_liger(model, ["rms_norm", "swiglu"]) == expected
        assert liger_streaming.apply_streaming_liger(model, ["swiglu", "rms_norm"]) == expected
        assert {name: id(param) for name, param in model.named_parameters()} == parameters
        assert set(model.state_dict()) == keys
        assert LlamaRMSNorm(32).forward.__func__ is LlamaRMSNorm.forward
        assert LlamaMLP(model.config).forward.__func__ is LlamaMLP.forward
        with pytest.raises(ValueError, match="cannot change"):
            liger_streaming.apply_streaming_liger(model, ["rms_norm"])
    finally:
        wrapper._close_stream_runtime()


def test_failed_liger_install_closes_disk_reader(tmp_path, monkeypatch):
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import layer_shard, liger_streaming

    _install_cpu_substitutes(monkeypatch)
    base = _tiny_streaming_base(tmp_path)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda _base: str(tmp_path / "shards"))
    wrapper = SFTTrainerWrapper(
        _config(base, tmp_path / "output", enabled=True, source="disk"), device="cpu"
    )

    def fail_install(_model, _components):
        raise RuntimeError("deliberate Liger installation failure")

    monkeypatch.setattr(liger_streaming, "apply_streaming_liger", fail_install)
    with pytest.raises(RuntimeError, match="deliberate"):
        wrapper.setup({"train": []})
    runtime = wrapper._stream_runtime
    assert runtime.hook is None
    assert runtime.source._closed and not runtime.source._thread.is_alive()


def test_stream_probe_sees_liger_patches(tmp_path, monkeypatch):
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import layer_shard

    _install_cpu_substitutes(monkeypatch)
    base = _tiny_streaming_base(tmp_path)
    monkeypatch.setattr(layer_shard, "resolve_shard_dir", lambda _base: str(tmp_path / "shards"))
    wrapper = SFTTrainerWrapper(_config(base, tmp_path / "output", enabled=True), device="cpu")
    plan = object()
    monkeypatch.setattr(wrapper, "_stream_budget_lines", lambda *_args, **_kwargs: ([], plan))

    def inspect_probe(model, actual_plan):
        assert actual_plan is plan
        assert model._soup_streaming_liger == (("rms_norm",), {"rms_norm": 9})
        raise RuntimeError("probe reached patched Liger graph")

    monkeypatch.setattr(wrapper, "_run_stream_vram_probe", inspect_probe)
    try:
        with pytest.raises(RuntimeError, match="probe reached"):
            wrapper.setup({"train": []})
    finally:
        wrapper._close_stream_runtime()


@pytest.mark.parametrize("model_type", ["llama", "qwen2"])
def test_streaming_liger_refuses_before_resolving_weights(tmp_path, monkeypatch, model_type):
    from types import SimpleNamespace

    from transformers import AutoConfig

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import spectrum_scan

    base = _tiny_streaming_base(tmp_path)
    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type=model_type),
    )

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("refused streamed Liger configuration reached weight resolution")

    monkeypatch.setattr(spectrum_scan, "resolve_model_weights", unexpected_load)
    wrapper = SFTTrainerWrapper(_config(base, tmp_path / "output", enabled=True), device="cpu")
    with pytest.raises(ValueError, match="CUDA" if model_type == "llama" else "Llama"):
        wrapper.setup({"train": []})


@pytest.mark.parametrize("installed", [None, "0.3.0", "0.8.2", "0.9.0"])
def test_streamed_liger_dependency_floor(monkeypatch, installed):
    import importlib.metadata

    from soup_cli.utils.liger_streaming import validate_streaming_liger_environment

    def package_version(_name):
        if installed is None:
            raise importlib.metadata.PackageNotFoundError("liger-kernel")
        return installed

    monkeypatch.setattr(importlib.metadata, "version", package_version)
    with pytest.raises(ValueError, match="liger-kernel==0.8.1"):
        validate_streaming_liger_environment("cuda")


@pytest.mark.parametrize(
    "target", ["norm_class", "mlp_class", "rope", "model_forward", "cross_entropy"]
)
def test_previous_global_liger_patch_is_refused(monkeypatch, target):
    import torch.nn.functional as functional
    from transformers.models.llama import modeling_llama

    from soup_cli.utils.liger_streaming import _assert_no_global_liger

    def foreign(*_args, **_kwargs):
        raise AssertionError("a global patch must not execute")

    foreign.__module__ = "liger_kernel.test_patch"
    owner, name = {
        "norm_class": (modeling_llama, "LlamaRMSNorm"),
        "mlp_class": (modeling_llama, "LlamaMLP"),
        "rope": (modeling_llama, "apply_rotary_pos_emb"),
        "model_forward": (modeling_llama.LlamaForCausalLM, "forward"),
        "cross_entropy": (functional, "cross_entropy"),
    }[target]
    monkeypatch.setattr(owner, name, foreign)
    with pytest.raises(ValueError, match="global Liger"):
        _assert_no_global_liger()
