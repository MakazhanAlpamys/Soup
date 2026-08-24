from __future__ import annotations

import math
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.test_issue502_qwen35_transformers5 import _REQUIRES_TRAINING_FLOOR


def _toy_qwen35_classes():
    """Register a tiny heterogeneous MoE under the real Qwen3.5 model type.

    The alternating attention modules are the property that matters here: a
    layer-0-derived stream spec cannot execute layer 1.  The small MoE is real
    computation rather than naming decoration, so resident-vs-streamed logits
    exercise the complete decoder graph on CPU.
    """
    from torch import nn
    from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutput

    cached = getattr(_toy_qwen35_classes, "_cached", None)
    if cached is not None:
        return cached

    class ToyQwen35Config(PretrainedConfig):
        model_type = "qwen3_5_moe"

        def __init__(
            self,
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            num_experts=2,
            layer_types=None,
            **kwargs,
        ):
            tie_word_embeddings = kwargs.pop("tie_word_embeddings", False)
            super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_experts = num_experts
            self.layer_types = layer_types or ["full_attention", "linear_attention"]

    class ToyMoe(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
            self.experts = nn.ModuleList(
                nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                for _ in range(config.num_experts)
            )

        def forward(self, hidden_states):
            routing = self.gate(hidden_states).softmax(dim=-1)
            expert_outputs = torch.stack(
                [expert(hidden_states) for expert in self.experts], dim=-2
            )
            return (expert_outputs * routing.unsqueeze(-1)).sum(dim=-2)

    class ToyFullAttentionLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.mlp = ToyMoe(config)

        def forward(self, hidden_states, *args, **kwargs):
            del args, kwargs
            return hidden_states + torch.tanh(
                self.self_attn.q_proj(hidden_states)
            ) + self.mlp(hidden_states)

    class ToyLinearAttentionLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj_qkv = nn.Linear(
                config.hidden_size, config.hidden_size, bias=False
            )
            self.mlp = ToyMoe(config)

        def forward(self, hidden_states, *args, **kwargs):
            del args, kwargs
            return hidden_states + torch.sigmoid(
                self.linear_attn.in_proj_qkv(hidden_states)
            ) + self.mlp(hidden_states)

    class ToyDecoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList(
                ToyFullAttentionLayer(config)
                if layer_type == "full_attention"
                else ToyLinearAttentionLayer(config)
                for layer_type in config.layer_types
            )
            self.norm = nn.LayerNorm(config.hidden_size)

        def forward(self, input_ids):
            hidden_states = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return self.norm(hidden_states)

    class ToyQwen35ForCausalLM(PreTrainedModel):
        config_class = ToyQwen35Config
        base_model_prefix = "model"

        def __init__(self, config):
            super().__init__(config)
            self.model = ToyDecoder(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        def _init_weights(self, module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, value):
            self.lm_head = value

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            del kwargs
            return {"input_ids": input_ids}

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            del kwargs
            if input_ids is None:
                if inputs_embeds is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = inputs_embeds
                for layer in self.model.layers:
                    hidden_states = layer(hidden_states)
                hidden_states = self.model.norm(hidden_states)
            else:
                hidden_states = self.model(input_ids)
            return CausalLMOutput(logits=self.lm_head(hidden_states))

    AutoModelForCausalLM.register(
        ToyQwen35Config, ToyQwen35ForCausalLM, exist_ok=True
    )
    result = ToyQwen35Config, ToyQwen35ForCausalLM
    _toy_qwen35_classes._cached = result
    return result


def _tiny_qwen35_lora():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "in_proj_qkv"],
        task_type=TaskType.CAUSAL_LM,
    )


def _copy_toy_lora(src, dst):
    def canonical(key):
        return key.replace(".inner.", ".")

    src_lora = {canonical(k): v for k, v in src.state_dict().items() if "lora_" in k}
    dst_lora = {canonical(k): v for k, v in dst.state_dict().items() if "lora_" in k}
    assert src_lora and set(src_lora) == set(dst_lora)
    with torch.no_grad():
        for key, value in src_lora.items():
            dst_lora[key].copy_(value)


def _heterogeneous_weights_dir(tmp_path: Path, *, vlm_prefix: bool = False) -> str:
    weights = tmp_path / "weights"
    weights.mkdir()
    prefix = "model.language_model." if vlm_prefix else "model."
    save_file(
        {
            prefix + "layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            prefix + "layers.0.mlp.gate_proj.weight": torch.randn(4, 4),
            prefix + "layers.1.linear_attn.in_proj_qkv.weight": torch.randn(4, 4),
            prefix + "layers.1.mlp.gate_proj.weight": torch.randn(4, 4),
            prefix + "embed_tokens.weight": torch.randn(8, 4),
        },
        str(weights / "model.safetensors"),
    )
    return str(weights)


def _heterogeneous_meta_model():
    import torch.nn as nn

    class _MetaLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 4, device="meta"))

    class _Layer0(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = _MetaLinear()
            self.mlp = nn.Module()
            self.mlp.gate_proj = _MetaLinear()

    class _Layer1(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = nn.Module()
            self.linear_attn.in_proj_qkv = _MetaLinear()
            self.mlp = nn.Module()
            self.mlp.gate_proj = _MetaLinear()

    class _Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer0(), _Layer1()])

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Decoder()

    return _Model()


class TestQwen35StreamingAliases:
    def test_stream_arch_of_accepts_qwen35_dense_text_aliases(self):
        from soup_cli.utils.layer_stream import stream_arch_of

        text = types.SimpleNamespace(model_type="qwen3_5_text")
        assert stream_arch_of(text) == "qwen3"

        wrapped = types.SimpleNamespace(
            model_type="qwen3_5",
            text_config=text,
        )
        assert stream_arch_of(wrapped) == "qwen3"

    def test_stream_arch_of_accepts_qwen35_moe_aliases(self):
        from soup_cli.utils.layer_stream import stream_arch_of

        cfg = types.SimpleNamespace(model_type="qwen3_5_moe")
        assert stream_arch_of(cfg) == "qwen3"

        wrapped = types.SimpleNamespace(
            model_type="qwen2_vl",
            text_config=types.SimpleNamespace(model_type="qwen3_5_moe_text"),
        )
        assert stream_arch_of(wrapped) == "qwen3"

    def test_multimodal_gemma3_wrapper_is_still_rejected(self):
        from soup_cli.utils.layer_stream import stream_arch_of

        wrapped = types.SimpleNamespace(
            model_type="gemma3",
            text_config=types.SimpleNamespace(model_type="gemma3_text"),
        )
        with pytest.raises(ValueError, match="gemma3"):
            stream_arch_of(wrapped)

    def test_stream_setup_reads_text_subconfig_and_moe_intermediate_size(self):
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        class _Wrapper(StreamingSetupMixin):
            pass

        wrapper = _Wrapper()
        cfg = types.SimpleNamespace(
            text_config=types.SimpleNamespace(
                hidden_size=4096,
                num_hidden_layers=48,
                vocab_size=151936,
                moe_intermediate_size=1536,
                num_experts_per_tok=8,
                shared_expert_intermediate_size=2048,
            )
        )
        assert wrapper._stream_shape_config(cfg) is cfg.text_config
        assert wrapper._stream_intermediate_size(cfg.text_config) == 1536 * 8 + 2048


class TestQwen35StreamingParity:
    @_REQUIRES_TRAINING_FLOOR
    def test_dense_text_decoder_logits_match_resident_bit_exactly(self, tmp_path):
        """Admit the real dense Qwen3.5 decoder graph, not just its model-type name.

        Qwen3.8-27B alternates linear-attention and full-attention blocks. This
        native Transformers fixture preserves both layer kinds while shrinking
        every dimension enough for a CPU parity control.
        """
        from peft import get_peft_model
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream import stream_arch_of
        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        config = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=128,
        )
        torch.manual_seed(31)
        resident = Qwen3_5ForCausalLM(config).to(torch.float32).eval()
        weights = tmp_path / "weights"
        resident.save_pretrained(weights, safe_serialization=True)

        arch = stream_arch_of(config)
        assert arch == "qwen3"
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(str(weights), shards, dtype="float32", arch=arch)

        streamed, runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=shards,
            index=index,
            lora_config=_tiny_qwen35_lora(),
            device="cpu",
            dtype="float32",
            buffers=2,
            pin=False,
            seed=37,
        )
        try:
            reference = get_peft_model(resident, _tiny_qwen35_lora())
            _copy_toy_lora(streamed, reference)
            streamed.eval()
            reference.eval()
            input_ids = torch.tensor([[1, 7, 3, 11, 5, 2]])
            with torch.no_grad():
                got = streamed(input_ids=input_ids).logits
                want = reference(input_ids=input_ids).logits
            assert torch.equal(got, want), (got - want).abs().max().item()
        finally:
            runtime.close()

    def test_heterogeneous_moe_logits_match_resident_bit_exactly(
        self, tmp_path, monkeypatch
    ):
        """Admission gate: exercise the alias through the complete CPU runtime.

        Layer 0 owns ``self_attn.q_proj`` while layer 1 owns
        ``linear_attn.in_proj_qkv``.  Reusing layer 0's stream spec therefore
        cannot pass this test, and equal logits prove more than buffer-copy
        fidelity: the substituted weights execute in the same decoder kernels
        as the resident reference.
        """
        from peft import get_peft_model
        from transformers import AutoConfig

        from soup_cli.utils.layer_shard import layer_shard_path, shard_checkpoint
        from soup_cli.utils.layer_stream import stream_arch_of
        from soup_cli.utils.layer_stream_runtime import build_streamed_model

        config_cls, model_cls = _toy_qwen35_classes()
        torch.manual_seed(23)
        config = config_cls()
        resident = model_cls(config).to(torch.float32).eval()
        weights = tmp_path / "weights"
        resident.save_pretrained(weights, safe_serialization=True)

        # Avoid registering the official Qwen3.5 model type globally.  That
        # would replace Transformers' native config in newer environments and
        # make this test order-dependent.  The runtime still receives a config
        # whose real model_type exercises the production alias.
        monkeypatch.setattr(
            AutoConfig,
            "from_pretrained",
            staticmethod(
                lambda model_id, **kwargs: config_cls.from_pretrained(model_id)
            ),
        )

        arch = stream_arch_of(config)
        assert arch == "qwen3"
        shards = str(tmp_path / "shards")
        index = shard_checkpoint(str(weights), shards, dtype="float32", arch=arch)
        layer0 = load_file(layer_shard_path(shards, 0))
        layer1 = load_file(layer_shard_path(shards, 1))
        assert "self_attn.q_proj.weight" in layer0
        assert "self_attn.q_proj.weight" not in layer1
        assert "linear_attn.in_proj_qkv.weight" in layer1
        assert "linear_attn.in_proj_qkv.weight" not in layer0

        streamed, runtime = build_streamed_model(
            model_id=str(weights),
            shard_dir=shards,
            index=index,
            lora_config=_tiny_qwen35_lora(),
            device="cpu",
            dtype="float32",
            buffers=2,
            pin=False,
            seed=29,
        )
        try:
            reference = get_peft_model(resident, _tiny_qwen35_lora())
            _copy_toy_lora(streamed, reference)
            streamed.eval()
            reference.eval()
            input_ids = torch.tensor([[1, 7, 3, 11, 5, 2]])
            with torch.no_grad():
                got = streamed(input_ids=input_ids).logits
                want = reference(input_ids=input_ids).logits
            assert torch.equal(got, want), (got - want).abs().max().item()
        finally:
            runtime.close()


class TestQwen35StreamingSharder:
    def test_sharder_accepts_heterogeneous_layers_and_canonicalises_vlm_prefix(self, tmp_path):
        from soup_cli.utils.layer_shard import (
            extras_shard_path,
            layer_shard_path,
            shard_checkpoint,
        )

        weights = _heterogeneous_weights_dir(tmp_path, vlm_prefix=True)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")

        assert "self_attn.q_proj.weight" in index.layer_keys
        assert "linear_attn.in_proj_qkv.weight" in index.layer_keys

        layer0 = load_file(layer_shard_path(out, 0))
        layer1 = load_file(layer_shard_path(out, 1))
        extras = load_file(extras_shard_path(out))

        assert "self_attn.q_proj.weight" in layer0
        assert "linear_attn.in_proj_qkv.weight" not in layer0
        assert "linear_attn.in_proj_qkv.weight" in layer1
        assert "self_attn.q_proj.weight" not in layer1
        assert "model.embed_tokens.weight" in extras

    def test_sharder_refuses_canonical_key_collisions(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint

        weights = tmp_path / "weights"
        weights.mkdir()
        save_file(
            {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
                "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
                "model.embed_tokens.weight": torch.randn(8, 4),
            },
            str(weights / "model.safetensors"),
        )

        with pytest.raises(ValueError, match="canonicalise"):
            shard_checkpoint(str(weights), str(tmp_path / "shards"), dtype="float32")

    def test_sharder_refuses_conflicting_layouts_for_a_shared_key(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint

        weights = tmp_path / "weights"
        weights.mkdir()
        save_file(
            {
                "model.layers.0.mlp.gate_proj.weight": torch.randn(4, 4),
                "model.layers.1.mlp.gate_proj.weight": torch.randn(4, 8),
                "model.embed_tokens.weight": torch.randn(8, 4),
            },
            str(weights / "model.safetensors"),
        )

        with pytest.raises(ValueError, match="stored shapes or dtypes"):
            shard_checkpoint(str(weights), str(tmp_path / "shards"), dtype="float32")


class TestQwen35StreamingRuntime:
    def test_quantised_layer_suffixes_union_all_decoder_variants(self, monkeypatch):
        import sys

        import torch.nn as nn

        bnb = types.ModuleType("bitsandbytes")
        bnb.nn = types.SimpleNamespace(Params4bit=nn.Parameter)
        monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)

        class _QuantLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(4, 4))

        class _Layer0(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Module()
                self.self_attn.q_proj = _QuantLinear()

        class _Layer1(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = nn.Module()
                self.linear_attn.in_proj_qkv = _QuantLinear()

        class _Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Layer0(), _Layer1()])

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _Decoder()

        from soup_cli.utils.layer_stream_runtime import quantised_layer_suffixes

        assert quantised_layer_suffixes(_Model()) == frozenset(
            {"self_attn.q_proj.weight", "linear_attn.in_proj_qkv.weight"}
        )

    def test_ram_source_explicitly_requests_cpu_allocations(self, tmp_path, monkeypatch):
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import RamSource

        weights = _heterogeneous_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")
        layer_specs = RamSource.layer_specs_from_shards(out, index.n_layers)

        requested_devices = []
        real_empty = torch.empty

        def _cpu_empty(*args, **kwargs):
            if "pin_memory" not in kwargs:
                return real_empty(*args, **kwargs)
            requested_devices.append(kwargs.get("device"))
            return torch.zeros(
                *args,
                dtype=kwargs.get("dtype"),
                device="cpu",
                pin_memory=kwargs.get("pin_memory", False),
            )

        monkeypatch.setattr(torch, "empty", _cpu_empty)
        source = RamSource(out, index.n_layers, layer_specs, pin=False)

        assert requested_devices
        assert all(str(device) == "cpu" for device in requested_devices)
        assert all(
            tensor.device.type == "cpu"
            for layer in source.store
            for tensor in layer.values()
        )

    @pytest.mark.parametrize("device_type", ["meta", "mps"])
    def test_ram_source_refuses_a_non_cpu_allocation(
        self, tmp_path, monkeypatch, device_type
    ):
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import RamSource

        weights = _heterogeneous_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")
        layer_specs = RamSource.layer_specs_from_shards(out, index.n_layers)
        def _misplaced_empty(*args, **kwargs):
            del args
            if kwargs.get("pin_memory"):
                return types.SimpleNamespace(device=torch.device(device_type))
            raise AssertionError("RamSource allocations should request pin_memory")

        monkeypatch.setattr(torch, "empty", _misplaced_empty)
        with pytest.raises(RuntimeError, match=rf"requested a CPU tensor.*{device_type}"):
            RamSource(out, index.n_layers, layer_specs, pin=True)

    def test_ram_source_refuses_pageable_memory_when_pinning_was_requested(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import RamSource

        weights = _heterogeneous_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")
        layer_specs = RamSource.layer_specs_from_shards(out, index.n_layers)
        real_empty = torch.empty

        def _pageable_empty(*args, **kwargs):
            allocation = dict(kwargs)
            allocation["pin_memory"] = False
            return real_empty(*args, **allocation)

        monkeypatch.setattr(torch, "empty", _pageable_empty)
        with pytest.raises(RuntimeError, match="returned pageable memory"):
            RamSource(out, index.n_layers, layer_specs, pin=True)

    def test_mps_gate_disables_pinning_for_direct_callers(
        self, tmp_path, monkeypatch
    ):
        import soup_cli.utils.layer_stream_runtime as runtime_module
        from soup_cli.utils.layer_shard import shard_checkpoint

        class _SourceCapturedError(Exception):
            pass

        captured = {}
        messages = []

        class _RecordingConsole:
            def print(self, *args, **_kwargs):
                messages.append(" ".join(str(arg) for arg in args))

        def _capture_source(*args, **kwargs):
            captured["pin"] = args[3]
            captured["require_pin"] = kwargs["require_pin"]
            raise _SourceCapturedError

        weights = _heterogeneous_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")
        monkeypatch.setattr(runtime_module, "_build_source", _capture_source)

        with pytest.raises(_SourceCapturedError):
            runtime_module.install_streaming(
                _heterogeneous_meta_model(),
                shard_dir=out,
                index=index,
                device="mps",
                pin=True,
                require_pin=True,
                console=_RecordingConsole(),
            )

        assert captured == {"pin": False, "require_pin": False}
        assert any("require_pin=True" in message for message in messages)
        assert any("pageable CPU source" in message for message in messages)

    @pytest.mark.skipif(
        not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        reason="needs Apple Silicon MPS",
    )
    def test_mps_target_keeps_the_host_store_on_cpu(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import install_streaming

        weights = _heterogeneous_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")
        runtime = install_streaming(
            _heterogeneous_meta_model(),
            shard_dir=out,
            index=index,
            device="mps",
            pin=True,
            require_pin=True,
        )
        try:
            assert runtime.pinned is False
            assert all(
                tensor.device.type == "cpu"
                for layer in runtime.source.store
                for tensor in layer.values()
            )
            assert all(
                tensor.device.type == "mps"
                for slot in runtime.pool.buffers
                for tensor in slot.values()
            )
            runtime.pool.load_async(0, runtime.source)
            copied = runtime.pool.wait(0)["self_attn.q_proj.weight"].cpu()
            source = runtime.source.get(0, "self_attn.q_proj.weight")
            assert torch.equal(copied, source)
        finally:
            runtime.close()

    def test_install_streaming_accepts_heterogeneous_layer_sets(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import install_streaming

        weights = _heterogeneous_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")
        model = _heterogeneous_meta_model()

        runtime = install_streaming(
            model, shard_dir=out, index=index, device="cpu", pin=False
        )
        try:
            for layer in runtime.source.store:
                for tensor in layer.values():
                    assert tensor.device.type == "cpu"
            runtime.pool.load_async(0, runtime.source)
            buffers0 = runtime.pool.wait(0)
            assert runtime.source.get(0, "self_attn.q_proj.weight").device.type == "cpu"
            assert buffers0["self_attn.q_proj.weight"].device.type == "cpu"
            assert torch.equal(
                buffers0["self_attn.q_proj.weight"],
                runtime.source.get(0, "self_attn.q_proj.weight"),
            )

            runtime.pool.load_async(1, runtime.source)
            buffers1 = runtime.pool.wait(1)
            assert runtime.source.get(1, "linear_attn.in_proj_qkv.weight").device.type == "cpu"
            assert buffers1["linear_attn.in_proj_qkv.weight"].device.type == "cpu"
            assert torch.equal(
                buffers1["linear_attn.in_proj_qkv.weight"],
                runtime.source.get(1, "linear_attn.in_proj_qkv.weight"),
            )
        finally:
            runtime.close()

    def test_build_stream_plan_accepts_actual_store_bytes_override(self):
        from soup_cli.utils.layer_stream import TIER_RAM, build_stream_plan

        plan = build_stream_plan(
            arch="qwen3",
            n_layers=2,
            layer_bytes=10,
            store_bytes=13,
            embed_bytes=0,
            available_ram_bytes=100,
            pinned_limit_bytes=None,
            buffers=2,
        )

        assert plan.tier == TIER_RAM
        assert plan.store_bytes == 13

    def test_merge_layer_specs_refuses_conflicting_shared_layouts(self):
        from soup_cli.utils.layer_stream_runtime import RamSource

        with pytest.raises(ValueError, match="stored shapes or dtypes"):
            RamSource.merge_layer_specs(
                [
                    {"mlp.gate_proj.weight": ((4, 4), "float32")},
                    {"mlp.gate_proj.weight": ((4, 8), "float32")},
                ]
            )

    def test_stream_layer_budget_bytes_uses_the_union_pool_spec(self, tmp_path):
        from soup_cli.trainer.stream_setup import StreamingSetupMixin
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream import dtype_bytes
        from soup_cli.utils.layer_stream_runtime import RamSource

        weights = _heterogeneous_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")
        layer_specs = RamSource.layer_specs_from_shards(out, index.n_layers)

        budget = StreamingSetupMixin._stream_layer_budget_bytes(layer_specs)
        union = RamSource.merge_layer_specs(layer_specs)
        actual = sum(
            math.prod(shape) * dtype_bytes(stored) for shape, stored in union.values()
        )
        per_layer = [
            sum(math.prod(shape) * dtype_bytes(stored) for shape, stored in spec.values())
            for spec in layer_specs
        ]

        assert budget == actual
        assert budget > max(per_layer)


class TestQwen35StreamingSetup:
    def test_stream_setup_threads_moe_targets_and_union_budget(self, monkeypatch, tmp_path):
        from soup_cli.trainer.stream_setup import StreamingSetupMixin

        class _Wrapper(StreamingSetupMixin):
            def __init__(self):
                self.device = "cpu"
                self._trust_remote_code = False

            def _stream_budget_lines(self, *_args, **_kwargs):
                return (), None

        cfg = types.SimpleNamespace(
            base="Qwen/Qwen3.5-35B-A3B",
            data=types.SimpleNamespace(max_length=64),
        )
        tcfg = types.SimpleNamespace(
            quantization="none",
            double_quant_on=True,
            stream_source="auto",
            stream_buffers=2,
            stream_disk_kind=None,
            stream_pin=None,
            seed=7,
            moe_lora=True,
            batch_size=1,
            gradient_accumulation_steps=1,
            stream_vram_probe=False,
            stream_vram_override=None,
            lora=types.SimpleNamespace(
                r=8,
                alpha=16,
                dropout=0.0,
                target_modules="auto",
                use_dora=False,
                use_rslora=False,
            ),
        )
        model_cfg = types.SimpleNamespace(
            model_type="qwen2_vl",
            text_config=types.SimpleNamespace(
                model_type="qwen3_5_moe_text",
                hidden_size=64,
                num_hidden_layers=2,
                vocab_size=128,
                moe_intermediate_size=32,
                num_experts_per_tok=2,
                num_experts=16,
            ),
        )
        index = types.SimpleNamespace(n_layers=2, total_params=100, quant="none", quant_specs={})
        runtime = types.SimpleNamespace(
            stats=lambda: {
                "tier": "ram",
                "store_bytes": 0,
                "pinned": False,
                "n_layers": 2,
                "buffers": 2,
                "buffer_bytes": 8,
            }
        )
        tokenizer = types.SimpleNamespace(pad_token=None, eos_token="</s>")
        captured = {}
        layer_specs = [
            {"self_attn.q_proj.weight": ((4, 4), "float32")},
            {"linear_attn.in_proj_qkv.weight": ((4, 4), "float32")},
        ]

        def fake_lora_config(**kwargs):
            captured["target_modules"] = kwargs["target_modules"]
            return types.SimpleNamespace(**kwargs)

        from soup_cli.utils.layer_stream import build_stream_plan as real_build_stream_plan

        def capture_stream_plan(**kwargs):
            captured["layer_bytes"] = kwargs["layer_bytes"]
            return real_build_stream_plan(**kwargs)

        monkeypatch.setattr(
            "transformers.AutoTokenizer.from_pretrained",
            lambda *_a, **_k: tokenizer,
        )
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda *_a, **_k: model_cfg,
        )
        monkeypatch.setattr("peft.LoraConfig", fake_lora_config)
        monkeypatch.setattr(
            "soup_cli.utils.layer_shard.resolve_shard_dir",
            lambda *_a, **_k: str(tmp_path / "shards"),
        )
        monkeypatch.setattr(
            "soup_cli.utils.layer_shard.shard_checkpoint",
            lambda *_a, **_k: index,
        )
        monkeypatch.setattr(
            "soup_cli.utils.layer_shard.source_weight_bytes",
            lambda *_a, **_k: 1024,
        )
        monkeypatch.setattr(
            "soup_cli.utils.spectrum_scan.resolve_model_weights",
            lambda *_a, **_k: str(tmp_path / "weights"),
        )
        monkeypatch.setattr("soup_cli.utils.layer_stream.free_ram_bytes", lambda: 1_000_000)
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream_runtime.build_meta_skeleton",
            lambda *_a, **_k: types.SimpleNamespace(),
        )
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream_runtime.RamSource.layer_specs_from_shards",
            lambda *_a, **_k: layer_specs,
        )
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream.build_stream_plan", capture_stream_plan
        )
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream_runtime.extras_resident_bytes",
            lambda *_a, **_k: 0,
        )
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream_runtime.build_streamed_model",
            lambda **_kwargs: (types.SimpleNamespace(), runtime),
        )
        monkeypatch.setattr(
            "soup_cli.utils.moe.detect_moe_model",
            lambda *_a, **_k: True,
        )
        monkeypatch.setattr(
            "soup_cli.utils.moe.get_moe_target_modules",
            lambda *_a, **_k: [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream.render_stream_panel",
            lambda *_a, **_k: "panel",
        )
        monkeypatch.setattr(
            "soup_cli.utils.layer_stream_runtime.expandable_segments_status",
            lambda: (True, ""),
        )

        wrapper = _Wrapper()
        wrapper._setup_streaming_transformers(cfg, tcfg)

        assert captured["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        assert captured["layer_bytes"] == 128
