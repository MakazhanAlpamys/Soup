from __future__ import annotations

import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


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
    def test_stream_arch_of_accepts_qwen35_moe_aliases(self):
        from soup_cli.utils.layer_stream import stream_arch_of

        cfg = types.SimpleNamespace(model_type="qwen3_5_moe")
        assert stream_arch_of(cfg) == "qwen3"

        wrapped = types.SimpleNamespace(
            model_type="qwen2_vl",
            text_config=types.SimpleNamespace(model_type="qwen3_5_moe_text"),
        )
        assert stream_arch_of(wrapped) == "qwen3"

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

    def test_install_streaming_accepts_heterogeneous_layer_sets(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import install_streaming

        weights = _heterogeneous_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")
        model = _heterogeneous_meta_model()

        runtime = install_streaming(model, shard_dir=out, index=index, device="cpu")
        try:
            runtime.pool.load_async(0, runtime.source)
            buffers0 = runtime.pool.wait(0)
            assert torch.equal(
                buffers0["self_attn.q_proj.weight"],
                runtime.source.get(0, "self_attn.q_proj.weight"),
            )

            runtime.pool.load_async(1, runtime.source)
            buffers1 = runtime.pool.wait(1)
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
