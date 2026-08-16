from __future__ import annotations

import math
import types
from pathlib import Path

import pytest
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

    def test_install_streaming_accepts_heterogeneous_layer_sets(self, tmp_path):
        from soup_cli.utils.layer_shard import shard_checkpoint
        from soup_cli.utils.layer_stream_runtime import install_streaming

        weights = _heterogeneous_weights_dir(tmp_path)
        out = str(tmp_path / "shards")
        index = shard_checkpoint(weights, out, dtype="float32", arch="qwen3")
        model = _heterogeneous_meta_model()

        # A process-wide accelerator default must not move the host store off CPU.
        with torch.device("meta"):
            runtime = install_streaming(model, shard_dir=out, index=index, device="cpu")
        try:
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
    def test_stream_setup_threads_moe_targets_into_lora_config(self, monkeypatch, tmp_path):
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
            stream_source="auto",
            stream_buffers=2,
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

        def fake_lora_config(**kwargs):
            captured["target_modules"] = kwargs["target_modules"]
            return types.SimpleNamespace(**kwargs)

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
            lambda *_a, **_k: [{"self_attn.q_proj.weight": ((4, 4), "float32")}] * 2,
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
