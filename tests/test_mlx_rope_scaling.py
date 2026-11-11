"""Native MLX RoPE scaling tests: config gates, numeric oracle, and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.mlx_rope import resolve_mlx_rope_scaling


def _refusal(yaml_str: str) -> str:
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(yaml_str)
    return " ".join(str(excinfo.value).split())


# ---------------------------------------------------------------------------
# 1. Portable Configuration and Schema Refusal Tests (No MLX package needed)
# ---------------------------------------------------------------------------


class TestMLXRoPESchemaGates:
    """Verify that schema.py strictly enforces MLX RoPE scaling boundaries."""

    def test_sft_mlx_linear_unquantized_loads(self):
        yaml_str = """
base: org/model
task: sft
backend: mlx
data:
  train: ./train.jsonl
  max_length: 512
training:
  quantization: none
  rope_scaling_type: linear
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.backend == "mlx"
        assert cfg.task == "sft"
        assert cfg.training.rope_scaling_type == "linear"
        assert cfg.training.quantization == "none"

    @pytest.mark.parametrize("rope_type", ["dynamic", "yarn", "llama3"])
    def test_sft_mlx_refuses_non_linear_types(self, rope_type):
        yaml_str = f"""
base: org/model
task: sft
backend: mlx
data:
  train: ./train.jsonl
training:
  quantization: none
  rope_scaling_type: {rope_type}
"""
        err = _refusal(yaml_str)
        assert (
            f"MLX text SFT currently supports rope_scaling_type='linear' only (got '{rope_type}')"
            in err
        )
        assert "Use rope_scaling_type: linear and training.quantization: none" in err

    @pytest.mark.parametrize("quant", ["4bit", "8bit"])
    def test_sft_mlx_refuses_quantization_with_rope_scaling(self, quant):
        yaml_str = f"""
base: org/model
task: sft
backend: mlx
data:
  train: ./train.jsonl
training:
  quantization: {quant}
  rope_scaling_type: linear
"""
        err = _refusal(yaml_str)
        assert (
            f"training.rope_scaling_type with backend='mlx' "
            f"requires training.quantization='none' (got '{quant}')"
            in err
        )
        assert (
            "Use training.quantization: none, or backend: transformers, or remove the key." in err
        )

    @pytest.mark.parametrize("modality", ["vision", "audio"])
    def test_sft_mlx_refuses_non_text_modalities(self, modality):
        yaml_str = f"""
base: org/model
task: sft
backend: mlx
modality: {modality}
data:
  train: ./train.jsonl
training:
  quantization: none
  rope_scaling_type: linear
"""
        err = _refusal(yaml_str)
        assert f"modality={modality!r}" in err
        assert "RoPE is rescaled only" in err


# ---------------------------------------------------------------------------
# 2. Checkpoint Inspection & Validation Tests (`resolve_mlx_rope_scaling`)
# ---------------------------------------------------------------------------


class TestResolveMLXRoPEScaling:
    """Validate checkpoint original context length, pre-scaled refusals, and factors."""

    def test_unset_rope_scaling_returns_none(self):
        rope_cfg, model_cfg = resolve_mlx_rope_scaling(
            {"max_position_embeddings": 2048}, 4096, None
        )
        assert rope_cfg is None
        assert model_cfg is None

    @pytest.mark.parametrize(
        "invalid_len",
        [None, True, False, -10, 0, "4096", 2048.5],
    )
    def test_missing_or_invalid_max_position_embeddings_refused(self, invalid_len):
        config = {}
        if invalid_len is not None:
            config["max_position_embeddings"] = invalid_len
        with pytest.raises(ValueError, match="original context length .* is missing or invalid"):
            resolve_mlx_rope_scaling(config, 4096, "linear")

    def test_checkpoint_already_scaled_is_refused(self):
        config = {
            "max_position_embeddings": 4096,
            "rope_scaling": {"type": "linear", "factor": 2.0},
        }
        with pytest.raises(ValueError, match="already carries RoPE scaling"):
            resolve_mlx_rope_scaling(config, 8192, "linear")

    def test_checkpoint_with_nested_per_layer_rope_is_refused(self):
        config = {
            "max_position_embeddings": 4096,
            "rope_scaling": {"layer_0": {"type": "linear"}},
        }
        with pytest.raises(ValueError, match="has per-layer or nested RoPE configuration"):
            resolve_mlx_rope_scaling(config, 8192, "linear")

    def test_no_extension_when_target_at_or_below_original_length(self):
        config = {"max_position_embeddings": 2048}
        # At boundary: target == original
        rope_cfg, model_cfg = resolve_mlx_rope_scaling(config, 2048, "linear")
        assert rope_cfg is None
        assert model_cfg is None

        # Below boundary: target < original
        rope_cfg, model_cfg = resolve_mlx_rope_scaling(config, 1024, "linear")
        assert rope_cfg is None
        assert model_cfg is None

    def test_extension_calculates_exact_factor(self):
        config = {"max_position_embeddings": 2048}
        rope_cfg, model_cfg = resolve_mlx_rope_scaling(config, 4096, "linear")
        assert rope_cfg == {
            "type": "linear",
            "factor": 2.0,
            "original_max_position_embeddings": 2048,
        }
        assert model_cfg == {
            "rope_scaling": {"type": "linear", "factor": 2.0},
            "max_position_embeddings": 4096,
        }


# ---------------------------------------------------------------------------
# 3. Adapter Metadata Validation Tests (`read_native_adapter`)
# ---------------------------------------------------------------------------


class TestReadNativeAdapterRoPEMetadata:
    def test_validates_malformed_rope_scaling_in_adapter(self, tmp_path):
        from soup_cli.utils.mlx_generation import read_native_adapter

        base = tmp_path / "base"
        base.mkdir()
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapters.safetensors").write_bytes(b"dummy")

        def _write(rope_meta):
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "model": str(base),
                        "fine_tune_type": "lora",
                        "num_layers": 1,
                        "lora_parameters": {
                            "rank": 2,
                            "scale": 1.0,
                            "dropout": 0.0,
                            "keys": ["q_proj"],
                        },
                        "rope_scaling": rope_meta,
                    }
                )
            )

        # Non-dict
        _write("linear")
        with pytest.raises(ValueError, match="rope_scaling must be a dictionary"):
            read_native_adapter(adapter)

        # Non-linear
        _write({"type": "dynamic", "factor": 2.0, "original_max_position_embeddings": 64})
        with pytest.raises(ValueError, match="supports linear RoPE scaling only"):
            read_native_adapter(adapter)

        # Factor <= 1.0
        _write({"type": "linear", "factor": 1.0, "original_max_position_embeddings": 64})
        with pytest.raises(ValueError, match="factor must be a finite number > 1.0"):
            read_native_adapter(adapter)

        # Invalid original_max_position_embeddings
        _write({"type": "linear", "factor": 2.0, "original_max_position_embeddings": -5})
        with pytest.raises(
            ValueError, match="original_max_position_embeddings must be a positive integer"
        ):
            read_native_adapter(adapter)


# ---------------------------------------------------------------------------
# 4. Independent Mathematical Numeric Oracle & Mutation Verification
# ---------------------------------------------------------------------------


class TestRoPENumericOracle:
    """Independent mathematical verification for linear RoPE position scaling."""

    @pytest.fixture(autouse=True)
    def _require_mlx(self):
        pytest.importorskip("mlx.core")
        pytest.importorskip("mlx.nn")

    def test_numeric_oracle_matches_mlx_fast_rope_at_all_positions(self):
        import mlx.core as mx
        import mlx.nn as nn

        dims = 32
        base_theta = 10000.0
        factor = 2.5
        scale = 1.0 / factor

        # Test positions: below boundary (10), at boundary (64), and beyond (120)
        positions = [10, 64, 120]

        half = dims // 2
        inv_freq = 1.0 / (base_theta ** (np.arange(0, dims, 2)[:half] / dims))
        rope = nn.RoPE(dims=dims, traditional=False, base=base_theta, scale=scale)

        np.random.seed(42)
        for pos in positions:
            x_np = np.random.randn(1, 1, dims).astype(np.float32)
            x_mx = mx.array(x_np)

            # Native MLX RoPE with linear scale at offset=pos
            out_mlx = np.array(rope(x_mx, offset=pos))

            # Independent Mathematical Oracle:
            # Scaled angle: (pos * scale) * inv_freq == (pos / factor) * inv_freq
            angles = (pos * scale) * inv_freq
            cos_angles = np.cos(angles)[None, None, :]
            sin_angles = np.sin(angles)[None, None, :]

            x1 = x_np[..., :half]
            x2 = x_np[..., half:]

            oracle_1 = x1 * cos_angles - x2 * sin_angles
            oracle_2 = x1 * sin_angles + x2 * cos_angles
            oracle_out = np.concatenate([oracle_1, oracle_2], axis=-1)

            diff = np.max(np.abs(out_mlx - oracle_out))
            assert diff < 1e-5, f"Oracle mismatch at pos {pos}: max diff {diff}"

    def test_mutation_unscaled_rope_fails_scaled_oracle_at_extended_positions(self):
        """Sensitivity check: if scaling is removed (scale=1.0), output must diverge."""
        import mlx.core as mx
        import mlx.nn as nn

        dims = 32
        base_theta = 10000.0
        factor = 2.0
        # Scaled vs unscaled
        scaled_rope = nn.RoPE(dims=dims, traditional=False, base=base_theta, scale=1.0 / factor)
        unscaled_rope = nn.RoPE(dims=dims, traditional=False, base=base_theta, scale=1.0)

        # At position 100 (beyond boundary of 64)
        np.random.seed(7)
        x_np = np.random.randn(1, 1, dims).astype(np.float32)
        x_mx = mx.array(x_np)

        out_scaled = np.array(scaled_rope(x_mx, offset=100))
        out_unscaled = np.array(unscaled_rope(x_mx, offset=100))

        diff = np.max(np.abs(out_scaled - out_unscaled))
        assert diff > 0.1, (
            f"Sensitivity failure: unscaled RoPE did not diverge from scaled RoPE (diff={diff})"
        )

    def test_position_scaling_identity(self):
        """Mathematical identity: RoPE_scaled(x, offset=2p) == RoPE_unscaled(x, offset=p)."""
        import mlx.core as mx
        import mlx.nn as nn

        dims = 32
        base_theta = 10000.0
        factor = 2.0

        scaled_rope = nn.RoPE(dims=dims, traditional=False, base=base_theta, scale=1.0 / factor)
        unscaled_rope = nn.RoPE(dims=dims, traditional=False, base=base_theta, scale=1.0)

        np.random.seed(99)
        x_mx = mx.array(np.random.randn(1, 1, dims).astype(np.float32))

        for p in [5, 20, 64, 128]:
            out_scaled = np.array(scaled_rope(x_mx, offset=int(factor * p)))
            out_unscaled = np.array(unscaled_rope(x_mx, offset=p))
            diff = np.max(np.abs(out_scaled - out_unscaled))
            assert diff < 1e-6, f"Identity failure at p={p}: diff={diff}"


# ---------------------------------------------------------------------------
# 5. Native Apple Silicon Training, Save, Reload, and Merge Verification
# ---------------------------------------------------------------------------


class TestMLXRoPELifecycleNative:
    """End-to-end native tests on Apple Silicon: train -> save -> infer reload -> merge."""

    @pytest.fixture(autouse=True)
    def _require_native(self):
        pytest.importorskip("mlx.core")
        pytest.importorskip("mlx_lm")

    def _create_tiny_llama(
        self, path: Path, *, orig_ctx: int = 64, quantized: bool = False
    ) -> Path:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm.models.llama import Model, ModelArgs
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        path.mkdir(parents=True, exist_ok=True)
        mx.random.seed(42)
        config = {
            "model_type": "llama",
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "vocab_size": 32,
            "max_position_embeddings": orig_ctx,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
        model = Model(ModelArgs(**config))
        if quantized:
            nn.quantize(model, group_size=32, bits=4)
            config["quantization"] = {"group_size": 32, "bits": 4}

        mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
        (path / "config.json").write_text(json.dumps(config))

        vocab = {
            w: i
            for i, w in enumerate(
                ["<unk>", "<eos>", "<pad>", "user", "assistant", "alpha", "beta", "gamma"]
            )
        }
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] }} "
            "{{ message['content'] }} <eos> {% endfor %}"
        )
        tokenizer.save_pretrained(path)
        return path

    def test_quantized_model_with_rope_scaling_refused_at_runtime(self, tmp_path):
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        base = self._create_tiny_llama(tmp_path / "quant_base", orig_ctx=64, quantized=True)
        output = tmp_path / "output"

        cfg = SoupConfig(
            base=str(base),
            backend="mlx",
            output=str(output),
            data={"train": "dummy.jsonl", "max_length": 128},
            training={
                "quantization": "none",
                "rope_scaling_type": "linear",
                "epochs": 1,
                "batch_size": 1,
                "lr": 0.001,
            },
        )
        trainer = MLXSFTTrainerWrapper(cfg)
        with pytest.raises(
            ValueError, match="MLX RoPE scaling is not supported on a quantized base model"
        ):
            trainer.setup({"train": [{"text": "alpha beta gamma"}]})

    def test_native_training_persistence_reload_and_merge(self, tmp_path, monkeypatch):
        """Verify full lifecycle: SFT beyond native length -> metadata -> reload -> merge."""
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
        from soup_cli.utils.mlx_generation import load_native_model, read_native_adapter
        from soup_cli.utils.mlx_merge import merge_native_adapter

        orig_len = 64
        extended_len = 128
        expected_factor = 2.0

        base = self._create_tiny_llama(tmp_path / "base", orig_ctx=orig_len, quantized=False)
        output = tmp_path / "adapter_output"

        # Train rows that repeat tokens past the original 64-token limit
        repeated_text = " ".join(["alpha", "beta", "gamma"] * 30)  # ~90 tokens > 64
        dataset = {"train": [{"text": repeated_text}] * 4}

        cfg = SoupConfig(
            base=str(base),
            backend="mlx",
            output=str(output),
            data={
                "train": "dummy.jsonl",
                "max_length": extended_len,
                "train_on_responses_only": False,
            },
            training={
                "quantization": "none",
                "rope_scaling_type": "linear",
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "lr": 0.01,
                "lora": {"r": 4, "alpha": 8, "dropout": 0, "target_modules": ["self_attn.q_proj"]},
            },
        )

        trainer = MLXSFTTrainerWrapper(cfg)
        trainer.setup(dataset)
        assert trainer._effective_rope_scaling == {
            "type": "linear",
            "factor": expected_factor,
            "original_max_position_embeddings": orig_len,
        }

        # Train and save adapter
        result = trainer.train()
        assert result["total_steps"] > 0
        assert (output / "adapters.safetensors").is_file()
        assert (output / "adapter_config.json").is_file()

        # 1. Inspect persisted adapter metadata
        metadata = read_native_adapter(output)
        assert metadata is not None
        assert "rope_scaling" in metadata
        assert metadata["rope_scaling"] == {
            "type": "linear",
            "factor": expected_factor,
            "original_max_position_embeddings": orig_len,
        }

        # 2. Reload via load_native_model (simulating soup infer / soup chat)
        model, tokenizer, loaded_config = load_native_model(str(output), return_config=True)
        assert loaded_config["max_position_embeddings"] == extended_len
        assert loaded_config["rope_scaling"]["type"] == "linear"
        assert loaded_config["rope_scaling"]["factor"] == expected_factor

        # 3. Refusal: load_native_model with mismatching base original length
        mismatch_adapter = tmp_path / "mismatch_adapter"
        mismatch_adapter.mkdir()
        (mismatch_adapter / "adapters.safetensors").write_bytes(
            (output / "adapters.safetensors").read_bytes()
        )
        bad_meta = dict(metadata)
        bad_meta["rope_scaling"] = dict(metadata["rope_scaling"])
        bad_meta["rope_scaling"]["original_max_position_embeddings"] = 32
        (mismatch_adapter / "adapter_config.json").write_text(json.dumps(bad_meta))
        with pytest.raises(
            ValueError, match="does not match adapter original_max_position_embeddings"
        ):
            load_native_model(str(mismatch_adapter))

        # 4. Standalone fusion via merge_native_adapter (soup merge --backend mlx)
        fused_output = tmp_path / "fused_model"
        destination = merge_native_adapter(str(output), str(fused_output), dtype="float32")
        assert destination.is_dir()
        assert (destination / "config.json").is_file()

        fused_config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        assert fused_config["rope_scaling"] == {
            "type": "linear",
            "factor": expected_factor,
            "original_max_position_embeddings": orig_len,
        }
        assert fused_config["max_position_embeddings"] == extended_len
