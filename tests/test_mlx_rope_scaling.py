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

        def _rope_scales(m):
            import mlx.nn as nn

            return {
                name: float(module.scale)
                for name, module in m.named_modules()
                if isinstance(module, nn.RoPE)
            }

        trainer = MLXSFTTrainerWrapper(cfg)
        trainer.setup(dataset)
        assert trainer._effective_rope_scaling == {
            "type": "linear",
            "factor": expected_factor,
            "original_max_position_embeddings": orig_len,
        }
        scales = _rope_scales(trainer.model)
        assert scales and set(scales.values()) == {1.0 / expected_factor}

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

        reloaded_scales = _rope_scales(model)
        assert reloaded_scales and set(reloaded_scales.values()) == {1.0 / expected_factor}

        # Verify forward execution at sequence length past original context (81 tokens > 64)
        import mlx.core as mx

        extended_tokens = mx.array([[1, 2, 3] * 27])  # 81 tokens
        logits = model(extended_tokens)
        assert logits.shape == (1, 81, 32)
        assert np.all(np.isfinite(np.array(logits)))

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

    def test_mutation_omitting_model_config_refuses_at_setup(
        self, tmp_path, monkeypatch
    ):
        """Mutation test: omitting model_config forwarding raises refusal on unscaled RoPE."""
        monkeypatch.chdir(tmp_path)
        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
        from soup_cli.utils import mlx as mlx_utils

        base = self._create_tiny_llama(tmp_path / "base_mut", orig_ctx=64)
        output = tmp_path / "mut_output"

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
                "lr": 0.01,
            },
        )
        trainer = MLXSFTTrainerWrapper(cfg)

        real_load = mlx_utils.load_mlx_model

        # Mutation: drop model_config override to verify mutation turns red
        def buggy_load(path, quantization="4bit", model_config=None):
            del model_config
            return real_load(path, quantization=quantization)

        monkeypatch.setattr(mlx_utils, "load_mlx_model", buggy_load)
        with pytest.raises(
            ValueError, match="does not apply native MLX RoPE scaling"
        ):
            trainer.setup({"train": [{"text": "alpha beta gamma"}]})

    def test_base_without_rope_refused_by_name(self, tmp_path, monkeypatch):
        """A base model architecture without RoPE is refused by name."""
        monkeypatch.chdir(tmp_path)
        import mlx.nn as nn

        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
        from soup_cli.utils import mlx as mlx_utils

        base = self._create_tiny_llama(tmp_path / "base_norope", orig_ctx=64)

        class DummyLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)

            def __call__(self, x):
                return self.linear(x)

        real_load = mlx_utils.load_mlx_model

        def load_norope(path, quantization="4bit", model_config=None):
            del model_config
            _, tok = real_load(path, quantization=quantization)
            return DummyLinearModel(), tok

        monkeypatch.setattr(mlx_utils, "load_mlx_model", load_norope)

        cfg = SoupConfig(
            base=str(base),
            backend="mlx",
            output=str(tmp_path / "out_norope"),
            data={"train": "dummy.jsonl", "max_length": 128},
            training={
                "quantization": "none",
                "rope_scaling_type": "linear",
                "epochs": 1,
                "batch_size": 1,
                "lr": 0.01,
            },
        )
        trainer = MLXSFTTrainerWrapper(cfg)
        with pytest.raises(
            ValueError, match="does not contain rotary embedding \\(nn.RoPE\\) modules"
        ):
            trainer.setup({"train": [{"text": "alpha beta gamma"}]})

    def test_patterned_adapter_with_rope_scaling_validation(self):
        """Patterned adapter (#2133) carrying rope_scaling is validated identically."""
        from soup_cli.utils.mlx_generation import _validate_adapter_rope_scaling

        patterned_valid = {
            "model": "meta-llama/Llama-3.2-1B",
            "fine_tune_type": "lora",
            "soup_lora": {"recipe": {"r": 8, "alpha": 16}},
            "rope_scaling": {
                "type": "linear",
                "factor": 2.0,
                "original_max_position_embeddings": 64,
            },
        }
        # Valid patterned adapter with rope_scaling passes
        _validate_adapter_rope_scaling(patterned_valid)

        bad_factor = dict(patterned_valid)
        bad_factor["rope_scaling"] = dict(patterned_valid["rope_scaling"])
        bad_factor["rope_scaling"]["factor"] = 0.5
        with pytest.raises(
            ValueError, match="rope_scaling factor must be a finite number > 1.0"
        ):
            _validate_adapter_rope_scaling(bad_factor)

        bad_orig = dict(patterned_valid)
        bad_orig["rope_scaling"] = dict(patterned_valid["rope_scaling"])
        bad_orig["rope_scaling"]["original_max_position_embeddings"] = -10
        with pytest.raises(
            ValueError, match="original_max_position_embeddings must be a positive integer"
        ):
            _validate_adapter_rope_scaling(bad_orig)

    def test_sft_when_max_length_at_or_below_base_context_length_preserves_unscaled_rope(
        self, tmp_path, monkeypatch
    ):
        """When data.max_length <= orig_ctx, native unscaled RoPE is preserved with a log line."""
        monkeypatch.chdir(tmp_path)
        from unittest.mock import patch

        import mlx.nn as nn

        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        base = self._create_tiny_llama(tmp_path / "base_unit", orig_ctx=64)
        output = tmp_path / "output_unit"

        cfg = SoupConfig(
            base=str(base),
            backend="mlx",
            output=str(output),
            data={"train": "dummy.jsonl", "max_length": 64},  # equal to orig_ctx=64
            training={
                "quantization": "none",
                "rope_scaling_type": "linear",
                "epochs": 1,
                "batch_size": 1,
                "lr": 0.01,
            },
        )
        trainer = MLXSFTTrainerWrapper(cfg)
        with patch("soup_cli.trainer.mlx_sft.console.print") as mock_print:
            trainer.setup({"train": [{"text": "alpha beta gamma"}]})

        assert trainer._effective_rope_scaling is None
        printed = [str(call.args[0]) for call in mock_print.call_args_list if call.args]
        assert any("unscaled native RoPE preserved" in line for line in printed)
        scales = {
            name: float(m.scale)
            for name, m in trainer.model.named_modules()
            if isinstance(m, nn.RoPE)
        }
        assert scales and set(scales.values()) == {1.0}


@pytest.fixture
def revision_snapshots(tmp_path, monkeypatch):
    """Real native assets behind a deterministic, moving HF transport boundary."""
    pytest.importorskip("mlx.core")
    upstream = pytest.importorskip("mlx_lm.utils")
    from types import SimpleNamespace

    import huggingface_hub
    import mlx.core as mx

    from soup_cli.utils import run_provenance as provenance

    first, second = "a" * 40, "b" * 40
    factory = TestMLXRoPELifecycleNative()._create_tiny_llama
    snapshots = {
        first: factory(tmp_path / "first", orig_ctx=64),
        second: factory(tmp_path / "second", orig_ctx=32),
    }
    # Same architecture, deliberately different weights, tokenizer and context.
    weights = mx.load(str(snapshots[second] / "model.safetensors"))
    weights["lm_head.weight"] = weights["lm_head.weight"] + 0.25
    mx.save_safetensors(str(snapshots[second] / "model.safetensors"), weights)
    tokenizer_path = snapshots[second] / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    vocab = tokenizer["model"]["vocab"]
    vocab["alpha"], vocab["beta"] = vocab["beta"], vocab["alpha"]
    tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    state = SimpleNamespace(first=first, second=second, snapshots=snapshots,
                            downloads=[], configs=[], lookups=[], cache_hit=False)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WORLD_SIZE", "RANK",
                "LOCAL_RANK", "MLX_RANK", "MLX_HOSTFILE", "MLX_WORLD_SIZE",
                "LOCAL_WORLD_SIZE", "MLXLM_USE_MODELSCOPE"):
        monkeypatch.delenv(key, raising=False)

    def model_info(base, **kwargs):
        assert base == "owner/revision-fixture"
        state.lookups.append(base)
        return SimpleNamespace(sha=first if len(state.lookups) == 1 else second)

    def snapshot_download(repo_id, *, revision=None, **kwargs):
        assert repo_id == "owner/revision-fixture"
        state.downloads.append(revision)
        return str(snapshots[revision or second])

    def config_download(*, repo_id, filename, revision=None, **kwargs):
        assert repo_id == "owner/revision-fixture" and filename == "config.json"
        state.configs.append((revision, kwargs.get("local_files_only", False)))
        if kwargs.get("local_files_only") and not state.cache_hit:
            raise FileNotFoundError("cold cache")
        return str(snapshots[revision or second] / "config.json")

    # Keep real upstream load/_download/load_model/load_tokenizer. Only HF's
    # transport is replaced; the selected directory contains real native bytes.
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(upstream, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", config_download)
    monkeypatch.setattr(provenance, "get_hf_api", lambda: SimpleNamespace(model_info=model_info))
    return state


@pytest.mark.parametrize("task,rank", [("sft", 4), ("sft", 0), ("dpo", 4), ("grpo", 4)])
def test_revision_native_setup_loads_matching_weights_tokenizer_and_config(
    tmp_path, revision_snapshots, task, rank,
):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.dispatch import build_trainer
    from soup_cli.utils.mlx import mlx_revision_binding_reason
    from soup_cli.utils.run_provenance import BaseRevision, base_revision_scope

    state = revision_snapshots
    assert mlx_revision_binding_reason() is None
    cfg = SoupConfig(
        base="owner/revision-fixture", backend="mlx", task=task,
        output=str(tmp_path / "trained"),
        data={"train": "unused.jsonl", "max_length": 128, "val_split": 0},
        training={"batch_size": 1, "gradient_accumulation_steps": 1,
                  "quantization": "none", "reward_fn": "format",
                  "lora": {"r": rank},
                  **({"rope_scaling_type": "linear"} if task == "sft" else {})},
    )
    wrapper = build_trainer(cfg, device="cpu")
    identity = BaseRevision(cfg.base)
    with base_revision_scope(identity):
        wrapper.setup({"train": [{"messages": [{"role": "user", "content": "alpha"}]}],
                       "val": []})
    assert identity.revision == state.first and identity.source == "hub_commit"
    assert state.lookups == [cfg.base]
    assert wrapper.tokenizer.encode("alpha", add_special_tokens=False) == [5]
    loaded = dict(tree_flatten(wrapper.model.parameters()))
    first_weights = mx.load(str(state.snapshots[state.first] / "model.safetensors"))
    assert mx.array_equal(loaded["lm_head.weight"], first_weights["lm_head.weight"]).item()
    wrong_weights = mx.load(str(state.snapshots[state.second] / "model.safetensors"))
    assert not mx.array_equal(loaded["lm_head.weight"], wrong_weights["lm_head.weight"]).item()
    assert state.downloads == [state.first]
    if task == "sft":
        scales = {float(module.scale) for _, module in wrapper.model.named_modules()
                  if isinstance(module, nn.RoPE)}
        assert scales == {0.5}
        assert wrapper._effective_rope_scaling["factor"] == 2.0
        assert state.configs == [(state.first, True), (state.first, False)]
        if rank == 0:
            assert wrapper._full_model_config["max_position_embeddings"] == 128
    else:
        assert state.configs == []
    assert cfg.base == "owner/revision-fixture"


@pytest.mark.parametrize("rank,cache_hit", [(4, True), (4, False), (0, False)])
def test_revision_native_train_cli_updates_saves_and_fresh_process_reloads(
    tmp_path, monkeypatch, revision_snapshots, rank, cache_hit,
):
    import hashlib
    import os
    import subprocess
    import sys

    import mlx.core as mx
    import yaml
    from mlx.utils import tree_flatten

    from soup_cli.cli import app
    from soup_cli.commands import train
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
    from soup_cli.utils.run_provenance import read_run_provenance
    from tests.conftest import make_cli_runner, strip_ansi

    state = revision_snapshots
    state.cache_hit = cache_hit
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr(train, "detect_device", lambda backend=None: ("cpu", "CPU"))
    monkeypatch.setattr(train, "get_gpu_info", lambda backend=None: {"memory_total": "N/A"})
    source = tmp_path / "train.jsonl"
    source.write_text('{"text":"alpha beta gamma alpha beta gamma"}\n' * 2, encoding="utf-8")
    cfg = {
        "base": "owner/revision-fixture", "backend": "mlx", "output": str(tmp_path / "trained"),
        "data": {"train": str(source), "format": "plaintext", "val_split": 0,
                 "max_length": 128, "train_on_responses_only": False},
        "training": {"quantization": "none", "rope_scaling_type": "linear", "epochs": 1,
                     "batch_size": 1, "gradient_accumulation_steps": 1, "lr": 0.01,
                     "scheduler": "constant", "warmup_ratio": 0, "weight_decay": 0,
                     "logging_steps": 1, "save_steps": 100, "rewind_log": False,
                     "lora": ({"r": rank, "alpha": 8, "dropout": 0,
                               "target_modules": ["self_attn.q_proj"]} if rank else {"r": 0})},
    }
    config_path = tmp_path / "soup.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    observed = {}
    setup = MLXSFTTrainerWrapper.setup

    def observe(wrapper, dataset):
        setup(wrapper, dataset)
        observed["wrapper"] = wrapper
        import mlx.nn as nn

        observed["rope_scales"] = {
            float(module.scale) for _, module in wrapper.model.named_modules()
            if isinstance(module, nn.RoPE)
        }

    from soup_cli.trainer import mlx_optim

    build_optimizer = mlx_optim.build_optimizer

    def capture_before_updates(plan):
        observed["before"] = dict(tree_flatten(observed["wrapper"].model.parameters()))
        mx.eval(observed["before"])
        return build_optimizer(plan)

    monkeypatch.setattr(MLXSFTTrainerWrapper, "setup", observe)
    monkeypatch.setattr(mlx_optim, "build_optimizer", capture_before_updates)
    result = make_cli_runner().invoke(app, [
        "--no-telemetry", "--no-audit-log", "train", "--config", str(config_path), "--yes",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert observed["rope_scales"] == {0.5}
    assert state.downloads == [state.first] and state.lookups == [cfg["base"]]
    assert state.configs == [(state.first, True)] + ([] if cache_hit else [(state.first, False)])
    wrapper = observed["wrapper"]
    after = dict(tree_flatten(wrapper.model.parameters()))
    before = observed["before"]
    trained = set(dict(tree_flatten(wrapper.model.trainable_parameters())))
    assert any(not mx.array_equal(after[name], before[name]).item() for name in trained)
    assert all(mx.array_equal(after[name], before[name]).item() for name in before.keys() - trained)
    wrapper.model.eval()
    expected = np.array(wrapper.model(mx.array([[3, 5, 6, 7]])))
    tracker = ExperimentTracker(tmp_path / "runs.db")
    try:
        row, = tracker.list_runs()
    finally:
        tracker.close()
    metadata = read_run_provenance(cfg["output"])
    assert metadata["base_revision"] == row["base_revision"] == state.first
    assert metadata["base_model"] == row["base_model"] == cfg["base"]
    assert metadata["training_data_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert row["status"] == "completed" and row["total_steps"] == 2
    plain = " ".join(strip_ansi(result.output).split())
    assert f"Base revision: {state.first} (bound Hub commit)" in plain
    # Independent process, real upstream native loader, entirely local paths.
    # This verifies this training artifact; it does not qualify other Soup consumers.
    script = """
import json, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
artifact, base, destination, rank = sys.argv[1:]
kwargs = {} if rank == '0' else {
    'adapter_path': artifact,
    'model_config': {'rope_scaling': {'type': 'linear', 'factor': 2.0},
                     'max_position_embeddings': 128},
}
model, tokenizer = load(artifact if rank == '0' else base, **kwargs)
model.eval()
Path(destination).write_text(json.dumps({'logits': model(mx.array([[3,5,6,7]])).tolist(),
    'tokens': tokenizer.encode('alpha', add_special_tokens=False)}))
"""
    reload_path = tmp_path / "reloaded.json"
    child = subprocess.run([
        sys.executable, "-c", script, cfg["output"], str(state.snapshots[state.first]),
        str(reload_path), str(rank),
    ], capture_output=True, text=True, timeout=60, env=dict(os.environ, HF_HUB_OFFLINE="1"))
    assert child.returncode == 0, child.stderr
    reloaded = json.loads(reload_path.read_text())
    np.testing.assert_allclose(reloaded["logits"], expected, atol=1e-5, rtol=1e-5)
    assert reloaded["tokens"] == [5]
