"""v0.53.0 Quant Menu II — UD GGUFs + KV cache + NVFP4 + LF parity + save formats.

Schema-only test suite. Live wiring deferred to v0.53.1 (mirrors v0.50.0 /
v0.51.0 / v0.52.0 single-file test layout).
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from soup_cli.config.loader import load_config_from_string

# ---------------------------------------------------------------------------
# Part A — Unsloth Dynamic 2.0 GGUF ladder
# ---------------------------------------------------------------------------


class TestUDGGUF:
    def test_ud_formats_frozenset(self):
        from soup_cli.utils.gguf_quant import UD_GGUF_FORMATS

        assert isinstance(UD_GGUF_FORMATS, frozenset)
        # 6 K-XL + 8 IQ variants = 14 total
        assert len(UD_GGUF_FORMATS) == 14
        assert "UD-Q8_K_XL" in UD_GGUF_FORMATS
        assert "UD-IQ1_M" in UD_GGUF_FORMATS
        assert "UD-IQ1_S" in UD_GGUF_FORMATS
        assert "UD-IQ2_XXS" in UD_GGUF_FORMATS

    @pytest.mark.parametrize(
        "name", [
            "UD-Q8_K_XL", "UD-Q4_K_XL", "UD-IQ1_M",
            "ud-q8_k_xl", "Ud-Iq1_M",
        ],
    )
    def test_validate_ud_canonical_case_insensitive(self, name):
        from soup_cli.utils.gguf_quant import UD_GGUF_FORMATS, validate_ud_gguf_format

        result = validate_ud_gguf_format(name)
        # Returns the canonical entry from the allowlist regardless of input case.
        assert result in UD_GGUF_FORMATS
        # Idempotent: validating the canonical form returns the same value.
        assert validate_ud_gguf_format(result) == result

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (123, TypeError),
            (b"UD-Q8_K_XL", TypeError),
            ("", ValueError),
            ("UD-Q8\x00_K_XL", ValueError),
            ("U" * 64, ValueError),
            ("UD-UNKNOWN", ValueError),
            ("Q4_K_M", ValueError),  # apple/arm — not UD
        ],
    )
    def test_validate_ud_rejects(self, bad, exc):
        from soup_cli.utils.gguf_quant import validate_ud_gguf_format

        with pytest.raises(exc):
            validate_ud_gguf_format(bad)

    def test_is_ud_gguf_format(self):
        from soup_cli.utils.gguf_quant import is_ud_gguf_format

        assert is_ud_gguf_format("UD-Q4_K_XL") is True
        assert is_ud_gguf_format("ud-iq1_m") is True
        assert is_ud_gguf_format("IQ4_XS") is False
        assert is_ud_gguf_format("Q4_K_M") is False
        assert is_ud_gguf_format(True) is False
        assert is_ud_gguf_format(None) is False
        assert is_ud_gguf_format(123) is False

    def test_spec_frozen(self):
        from soup_cli.utils.gguf_quant import get_gguf_spec

        spec = get_gguf_spec("UD-Q4_K_XL")
        assert spec.family == "ud"
        assert spec.live_wired is False
        with pytest.raises(Exception):
            spec.live_wired = True  # type: ignore[misc]

    def test_calibration_data_validator(self):
        from soup_cli.utils.gguf_quant import validate_calibration_data_path

        assert validate_calibration_data_path("calib.jsonl") == "calib.jsonl"

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            (123, TypeError),
            ("", ValueError),
            ("foo\x00bar", ValueError),
            ("x" * 5000, ValueError),
        ],
    )
    def test_calibration_data_rejects(self, bad, exc):
        from soup_cli.utils.gguf_quant import validate_calibration_data_path

        with pytest.raises(exc):
            validate_calibration_data_path(bad)

    def test_export_now_live(self):
        """v0.53.1 #139 — live wiring landed; stub is gone.

        ``export_advanced_gguf`` now requires keyword-only args. Calling
        without args raises ``TypeError`` (not ``NotImplementedError``)
        which is exactly the regression we want as proof the live wiring
        is in place.
        """
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        with pytest.raises(TypeError):
            export_advanced_gguf()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Part B — IQ + Apple/ARM GGUF
# ---------------------------------------------------------------------------


class TestIQAppleARMGGUF:
    def test_iq_frozenset(self):
        from soup_cli.utils.gguf_quant import IQ_GGUF_FORMATS

        assert isinstance(IQ_GGUF_FORMATS, frozenset)
        assert "IQ1_S" in IQ_GGUF_FORMATS
        assert "IQ4_NL" in IQ_GGUF_FORMATS
        # Should not have UD- prefix
        assert all(not n.startswith("UD-") for n in IQ_GGUF_FORMATS)

    def test_apple_arm_frozenset(self):
        from soup_cli.utils.gguf_quant import APPLE_ARM_GGUF_FORMATS

        assert isinstance(APPLE_ARM_GGUF_FORMATS, frozenset)
        assert "Q4_NL" in APPLE_ARM_GGUF_FORMATS
        assert "Q4_0_4_4" in APPLE_ARM_GGUF_FORMATS
        assert "Q5_K_M" in APPLE_ARM_GGUF_FORMATS

    @pytest.mark.parametrize(
        "name", ["IQ1_S", "iq2_m", "IQ3_XXS", "IQ4_NL"],
    )
    def test_validate_iq_canonical(self, name):
        from soup_cli.utils.gguf_quant import validate_iq_gguf_format

        result = validate_iq_gguf_format(name)
        assert result.startswith("IQ")

    def test_validate_iq_rejects_ud(self):
        from soup_cli.utils.gguf_quant import validate_iq_gguf_format

        with pytest.raises(ValueError, match="not supported"):
            validate_iq_gguf_format("UD-IQ1_M")

    def test_validate_apple_arm_canonical(self):
        from soup_cli.utils.gguf_quant import validate_apple_arm_gguf_format

        assert validate_apple_arm_gguf_format("Q4_NL") == "Q4_NL"
        assert validate_apple_arm_gguf_format("q5_k_m") == "Q5_K_M"

    def test_is_iq_and_apple_arm(self):
        from soup_cli.utils.gguf_quant import (
            is_apple_arm_gguf_format,
            is_iq_gguf_format,
        )

        assert is_iq_gguf_format("IQ2_M") is True
        assert is_iq_gguf_format("UD-IQ2_M") is False
        assert is_apple_arm_gguf_format("Q4_NL") is True
        assert is_apple_arm_gguf_format("IQ1_S") is False

    def test_is_advanced_gguf_format(self):
        from soup_cli.utils.gguf_quant import is_advanced_gguf_format

        assert is_advanced_gguf_format("UD-Q4_K_XL") is True
        assert is_advanced_gguf_format("IQ2_M") is True
        assert is_advanced_gguf_format("Q4_NL") is True
        assert is_advanced_gguf_format("q4_0") is False  # legacy llama.cpp
        assert is_advanced_gguf_format(123) is False

    def test_all_advanced_no_overlap(self):
        from soup_cli.utils.gguf_quant import (
            APPLE_ARM_GGUF_FORMATS,
            IQ_GGUF_FORMATS,
            UD_GGUF_FORMATS,
        )

        # UD-prefixed entries belong only to UD
        assert UD_GGUF_FORMATS & IQ_GGUF_FORMATS == frozenset()
        assert UD_GGUF_FORMATS & APPLE_ARM_GGUF_FORMATS == frozenset()
        assert IQ_GGUF_FORMATS & APPLE_ARM_GGUF_FORMATS == frozenset()


# ---------------------------------------------------------------------------
# Part C — KV cache types
# ---------------------------------------------------------------------------


class TestKVCache:
    def test_kv_cache_types_frozenset(self):
        from soup_cli.utils.kv_cache import KV_CACHE_TYPES

        assert isinstance(KV_CACHE_TYPES, frozenset)
        assert KV_CACHE_TYPES == {"q8_0", "bf16", "f16", "fp8"}

    @pytest.mark.parametrize(
        "name", ["q8_0", "Q8_0", "bf16", "BF16", "f16", "fp8", "FP8"],
    )
    def test_validate_kv_canonical(self, name):
        from soup_cli.utils.kv_cache import validate_kv_cache_type

        assert validate_kv_cache_type(name) == name.lower()

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("q8\x00_0", ValueError),
            ("q" * 20, ValueError),
            ("int8", ValueError),
            ("q4_0", ValueError),
        ],
    )
    def test_validate_kv_rejects(self, bad, exc):
        from soup_cli.utils.kv_cache import validate_kv_cache_type

        with pytest.raises(exc):
            validate_kv_cache_type(bad)

    def test_requires_hopper(self):
        from soup_cli.utils.kv_cache import requires_hopper

        assert requires_hopper("fp8") is True
        assert requires_hopper("FP8") is True
        assert requires_hopper("q8_0") is False
        assert requires_hopper(True) is False
        assert requires_hopper(None) is False

    def test_spec_frozen_and_metadata(self):
        from soup_cli.utils.kv_cache import get_kv_cache_spec

        spec = get_kv_cache_spec("fp8")
        assert spec.requires_hopper is True
        assert spec.bits == 8
        with pytest.raises(Exception):
            spec.requires_hopper = False  # type: ignore[misc]

    def test_apply_deferred(self):
        from soup_cli.utils.kv_cache import apply_kv_cache_type

        with pytest.raises(NotImplementedError, match="v0.53.1"):
            apply_kv_cache_type()

    def test_metadata_immutable(self):
        from soup_cli.utils.kv_cache import _KV_CACHE_METADATA

        assert isinstance(_KV_CACHE_METADATA, MappingProxyType)
        with pytest.raises(TypeError):
            _KV_CACHE_METADATA["evil"] = None  # type: ignore[index]


class TestKVCacheSchema:
    def test_schema_default_none(self):
        cfg = load_config_from_string(
            "base: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
        )
        assert cfg.training.kv_cache_type is None

    def test_schema_accept_q8(self):
        cfg = load_config_from_string(
            "base: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
            "training: {kv_cache_type: q8_0}\n"
        )
        assert cfg.training.kv_cache_type == "q8_0"

    def test_schema_case_insensitive_normalisation(self):
        cfg = load_config_from_string(
            "base: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
            "training: {kv_cache_type: BF16}\n"
        )
        assert cfg.training.kv_cache_type == "bf16"

    def test_schema_rejects_unknown(self):
        with pytest.raises(ValueError, match="not supported|kv_cache_type"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                "training: {kv_cache_type: int8}\n"
            )

    def test_schema_fp8_rejected_on_mlx(self):
        with pytest.raises(
            ValueError,
            match="kv_cache_type='fp8' is not supported on the mlx",
        ):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "backend: mlx\n"
                "data: {train: x.jsonl}\n"
                "training: {kv_cache_type: fp8}\n"
            )

    def test_schema_q8_0_allowed_on_mlx(self):
        cfg = load_config_from_string(
            "base: mlx-community/foo\n"
            "task: sft\n"
            "backend: mlx\n"
            "data: {train: x.jsonl}\n"
            "training: {kv_cache_type: q8_0}\n"
        )
        assert cfg.training.kv_cache_type == "q8_0"


# ---------------------------------------------------------------------------
# Part D — FP8 attention + NVFP4 + unsloth_bnb_4bit
# ---------------------------------------------------------------------------


class TestFP8Attention:
    def test_compat_off_no_check(self):
        from soup_cli.utils.advanced_precision import (
            validate_fp8_attention_compat,
        )

        validate_fp8_attention_compat(
            fp8_attention=False, quantization_aware=False, backend="transformers",
        )

    def test_compat_happy(self):
        from soup_cli.utils.advanced_precision import (
            validate_fp8_attention_compat,
        )

        validate_fp8_attention_compat(
            fp8_attention=True, quantization_aware="fp8", backend="transformers",
        )

    def test_compat_requires_fp8_qat(self):
        from soup_cli.utils.advanced_precision import (
            validate_fp8_attention_compat,
        )

        with pytest.raises(ValueError, match="quantization_aware='fp8'"):
            validate_fp8_attention_compat(
                fp8_attention=True, quantization_aware=False,
                backend="transformers",
            )

    def test_compat_rejects_mlx(self):
        from soup_cli.utils.advanced_precision import (
            validate_fp8_attention_compat,
        )

        with pytest.raises(ValueError, match="mlx"):
            validate_fp8_attention_compat(
                fp8_attention=True, quantization_aware="fp8", backend="mlx",
            )

    def test_compat_bool_guard(self):
        from soup_cli.utils.advanced_precision import (
            validate_fp8_attention_compat,
        )

        with pytest.raises(TypeError, match="bool"):
            validate_fp8_attention_compat(
                fp8_attention=1,  # type: ignore[arg-type]
                quantization_aware="fp8",
                backend="transformers",
            )

    def test_schema_default_false(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
        )
        assert cfg.training.fp8_attention is False

    def test_schema_happy(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
            "training: {fp8_attention: true, quantization_aware: fp8}\n"
        )
        assert cfg.training.fp8_attention is True

    def test_schema_rejects_without_fp8_qat(self):
        with pytest.raises(ValueError, match="quantization_aware='fp8'"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                "training: {fp8_attention: true}\n"
            )

    def test_apply_deferred(self):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        with pytest.raises(NotImplementedError, match="v0.53.1"):
            apply_fp8_attention()


class TestNVFP4:
    def test_compat_off_no_check(self):
        from soup_cli.utils.advanced_precision import validate_nvfp4_compat

        validate_nvfp4_compat(
            nvfp4=False, backend="transformers", modality="text",
        )

    def test_compat_happy(self):
        from soup_cli.utils.advanced_precision import validate_nvfp4_compat

        validate_nvfp4_compat(
            nvfp4=True, backend="transformers", modality="text",
        )

    def test_compat_rejects_mlx(self):
        from soup_cli.utils.advanced_precision import validate_nvfp4_compat

        with pytest.raises(ValueError, match="mlx"):
            validate_nvfp4_compat(nvfp4=True, backend="mlx", modality="text")

    def test_compat_rejects_vision(self):
        from soup_cli.utils.advanced_precision import validate_nvfp4_compat

        with pytest.raises(ValueError, match="text"):
            validate_nvfp4_compat(
                nvfp4=True, backend="transformers", modality="vision",
            )

    def test_compat_bool_guard(self):
        from soup_cli.utils.advanced_precision import validate_nvfp4_compat

        with pytest.raises(TypeError, match="bool"):
            validate_nvfp4_compat(
                nvfp4=1,  # type: ignore[arg-type]
                backend="transformers",
                modality="text",
            )

    def test_schema_default_false(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
        )
        assert cfg.training.nvfp4 is False

    def test_schema_happy(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
            "training: {nvfp4: true}\n"
        )
        assert cfg.training.nvfp4 is True

    def test_schema_rejects_mlx(self):
        with pytest.raises(ValueError, match="mlx"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "backend: mlx\n"
                "data: {train: x.jsonl}\n"
                "training: {nvfp4: true}\n"
            )

    def test_schema_rejects_vision(self):
        with pytest.raises(ValueError, match="text"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "modality: vision\n"
                "data: {train: x.jsonl, format: llava}\n"
                "training: {nvfp4: true}\n"
            )

    def test_apply_deferred(self):
        from soup_cli.utils.advanced_precision import apply_nvfp4

        with pytest.raises(NotImplementedError, match="v0.53.1"):
            apply_nvfp4()


class TestUnslothBNB4Bit:
    def test_compat_off_no_check(self):
        from soup_cli.utils.advanced_precision import (
            validate_unsloth_bnb_4bit_compat,
        )

        validate_unsloth_bnb_4bit_compat(
            unsloth_bnb_4bit=False, backend="transformers", quantization="none",
        )

    def test_compat_happy(self):
        from soup_cli.utils.advanced_precision import (
            validate_unsloth_bnb_4bit_compat,
        )

        validate_unsloth_bnb_4bit_compat(
            unsloth_bnb_4bit=True, backend="unsloth", quantization="4bit",
        )

    def test_compat_requires_unsloth(self):
        from soup_cli.utils.advanced_precision import (
            validate_unsloth_bnb_4bit_compat,
        )

        with pytest.raises(ValueError, match="backend='unsloth'"):
            validate_unsloth_bnb_4bit_compat(
                unsloth_bnb_4bit=True, backend="transformers",
                quantization="4bit",
            )

    def test_compat_requires_4bit(self):
        from soup_cli.utils.advanced_precision import (
            validate_unsloth_bnb_4bit_compat,
        )

        with pytest.raises(ValueError, match="quantization='4bit'"):
            validate_unsloth_bnb_4bit_compat(
                unsloth_bnb_4bit=True, backend="unsloth", quantization="8bit",
            )

    def test_schema_happy(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: sft\n"
            "backend: unsloth\n"
            "data: {train: x.jsonl}\n"
            "training: {unsloth_bnb_4bit: true, quantization: 4bit}\n"
        )
        assert cfg.training.unsloth_bnb_4bit is True

    def test_schema_rejects_non_unsloth(self):
        with pytest.raises(ValueError, match="backend='unsloth'"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                "training: {unsloth_bnb_4bit: true, quantization: 4bit}\n"
            )


# ---------------------------------------------------------------------------
# Part E — LF / Axolotl parity
# ---------------------------------------------------------------------------


class TestLFParity:
    def test_double_quant_default_false(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
        )
        assert cfg.training.bnb_4bit_use_double_quant is False

    def test_double_quant_happy_with_4bit(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
            "training: {bnb_4bit_use_double_quant: true, quantization: 4bit}\n"
        )
        assert cfg.training.bnb_4bit_use_double_quant is True

    def test_double_quant_rejects_8bit(self):
        with pytest.raises(ValueError, match="quantization='4bit'"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                "training: "
                "{bnb_4bit_use_double_quant: true, quantization: 8bit}\n"
            )

    def test_llm_int8_requires_8bit(self):
        with pytest.raises(ValueError, match="quantization='8bit'"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                "training: {llm_int8: true, quantization: 4bit}\n"
            )

    def test_llm_int8_happy(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: sft\n"
            "data: {train: x.jsonl}\n"
            "training: {llm_int8: true, quantization: 8bit}\n"
        )
        assert cfg.training.llm_int8 is True

    def test_quantize_ref_model_happy_dpo(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: dpo\n"
            "data: {train: x.jsonl, format: dpo}\n"
            "training: {quantize_ref_model: true}\n"
        )
        assert cfg.training.quantize_ref_model is True

    def test_quantize_ref_model_rejects_sft(self):
        with pytest.raises(ValueError, match="reference model"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                "training: {quantize_ref_model: true}\n"
            )

    def test_quantize_reward_model_happy_ppo(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: ppo\n"
            "data: {train: x.jsonl}\n"
            "training: {quantize_reward_model: true, reward_model: r/m}\n"
        )
        assert cfg.training.quantize_reward_model is True

    def test_quantize_reward_model_rejects_dpo(self):
        with pytest.raises(ValueError, match="ppo|reward_model"):
            load_config_from_string(
                "base: a/b\n"
                "task: dpo\n"
                "data: {train: x.jsonl, format: dpo}\n"
                "training: {quantize_reward_model: true}\n"
            )

    @pytest.mark.parametrize(
        "field", [
            "fp8_attention", "nvfp4", "unsloth_bnb_4bit",
            "bnb_4bit_use_double_quant", "llm_int8",
            "quantize_ref_model", "quantize_reward_model",
        ],
    )
    def test_bool_guards_reject_int(self, field):
        # The inner TypeError from `_validate_v053_bool_fields` propagates
        # through Pydantic's field_validator(mode='before') unchanged.
        with pytest.raises(TypeError, match="v0.53.0 flag must be bool"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                f"training: {{{field}: 7}}\n"
            )

    def test_quantize_ref_model_happy_grpo(self):
        # Code-review HIGH fix — GRPO has a KL-to-ref policy so
        # quantize_ref_model is meaningful on grpo.
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: grpo\n"
            "data: {train: x.jsonl}\n"
            "training: {quantize_ref_model: true, reward_fn: accuracy}\n"
        )
        assert cfg.training.quantize_ref_model is True

    def test_quantize_ref_model_happy_kto(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: kto\n"
            "data: {train: x.jsonl, format: kto}\n"
            "training: {quantize_ref_model: true}\n"
        )
        assert cfg.training.quantize_ref_model is True

    def test_quantize_ref_model_rejects_pretrain(self):
        with pytest.raises(ValueError, match="reference model"):
            load_config_from_string(
                "base: a/b\n"
                "task: pretrain\n"
                "data: {train: x.txt, format: plaintext}\n"
                "training: {quantize_ref_model: true}\n"
            )

    @pytest.mark.parametrize("quant", ["none", "8bit", "gptq"])
    def test_double_quant_rejects_non_4bit(self, quant):
        with pytest.raises(ValueError, match="quantization='4bit'"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                f"training: {{bnb_4bit_use_double_quant: true, quantization: {quant}}}\n"
            )

    def test_llm_int8_rejects_default_none(self):
        with pytest.raises(ValueError, match="quantization='8bit'"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                "training: {llm_int8: true}\n"
            )

    def test_quantize_reward_model_happy_reward_model_task(self):
        cfg = load_config_from_string(
            "base: a/b\n"
            "task: reward_model\n"
            "data: {train: x.jsonl, format: dpo}\n"
            "training: {quantize_reward_model: true}\n"
        )
        assert cfg.training.quantize_reward_model is True

    def test_v053_bool_field_explicit_null_rejected(self):
        # Review fix — `None` should NOT silently coerce to False on a
        # non-optional bool field. Pydantic rejects with "valid boolean"
        # error so a YAML typo like `fp8_attention: ~` is surfaced loudly
        # rather than masquerading as `False`.
        with pytest.raises(ValueError, match="valid boolean"):
            load_config_from_string(
                "base: a/b\n"
                "task: sft\n"
                "data: {train: x.jsonl}\n"
                "training: {fp8_attention: null}\n"
            )


# ---------------------------------------------------------------------------
# Part F — Save formats (merge 4bit + torchao export)
# ---------------------------------------------------------------------------


class TestSaveFormats:
    def test_merge_save_formats_frozenset(self):
        from soup_cli.utils.save_formats import MERGE_SAVE_FORMATS

        assert isinstance(MERGE_SAVE_FORMATS, frozenset)
        assert MERGE_SAVE_FORMATS == {"fp16", "4bit", "4bit_forced"}

    def test_torchao_schemes_frozenset(self):
        from soup_cli.utils.save_formats import TORCHAO_PTQ_SCHEMES

        assert isinstance(TORCHAO_PTQ_SCHEMES, frozenset)
        assert "Int4WeightOnly" in TORCHAO_PTQ_SCHEMES
        assert "NVFP4" in TORCHAO_PTQ_SCHEMES

    @pytest.mark.parametrize(
        "name", ["fp16", "FP16", "4bit", "4BIT", "4bit_forced"],
    )
    def test_validate_merge_canonical(self, name):
        from soup_cli.utils.save_formats import validate_merge_save_format

        assert validate_merge_save_format(name) == name.lower()

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("4bit\x00", ValueError),
            ("x" * 100, ValueError),
            ("2bit", ValueError),
        ],
    )
    def test_validate_merge_rejects(self, bad, exc):
        from soup_cli.utils.save_formats import validate_merge_save_format

        with pytest.raises(exc):
            validate_merge_save_format(bad)

    def test_validate_torchao_case_sensitive(self):
        from soup_cli.utils.save_formats import validate_torchao_scheme

        assert validate_torchao_scheme("Int4WeightOnly") == "Int4WeightOnly"
        with pytest.raises(ValueError, match="not supported"):
            validate_torchao_scheme("int4weightonly")

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (123, TypeError),
            ("", ValueError),
            ("X\x00Y", ValueError),
            ("Unknown", ValueError),
        ],
    )
    def test_validate_torchao_rejects(self, bad, exc):
        from soup_cli.utils.save_formats import validate_torchao_scheme

        with pytest.raises(exc):
            validate_torchao_scheme(bad)

    def test_quant_config_path(self):
        from soup_cli.utils.save_formats import validate_quant_config_path

        assert validate_quant_config_path("cfg.yaml") == "cfg.yaml"

    @pytest.mark.parametrize(
        "bad,exc", [
            (True, TypeError),
            (None, TypeError),
            ("", ValueError),
            ("a\x00b", ValueError),
        ],
    )
    def test_quant_config_path_rejects(self, bad, exc):
        from soup_cli.utils.save_formats import validate_quant_config_path

        with pytest.raises(exc):
            validate_quant_config_path(bad)

    def test_spec_frozen(self):
        from soup_cli.utils.save_formats import (
            get_merge_save_spec,
            get_torchao_spec,
        )

        m = get_merge_save_spec("4bit")
        t = get_torchao_spec("NVFP4")
        assert m.bits == 4
        assert t.bits == 4
        with pytest.raises(Exception):
            m.bits = 8  # type: ignore[misc]
        with pytest.raises(Exception):
            t.live_wired = True  # type: ignore[misc]

    def test_merge_metadata_immutable(self):
        from soup_cli.utils.save_formats import (
            _MERGE_METADATA,
            _TORCHAO_METADATA,
        )

        assert isinstance(_MERGE_METADATA, MappingProxyType)
        assert isinstance(_TORCHAO_METADATA, MappingProxyType)

    def test_merge_4bit_now_live(self):
        """v0.53.1 #142 — live wiring landed; signature now requires kwargs."""
        from soup_cli.utils.save_formats import merge_4bit

        with pytest.raises(TypeError):
            merge_4bit()  # type: ignore[call-arg]

    def test_export_torchao_now_live(self):
        """v0.53.1 #142 — live wiring landed; signature now requires kwargs."""
        from soup_cli.utils.save_formats import export_torchao

        with pytest.raises(TypeError):
            export_torchao()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Cross-cutting — round-trip + immutability
# ---------------------------------------------------------------------------


class TestCrossCutting:
    def test_full_v053_roundtrip(self):
        yaml_text = (
            "base: a/b\n"
            "task: sft\n"
            "backend: unsloth\n"
            "data: {train: x.jsonl}\n"
            "training:\n"
            "  quantization: 4bit\n"
            "  quantization_aware: fp8\n"
            "  fp8_attention: true\n"
            "  kv_cache_type: q8_0\n"
            "  unsloth_bnb_4bit: true\n"
            "  bnb_4bit_use_double_quant: true\n"
        )
        cfg = load_config_from_string(yaml_text)
        assert cfg.training.fp8_attention is True
        assert cfg.training.kv_cache_type == "q8_0"
        assert cfg.training.unsloth_bnb_4bit is True
        assert cfg.training.bnb_4bit_use_double_quant is True

    def test_gguf_metadata_immutable(self):
        from soup_cli.utils.gguf_quant import _GGUF_METADATA

        assert isinstance(_GGUF_METADATA, MappingProxyType)
        with pytest.raises(TypeError):
            _GGUF_METADATA["evil"] = None  # type: ignore[index]

    def test_total_advanced_gguf_count(self):
        from soup_cli.utils.gguf_quant import (
            ALL_ADVANCED_GGUF_FORMATS,
            APPLE_ARM_GGUF_FORMATS,
            IQ_GGUF_FORMATS,
            UD_GGUF_FORMATS,
        )

        assert len(ALL_ADVANCED_GGUF_FORMATS) == (
            len(UD_GGUF_FORMATS)
            + len(IQ_GGUF_FORMATS)
            + len(APPLE_ARM_GGUF_FORMATS)
        )


class TestReviewFollowups:
    """Coverage-gap fixes from the v0.53.0 review round."""

    # --- gguf_quant ----------------------------------------------------------
    def test_get_gguf_spec_unknown_raises(self):
        from soup_cli.utils.gguf_quant import get_gguf_spec

        with pytest.raises(ValueError, match="not in v0.53.0 catalog"):
            get_gguf_spec("Q4_0")  # legacy llama.cpp, not in v0.53.0 allowlist

    def test_get_gguf_spec_non_string_raises(self):
        from soup_cli.utils.gguf_quant import get_gguf_spec

        with pytest.raises(TypeError, match="str"):
            get_gguf_spec(123)  # type: ignore[arg-type]

    def test_calibration_path_exact_boundary(self):
        from soup_cli.utils.gguf_quant import validate_calibration_data_path

        # 4096 chars — accepted at boundary
        ok = "a" * 4096
        assert validate_calibration_data_path(ok) == ok
        # 4097 chars — rejected
        with pytest.raises(ValueError, match="too int"):
            validate_calibration_data_path("a" * 4097)

    def test_lower_index_immutable(self):
        from soup_cli.utils.gguf_quant import _LOWER_INDEX

        assert isinstance(_LOWER_INDEX, MappingProxyType)
        with pytest.raises(TypeError):
            _LOWER_INDEX["evil"] = "x"  # type: ignore[index]

    # --- kv_cache ------------------------------------------------------------
    def test_get_kv_cache_spec_unknown(self):
        from soup_cli.utils.kv_cache import get_kv_cache_spec

        with pytest.raises(ValueError, match="not supported"):
            get_kv_cache_spec("int8")

    def test_requires_hopper_reads_from_spec(self):
        # requires_hopper now delegates to _KV_CACHE_METADATA so a future
        # spec edit would be picked up automatically.
        from soup_cli.utils.kv_cache import (
            _KV_CACHE_METADATA,
            requires_hopper,
        )

        for name, spec in _KV_CACHE_METADATA.items():
            assert requires_hopper(name) is spec.requires_hopper

    # --- advanced_precision bool guards on string params --------------------
    def test_fp8_attention_backend_bool_rejected(self):
        from soup_cli.utils.advanced_precision import (
            validate_fp8_attention_compat,
        )

        with pytest.raises(TypeError, match="backend must not be bool"):
            validate_fp8_attention_compat(
                fp8_attention=True,
                quantization_aware="fp8",
                backend=True,  # type: ignore[arg-type]
            )

    def test_nvfp4_backend_bool_rejected(self):
        from soup_cli.utils.advanced_precision import validate_nvfp4_compat

        with pytest.raises(TypeError, match="backend must not be bool"):
            validate_nvfp4_compat(
                nvfp4=True,
                backend=True,  # type: ignore[arg-type]
                modality="text",
            )

    def test_nvfp4_modality_bool_rejected(self):
        from soup_cli.utils.advanced_precision import validate_nvfp4_compat

        with pytest.raises(TypeError, match="modality must not be bool"):
            validate_nvfp4_compat(
                nvfp4=True,
                backend="transformers",
                modality=True,  # type: ignore[arg-type]
            )

    def test_unsloth_bnb_backend_bool_rejected(self):
        from soup_cli.utils.advanced_precision import (
            validate_unsloth_bnb_4bit_compat,
        )

        with pytest.raises(TypeError, match="backend must not be bool"):
            validate_unsloth_bnb_4bit_compat(
                unsloth_bnb_4bit=True,
                backend=True,  # type: ignore[arg-type]
                quantization="4bit",
            )

    def test_unsloth_bnb_quantization_bool_rejected(self):
        from soup_cli.utils.advanced_precision import (
            validate_unsloth_bnb_4bit_compat,
        )

        with pytest.raises(TypeError, match="quantization must not be bool"):
            validate_unsloth_bnb_4bit_compat(
                unsloth_bnb_4bit=True,
                backend="unsloth",
                quantization=True,  # type: ignore[arg-type]
            )

    # --- save_formats boundary + symmetric behavior --------------------------
    def test_quant_config_path_exact_boundary(self):
        from soup_cli.utils.save_formats import validate_quant_config_path

        ok = "a" * 4096
        assert validate_quant_config_path(ok) == ok
        with pytest.raises(ValueError, match="too int"):
            validate_quant_config_path("a" * 4097)

    def test_merge_save_format_normalises_to_lowercase(self):
        from soup_cli.utils.save_formats import validate_merge_save_format

        assert validate_merge_save_format("4BIT_FORCED") == "4bit_forced"
        assert validate_merge_save_format("FP16") == "fp16"
