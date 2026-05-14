"""v0.49.0 — Long Context & Architecture: YaRN, Dynamic NTK, LongLoRA, Llama 3.1 NTK."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string

_BASE_DATA_LINE = "data:\n  train: ./train.jsonl\n"


def _load(yaml_text: str):
    # Ensure required `data.train` is present without bloating each fixture.
    if "data:" not in yaml_text:
        yaml_text = yaml_text + "\n" + _BASE_DATA_LINE
    return load_config_from_string(yaml_text)


# =====================================================================
# Part A — YaRN RoPE scaling
# =====================================================================


class TestYarnSchema:
    def test_yarn_default_fields(self):
        cfg = _load("base: meta-llama/Llama-3.1-8B\ntask: sft\n")
        # All YaRN fields default to None until rope_scaling_type=yarn.
        assert cfg.training.yarn_factor is None
        assert cfg.training.yarn_attn_factor is None
        assert cfg.training.yarn_beta_fast is None
        assert cfg.training.yarn_beta_slow is None

    def test_yarn_accept_valid(self):
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  rope_scaling_type: yarn
  yarn_factor: 4.0
  yarn_attn_factor: 1.0
  yarn_beta_fast: 32
  yarn_beta_slow: 1
"""
        cfg = _load(yaml_in)
        assert cfg.training.rope_scaling_type == "yarn"
        assert cfg.training.yarn_factor == 4.0
        assert cfg.training.yarn_beta_fast == 32

    @pytest.mark.parametrize(
        "field,value",
        [
            ("yarn_factor", 0.0),  # gt=1.0
            ("yarn_factor", 1.0),
            ("yarn_factor", 1025.0),  # le=1024
            ("yarn_attn_factor", -0.1),
            ("yarn_attn_factor", 10.1),
            ("yarn_beta_fast", 0),
            ("yarn_beta_fast", 1025),
            ("yarn_beta_slow", 0),
            ("yarn_beta_slow", 1025),
        ],
    )
    def test_yarn_field_bounds(self, field, value):
        yaml_in = f"""
base: meta-llama/Llama-3.1-8B
task: sft
training:
  rope_scaling_type: yarn
  {field}: {value}
"""
        with pytest.raises((ValidationError, ValueError)):
            _load(yaml_in)

    @pytest.mark.parametrize(
        "field", ["yarn_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow"]
    )
    def test_yarn_field_bool_rejected(self, field):
        yaml_in = f"""
base: meta-llama/Llama-3.1-8B
task: sft
training:
  rope_scaling_type: yarn
  {field}: true
"""
        with pytest.raises((ValidationError, ValueError)):
            _load(yaml_in)

    def test_yarn_field_requires_yarn_type(self):
        """Setting yarn_factor with rope_scaling_type != yarn must raise."""
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  rope_scaling_type: linear
  yarn_factor: 4.0
"""
        with pytest.raises((ValidationError, ValueError), match="yarn"):
            _load(yaml_in)

    def test_yarn_no_extra_fields_without_yarn(self):
        """rope_scaling_type=yarn without yarn_factor is allowed (defaults will fill)."""
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  rope_scaling_type: yarn
"""
        cfg = _load(yaml_in)
        assert cfg.training.rope_scaling_type == "yarn"


class TestYarnMath:
    def test_yarn_find_correction_dim(self):
        from soup_cli.utils.long_context import yarn_find_correction_dim

        # Wavelength of a given num_rotations is invertible to dim:
        d = yarn_find_correction_dim(
            num_rotations=32, dim=64, base=10000.0, max_position_embeddings=4096
        )
        assert isinstance(d, float)
        assert 0 <= d <= 64

    def test_yarn_find_correction_range_ordered(self):
        from soup_cli.utils.long_context import yarn_find_correction_range

        low, high = yarn_find_correction_range(
            beta_fast=32, beta_slow=1, dim=64, base=10000.0, max_position_embeddings=4096
        )
        assert isinstance(low, int)
        assert isinstance(high, int)
        assert low <= high
        assert 0 <= low <= 64
        assert 0 <= high <= 64

    def test_yarn_find_correction_range_clamped(self):
        """Range must be clamped to [0, dim/2]; high>=low+1 guarantees no div0."""
        from soup_cli.utils.long_context import yarn_find_correction_range

        low, high = yarn_find_correction_range(
            beta_fast=10_000, beta_slow=10_000, dim=64, base=10000.0, max_position_embeddings=64
        )
        assert high > low  # disambiguated

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_yarn_find_correction_dim_bad_input(self, bad):
        from soup_cli.utils.long_context import yarn_find_correction_dim

        with pytest.raises(ValueError):
            yarn_find_correction_dim(
                num_rotations=bad, dim=64, base=10000.0, max_position_embeddings=4096
            )

    def test_yarn_linear_ramp_mask_shape_and_bounds(self):
        from soup_cli.utils.long_context import yarn_linear_ramp_mask

        mask = yarn_linear_ramp_mask(low=4, high=8, dim=16)
        assert len(mask) == 16
        for value in mask:
            assert 0.0 <= value <= 1.0

    def test_yarn_linear_ramp_mask_degenerate_low_eq_high(self):
        """Avoid div-by-zero: low == high must be auto-disambiguated."""
        from soup_cli.utils.long_context import yarn_linear_ramp_mask

        mask = yarn_linear_ramp_mask(low=4, high=4, dim=16)
        assert len(mask) == 16

    def test_yarn_get_attn_scale(self):
        from soup_cli.utils.long_context import yarn_get_mscale

        # mscale = 0.1 * ln(factor) + 1.0
        assert yarn_get_mscale(1.0) == pytest.approx(1.0)
        v = yarn_get_mscale(4.0)
        assert v == pytest.approx(0.1 * math.log(4.0) + 1.0)

    def test_yarn_get_mscale_factor_below_one_returns_one(self):
        from soup_cli.utils.long_context import yarn_get_mscale

        assert yarn_get_mscale(0.5) == 1.0

    def test_get_rope_scaling_config_yarn_with_overrides(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        cfg = get_rope_scaling_config(
            scaling_type="yarn",
            target_length=32768,
            original_length=8192,
            yarn_factor=4.0,
            yarn_attn_factor=1.0,
            yarn_beta_fast=32,
            yarn_beta_slow=1,
        )
        assert cfg["type"] == "yarn"
        assert cfg["factor"] == 4.0
        assert cfg["original_max_position_embeddings"] == 8192
        assert cfg["beta_fast"] == 32
        assert cfg["beta_slow"] == 1
        assert cfg["attention_factor"] == 1.0


# =====================================================================
# Part B — Dynamic NTK (existing path) — verify hardened behaviour
# =====================================================================


class TestDynamicNTK:
    def test_dynamic_basic_factor(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        cfg = get_rope_scaling_config("dynamic", target_length=32768, original_length=8192)
        assert cfg == {"type": "dynamic", "factor": 4.0}

    def test_dynamic_no_scaling_when_target_le_original(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        cfg = get_rope_scaling_config("dynamic", target_length=4096, original_length=8192)
        assert cfg == {}


# =====================================================================
# Part C — LongLoRA S² shifted-sparse attention (schema gate)
# =====================================================================


class TestLongLoraSchema:
    def test_default_off(self):
        cfg = _load("base: meta-llama/Llama-3.1-8B\ntask: sft\n")
        assert cfg.training.use_longlora is False

    def test_accept_on_llama_sft(self):
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  use_longlora: true
"""
        cfg = _load(yaml_in)
        assert cfg.training.use_longlora is True

    def test_reject_non_llama(self):
        # v0.53.4 #120 — allowlist now covers Mistral / Qwen / Phi too;
        # use Gemma (still outside the allowlist) for the rejection test.
        yaml_in = """
base: google/gemma-2-9b
task: sft
training:
  use_longlora: true
"""
        with pytest.raises((ValidationError, ValueError), match="LongLoRA"):
            _load(yaml_in)

    def test_reject_non_sft(self):
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: dpo
training:
  use_longlora: true
"""
        with pytest.raises((ValidationError, ValueError), match="sft"):
            _load(yaml_in)

    def test_reject_mlx(self):
        yaml_in = """
base: mlx-community/Llama-3.1-8B-Instruct-4bit
task: sft
backend: mlx
training:
  use_longlora: true
"""
        with pytest.raises((ValidationError, ValueError), match="mlx"):
            _load(yaml_in)

    def test_reject_with_ring_attention(self):
        """LongLoRA's S² shifted-sparse attention is incompatible with Ring/FA v3
        custom-mask attention path."""
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  use_longlora: true
  use_ring_attention: true
"""
        with pytest.raises((ValidationError, ValueError), match="ring|LongLoRA"):
            _load(yaml_in)


class TestLongLoraHelpers:
    def test_is_llama_model_word_boundary(self):
        from soup_cli.utils.longlora import is_llama_model

        # Word-boundary regex: rejects substrings, accepts variants.
        assert is_llama_model("meta-llama/Llama-3.1-8B") is True
        assert is_llama_model("meta-llama/Llama-2-7B") is True
        assert is_llama_model("codellama/CodeLlama-7B") is True
        assert is_llama_model("meta-llama/Meta-Llama-3-8B") is True
        # Not Llama:
        assert is_llama_model("mistralai/Mistral-7B-v0.1") is False
        assert is_llama_model("Qwen/Qwen2-7B") is False
        assert is_llama_model("") is False

    def test_is_llama_model_rejects_null_byte(self):
        from soup_cli.utils.longlora import is_llama_model

        with pytest.raises(ValueError):
            is_llama_model("meta-llama/\x00Llama-3")

    def test_is_llama_model_rejects_non_string(self):
        from soup_cli.utils.longlora import is_llama_model

        with pytest.raises(TypeError):
            is_llama_model(None)

    def test_validate_longlora_compat_happy(self):
        from soup_cli.utils.longlora import validate_longlora_compat

        # No raise.
        validate_longlora_compat(
            model_name="meta-llama/Llama-3.1-8B",
            task="sft",
            backend="transformers",
            use_ring_attention=False,
        )

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (
                # v0.53.4 #120 — Mistral is now allowlisted; use Gemma instead.
                dict(
                    model_name="google/gemma-2-9b",
                    task="sft",
                    backend="transformers",
                    use_ring_attention=False,
                ),
                "allowlist",
            ),
            (
                dict(
                    model_name="meta-llama/Llama-3.1-8B",
                    task="dpo",
                    backend="transformers",
                    use_ring_attention=False,
                ),
                "sft",
            ),
            (
                dict(
                    model_name="meta-llama/Llama-3.1-8B",
                    task="sft",
                    backend="mlx",
                    use_ring_attention=False,
                ),
                "mlx",
            ),
            (
                dict(
                    model_name="meta-llama/Llama-3.1-8B",
                    task="sft",
                    backend="transformers",
                    use_ring_attention=True,
                ),
                "ring",
            ),
        ],
    )
    def test_validate_longlora_compat_rejects(self, kwargs, match):
        from soup_cli.utils.longlora import validate_longlora_compat

        with pytest.raises(ValueError, match=match):
            validate_longlora_compat(**kwargs)

    def test_apply_longlora_forward_override_now_live(self):
        """v0.53.11 #119 lifted the stub — returns a LongLoRAForwardOverride context."""
        from soup_cli.utils.longlora import (
            LongLoRAForwardOverride,
            apply_longlora_forward_override,
        )

        class _StubModel:
            def modules(self):
                return iter([])

        result = apply_longlora_forward_override(_StubModel(), group_size=4)
        assert isinstance(result, LongLoRAForwardOverride)


# =====================================================================
# Part D — Llama 3.1 NTK-aware (full impl)
# =====================================================================


class TestLlama3RopeSchema:
    def test_llama3_in_literal(self):
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  rope_scaling_type: llama3
"""
        cfg = _load(yaml_in)
        assert cfg.training.rope_scaling_type == "llama3"

    def test_llama3_constants_present(self):
        from soup_cli.utils.long_context import (
            LLAMA3_DEFAULT_HIGH_FREQ_FACTOR,
            LLAMA3_DEFAULT_LOW_FREQ_FACTOR,
            LLAMA3_DEFAULT_OLD_CONTEXT_LEN,
            LLAMA3_DEFAULT_SCALE_FACTOR,
        )

        assert LLAMA3_DEFAULT_SCALE_FACTOR == 8.0
        assert LLAMA3_DEFAULT_LOW_FREQ_FACTOR == 1.0
        assert LLAMA3_DEFAULT_HIGH_FREQ_FACTOR == 4.0
        assert LLAMA3_DEFAULT_OLD_CONTEXT_LEN == 8192

    def test_llama3_get_rope_scaling_config(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        cfg = get_rope_scaling_config(
            scaling_type="llama3",
            target_length=131072,
            original_length=8192,
        )
        assert cfg["type"] == "llama3"
        assert cfg["factor"] == 16.0
        assert cfg["original_max_position_embeddings"] == 8192
        assert cfg["low_freq_factor"] == 1.0
        assert cfg["high_freq_factor"] == 4.0


class TestLlama3InvFreqScale:
    def test_low_freq_passthrough(self):
        """Wavelength >> high_freq_threshold => scaled by 1/factor."""
        from soup_cli.utils.long_context import scale_inv_freq_llama3

        # A very low freq (long wavelength) should be divided by scale_factor.
        inv_freq = 1e-6  # huge wavelength
        scaled = scale_inv_freq_llama3(
            inv_freq=inv_freq,
            scale_factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            old_context_len=8192,
        )
        assert scaled == pytest.approx(inv_freq / 8.0)

    def test_high_freq_passthrough(self):
        """Wavelength << low_freq_threshold => unchanged."""
        from soup_cli.utils.long_context import scale_inv_freq_llama3

        # Very high freq (short wavelength) — no scaling.
        inv_freq = 1.0  # tiny wavelength
        scaled = scale_inv_freq_llama3(
            inv_freq=inv_freq,
            scale_factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            old_context_len=8192,
        )
        assert scaled == pytest.approx(inv_freq)

    def test_smooth_transition_zone(self):
        """Mid-band frequencies are smooth-interpolated between the two regions."""
        from soup_cli.utils.long_context import scale_inv_freq_llama3

        # Pick a wavelength in the transition band: 2π·(8192/2) ≈ ~25_700; an inv_freq
        # such that 2π/inv_freq lands between low and high thresholds.
        # low_freq_wavelen = old_context_len / low_freq_factor = 8192
        # high_freq_wavelen = old_context_len / high_freq_factor = 2048
        # pick wavelen ~4096 -> inv_freq = 2π / 4096
        inv_freq = (2.0 * math.pi) / 4096.0
        scaled = scale_inv_freq_llama3(
            inv_freq=inv_freq,
            scale_factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            old_context_len=8192,
        )
        # Must be strictly between the two endpoint behaviours.
        assert inv_freq / 8.0 < scaled < inv_freq

    @pytest.mark.parametrize(
        "param,value",
        [
            ("inv_freq", True),
            ("scale_factor", True),
            ("low_freq_factor", False),
            ("high_freq_factor", True),
            ("old_context_len", True),
        ],
    )
    def test_scale_inv_freq_llama3_rejects_bool_on_every_param(self, param, value):
        from soup_cli.utils.long_context import scale_inv_freq_llama3

        kwargs = dict(
            inv_freq=1e-4,
            scale_factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            old_context_len=8192,
        )
        kwargs[param] = value
        with pytest.raises(ValueError):
            scale_inv_freq_llama3(**kwargs)

    def test_llama3_emit_uses_original_length(self):
        """Review-fix: when llama3_old_context_len is omitted, the emitted
        ``original_max_position_embeddings`` must mirror the caller's
        ``original_length`` — not silently snap to the 8192 default."""
        from soup_cli.utils.long_context import get_rope_scaling_config

        cfg = get_rope_scaling_config(
            scaling_type="llama3",
            target_length=131072,
            original_length=16384,
        )
        assert cfg["original_max_position_embeddings"] == 16384
        # Explicit override still wins.
        cfg = get_rope_scaling_config(
            scaling_type="llama3",
            target_length=131072,
            original_length=16384,
            llama3_old_context_len=8192,
        )
        assert cfg["original_max_position_embeddings"] == 8192

    def test_scale_inv_freq_llama3_rejects_bad_inputs(self):
        from soup_cli.utils.long_context import scale_inv_freq_llama3

        with pytest.raises(ValueError):
            scale_inv_freq_llama3(
                inv_freq=1e-4,
                scale_factor=0.0,  # must be > 1
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                old_context_len=8192,
            )
        with pytest.raises(ValueError):
            scale_inv_freq_llama3(
                inv_freq=1e-4,
                scale_factor=8.0,
                low_freq_factor=4.0,
                high_freq_factor=1.0,  # high must be > low
                old_context_len=8192,
            )
        with pytest.raises(ValueError):
            scale_inv_freq_llama3(
                inv_freq=float("nan"),
                scale_factor=8.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                old_context_len=8192,
            )

    def test_auto_detect_llama3_from_config(self):
        """Auto-detect rope_scaling.type='llama3' inside an HF-style config dict."""
        from soup_cli.utils.long_context import detect_llama3_rope_in_config

        cfg = {"rope_scaling": {"type": "llama3", "factor": 8.0}}
        assert detect_llama3_rope_in_config(cfg) is True

        cfg = {"rope_scaling": {"rope_type": "llama3"}}
        assert detect_llama3_rope_in_config(cfg) is True

        cfg = {"rope_scaling": {"type": "linear"}}
        assert detect_llama3_rope_in_config(cfg) is False

        assert detect_llama3_rope_in_config({}) is False
        assert detect_llama3_rope_in_config({"rope_scaling": None}) is False

    def test_detect_llama3_rejects_non_dict(self):
        from soup_cli.utils.long_context import detect_llama3_rope_in_config

        with pytest.raises(TypeError):
            detect_llama3_rope_in_config("not a dict")  # type: ignore[arg-type]


# =====================================================================
# Cross-cutting validators
# =====================================================================


class TestGetRopeScalingConfigInputValidation:
    """Security review fix — `get_rope_scaling_config` is a public API; reject
    bool / NaN / Inf / non-positive values at the boundary so a non-schema
    caller cannot emit a corrupt ``{"factor": NaN}`` HF config."""

    def test_rejects_bool_target_length(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        with pytest.raises(ValueError, match="bool"):
            get_rope_scaling_config("linear", target_length=True, original_length=8192)

    def test_rejects_bool_original_length(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        with pytest.raises(ValueError, match="bool"):
            get_rope_scaling_config("linear", target_length=32768, original_length=True)

    def test_rejects_nan_target_length(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        with pytest.raises(ValueError, match="finite"):
            get_rope_scaling_config("linear", target_length=float("nan"), original_length=8192)

    def test_rejects_zero_original_length(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        with pytest.raises(ValueError, match="positive"):
            get_rope_scaling_config("linear", target_length=32768, original_length=0)

    def test_rejects_nan_yarn_factor(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        with pytest.raises(ValueError):
            get_rope_scaling_config(
                "yarn",
                target_length=32768,
                original_length=8192,
                yarn_factor=float("nan"),
            )


class TestCoverageGapFixes:
    """tdd-review follow-ups — close coverage gaps surfaced by the agent."""

    def test_yarn_find_correction_dim_rejects_bad_dim_base(self):
        from soup_cli.utils.long_context import yarn_find_correction_dim

        with pytest.raises(ValueError):
            yarn_find_correction_dim(
                num_rotations=32, dim=0, base=10000.0, max_position_embeddings=4096
            )
        with pytest.raises(ValueError):
            yarn_find_correction_dim(
                num_rotations=32, dim=True, base=10000.0, max_position_embeddings=4096
            )
        with pytest.raises(ValueError):
            yarn_find_correction_dim(
                num_rotations=32, dim=64, base=0.0, max_position_embeddings=4096
            )
        with pytest.raises(ValueError):
            yarn_find_correction_dim(
                num_rotations=32, dim=64, base=10000.0, max_position_embeddings=0
            )

    def test_yarn_linear_ramp_mask_rejects_bad_inputs(self):
        from soup_cli.utils.long_context import yarn_linear_ramp_mask

        with pytest.raises(ValueError):
            yarn_linear_ramp_mask(low=4, high=8, dim=0)
        with pytest.raises(ValueError):
            yarn_linear_ramp_mask(low=4, high=8, dim=True)
        with pytest.raises(ValueError):
            yarn_linear_ramp_mask(low=True, high=8, dim=16)
        with pytest.raises(ValueError):
            yarn_linear_ramp_mask(low=-1, high=8, dim=16)

    def test_yarn_get_mscale_rejects_non_finite(self):
        from soup_cli.utils.long_context import yarn_get_mscale

        with pytest.raises(ValueError):
            yarn_get_mscale(float("nan"))
        with pytest.raises(ValueError):
            yarn_get_mscale(float("inf"))
        with pytest.raises(ValueError):
            yarn_get_mscale(True)

    def test_linear_config(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        cfg = get_rope_scaling_config("linear", target_length=32768, original_length=8192)
        assert cfg == {"type": "linear", "factor": 4.0}

    def test_longrope_config(self):
        from soup_cli.utils.long_context import get_rope_scaling_config

        cfg = get_rope_scaling_config("longrope", target_length=131072, original_length=8192)
        assert cfg["type"] == "longrope"
        assert cfg["original_max_position_embeddings"] == 8192

    def test_is_llama_model_oversize_returns_false(self):
        from soup_cli.utils.longlora import is_llama_model

        big_name = "meta-llama/Llama-3" + "x" * 1024
        # > 512-char cap returns False (not raise) per the size-guard branch.
        assert is_llama_model(big_name) is False

    def test_validate_longlora_compat_rejects_unsloth(self):
        from soup_cli.utils.longlora import validate_longlora_compat

        with pytest.raises(ValueError, match="transformers"):
            validate_longlora_compat(
                model_name="meta-llama/Llama-3.1-8B",
                task="sft",
                backend="unsloth",
                use_ring_attention=False,
            )

    def test_scale_inv_freq_llama3_rejects_zero_old_context_len(self):
        from soup_cli.utils.long_context import scale_inv_freq_llama3

        with pytest.raises(ValueError, match="old_context_len"):
            scale_inv_freq_llama3(
                inv_freq=1e-4,
                scale_factor=8.0,
                low_freq_factor=1.0,
                high_freq_factor=4.0,
                old_context_len=0,
            )


class TestRopeScalingTypeLiteral:
    @pytest.mark.parametrize("rt", ["linear", "dynamic", "yarn", "longrope", "llama3"])
    def test_all_types_accepted(self, rt):
        yaml_in = f"""
base: meta-llama/Llama-3.1-8B
task: sft
training:
  rope_scaling_type: {rt}
"""
        cfg = _load(yaml_in)
        assert cfg.training.rope_scaling_type == rt

    def test_unknown_type_rejected(self):
        yaml_in = """
base: meta-llama/Llama-3.1-8B
task: sft
training:
  rope_scaling_type: ntk
"""
        with pytest.raises((ValidationError, ValueError)):
            _load(yaml_in)
