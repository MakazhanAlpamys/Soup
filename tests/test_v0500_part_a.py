"""Tests for v0.50.0 Part A — GRPO objective variants.

Covers allowlist validation, metadata immutability, delta-required gates,
deferred live-wiring stubs, and schema integration.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig, TrainingConfig
from soup_cli.utils import grpo_variants
from soup_cli.utils.grpo_variants import (
    SUPPORTED_GRPO_VARIANTS,
    GRPOVariantSpec,
    apply_variant_loss,
    get_variant_spec,
    list_variants,
    validate_grpo_delta,
    validate_grpo_variant,
    variant_is_live_wired,
    variant_requires_delta,
)

# ---------------------------------------------------------------------------
# Allowlist surface
# ---------------------------------------------------------------------------


def test_supported_variants_is_frozenset():
    assert isinstance(SUPPORTED_GRPO_VARIANTS, frozenset)
    with pytest.raises(AttributeError):
        SUPPORTED_GRPO_VARIANTS.add("evil")  # type: ignore[attr-defined]


def test_supported_variants_includes_all_v0500():
    for name in ("gspo", "dapo", "dr_grpo", "bnpo", "two_sided", "rft"):
        assert name in SUPPORTED_GRPO_VARIANTS


def test_supported_variants_includes_standard():
    assert "standard" in SUPPORTED_GRPO_VARIANTS


def test_list_variants_sorted_tuple():
    result = list_variants()
    assert isinstance(result, tuple)
    assert list(result) == sorted(result)


# ---------------------------------------------------------------------------
# validate_grpo_variant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["gspo", "dapo", "dr_grpo", "bnpo", "two_sided", "rft", "standard"]
)
def test_validate_grpo_variant_happy(name):
    assert validate_grpo_variant(name) == name


def test_validate_grpo_variant_case_insensitive():
    assert validate_grpo_variant("GSPO") == "gspo"
    assert validate_grpo_variant("DAPO") == "dapo"


def test_validate_grpo_variant_non_string():
    with pytest.raises(ValueError, match="must be a string"):
        validate_grpo_variant(123)
    with pytest.raises(ValueError, match="must be a string"):
        validate_grpo_variant(None)


def test_validate_grpo_variant_empty():
    with pytest.raises(ValueError, match="non-empty"):
        validate_grpo_variant("")


def test_validate_grpo_variant_null_byte():
    with pytest.raises(ValueError, match="null byte"):
        validate_grpo_variant("gspo\x00")


def test_validate_grpo_variant_oversize():
    with pytest.raises(ValueError, match="exceeds"):
        validate_grpo_variant("x" * 100)


def test_validate_grpo_variant_unknown():
    with pytest.raises(ValueError, match="not supported"):
        validate_grpo_variant("trpo")


def test_validate_grpo_variant_bool_rejected():
    """tdd-guide HIGH fix: explicit bool guard test."""
    with pytest.raises(ValueError, match="bool"):
        validate_grpo_variant(True)
    with pytest.raises(ValueError, match="bool"):
        validate_grpo_variant(False)


def test_deferred_live_invariant():
    """tdd-guide LOW fix: catch future drift in allowlist split.

    v0.53.11 #123 lifted every entry — ``_DEFERRED_LIVE`` is now empty
    because all 6 variants ship with live math kernels.
    """
    from soup_cli.utils.grpo_variants import _DEFERRED_LIVE
    assert len(SUPPORTED_GRPO_VARIANTS) == 7
    assert _DEFERRED_LIVE == frozenset()


def test_variant_spec_description_and_requires_delta():
    """tdd-guide LOW fix: exercise unread spec fields."""
    spec = get_variant_spec("two_sided")
    assert spec.requires_delta is True
    assert "two-sided" in spec.description.lower() or "two_sided" in spec.description.lower()
    spec_gspo = get_variant_spec("gspo")
    assert spec_gspo.requires_delta is False
    assert "stabilized" in spec_gspo.description.lower()


# ---------------------------------------------------------------------------
# Variant metadata
# ---------------------------------------------------------------------------


def test_get_variant_spec_returns_frozen_dataclass():
    spec = get_variant_spec("gspo")
    assert isinstance(spec, GRPOVariantSpec)
    assert spec.name == "gspo"
    with pytest.raises(Exception):
        spec.name = "evil"  # type: ignore[misc]


def test_get_variant_spec_unknown_raises():
    with pytest.raises(ValueError):
        get_variant_spec("trpo")


def test_two_sided_requires_delta():
    assert variant_requires_delta("two_sided") is True
    assert variant_requires_delta("TWO_SIDED") is True


@pytest.mark.parametrize("name", ["gspo", "dapo", "dr_grpo", "bnpo", "rft", "standard"])
def test_other_variants_do_not_require_delta(name):
    assert variant_requires_delta(name) is False


def test_variant_requires_delta_non_string():
    assert variant_requires_delta(123) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Live-wiring flags
# ---------------------------------------------------------------------------


def test_standard_is_live_wired():
    assert variant_is_live_wired("standard") is True


@pytest.mark.parametrize("name", ["gspo", "dapo", "dr_grpo", "bnpo", "two_sided", "rft"])
def test_v0500_variants_now_live(name):
    """v0.53.11 #123 lifted every deferred variant — all are live wired now."""
    assert variant_is_live_wired(name) is True


def test_variant_is_live_wired_non_string():
    assert variant_is_live_wired(None) is False  # type: ignore[arg-type]


def test_apply_variant_loss_standard_is_noop():
    # standard variant returns None (delegates to existing GRPOTrainerWrapper).
    # v0.53.11 #123 changed the signature to require logp_new/logp_old/advantages
    # as kwargs.
    torch = pytest.importorskip("torch")
    logp_new = torch.zeros(2, 4)
    logp_old = torch.zeros(2, 4)
    advantages = torch.zeros(2)
    assert apply_variant_loss(
        "standard",
        logp_new=logp_new,
        logp_old=logp_old,
        advantages=advantages,
    ) is None


def test_apply_variant_loss_unknown_raises_validation():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="not supported"):
        apply_variant_loss(
            "trpo",
            logp_new=torch.zeros(2, 4),
            logp_old=torch.zeros(2, 4),
            advantages=torch.zeros(2),
        )


# ---------------------------------------------------------------------------
# validate_grpo_delta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", [0.1, 0.5, 1.0, 0.01])
def test_validate_grpo_delta_happy(v):
    assert validate_grpo_delta(v) == v


def test_validate_grpo_delta_int_coerced():
    assert validate_grpo_delta(1) == 1.0


def test_validate_grpo_delta_bool_rejected():
    with pytest.raises(ValueError, match="bool"):
        validate_grpo_delta(True)
    with pytest.raises(ValueError, match="bool"):
        validate_grpo_delta(False)


def test_validate_grpo_delta_non_number():
    with pytest.raises(ValueError, match="must be a number"):
        validate_grpo_delta("0.5")


def test_validate_grpo_delta_nan_inf():
    with pytest.raises(ValueError, match="finite"):
        validate_grpo_delta(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        validate_grpo_delta(float("inf"))


@pytest.mark.parametrize("v", [0.0, -0.1, 1.5, 2.0])
def test_validate_grpo_delta_out_of_range(v):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        validate_grpo_delta(v)


# ---------------------------------------------------------------------------
# Module-level immutability
# ---------------------------------------------------------------------------


def test_variant_metadata_immutable():
    with pytest.raises(TypeError):
        grpo_variants._VARIANT_METADATA["evil"] = "x"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Schema integration (TrainingConfig.grpo_variant + grpo_delta + grpo_fp16)
# ---------------------------------------------------------------------------


def test_training_config_default_grpo_variant_is_none():
    tc = TrainingConfig()
    assert tc.grpo_variant is None
    assert tc.grpo_delta is None
    assert tc.grpo_fp16 is False


@pytest.mark.parametrize("name", ["gspo", "dapo", "dr_grpo", "bnpo", "rft"])
def test_training_config_accepts_v0500_variant(name):
    tc = TrainingConfig(grpo_variant=name)
    assert tc.grpo_variant == name


def test_training_config_two_sided_requires_delta():
    with pytest.raises(ValidationError, match="grpo_delta"):
        TrainingConfig(grpo_variant="two_sided")


def test_training_config_two_sided_with_delta_ok():
    tc = TrainingConfig(grpo_variant="two_sided", grpo_delta=0.3)
    assert tc.grpo_variant == "two_sided"
    assert tc.grpo_delta == 0.3


def test_training_config_delta_without_two_sided_rejected():
    with pytest.raises(ValidationError, match="two_sided"):
        TrainingConfig(grpo_variant="gspo", grpo_delta=0.3)


def test_training_config_unknown_variant_rejected():
    with pytest.raises(ValidationError, match="literal_error|Input should be"):
        TrainingConfig(grpo_variant="trpo")


def test_training_config_grpo_delta_nan_rejected():
    """Security review fix: NaN rejected at schema layer (either Pydantic
    bounds or the explicit field_validator catches it)."""
    with pytest.raises(ValidationError):
        TrainingConfig(grpo_variant="two_sided", grpo_delta=float("nan"))


def test_training_config_grpo_delta_inf_rejected():
    """Security review fix: explicit Inf rejection at schema layer."""
    with pytest.raises(ValidationError):
        TrainingConfig(grpo_variant="two_sided", grpo_delta=float("inf"))


def test_training_config_grpo_fp16_accepted():
    tc = TrainingConfig(grpo_fp16=True)
    assert tc.grpo_fp16 is True


def test_training_config_grpo_fp16_int_coerced_to_bool():
    # Pydantic v2 default coerces "yes"-like strings to bool; assert behaviour
    # is consistent (1/0 -> True/False; arbitrary strings rejected).
    tc = TrainingConfig(grpo_fp16=1)
    assert tc.grpo_fp16 is True
    with pytest.raises(ValidationError):
        TrainingConfig(grpo_fp16="garbage")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SoupConfig cross-validators
# ---------------------------------------------------------------------------


def _grpo_yaml(extra: str = "") -> str:
    return f"""
base: test-llama
task: grpo
data:
  train: ./data.jsonl
  format: chatml
output: ./out
training:
  epochs: 1
  lr: 1e-4
  reward_fn: accuracy
{extra}
"""


def test_soupconfig_grpo_variant_happy():
    yaml = _grpo_yaml("  grpo_variant: dapo\n")
    cfg = load_config_from_string(yaml)
    assert isinstance(cfg, SoupConfig)
    assert cfg.training.grpo_variant == "dapo"


def test_soupconfig_grpo_variant_rejected_on_non_grpo_task():
    yaml = """
base: test-llama
task: sft
data:
  train: ./data.jsonl
  format: chatml
output: ./out
training:
  epochs: 1
  lr: 1e-4
  grpo_variant: dapo
"""
    with pytest.raises((ValidationError, ValueError), match="task='grpo'|task=.grpo."):
        load_config_from_string(yaml)


def test_soupconfig_two_sided_with_delta_happy():
    yaml = _grpo_yaml("  grpo_variant: two_sided\n  grpo_delta: 0.3\n")
    cfg = load_config_from_string(yaml)
    assert cfg.training.grpo_variant == "two_sided"
    assert math.isclose(cfg.training.grpo_delta, 0.3)


def test_soupconfig_mlx_backend_rejected():
    yaml = """
base: test-llama
task: grpo
backend: mlx
data:
  train: ./data.jsonl
  format: chatml
output: ./out
training:
  epochs: 1
  lr: 1e-4
  reward_fn: accuracy
  grpo_variant: gspo
"""
    with pytest.raises((ValidationError, ValueError), match="mlx"):
        load_config_from_string(yaml)
