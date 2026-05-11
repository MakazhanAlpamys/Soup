"""Tests for v0.50.0 Part D — GRPO stability / efficiency knobs.

7 schema fields (ref_model_ema_alpha, replay_buffer_size, async_grpo_prefetch,
tis_threshold, mask_truncated_completions, defer_rerolling, skip_zero_advantage,
off_policy_mask_threshold) — schema-only this release.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TrainingConfig

# ---------------------------------------------------------------------------
# TrainingConfig defaults
# ---------------------------------------------------------------------------


def test_defaults_all_disabled():
    tc = TrainingConfig()
    assert tc.ref_model_ema_alpha is None
    assert tc.replay_buffer_size is None
    assert tc.async_grpo_prefetch is False
    assert tc.tis_threshold is None
    assert tc.mask_truncated_completions is False
    assert tc.defer_rerolling is False
    assert tc.skip_zero_advantage is False
    assert tc.off_policy_mask_threshold is None


# ---------------------------------------------------------------------------
# Field bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("v", [0.001, 0.5, 1.0])
def test_ref_model_ema_alpha_valid(v):
    tc = TrainingConfig(ref_model_ema_alpha=v)
    assert tc.ref_model_ema_alpha == v


@pytest.mark.parametrize("v", [0.0, -0.1, 1.5])
def test_ref_model_ema_alpha_out_of_range(v):
    with pytest.raises(ValidationError):
        TrainingConfig(ref_model_ema_alpha=v)


@pytest.mark.parametrize("v", [1, 1000, 1_000_000])
def test_replay_buffer_size_valid(v):
    tc = TrainingConfig(replay_buffer_size=v)
    assert tc.replay_buffer_size == v


@pytest.mark.parametrize("v", [0, -1, 1_000_001])
def test_replay_buffer_size_out_of_range(v):
    with pytest.raises(ValidationError):
        TrainingConfig(replay_buffer_size=v)


@pytest.mark.parametrize("v", [0.01, 1.0, 100.0])
def test_tis_threshold_valid(v):
    tc = TrainingConfig(tis_threshold=v)
    assert tc.tis_threshold == v


@pytest.mark.parametrize("v", [0.0, -0.1, 100.1])
def test_tis_threshold_out_of_range(v):
    with pytest.raises(ValidationError):
        TrainingConfig(tis_threshold=v)


@pytest.mark.parametrize("v", [0.0, 0.5, 1.0])
def test_off_policy_mask_threshold_valid(v):
    tc = TrainingConfig(off_policy_mask_threshold=v)
    assert tc.off_policy_mask_threshold == v


@pytest.mark.parametrize("v", [-0.1, 1.1])
def test_off_policy_mask_threshold_out_of_range(v):
    with pytest.raises(ValidationError):
        TrainingConfig(off_policy_mask_threshold=v)


# ---------------------------------------------------------------------------
# Cross-validator: mask_truncated_completions requires tis_threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", [
    "ref_model_ema_alpha", "tis_threshold",
    "off_policy_mask_threshold", "replay_buffer_size",
])
def test_stability_numeric_bool_rejected(field):
    """tdd-guide HIGH fix: bool-before-int/float guard per project policy."""
    with pytest.raises(ValidationError):
        TrainingConfig(**{field: True})


def test_mask_without_tis_rejected():
    with pytest.raises(ValidationError, match="tis_threshold"):
        TrainingConfig(mask_truncated_completions=True)


def test_mask_with_tis_ok():
    tc = TrainingConfig(mask_truncated_completions=True, tis_threshold=2.0)
    assert tc.mask_truncated_completions is True
    assert tc.tis_threshold == 2.0


def test_tis_without_mask_ok():
    # tis_threshold alone is fine (use IS without truncation masking).
    tc = TrainingConfig(tis_threshold=2.0)
    assert tc.tis_threshold == 2.0
    assert tc.mask_truncated_completions is False


# ---------------------------------------------------------------------------
# SoupConfig task gate
# ---------------------------------------------------------------------------


def _grpo_yaml(extra: str) -> str:
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


def _sft_yaml(extra: str) -> str:
    return f"""
base: test-llama
task: sft
data:
  train: ./data.jsonl
  format: chatml
output: ./out
training:
  epochs: 1
  lr: 1e-4
{extra}
"""


@pytest.mark.parametrize("field,value", [
    ("ref_model_ema_alpha", 0.99),
    ("replay_buffer_size", 1000),
    ("async_grpo_prefetch", True),
    ("tis_threshold", 2.0),
    ("defer_rerolling", True),
    ("skip_zero_advantage", True),
    ("off_policy_mask_threshold", 0.5),
])
def test_stability_field_on_grpo_happy(field, value):
    cfg = load_config_from_string(_grpo_yaml(f"  {field}: {value}\n"))
    assert getattr(cfg.training, field) == value


@pytest.mark.parametrize("field,value", [
    ("ref_model_ema_alpha", 0.99),
    ("replay_buffer_size", 1000),
    ("async_grpo_prefetch", True),
    ("tis_threshold", 2.0),
    ("defer_rerolling", True),
    ("skip_zero_advantage", True),
    ("off_policy_mask_threshold", 0.5),
])
def test_stability_field_on_sft_rejected(field, value):
    with pytest.raises((ValidationError, ValueError), match="task='grpo'|task=.grpo."):
        load_config_from_string(_sft_yaml(f"  {field}: {value}\n"))


def test_stability_field_on_mlx_rejected():
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
  ref_model_ema_alpha: 0.99
"""
    with pytest.raises((ValidationError, ValueError), match="mlx"):
        load_config_from_string(yaml)


def test_grpo_fp16_on_sft_rejected():
    """Code-review HIGH fix: grpo_fp16 requires task='grpo'."""
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
  grpo_fp16: true
"""
    with pytest.raises((ValidationError, ValueError), match="task='grpo'|task=.grpo."):
        load_config_from_string(yaml)


def test_grpo_fp16_on_grpo_happy():
    yaml = _grpo_yaml("  grpo_fp16: true\n")
    cfg = load_config_from_string(yaml)
    assert cfg.training.grpo_fp16 is True


def test_combined_stability_fields_named_in_error():
    yaml = _sft_yaml(
        "  ref_model_ema_alpha: 0.99\n"
        "  replay_buffer_size: 1000\n"
    )
    with pytest.raises((ValidationError, ValueError), match="ref_model_ema_alpha"):
        load_config_from_string(yaml)
