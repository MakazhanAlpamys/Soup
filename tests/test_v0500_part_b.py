"""Tests for v0.50.0 Part B — Long-context GRPO + vLLM sleep mode.

Schema-only release; live wiring (Tiled MLP, vLLM sleep) is deferred.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TrainingConfig
from soup_cli.utils.grpo_long_context import (
    apply_vllm_sleep_mode,
    validate_long_context_grpo_compat,
    validate_vllm_sleep_mode_compat,
)

# ---------------------------------------------------------------------------
# TrainingConfig field defaults + acceptance
# ---------------------------------------------------------------------------


def test_long_context_grpo_default_false():
    assert TrainingConfig().long_context_grpo is False


def test_vllm_sleep_mode_default_false():
    assert TrainingConfig().vllm_sleep_mode is False


def test_long_context_grpo_accepts_bool():
    tc = TrainingConfig(long_context_grpo=True)
    assert tc.long_context_grpo is True


def test_vllm_sleep_mode_accepts_bool():
    tc = TrainingConfig(vllm_sleep_mode=True)
    assert tc.vllm_sleep_mode is True


# ---------------------------------------------------------------------------
# validate_long_context_grpo_compat
# ---------------------------------------------------------------------------


def test_long_context_grpo_happy():
    validate_long_context_grpo_compat(
        task="grpo", backend="transformers", use_ring_attention=False,
    )


def test_long_context_grpo_rejects_non_grpo():
    with pytest.raises(ValueError, match="task='grpo'"):
        validate_long_context_grpo_compat(
            task="sft", backend="transformers", use_ring_attention=False,
        )


def test_long_context_grpo_rejects_mlx():
    with pytest.raises(ValueError, match="mlx"):
        validate_long_context_grpo_compat(
            task="grpo", backend="mlx", use_ring_attention=False,
        )


def test_long_context_grpo_rejects_ring_attention():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_long_context_grpo_compat(
            task="grpo", backend="transformers", use_ring_attention=True,
        )


def test_long_context_grpo_task_must_be_string():
    with pytest.raises(ValueError, match="task"):
        validate_long_context_grpo_compat(
            task=None,  # type: ignore[arg-type]
            backend="transformers",
            use_ring_attention=False,
        )


def test_long_context_grpo_empty_task():
    with pytest.raises(ValueError):
        validate_long_context_grpo_compat(
            task="", backend="transformers", use_ring_attention=False,
        )


def test_long_context_grpo_null_byte_task():
    """Security review MEDIUM fix."""
    with pytest.raises(ValueError, match="null byte"):
        validate_long_context_grpo_compat(
            task="grpo\x00", backend="transformers", use_ring_attention=False,
        )


def test_long_context_grpo_null_byte_backend():
    """Security review MEDIUM fix."""
    with pytest.raises(ValueError, match="null byte"):
        validate_long_context_grpo_compat(
            task="grpo", backend="transformers\x00", use_ring_attention=False,
        )


def test_long_context_grpo_use_ring_attention_must_be_bool():
    """Security review MEDIUM fix."""
    with pytest.raises(ValueError, match="bool"):
        validate_long_context_grpo_compat(
            task="grpo", backend="transformers", use_ring_attention=1,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# validate_vllm_sleep_mode_compat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["transformers", "unsloth"])
def test_vllm_sleep_mode_happy(backend):
    validate_vllm_sleep_mode_compat(backend=backend)


def test_vllm_sleep_mode_rejects_mlx():
    with pytest.raises(ValueError, match="vllm_sleep_mode"):
        validate_vllm_sleep_mode_compat(backend="mlx")


def test_vllm_sleep_mode_backend_must_be_string():
    with pytest.raises(ValueError, match="backend"):
        validate_vllm_sleep_mode_compat(backend=None)  # type: ignore[arg-type]


def test_vllm_sleep_mode_empty_backend():
    with pytest.raises(ValueError):
        validate_vllm_sleep_mode_compat(backend="")


def test_vllm_sleep_mode_null_byte_backend():
    """tdd-guide MEDIUM fix: null-byte rejection on backend string."""
    with pytest.raises(ValueError, match="null byte"):
        validate_vllm_sleep_mode_compat(backend="transformers\x00")


# ---------------------------------------------------------------------------
# Deferred live wiring
# ---------------------------------------------------------------------------


def test_apply_vllm_sleep_mode_deferred():
    """tdd-guide MEDIUM fix: assert v0.50.1 marker in error msg."""
    with pytest.raises(NotImplementedError, match=r"v0\.50\.1"):
        apply_vllm_sleep_mode(object())


# ---------------------------------------------------------------------------
# SoupConfig integration
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


def test_soupconfig_long_context_grpo_happy():
    cfg = load_config_from_string(_grpo_yaml("  long_context_grpo: true\n"))
    assert cfg.training.long_context_grpo is True


def test_soupconfig_long_context_grpo_with_ring_attention_rejected():
    yaml = _grpo_yaml("  long_context_grpo: true\n  use_ring_attention: true\n")
    with pytest.raises((ValidationError, ValueError), match="mutually exclusive"):
        load_config_from_string(yaml)


def test_soupconfig_long_context_grpo_on_sft_rejected():
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
  long_context_grpo: true
"""
    with pytest.raises((ValidationError, ValueError), match="task='grpo'|task=.grpo."):
        load_config_from_string(yaml)


def test_soupconfig_vllm_sleep_mode_happy():
    cfg = load_config_from_string(_grpo_yaml("  vllm_sleep_mode: true\n"))
    assert cfg.training.vllm_sleep_mode is True


def test_soupconfig_vllm_sleep_mode_on_sft_rejected():
    """Code-review HIGH fix: vllm_sleep_mode requires task='grpo'."""
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
  vllm_sleep_mode: true
"""
    with pytest.raises((ValidationError, ValueError), match="task='grpo'|task=.grpo."):
        load_config_from_string(yaml)


def test_soupconfig_vllm_sleep_mode_on_mlx_rejected():
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
  vllm_sleep_mode: true
"""
    with pytest.raises((ValidationError, ValueError), match="vllm_sleep_mode"):
        load_config_from_string(yaml)
