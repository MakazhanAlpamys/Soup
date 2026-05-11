"""Tests for v0.50.0 Part E — PRM task + Vision RL."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TrainingConfig
from soup_cli.utils.prm import (
    build_prm_trainer,
    validate_prm_compat,
    validate_vision_grpo_compat,
)

# ---------------------------------------------------------------------------
# validate_prm_compat
# ---------------------------------------------------------------------------


def test_prm_compat_happy_prm_format():
    validate_prm_compat(task="prm", data_format="prm", backend="transformers", modality="text")


def test_prm_compat_happy_auto_format():
    validate_prm_compat(task="prm", data_format="auto", backend="transformers", modality="text")


def test_prm_compat_rejects_wrong_format():
    with pytest.raises(ValueError, match="format"):
        validate_prm_compat(
            task="prm", data_format="alpaca", backend="transformers", modality="text",
        )


def test_prm_compat_rejects_mlx():
    with pytest.raises(ValueError, match="mlx"):
        validate_prm_compat(task="prm", data_format="prm", backend="mlx", modality="text")


def test_prm_compat_rejects_vision_modality():
    with pytest.raises(ValueError, match="modality"):
        validate_prm_compat(
            task="prm", data_format="prm", backend="transformers", modality="vision",
        )


def test_prm_compat_rejects_audio_modality():
    with pytest.raises(ValueError, match="modality"):
        validate_prm_compat(
            task="prm", data_format="prm", backend="transformers", modality="audio",
        )


def test_prm_compat_rejects_non_prm_task():
    with pytest.raises(ValueError, match="prm"):
        validate_prm_compat(
            task="sft", data_format="prm", backend="transformers", modality="text",
        )


def test_prm_compat_empty_task():
    with pytest.raises(ValueError, match="task"):
        validate_prm_compat(task="", data_format="prm", backend="transformers", modality="text")


def test_prm_compat_empty_format():
    with pytest.raises(ValueError, match="format"):
        validate_prm_compat(task="prm", data_format="", backend="transformers", modality="text")


def test_prm_compat_none_task():
    """tdd-guide MEDIUM fix: non-string task rejection."""
    with pytest.raises(ValueError, match="task"):
        validate_prm_compat(
            task=None,  # type: ignore[arg-type]
            data_format="prm",
            backend="transformers",
            modality="text",
        )


def test_prm_compat_none_format():
    """tdd-guide MEDIUM fix: non-string data_format rejection."""
    with pytest.raises(ValueError, match="format"):
        validate_prm_compat(
            task="prm",
            data_format=None,  # type: ignore[arg-type]
            backend="transformers",
            modality="text",
        )


def test_vision_grpo_soupconfig_ppo_happy():
    """tdd-guide MEDIUM fix: confirm ppo path is wired at SoupConfig level."""
    yaml = """
base: test-llama
task: ppo
modality: vision
data:
  train: ./data.jsonl
  format: chatml
output: ./out
training:
  epochs: 1
  lr: 1e-4
  vision_grpo: true
"""
    cfg = load_config_from_string(yaml)
    assert cfg.training.vision_grpo is True
    assert cfg.task == "ppo"


def test_build_prm_trainer_deferred():
    with pytest.raises(NotImplementedError, match="v0.50.1"):
        build_prm_trainer()


# ---------------------------------------------------------------------------
# validate_vision_grpo_compat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["grpo", "ppo"])
def test_vision_grpo_happy(task):
    validate_vision_grpo_compat(task=task, modality="vision", backend="transformers")


def test_vision_grpo_rejects_sft():
    with pytest.raises(ValueError, match="task"):
        validate_vision_grpo_compat(task="sft", modality="vision", backend="transformers")


def test_vision_grpo_rejects_text_modality():
    with pytest.raises(ValueError, match="modality"):
        validate_vision_grpo_compat(task="grpo", modality="text", backend="transformers")


def test_vision_grpo_rejects_audio():
    with pytest.raises(ValueError, match="modality"):
        validate_vision_grpo_compat(task="grpo", modality="audio", backend="transformers")


def test_vision_grpo_rejects_mlx():
    with pytest.raises(ValueError, match="mlx"):
        validate_vision_grpo_compat(task="grpo", modality="vision", backend="mlx")


def test_vision_grpo_empty_task():
    with pytest.raises(ValueError, match="task"):
        validate_vision_grpo_compat(task="", modality="vision", backend="transformers")


# ---------------------------------------------------------------------------
# Schema field defaults + acceptance
# ---------------------------------------------------------------------------


def test_vision_grpo_default_false():
    assert TrainingConfig().vision_grpo is False


def test_vision_grpo_accepts_bool():
    tc = TrainingConfig(vision_grpo=True)
    assert tc.vision_grpo is True


# ---------------------------------------------------------------------------
# SoupConfig integration
# ---------------------------------------------------------------------------


def test_prm_task_accepted():
    yaml = """
base: test-llama
task: prm
data:
  train: ./data.jsonl
  format: prm
output: ./out
training:
  epochs: 1
  lr: 1e-4
"""
    cfg = load_config_from_string(yaml)
    assert cfg.task == "prm"
    assert cfg.data.format == "prm"


def test_prm_with_wrong_format_rejected():
    yaml = """
base: test-llama
task: prm
data:
  train: ./data.jsonl
  format: alpaca
output: ./out
training:
  epochs: 1
  lr: 1e-4
"""
    with pytest.raises((ValidationError, ValueError), match="format"):
        load_config_from_string(yaml)


def test_prm_on_mlx_rejected():
    yaml = """
base: test-llama
task: prm
backend: mlx
data:
  train: ./data.jsonl
  format: prm
output: ./out
training:
  epochs: 1
  lr: 1e-4
"""
    with pytest.raises((ValidationError, ValueError), match="mlx"):
        load_config_from_string(yaml)


def test_vision_grpo_soupconfig_happy():
    yaml = """
base: test-llama
task: grpo
modality: vision
data:
  train: ./data.jsonl
  format: chatml
output: ./out
training:
  epochs: 1
  lr: 1e-4
  reward_fn: accuracy
  vision_grpo: true
"""
    cfg = load_config_from_string(yaml)
    assert cfg.training.vision_grpo is True


def test_vision_grpo_on_text_modality_rejected():
    yaml = """
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
  vision_grpo: true
"""
    with pytest.raises((ValidationError, ValueError), match="modality"):
        load_config_from_string(yaml)


def test_vision_grpo_on_sft_rejected():
    yaml = """
base: test-llama
task: sft
modality: vision
data:
  train: ./data.jsonl
  format: chatml
output: ./out
training:
  epochs: 1
  lr: 1e-4
  vision_grpo: true
"""
    with pytest.raises((ValidationError, ValueError), match="task"):
        load_config_from_string(yaml)
