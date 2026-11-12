"""#2765 - ``training.vllm_sleep_mode`` is refused at config load.

GRPO never sets TRL's ``use_vllm``, so no vLLM engine exists to put to sleep, yet the
key used to load and the trainer printed "vLLM sleep mode enabled". It is now refused
on every backend, with a message that names the cause and the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string

_DOCS = Path(__file__).resolve().parent.parent / "docs"


def _grpo_yaml(backend: str, sleep_line: str = "") -> str:
    return f"""
base: test-llama
task: grpo
backend: {backend}
data:
  train: ./data.jsonl
  format: chatml
output: ./out
training:
  epochs: 1
  lr: 1e-4
  reward_fn: accuracy
{sleep_line}
"""


@pytest.mark.parametrize("backend", ["transformers", "unsloth"])
def test_grpo_vllm_sleep_mode_is_refused_at_load(backend):
    with pytest.raises((ValidationError, ValueError)) as excinfo:
        load_config_from_string(_grpo_yaml(backend, "  vllm_sleep_mode: true\n"))
    message = " ".join(str(excinfo.value).split())
    assert "training.vllm_sleep_mode has no effect" in message
    assert "use_vllm" in message
    assert "v0.78.0" in message
    assert "Delete the line" in message


@pytest.mark.parametrize("backend", ["transformers", "unsloth"])
def test_grpo_without_vllm_sleep_mode_still_loads(backend):
    cfg = load_config_from_string(_grpo_yaml(backend))
    assert cfg.training.vllm_sleep_mode is False


def test_explicit_false_still_loads():
    cfg = load_config_from_string(_grpo_yaml("transformers", "  vllm_sleep_mode: false\n"))
    assert cfg.training.vllm_sleep_mode is False


def test_training_guide_no_longer_calls_the_key_live():
    text = (_DOCS / "training.md").read_text(encoding="utf-8")
    sleep_lines = [line for line in text.splitlines() if "vllm_sleep_mode" in line]
    assert sleep_lines, "docs/training.md should still explain the refused key"
    for line in sleep_lines:
        assert "LIVE" not in line, line
    assert "`vllm_sleep_mode`, `openenv` rollout, and PRM training" not in text
