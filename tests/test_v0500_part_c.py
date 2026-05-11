"""Tests for v0.50.0 Part C — Multi-turn agent rollout backends."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import TrainingConfig
from soup_cli.utils import agent_rollout
from soup_cli.utils.agent_rollout import (
    SUPPORTED_ROLLOUT_BACKENDS,
    RolloutBackendSpec,
    get_rollout_backend_spec,
    launch_rollout,
    list_rollout_backends,
    required_rollout_package,
    validate_rollout_backend,
)


def test_supported_rollout_backends_frozenset():
    assert isinstance(SUPPORTED_ROLLOUT_BACKENDS, frozenset)


def test_supported_rollout_backends_contents():
    assert SUPPORTED_ROLLOUT_BACKENDS == {"art", "ruler", "nemo_gym", "openenv"}


def test_list_rollout_backends_sorted_tuple():
    result = list_rollout_backends()
    assert isinstance(result, tuple)
    assert list(result) == sorted(result)


@pytest.mark.parametrize("name", ["art", "ruler", "nemo_gym", "openenv"])
def test_validate_rollout_backend_happy(name):
    assert validate_rollout_backend(name) == name


def test_validate_rollout_backend_case_insensitive():
    assert validate_rollout_backend("ART") == "art"
    assert validate_rollout_backend("NeMo_Gym") == "nemo_gym"


def test_validate_rollout_backend_non_string():
    with pytest.raises(ValueError, match="must be a string"):
        validate_rollout_backend(123)


def test_validate_rollout_backend_empty():
    with pytest.raises(ValueError, match="non-empty"):
        validate_rollout_backend("")


def test_validate_rollout_backend_null_byte():
    with pytest.raises(ValueError, match="null byte"):
        validate_rollout_backend("art\x00")


def test_validate_rollout_backend_oversize():
    with pytest.raises(ValueError, match="exceeds"):
        validate_rollout_backend("x" * 100)


def test_validate_rollout_backend_unknown():
    with pytest.raises(ValueError, match="not supported"):
        validate_rollout_backend("trlx")


def test_validate_rollout_backend_bool_rejected():
    """tdd-guide HIGH fix: explicit bool guard."""
    with pytest.raises(ValueError, match="bool"):
        validate_rollout_backend(True)


def test_required_rollout_package_unknown_raises():
    """tdd-guide MEDIUM fix: rejection path through helper."""
    with pytest.raises(ValueError, match="not supported"):
        required_rollout_package("trlx")


@pytest.mark.parametrize("name", ["art", "ruler", "nemo_gym", "openenv"])
def test_rollout_backend_spec_not_live_wired(name):
    """tdd-guide LOW fix: exercise live_wired field."""
    assert get_rollout_backend_spec(name).live_wired is False


def test_get_rollout_backend_spec_frozen():
    spec = get_rollout_backend_spec("art")
    assert isinstance(spec, RolloutBackendSpec)
    with pytest.raises(Exception):
        spec.name = "evil"  # type: ignore[misc]


@pytest.mark.parametrize("name,pkg", [
    ("art", "openpipe-art"),
    ("ruler", "ruler-eval"),
    ("nemo_gym", "nemo-gym"),
    ("openenv", None),
])
def test_required_rollout_package(name, pkg):
    assert required_rollout_package(name) == pkg


def test_metadata_immutable():
    with pytest.raises(TypeError):
        agent_rollout._BACKEND_METADATA["evil"] = None  # type: ignore[index]


@pytest.mark.parametrize("name", ["art", "ruler", "nemo_gym", "openenv"])
def test_launch_rollout_deferred(name):
    with pytest.raises(NotImplementedError, match="v0.50.1"):
        launch_rollout(name)


def test_launch_rollout_unknown_validation_first():
    with pytest.raises(ValueError, match="not supported"):
        launch_rollout("trlx")


# ---------------------------------------------------------------------------
# Schema integration
# ---------------------------------------------------------------------------


def test_training_config_default_rollout_none():
    assert TrainingConfig().rollout_backend is None


@pytest.mark.parametrize("name", ["art", "ruler", "nemo_gym", "openenv"])
def test_training_config_accepts_rollout(name):
    tc = TrainingConfig(rollout_backend=name)
    assert tc.rollout_backend == name


def test_training_config_unknown_rollout_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig(rollout_backend="trlx")


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


def test_soupconfig_rollout_happy():
    cfg = load_config_from_string(_grpo_yaml("  rollout_backend: art\n"))
    assert cfg.training.rollout_backend == "art"


def test_soupconfig_rollout_on_sft_rejected():
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
  rollout_backend: art
"""
    with pytest.raises((ValidationError, ValueError), match="task='grpo'|task=.grpo."):
        load_config_from_string(yaml)


def test_soupconfig_rollout_on_mlx_rejected():
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
  rollout_backend: art
"""
    with pytest.raises((ValidationError, ValueError), match="mlx"):
        load_config_from_string(yaml)
