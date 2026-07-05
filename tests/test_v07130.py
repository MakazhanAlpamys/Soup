"""v0.71.30 — PRM-guided GRPO + bundled rollout envs.

Tests the pure PRM reward kernels (split_steps / aggregate_step_scores), the
schema fields + cross-validators, the torch-lazy PRMScorer (safetensors head
load + scoring), the GRPO wiring, the bundled rollout envs, and the recipes.
"""
import ast
import inspect

import pytest

from soup_cli.utils.prm_reward import (
    AGGREGATE_MODES,
    aggregate_step_scores,
    split_steps,
)


# ---------------------------------------------------------------------------
# Task 1 — pure kernels
# ---------------------------------------------------------------------------
class TestSplitSteps:
    def test_splits_on_newlines(self):
        assert split_steps("a\nb\nc") == ["a", "b", "c"]

    def test_drops_empty_and_whitespace(self):
        assert split_steps("a\n\n  \nb\n") == ["a", "b"]

    def test_strips_each_step(self):
        assert split_steps("  a  \n\tb\t") == ["a", "b"]

    def test_non_string_returns_empty(self):
        assert split_steps(None) == []
        assert split_steps(123) == []

    def test_empty_returns_empty(self):
        assert split_steps("") == []
        assert split_steps("   \n  ") == []

    def test_caps_step_count(self):
        from soup_cli.utils.prm_reward import _MAX_STEPS

        text = "\n".join(str(i) for i in range(_MAX_STEPS + 50))
        assert len(split_steps(text)) == _MAX_STEPS

    def test_caps_step_chars(self):
        from soup_cli.utils.prm_reward import _MAX_STEP_CHARS

        long = "x" * (_MAX_STEP_CHARS + 100)
        out = split_steps(long)
        assert len(out) == 1
        assert len(out[0]) == _MAX_STEP_CHARS


class TestAggregate:
    def test_min(self):
        assert aggregate_step_scores([0.9, 0.2, 0.7], "min") == pytest.approx(0.2)

    def test_last(self):
        assert aggregate_step_scores([0.9, 0.2, 0.7], "last") == pytest.approx(0.7)

    def test_prod(self):
        assert aggregate_step_scores([0.5, 0.5, 0.5], "prod") == pytest.approx(0.125)

    def test_empty_returns_zero(self):
        assert aggregate_step_scores([], "min") == 0.0
        assert aggregate_step_scores([], "prod") == 0.0
        assert aggregate_step_scores([], "last") == 0.0

    def test_single(self):
        assert aggregate_step_scores([0.42], "min") == pytest.approx(0.42)

    def test_non_finite_is_safe(self):
        # NaN / inf must not propagate — coerced to 0.0
        out = aggregate_step_scores([float("nan"), 0.5], "min")
        assert out == 0.0

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError, match="min|prod|last"):
            aggregate_step_scores([0.5], "mean")

    def test_bool_mode_rejected(self):
        with pytest.raises(ValueError):
            aggregate_step_scores([0.5], True)

    def test_aggregate_modes_constant(self):
        assert set(AGGREGATE_MODES) == {"min", "prod", "last"}


# ---------------------------------------------------------------------------
# Task 2 — schema fields + cross-validators
# ---------------------------------------------------------------------------
def _prm_yaml(
    *,
    task: str = "grpo",
    backend: str = "transformers",
    modality: str = "text",
    prm_reward: str | None = "./prm",
    prm_aggregate: str | None = None,
) -> str:
    lines = [
        "base: HuggingFaceTB/SmolLM2-135M",
        f"task: {task}",
        f"backend: {backend}",
        f"modality: {modality}",
        "data:",
        "  train: ./data/train.jsonl",
        "  format: chatml",
        "training:",
    ]
    if prm_reward is not None:
        lines.append(f"  prm_reward: {prm_reward}")
    if prm_aggregate is not None:
        lines.append(f"  prm_aggregate: {prm_aggregate}")
    return "\n".join(lines) + "\n"


class TestPrmSchema:
    def test_default_fields(self):
        from soup_cli.config.schema import TrainingConfig

        tc = TrainingConfig()
        assert tc.prm_reward is None
        assert tc.prm_aggregate == "min"

    def test_happy_grpo_parses(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_prm_yaml(prm_aggregate="prod"))
        assert cfg.training.prm_reward == "./prm"
        assert cfg.training.prm_aggregate == "prod"

    def test_rejects_non_grpo_task(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="task='grpo'"):
            load_config_from_string(_prm_yaml(task="sft"))

    def test_rejects_mlx_backend(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="transformers"):
            load_config_from_string(_prm_yaml(backend="mlx"))

    def test_rejects_unsloth_backend(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="transformers"):
            load_config_from_string(_prm_yaml(backend="unsloth"))

    def test_rejects_non_text_modality(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="modality='text'"):
            load_config_from_string(_prm_yaml(modality="vision"))

    def test_aggregate_without_prm_reward_is_footgun(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="prm_reward"):
            load_config_from_string(
                _prm_yaml(prm_reward=None, prm_aggregate="prod")
            )

    def test_default_aggregate_without_prm_reward_ok(self):
        from soup_cli.config.loader import load_config_from_string

        # prm_aggregate at its default is fine even without prm_reward.
        cfg = load_config_from_string(
            _prm_yaml(task="sft", prm_reward=None, prm_aggregate="min")
        )
        assert cfg.training.prm_reward is None

    def test_rejects_null_byte(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError, match="null"):
            TrainingConfig(prm_reward="./prm\x00evil")

    def test_rejects_oversize(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValueError):
            TrainingConfig(prm_reward="x" * 5000)


class TestNoTopLevelTorch:
    def test_prm_reward_has_no_top_level_torch(self):
        import soup_cli.utils.prm_reward as mod

        source = inspect.getsource(mod)
        tree = ast.parse(source)
        heavy = {"torch", "transformers", "peft", "safetensors"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in heavy, alias.name
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in heavy, node.module
