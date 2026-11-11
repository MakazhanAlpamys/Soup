"""Tests for Transformers v5 warmup_ratio compatibility and TrainingArguments translation.

Verifies:
1. inspect_training_arguments_warmup_support detects warmup_ratio in TrainingArguments.__init__.
2. resolve_trainer_warmup_steps supports:
   - Dynamic calculation max(1, int(total_training_steps * warmup_ratio)) when steps are given.
   - Float warmup ratio passthrough for HF internal dynamic resolution when total_steps is None.
   - Explicit warmup_steps takes precedence over warmup_ratio.
3. build_warmup_kwargs and sanitize_training_kwargs omit warmup_ratio on modern
   transformers (v5+) preventing TypeError on unexpected 'warmup_ratio'.
4. Instantiating TrainingArguments with sanitized kwargs succeeds without TypeError.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from transformers import TrainingArguments

from soup_cli.utils.warmup import (
    build_warmup_kwargs,
    inspect_training_arguments_warmup_support,
    resolve_trainer_warmup_steps,
    sanitize_training_kwargs,
)


class MockLegacyTrainingArguments:
    """Mock TrainingArguments from Transformers v4 accepting warmup_ratio."""

    def __init__(
        self,
        output_dir: str,
        warmup_ratio: float = 0.0,
        warmup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        self.output_dir = output_dir
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = warmup_steps


class MockModernTrainingArguments:
    """Mock TrainingArguments from Transformers v5 rejecting warmup_ratio."""

    def __init__(
        self,
        output_dir: str,
        warmup_steps: float | int = 0,
        **kwargs: Any,
    ) -> None:
        self.output_dir = output_dir
        self.warmup_steps = warmup_steps


class TestWarmupSupportInspection:
    def test_inspect_real_transformers_v5(self) -> None:
        # In transformers 5.x, warmup_ratio is not in TrainingArguments.__init__
        has_ratio = inspect_training_arguments_warmup_support(TrainingArguments)
        sig = inspect.signature(TrainingArguments.__init__)
        expected = "warmup_ratio" in sig.parameters
        assert has_ratio is expected

    def test_inspect_mock_classes(self) -> None:
        assert inspect_training_arguments_warmup_support(MockLegacyTrainingArguments) is True
        assert inspect_training_arguments_warmup_support(MockModernTrainingArguments) is False

    def test_inspect_none_or_invalid_class(self) -> None:
        default_support = inspect_training_arguments_warmup_support(TrainingArguments)
        assert inspect_training_arguments_warmup_support(None) == default_support
        assert inspect_training_arguments_warmup_support(object) is False


class TestResolveTrainerWarmupSteps:
    @pytest.mark.parametrize(
        ("ratio", "total_steps", "expected"),
        [
            (0.1, 100, 10),
            (0.03, 1000, 30),
            (0.05, 500, 25),
            (0.5, 200, 100),
            (0.001, 100, 1),  # max(1, int(100 * 0.001)) = max(1, 0) = 1
            (0.0001, 50, 1),  # clamped to minimum 1 when ratio > 0
            (0.0, 100, 0),  # ratio=0 yields 0 steps
            (0.0, 1000, 0),
        ],
    )
    def test_dynamic_step_calculation(self, ratio: float, total_steps: int, expected: int) -> None:
        assert (
            resolve_trainer_warmup_steps(warmup_ratio=ratio, total_training_steps=total_steps)
            == expected
        )

    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [
            (0.0, 0.0),
            (0.03, 0.03),
            (0.1, 0.1),
            (0.5, 0.5),
            (None, 0.0),
        ],
    )
    def test_float_ratio_passthrough_when_total_steps_none(
        self, ratio: float | None, expected: float
    ) -> None:
        assert resolve_trainer_warmup_steps(warmup_ratio=ratio) == expected

    def test_explicit_warmup_steps_takes_precedence(self) -> None:
        # Explicit warmup_steps overrides warmup_ratio and total_training_steps
        assert resolve_trainer_warmup_steps(warmup_steps=42) == 42
        assert (
            resolve_trainer_warmup_steps(
                warmup_ratio=0.1, warmup_steps=42, total_training_steps=1000
            )
            == 42
        )
        assert (
            resolve_trainer_warmup_steps(warmup_ratio=0.5, warmup_steps=0, total_training_steps=500)
            == 0
        )

    def test_invalid_arguments_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="warmup_ratio must be in"):
            resolve_trainer_warmup_steps(warmup_ratio=-0.05)
        with pytest.raises(ValueError, match="warmup_ratio must be in"):
            resolve_trainer_warmup_steps(warmup_ratio=1.0)
        with pytest.raises(ValueError, match="warmup_ratio must be in"):
            resolve_trainer_warmup_steps(warmup_ratio=1.5)
        with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
            resolve_trainer_warmup_steps(warmup_steps=-1)


class TestBuildWarmupKwargs:
    def test_build_for_modern_transformers_omits_warmup_ratio(self) -> None:
        kwargs = build_warmup_kwargs(
            warmup_ratio=0.03,
            training_args_cls=MockModernTrainingArguments,
        )
        assert "warmup_ratio" not in kwargs
        assert kwargs["warmup_steps"] == 0.03

    def test_build_for_modern_transformers_with_total_steps(self) -> None:
        kwargs = build_warmup_kwargs(
            warmup_ratio=0.1,
            total_training_steps=500,
            training_args_cls=MockModernTrainingArguments,
        )
        assert "warmup_ratio" not in kwargs
        assert kwargs["warmup_steps"] == 50

    def test_build_for_legacy_transformers_includes_warmup_ratio(self) -> None:
        kwargs = build_warmup_kwargs(
            warmup_ratio=0.05,
            training_args_cls=MockLegacyTrainingArguments,
        )
        assert kwargs.get("warmup_ratio") == 0.05

    def test_build_with_explicit_warmup_steps(self) -> None:
        kwargs = build_warmup_kwargs(
            warmup_steps=100,
            training_args_cls=MockModernTrainingArguments,
        )
        assert kwargs == {"warmup_steps": 100}

    def test_build_with_none_or_zero(self) -> None:
        assert build_warmup_kwargs(
            warmup_ratio=None,
            training_args_cls=MockModernTrainingArguments,
        ) == {"warmup_steps": 0}
        assert build_warmup_kwargs(
            warmup_ratio=0.0,
            training_args_cls=MockModernTrainingArguments,
        ) == {"warmup_steps": 0}


class TestSanitizeTrainingKwargs:
    def test_sanitize_removes_warmup_ratio_on_v5(self, tmp_path: Path) -> None:
        raw_kwargs = {
            "output_dir": str(tmp_path / "out"),
            "warmup_ratio": 0.04,
            "learning_rate": 2e-5,
        }
        sanitized = sanitize_training_kwargs(
            raw_kwargs,
            training_args_cls=MockModernTrainingArguments,
        )
        assert "warmup_ratio" not in sanitized
        assert sanitized["warmup_steps"] == 0.04

    def test_sanitize_with_total_steps_on_v5(self, tmp_path: Path) -> None:
        raw_kwargs = {
            "output_dir": str(tmp_path / "out"),
            "warmup_ratio": 0.1,
        }
        sanitized = sanitize_training_kwargs(
            raw_kwargs,
            total_training_steps=200,
            training_args_cls=MockModernTrainingArguments,
        )
        assert "warmup_ratio" not in sanitized
        assert sanitized["warmup_steps"] == 20

    def test_sanitize_preserves_existing_warmup_steps_on_v5(self) -> None:
        raw_kwargs = {
            "warmup_ratio": 0.1,
            "warmup_steps": 45,
        }
        sanitized = sanitize_training_kwargs(
            raw_kwargs,
            training_args_cls=MockModernTrainingArguments,
        )
        assert "warmup_ratio" not in sanitized
        assert sanitized["warmup_steps"] == 45

    def test_sanitize_preserves_warmup_ratio_on_legacy_v4(self) -> None:
        raw_kwargs = {
            "warmup_ratio": 0.05,
            "warmup_steps": 0,
        }
        sanitized = sanitize_training_kwargs(
            raw_kwargs,
            training_args_cls=MockLegacyTrainingArguments,
        )
        assert sanitized.get("warmup_ratio") == 0.05

    def test_real_training_arguments_instantiation_without_type_error(self, tmp_path: Path) -> None:
        raw_kwargs: dict[str, Any] = {
            "output_dir": str(tmp_path / "out"),
            "warmup_ratio": 0.05,
        }
        # In transformers v5, TrainingArguments(**raw_kwargs) would raise TypeError.
        # sanitize_training_kwargs makes it compatible without error.
        sanitized = sanitize_training_kwargs(dict(raw_kwargs))
        args = TrainingArguments(**sanitized)
        assert args.warmup_steps == 0.05
        # Verify dynamic step resolution works
        assert args.get_warmup_steps(100) == 5

    def test_real_training_arguments_with_explicit_warmup_steps(self, tmp_path: Path) -> None:
        raw_kwargs: dict[str, Any] = {
            "output_dir": str(tmp_path / "out"),
            "warmup_steps": 25,
        }
        sanitized = sanitize_training_kwargs(dict(raw_kwargs))
        args = TrainingArguments(**sanitized)
        assert args.warmup_steps == 25
        assert args.get_warmup_steps(100) == 25
