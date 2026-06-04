"""v0.70.0 Part F — Live echo-trap detector.

RAGEN-style detection of trajectory degeneration in multi-turn agent RL.
When the policy collapses to self-repeating outputs (echo trap), the
reward stops improving and the policy drifts. Schema + math kernels
live; trainer-callback wiring deferred to v0.70.1.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest


class TestEchoTrapPublicSurface:
    def test_module_imports(self):
        from soup_cli.utils import echo_trap

        assert hasattr(echo_trap, "score_trajectory_repetition")
        assert hasattr(echo_trap, "score_trajectory_repetition_tokenized")
        assert hasattr(echo_trap, "score_echo_signal")
        assert hasattr(echo_trap, "score_echo_signal_tokenized")
        assert hasattr(echo_trap, "classify_echo_signal")
        assert hasattr(echo_trap, "EchoTrapReport")
        assert hasattr(echo_trap, "build_echo_trap_callback")
        assert hasattr(echo_trap, "VERDICTS")


class TestScoreTrajectoryRepetition:
    """Per-trajectory n-gram repetition score. Higher = more repetition.

    Tail-mass = fraction of n-grams that repeat more than once. Returns
    a float in [0, 1].
    """

    def test_unique_trajectory_zero(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        # All unique tokens → 0 repetition.
        tokens = ["a", "b", "c", "d", "e"]
        score = score_trajectory_repetition(tokens, ngram_n=2)
        assert score == 0.0

    def test_full_repetition(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        # Identical token throughout → near-perfect repetition.
        tokens = ["a"] * 10
        score = score_trajectory_repetition(tokens, ngram_n=2)
        assert score > 0.5

    def test_score_bounded(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        tokens = ["a", "b", "a", "b", "a", "b"]
        score = score_trajectory_repetition(tokens, ngram_n=2)
        assert 0.0 <= score <= 1.0

    def test_short_returns_zero(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        # Fewer tokens than ngram_n → no n-grams possible.
        assert score_trajectory_repetition(["a"], ngram_n=2) == 0.0

    def test_empty_returns_zero(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        assert score_trajectory_repetition([], ngram_n=2) == 0.0

    def test_invalid_ngram_n_rejected(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        with pytest.raises(ValueError, match="ngram_n"):
            score_trajectory_repetition(["a", "b"], ngram_n=0)
        with pytest.raises(ValueError, match="ngram_n"):
            score_trajectory_repetition(["a", "b"], ngram_n=-1)

    def test_bool_ngram_n_rejected(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        with pytest.raises(ValueError, match="bool"):
            score_trajectory_repetition(["a", "b"], ngram_n=True)

    def test_ngram_n_max_cap(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        with pytest.raises(ValueError, match="32"):
            score_trajectory_repetition(["a", "b"], ngram_n=33)

    def test_non_string_token_rejected(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        with pytest.raises(TypeError, match="tokens"):
            score_trajectory_repetition([1, 2, 3], ngram_n=2)  # type: ignore[list-item]

    def test_non_list_tokens_rejected(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        with pytest.raises(TypeError):
            score_trajectory_repetition("abc", ngram_n=2)


class TestScoreEchoSignal:
    """Aggregate echo signal over a batch of trajectories.

    Returns the mean repetition score across the batch.
    """

    def test_clean_trajectories(self):
        from soup_cli.utils.echo_trap import score_echo_signal

        batch = [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
        ]
        score = score_echo_signal(batch, ngram_n=2)
        assert math.isfinite(score)
        assert 0.0 <= score < 0.1

    def test_collapsed_batch(self):
        from soup_cli.utils.echo_trap import score_echo_signal

        batch = [["a"] * 10, ["b"] * 10]
        score = score_echo_signal(batch, ngram_n=2)
        assert score > 0.5

    def test_empty_batch(self):
        from soup_cli.utils.echo_trap import score_echo_signal

        assert score_echo_signal([], ngram_n=2) == 0.0

    def test_mixed_batch(self):
        from soup_cli.utils.echo_trap import score_echo_signal

        batch = [["a"] * 10, ["x", "y", "z"]]
        score = score_echo_signal(batch, ngram_n=2)
        # Average of repetitive + clean.
        assert 0.0 < score < 1.0

    def test_non_list_batch_rejected(self):
        from soup_cli.utils.echo_trap import score_echo_signal

        with pytest.raises(TypeError):
            score_echo_signal("not a list", ngram_n=2)

    def test_batch_size_cap(self):
        from soup_cli.utils.echo_trap import score_echo_signal

        big = [["x"] for _ in range(100_001)]
        with pytest.raises(ValueError, match="batch"):
            score_echo_signal(big, ngram_n=2)


class TestTokenizedEchoSignal:
    def test_tokenized_repetition_catches_subword_echo_trap(self):
        from soup_cli.utils.echo_trap import (
            classify_echo_signal,
            score_echo_signal,
            score_echo_signal_tokenized,
        )

        decoded_tokens = ["ha,", "ha.", "ha!", "ha?", "ha;", "ha:"]
        repeated_token_ids = [101, 202, 101, 202, 101, 202, 101, 202]

        whitespace_score = score_echo_signal([decoded_tokens], ngram_n=2)
        tokenized_score = score_echo_signal_tokenized([repeated_token_ids], ngram_n=2)

        assert classify_echo_signal(whitespace_score) == "OK"
        assert classify_echo_signal(tokenized_score) == "TRAP"

    def test_tokenized_trajectory_rejects_non_int_ids(self):
        from soup_cli.utils.echo_trap import score_trajectory_repetition_tokenized

        with pytest.raises(TypeError, match="token_ids"):
            score_trajectory_repetition_tokenized([1, "2", 3], ngram_n=2)
        with pytest.raises(TypeError, match="token_ids"):
            score_trajectory_repetition_tokenized([1, True, 3], ngram_n=2)

    def test_tokenized_batch_rejects_str(self):
        from soup_cli.utils.echo_trap import score_echo_signal_tokenized

        with pytest.raises(TypeError, match="token-id"):
            score_echo_signal_tokenized("not ids", ngram_n=2)


class TestClassifyEchoSignal:
    """OK / WARN / TRAP taxonomy (mirrors v0.26 / v0.56 / v0.70 Part A).

    - signal < 0.30: OK
    - 0.30 <= signal < 0.60: WARN
    - signal >= 0.60: TRAP
    """

    def test_ok(self):
        from soup_cli.utils.echo_trap import classify_echo_signal

        assert classify_echo_signal(0.0) == "OK"
        assert classify_echo_signal(0.1) == "OK"
        assert classify_echo_signal(0.29) == "OK"

    def test_warn(self):
        from soup_cli.utils.echo_trap import classify_echo_signal

        assert classify_echo_signal(0.30) == "WARN"
        assert classify_echo_signal(0.45) == "WARN"
        assert classify_echo_signal(0.59) == "WARN"

    def test_trap(self):
        from soup_cli.utils.echo_trap import classify_echo_signal

        assert classify_echo_signal(0.60) == "TRAP"
        assert classify_echo_signal(0.99) == "TRAP"
        assert classify_echo_signal(1.0) == "TRAP"

    def test_invalid_signal_rejected(self):
        from soup_cli.utils.echo_trap import classify_echo_signal

        with pytest.raises(ValueError, match="finite"):
            classify_echo_signal(float("nan"))
        with pytest.raises(ValueError):
            classify_echo_signal(-0.1)
        with pytest.raises(ValueError):
            classify_echo_signal(1.5)

    def test_bool_rejected(self):
        from soup_cli.utils.echo_trap import classify_echo_signal

        with pytest.raises(ValueError, match="bool"):
            classify_echo_signal(True)


class TestEchoTrapReport:
    def test_basic(self):
        from soup_cli.utils.echo_trap import EchoTrapReport

        report = EchoTrapReport(
            signal=0.4,
            verdict="WARN",
            step=200,
            trajectories_seen=64,
            details=("longest streak: a a a a a",),
        )
        assert report.signal == 0.4
        assert report.verdict == "WARN"

    def test_frozen(self):
        from soup_cli.utils.echo_trap import EchoTrapReport

        report = EchoTrapReport(
            signal=0.0,
            verdict="OK",
            step=0,
            trajectories_seen=0,
            details=(),
        )
        with pytest.raises(FrozenInstanceError):
            report.signal = 1.0  # type: ignore[misc]

    def test_invalid_verdict_rejected(self):
        from soup_cli.utils.echo_trap import EchoTrapReport

        with pytest.raises(ValueError, match="verdict"):
            EchoTrapReport(
                signal=0.0,
                verdict="EVIL",
                step=0,
                trajectories_seen=0,
                details=(),
            )

    def test_signal_out_of_range_rejected(self):
        from soup_cli.utils.echo_trap import EchoTrapReport

        with pytest.raises(ValueError):
            EchoTrapReport(
                signal=1.5,
                verdict="OK",
                step=0,
                trajectories_seen=0,
                details=(),
            )

    def test_bool_step_rejected(self):
        from soup_cli.utils.echo_trap import EchoTrapReport

        with pytest.raises(ValueError, match="bool"):
            EchoTrapReport(
                signal=0.0,
                verdict="OK",
                step=True,
                trajectories_seen=0,
                details=(),
            )

    def test_negative_trajectories_rejected(self):
        from soup_cli.utils.echo_trap import EchoTrapReport

        with pytest.raises(ValueError, match="trajectories"):
            EchoTrapReport(
                signal=0.0,
                verdict="OK",
                step=0,
                trajectories_seen=-1,
                details=(),
            )

    def test_details_must_be_tuple(self):
        from soup_cli.utils.echo_trap import EchoTrapReport

        with pytest.raises(TypeError, match="tuple"):
            EchoTrapReport(
                signal=0.0,
                verdict="OK",
                step=0,
                trajectories_seen=0,
                details=["not tuple"],  # type: ignore[arg-type]
            )


class TestBuildEchoTrapCallbackDeferred:
    def test_invalid_threshold_rejected_first(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        with pytest.raises(ValueError, match="threshold"):
            build_echo_trap_callback(threshold=2.0)

    def test_invalid_threshold_bool(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        with pytest.raises(ValueError, match="bool"):
            build_echo_trap_callback(threshold=True)

    def test_live_returns_callback(self):
        from soup_cli.utils.echo_trap import (
            EchoTrapCallback,
            build_echo_trap_callback,
        )

        assert isinstance(build_echo_trap_callback(threshold=0.5), EchoTrapCallback)

    def test_halt_must_be_bool(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        with pytest.raises(TypeError, match="halt"):
            build_echo_trap_callback(threshold=0.5, halt_on_trap="yes")  # type: ignore[arg-type]

    def test_tokenizer_aware_must_be_bool(self):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        with pytest.raises(TypeError, match="tokenizer_aware"):
            build_echo_trap_callback(
                threshold=0.5,
                tokenizer_aware="yes",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Schema integration — TrainingConfig + SoupConfig
# ---------------------------------------------------------------------------


class TestSchemaTrainingConfig:
    def test_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig()
        assert tcfg.echo_trap_enabled is False
        assert tcfg.echo_trap_threshold == 0.6
        assert tcfg.echo_trap_halt is False
        assert tcfg.echo_trap_tokenizer_aware is False

    def test_threshold_bounds(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        tcfg = TrainingConfig(echo_trap_threshold=0.45)
        assert tcfg.echo_trap_threshold == 0.45

        with pytest.raises(ValidationError):
            TrainingConfig(echo_trap_threshold=-0.1)
        with pytest.raises(ValidationError):
            TrainingConfig(echo_trap_threshold=1.5)


class TestSchemaSoupConfigTaskGate:
    """echo_trap_enabled only meaningful on RL agent tasks (grpo / ppo)."""

    def _yaml(self, task: str = "grpo") -> str:
        return f"""
base: meta-llama/Llama-3.1-8B
task: {task}
data:
  train: ./data/train.jsonl
  format: chatml
training:
  echo_trap_enabled: true
  echo_trap_threshold: 0.55
  echo_trap_tokenizer_aware: true
"""

    def test_grpo_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(self._yaml("grpo"))
        assert cfg.training.echo_trap_enabled is True
        assert cfg.training.echo_trap_tokenizer_aware is True

    def test_ppo_accepted(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(self._yaml("ppo"))
        assert cfg.training.echo_trap_enabled is True

    def test_sft_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="echo_trap"):
            load_config_from_string(self._yaml("sft"))

    def test_halt_without_enabled_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="echo_trap_enabled"):
            load_config_from_string(
                """
base: meta-llama/Llama-3.1-8B
task: grpo
data:
  train: ./data/train.jsonl
  format: chatml
training:
  echo_trap_halt: true
"""
            )

    def test_tokenizer_aware_without_enabled_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="echo_trap_enabled"):
            load_config_from_string(
                """
base: meta-llama/Llama-3.1-8B
task: grpo
data:
  train: ./data/train.jsonl
  format: chatml
training:
  echo_trap_tokenizer_aware: true
"""
            )


# ---------------------------------------------------------------------------
# Source wiring guards
# ---------------------------------------------------------------------------


class TestSourceWiring:
    def test_module_no_top_level_torch(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli"
            / "utils"
            / "echo_trap.py"
        )
        body = src.read_text(encoding="utf-8")
        assert "\nimport torch" not in body
        assert "\nfrom torch" not in body
