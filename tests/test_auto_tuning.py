"""Tests for v0.32.0 — Training Stability & Auto-Tuning.

Covers:
- Part A: LR range finder (utils/lr_finder.py)
- Part B: Live grad-accum auto-tuning (utils/grad_accum.py)
- Part C: Auto mixed-precision picker (utils/mixed_precision.py)
- Part D: Warmup auto-schedule (utils/warmup.py)
- Part E: Loss spike auto-recovery (utils/spike_recovery.py + callback wiring)
- Part F: Convergence detector (utils/convergence.py)
- Part G: Autopilot integration (autopilot/decisions.py extensions)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

# Rich/Typer emits per-character ANSI escapes in CI ("--\x1b[m-find\x1b[m-lr"),
# so substring assertions on the raw `result.output` fail on Linux runners
# even though they pass on Windows where Rich auto-disables colour. Strip
# escapes before comparing — same pattern used by test_hf_integration.py
# and test_eval_platform.py.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)

# --------------------------------------------------------------------------- #
# Part A — LR range finder                                                    #
# --------------------------------------------------------------------------- #

class TestLRFinderSchedule:
    """compute_lr_schedule produces a logarithmic LR sweep."""

    def test_log_schedule_endpoints(self):
        from soup_cli.utils.lr_finder import compute_lr_schedule

        lrs = compute_lr_schedule(start_lr=1e-7, end_lr=1e-1, num_steps=50)

        assert len(lrs) == 50
        assert lrs[0] == pytest.approx(1e-7, rel=1e-6)
        assert lrs[-1] == pytest.approx(1e-1, rel=1e-6)

    def test_log_schedule_monotonic(self):
        from soup_cli.utils.lr_finder import compute_lr_schedule

        lrs = compute_lr_schedule(1e-6, 1e-1, 30)
        for left, right in zip(lrs, lrs[1:]):
            assert right > left

    def test_log_schedule_geometric(self):
        from soup_cli.utils.lr_finder import compute_lr_schedule

        lrs = compute_lr_schedule(1e-6, 1e-1, 11)
        ratios = [lrs[i + 1] / lrs[i] for i in range(len(lrs) - 1)]
        assert all(math.isclose(ratios[0], r, rel_tol=1e-6) for r in ratios)

    def test_schedule_rejects_non_positive_start(self):
        from soup_cli.utils.lr_finder import compute_lr_schedule

        with pytest.raises(ValueError, match="start_lr"):
            compute_lr_schedule(0.0, 1e-1, 10)

    def test_schedule_rejects_inverted_range(self):
        from soup_cli.utils.lr_finder import compute_lr_schedule

        with pytest.raises(ValueError, match="end_lr"):
            compute_lr_schedule(1e-2, 1e-4, 10)

    def test_schedule_rejects_too_few_steps(self):
        from soup_cli.utils.lr_finder import compute_lr_schedule

        with pytest.raises(ValueError, match="num_steps"):
            compute_lr_schedule(1e-7, 1e-1, 1)

    def test_schedule_caps_steps(self):
        from soup_cli.utils.lr_finder import compute_lr_schedule

        with pytest.raises(ValueError, match="num_steps"):
            compute_lr_schedule(1e-7, 1e-1, 100_000)

    def test_schedule_min_steps_accepted(self):
        from soup_cli.utils.lr_finder import compute_lr_schedule

        lrs = compute_lr_schedule(1e-7, 1e-1, 2)
        assert len(lrs) == 2
        assert lrs[0] == pytest.approx(1e-7, rel=1e-6)
        assert lrs[-1] == pytest.approx(1e-1, rel=1e-6)


class TestLRFinderRecommendation:
    """find_optimal_lr picks the LR with steepest negative gradient
    (excluding the explosion tail)."""

    def test_picks_steepest_descent(self):
        from soup_cli.utils.lr_finder import find_optimal_lr

        # Loss decreases through 1e-3 then explodes
        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [3.0, 2.8, 2.4, 1.5, 4.0, 12.0]

        result = find_optimal_lr(lrs, losses)

        assert "recommended_lr" in result
        assert "min_loss_lr" in result
        assert "diverged_at" in result
        # Recommended must be <= min_loss_lr (steepest descent comes earlier)
        assert result["recommended_lr"] <= result["min_loss_lr"]
        # Recommended LR must be in the descent region, not the explosion tail
        assert result["recommended_lr"] < 1e-2

    def test_returns_smoothed_curve(self):
        from soup_cli.utils.lr_finder import find_optimal_lr

        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [3.0, 2.8, 2.4, 1.5, 4.0, 12.0]

        result = find_optimal_lr(lrs, losses)
        assert "smoothed_losses" in result
        assert len(result["smoothed_losses"]) == len(losses)

    def test_detects_divergence(self):
        from soup_cli.utils.lr_finder import find_optimal_lr

        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        losses = [3.0, 2.8, 2.4, 1.5, 100.0]

        result = find_optimal_lr(lrs, losses)
        assert result["diverged_at"] is not None
        assert result["diverged_at"] <= 1e-2

    def test_no_divergence_when_loss_stable(self):
        from soup_cli.utils.lr_finder import find_optimal_lr

        lrs = [1e-6, 1e-5, 1e-4, 1e-3]
        losses = [3.0, 2.8, 2.6, 2.4]

        result = find_optimal_lr(lrs, losses)
        assert result["diverged_at"] is None

    def test_mismatched_lengths_rejected(self):
        from soup_cli.utils.lr_finder import find_optimal_lr

        with pytest.raises(ValueError, match="length"):
            find_optimal_lr([1e-6, 1e-5], [3.0, 2.8, 2.6])

    def test_too_few_points_rejected(self):
        from soup_cli.utils.lr_finder import find_optimal_lr

        with pytest.raises(ValueError, match="at least"):
            find_optimal_lr([1e-6, 1e-5], [3.0, 2.8])

    def test_monotonic_increase_falls_back_to_first_lr(self):
        from soup_cli.utils.lr_finder import find_optimal_lr

        lrs = [1e-6, 1e-5, 1e-4, 1e-3]
        losses = [1.0, 2.0, 4.0, 8.0]  # explodes immediately
        result = find_optimal_lr(lrs, losses)
        assert result["recommended_lr"] == pytest.approx(lrs[0], rel=1e-6)


class TestLRFinderCLI:
    """`soup train --find-lr` flag is registered and surfaces in help."""

    def test_flag_in_help(self):
        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "--find-lr" in _strip_ansi(result.output)


# --------------------------------------------------------------------------- #
# Part B — Live grad-accum auto-tuning                                        #
# --------------------------------------------------------------------------- #

class TestGradAccumMonitor:
    def test_should_adjust_when_high_pressure(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0, threshold=0.92)
        # 23.5/24 = 0.979 → above 0.92 threshold
        assert mon.should_adjust(used_vram_gb=23.5) is True

    def test_should_not_adjust_when_low_pressure(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0, threshold=0.92)
        assert mon.should_adjust(used_vram_gb=10.0) is False

    def test_recommend_doubles_grad_accum(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0)
        new_batch, new_accum = mon.recommend(current_batch=8, current_accum=2)
        assert new_batch == 4
        assert new_accum == 4

    def test_recommend_keeps_effective_batch(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0)
        for batch, accum in [(16, 1), (8, 4), (32, 1)]:
            new_batch, new_accum = mon.recommend(batch, accum)
            assert new_batch * new_accum == batch * accum

    def test_recommend_floor_at_one(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0)
        new_batch, new_accum = mon.recommend(current_batch=1, current_accum=8)
        assert new_batch == 1
        assert new_accum == 8

    def test_recommend_caps_accum_at_max(self):
        from soup_cli.utils.grad_accum import MAX_ACCUM, GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0)
        # Push accum past the cap; new should equal MAX_ACCUM.
        _, new_accum = mon.recommend(current_batch=2, current_accum=MAX_ACCUM)
        assert new_accum == MAX_ACCUM

    def test_observe_updates_peak(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0)
        mon.observe(10.0)
        mon.observe(15.0)
        mon.observe(12.0)
        assert mon.peak_used_gb == pytest.approx(15.0)

    def test_observe_rejects_negative(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0)
        with pytest.raises(ValueError, match="used_vram_gb"):
            mon.observe(-1.0)

    def test_should_adjust_rejects_negative(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0)
        with pytest.raises(ValueError, match="used_vram_gb"):
            mon.should_adjust(-0.1)

    def test_recommend_rejects_zero_inputs(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        mon = GradAccumMonitor(total_vram_gb=24.0)
        with pytest.raises(ValueError, match="current_batch"):
            mon.recommend(current_batch=0, current_accum=2)
        with pytest.raises(ValueError, match="current_batch"):
            mon.recommend(current_batch=2, current_accum=0)

    def test_threshold_bounds(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        with pytest.raises(ValueError, match="threshold"):
            GradAccumMonitor(total_vram_gb=24.0, threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            GradAccumMonitor(total_vram_gb=24.0, threshold=0.0)

    def test_total_vram_must_be_positive(self):
        from soup_cli.utils.grad_accum import GradAccumMonitor

        with pytest.raises(ValueError, match="total_vram_gb"):
            GradAccumMonitor(total_vram_gb=-1.0)


# --------------------------------------------------------------------------- #
# Part C — Auto mixed-precision picker                                        #
# --------------------------------------------------------------------------- #

class TestMixedPrecisionPicker:
    def test_llama_ampere_picks_bf16(self):
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        assert pick_mixed_precision("meta-llama/Llama-3-8B", 8.0) == "bf16"

    def test_qwen2_picks_fp16(self):
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        assert pick_mixed_precision("Qwen/Qwen2-7B", 8.0) == "fp16"

    def test_pre_ampere_drops_to_fp16(self):
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        # Turing T4 (cc=7.5) — no bf16
        assert pick_mixed_precision("meta-llama/Llama-3-8B", 7.5) == "fp16"

    def test_pre_pascal_returns_no(self):
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        # cc<6.0 doesn't support fp16 reliably
        assert pick_mixed_precision("meta-llama/Llama-3-8B", 5.0) == "no"

    def test_unknown_model_default_bf16_on_ampere(self):
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        assert pick_mixed_precision("some/unknown-model", 9.0) == "bf16"

    def test_invalid_model_name_rejected(self):
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        with pytest.raises(ValueError, match="model"):
            pick_mixed_precision("", 8.0)
        with pytest.raises(ValueError, match="model"):
            pick_mixed_precision("a\x00b", 8.0)
        with pytest.raises(ValueError, match="200"):
            pick_mixed_precision("a" * 201, 8.0)

    def test_qwen25_picks_fp16_not_qwen2_default(self):
        """Longer 'qwen2.5' substring must win over 'qwen2'."""
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        # Both entries map to fp16 today, but the test guards the iteration
        # order: if someone changes qwen2.5 to bf16, this catches it.
        assert pick_mixed_precision("Qwen/Qwen2.5-7B", 8.0) == "fp16"

    def test_negative_cc_rejected(self):
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        with pytest.raises(ValueError, match="compute_capability"):
            pick_mixed_precision("meta-llama/Llama-3-8B", -1.0)

    def test_known_quirk_mapping_includes_qwen_and_phi(self):
        from soup_cli.utils.mixed_precision import KNOWN_PRECISION_QUIRKS

        keys = [k.lower() for k in KNOWN_PRECISION_QUIRKS]
        assert any("qwen" in k for k in keys)
        assert any("phi" in k for k in keys)

    def test_cc_exactly_fp16_boundary(self):
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        # cc == 6.0 should accept fp16 (the comparison is `< FP16_MIN_CC`)
        assert pick_mixed_precision("meta-llama/Llama-3-8B", 6.0) == "fp16"

    def test_cc_exactly_bf16_boundary(self):
        from soup_cli.utils.mixed_precision import pick_mixed_precision

        # cc == 8.0 should pick bf16 (the comparison is `< BF16_MIN_CC`)
        assert pick_mixed_precision("meta-llama/Llama-3-8B", 8.0) == "bf16"


# --------------------------------------------------------------------------- #
# Part D — Warmup auto-schedule                                               #
# --------------------------------------------------------------------------- #

class TestWarmupAutoSchedule:
    def test_basic_formula(self):
        from soup_cli.utils.warmup import compute_warmup_steps

        # 10000 examples / batch 4 / accum 2 / 3 epochs = 3750 update steps
        # 3% = 112 steps
        steps = compute_warmup_steps(
            num_examples=10000,
            batch_size=4,
            grad_accum=2,
            epochs=3,
            ratio=0.03,
        )
        assert 100 <= steps <= 130

    def test_clamps_to_min(self):
        from soup_cli.utils.warmup import compute_warmup_steps

        steps = compute_warmup_steps(
            num_examples=10, batch_size=1, grad_accum=1, epochs=1, ratio=0.03,
        )
        assert steps >= 10  # MIN_WARMUP

    def test_clamps_to_max(self):
        from soup_cli.utils.warmup import compute_warmup_steps

        steps = compute_warmup_steps(
            num_examples=10_000_000,
            batch_size=1,
            grad_accum=1,
            epochs=1,
            ratio=0.03,
        )
        assert steps <= 1000  # MAX_WARMUP

    def test_invalid_ratio_rejected(self):
        from soup_cli.utils.warmup import compute_warmup_steps

        for bad_ratio in [-0.01, 0.51]:
            with pytest.raises(ValueError, match="ratio"):
                compute_warmup_steps(
                    num_examples=1000, batch_size=1, grad_accum=1, epochs=1,
                    ratio=bad_ratio,
                )

    def test_ratio_zero_means_no_warmup(self):
        from soup_cli.utils.warmup import compute_warmup_steps

        steps = compute_warmup_steps(
            num_examples=1000, batch_size=1, grad_accum=1, epochs=1, ratio=0.0,
        )
        assert steps == 0

    def test_invalid_inputs_rejected(self):
        from soup_cli.utils.warmup import compute_warmup_steps

        with pytest.raises(ValueError, match="num_examples"):
            compute_warmup_steps(num_examples=0, batch_size=1, grad_accum=1, epochs=1)
        with pytest.raises(ValueError, match="batch_size"):
            compute_warmup_steps(num_examples=10, batch_size=0, grad_accum=1, epochs=1)
        with pytest.raises(ValueError, match="grad_accum"):
            compute_warmup_steps(num_examples=10, batch_size=1, grad_accum=0, epochs=1)
        with pytest.raises(ValueError, match="epochs"):
            compute_warmup_steps(num_examples=10, batch_size=1, grad_accum=1, epochs=0)

    def test_ratio_at_max_accepted(self):
        from soup_cli.utils.warmup import MAX_WARMUP, compute_warmup_steps

        # ratio == MAX_RATIO (0.5) is the inclusive upper bound.
        steps = compute_warmup_steps(
            num_examples=100_000, batch_size=1, grad_accum=1, epochs=1, ratio=0.5,
        )
        assert steps == MAX_WARMUP


class TestWarmupConfigField:
    def test_warmup_auto_default_false(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.warmup_auto is False

    def test_warmup_auto_can_be_set(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig(warmup_auto=True)
        assert cfg.warmup_auto is True


# --------------------------------------------------------------------------- #
# Part E — Loss spike auto-recovery                                           #
# --------------------------------------------------------------------------- #

class TestSpikeRecoveryStrategy:
    def test_should_recover_within_budget(self):
        from soup_cli.utils.spike_recovery import SpikeRecoveryStrategy

        strat = SpikeRecoveryStrategy(max_attempts=3, lr_decay=0.5)
        assert strat.should_recover(attempts=0) is True
        assert strat.should_recover(attempts=2) is True

    def test_should_not_recover_at_limit(self):
        from soup_cli.utils.spike_recovery import SpikeRecoveryStrategy

        strat = SpikeRecoveryStrategy(max_attempts=3, lr_decay=0.5)
        assert strat.should_recover(attempts=3) is False
        assert strat.should_recover(attempts=4) is False

    def test_compute_new_lr_decays(self):
        from soup_cli.utils.spike_recovery import SpikeRecoveryStrategy

        strat = SpikeRecoveryStrategy(max_attempts=3, lr_decay=0.5)
        assert strat.compute_new_lr(2e-4) == pytest.approx(1e-4)
        assert strat.compute_new_lr(1e-3) == pytest.approx(5e-4)

    def test_lr_decay_bounds(self):
        from soup_cli.utils.spike_recovery import SpikeRecoveryStrategy

        with pytest.raises(ValueError, match="lr_decay"):
            SpikeRecoveryStrategy(max_attempts=3, lr_decay=0.0)
        with pytest.raises(ValueError, match="lr_decay"):
            SpikeRecoveryStrategy(max_attempts=3, lr_decay=1.0)

    def test_max_attempts_bounds(self):
        from soup_cli.utils.spike_recovery import SpikeRecoveryStrategy

        with pytest.raises(ValueError, match="max_attempts"):
            SpikeRecoveryStrategy(max_attempts=0)
        with pytest.raises(ValueError, match="max_attempts"):
            SpikeRecoveryStrategy(max_attempts=100)

    def test_minimum_lr_floor(self):
        from soup_cli.utils.spike_recovery import SpikeRecoveryStrategy

        strat = SpikeRecoveryStrategy(max_attempts=3, lr_decay=0.5, min_lr=1e-7)
        # New LR is below floor → return floor
        assert strat.compute_new_lr(1e-8) == pytest.approx(1e-7)

    def test_compute_new_lr_rejects_non_positive(self):
        from soup_cli.utils.spike_recovery import SpikeRecoveryStrategy

        strat = SpikeRecoveryStrategy(max_attempts=3, lr_decay=0.5)
        with pytest.raises(ValueError, match="current_lr"):
            strat.compute_new_lr(0.0)
        with pytest.raises(ValueError, match="current_lr"):
            strat.compute_new_lr(-1e-4)

    def test_min_lr_must_be_positive(self):
        from soup_cli.utils.spike_recovery import SpikeRecoveryStrategy

        with pytest.raises(ValueError, match="min_lr"):
            SpikeRecoveryStrategy(max_attempts=3, lr_decay=0.5, min_lr=0.0)


class TestSpikeRecoveryConfig:
    def test_field_default(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.loss_spike_recovery is False
        assert cfg.loss_spike_recovery_max_attempts == 3

    def test_max_attempts_bounds(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(loss_spike_recovery_max_attempts=0)
        with pytest.raises(ValidationError):
            TrainingConfig(loss_spike_recovery_max_attempts=100)

    def test_recovery_requires_watchdog(self):
        from soup_cli.config.schema import TrainingConfig

        # Recovery without watchdog enabled is rejected.
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(loss_spike_recovery=True, loss_watchdog=False)
        messages = [err["msg"] for err in exc_info.value.errors()]
        assert any(
            "loss_watchdog" in msg and "loss_spike_recovery" in msg
            for msg in messages
        ), messages


# --------------------------------------------------------------------------- #
# Part F — Convergence detector                                               #
# --------------------------------------------------------------------------- #

class TestConvergenceDetector:
    def test_detects_plateau(self):
        from soup_cli.utils.convergence import detect_plateau

        # Last 50 losses essentially flat
        losses = [3.0 - 0.001 * i for i in range(150)] + [2.85] * 50
        assert detect_plateau(losses, window=50, rel_tol=0.005) is True

    def test_does_not_detect_when_decreasing(self):
        from soup_cli.utils.convergence import detect_plateau

        losses = [3.0 - 0.005 * i for i in range(200)]
        assert detect_plateau(losses, window=50, rel_tol=0.005) is False

    def test_too_few_points(self):
        from soup_cli.utils.convergence import detect_plateau

        losses = [3.0, 2.9, 2.8]
        assert detect_plateau(losses, window=50, rel_tol=0.005) is False

    def test_window_bounds(self):
        from soup_cli.utils.convergence import detect_plateau

        with pytest.raises(ValueError, match="window"):
            detect_plateau([3.0] * 100, window=0)
        with pytest.raises(ValueError, match="window"):
            detect_plateau([3.0] * 100, window=10001)

    def test_rel_tol_bounds(self):
        from soup_cli.utils.convergence import detect_plateau

        with pytest.raises(ValueError, match="rel_tol"):
            detect_plateau([3.0] * 100, window=50, rel_tol=-0.001)
        with pytest.raises(ValueError, match="rel_tol"):
            detect_plateau([3.0] * 100, window=50, rel_tol=2.0)


class TestRecommendAction:
    def test_recommends_continue_for_decreasing(self):
        from soup_cli.utils.convergence import recommend_action

        losses = [3.0 - 0.005 * i for i in range(200)]
        assert recommend_action(losses) == "continue"

    def test_recommends_early_stop_for_long_plateau(self):
        from soup_cli.utils.convergence import recommend_action

        losses = [3.0 - 0.001 * i for i in range(150)] + [2.85] * 100
        assert recommend_action(losses) == "early_stop"

    def test_recommends_lower_lr_for_oscillation(self):
        # Oscillating losses (high variance, no trend)
        import random

        from soup_cli.utils.convergence import recommend_action

        rng = random.Random(42)
        losses = [2.5 + rng.uniform(-0.4, 0.4) for _ in range(200)]
        action = recommend_action(losses)
        assert action in {"lower_lr", "continue"}

    def test_too_few_points_returns_continue(self):
        from soup_cli.utils.convergence import recommend_action

        assert recommend_action([3.0, 2.9, 2.8]) == "continue"


class TestConvergenceConfig:
    def test_field_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.convergence_detection is False

    def test_convergence_window_lower_bound(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(convergence_window=4)

    def test_convergence_window_upper_bound(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(convergence_window=10_001)

    def test_convergence_rel_tol_upper_bound(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(convergence_rel_tol=1.1)

    def test_convergence_rel_tol_zero_rejected(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(convergence_rel_tol=0.0)


class TestRecoveryFieldBounds:
    def test_lr_decay_upper_bound(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(loss_spike_recovery_lr_decay=1.0)

    def test_lr_decay_lower_bound(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(loss_spike_recovery_lr_decay=0.0)


class TestGradAccumThresholdField:
    def test_pressure_threshold_upper_bound(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(grad_accum_pressure_threshold=0.99)

    def test_pressure_threshold_lower_bound(self):
        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(grad_accum_pressure_threshold=0.05)


class TestPlateauNonPositiveMean:
    def test_plateau_non_positive_mean_returns_false(self):
        from soup_cli.utils.convergence import detect_plateau

        # All-negative losses → mean < 0 → refuse to assess.
        losses = [-2.0] * 60
        assert detect_plateau(losses, window=50, rel_tol=0.005) is False

    def test_recommend_action_non_positive_mean_returns_continue(self):
        from soup_cli.utils.convergence import recommend_action

        losses = [-3.0 - 0.001 * i for i in range(200)]
        assert recommend_action(losses) == "continue"


# --------------------------------------------------------------------------- #
# Part G — Autopilot integration                                              #
# --------------------------------------------------------------------------- #

class TestAutopilotIntegration:
    def test_decide_warmup_returns_int(self):
        from soup_cli.autopilot.decisions import decide_warmup

        steps = decide_warmup(
            num_examples=10000, batch_size=4, grad_accum=2, epochs=3,
        )
        assert isinstance(steps, int)
        assert steps > 0

    def test_decide_mixed_precision_routes_to_picker(self):
        from soup_cli.autopilot.decisions import decide_mixed_precision

        prec = decide_mixed_precision("meta-llama/Llama-3-8B", 8.0)
        assert prec in {"bf16", "fp16", "no"}

    def test_decide_mixed_precision_invalid_inputs(self):
        from soup_cli.autopilot.decisions import decide_mixed_precision

        with pytest.raises(ValueError):
            decide_mixed_precision("", 8.0)


class TestAutopilotConfigEmission:
    def test_generated_config_includes_warmup_auto_when_set(
        self, tmp_path, monkeypatch,
    ):
        """Generated config has warmup_auto=true so train.py picks it up."""
        from soup_cli.autopilot.generate_config import generate_config

        monkeypatch.chdir(tmp_path)
        decisions = {
            "task": "sft",
            "format": "alpaca",
            "max_length": 2048,
            "quantization": "4bit",
            "lora": {"r": 16, "alpha": 32, "use_dora": False},
            "lr": 2e-4,
            "epochs": 3,
            "batch_size": 4,
            "grad_accum": 2,
            "perf": {
                "use_flash_attn": True, "use_liger": True,
                "gradient_checkpointing": False,
            },
            "warmup_auto": True,
            "mixed_precision": "bf16",
        }
        out = Path("soup.yaml")
        generate_config(
            base="meta-llama/Llama-3-8B",
            data_path="data.jsonl",
            decisions=decisions,
            output_path=out,
        )
        text = out.read_text(encoding="utf-8")
        cfg = yaml.safe_load(text)
        assert cfg["training"].get("warmup_auto") is True

    def test_decisions_output_must_stay_under_cwd(self, tmp_path, monkeypatch):
        from soup_cli.autopilot.generate_config import generate_config

        monkeypatch.chdir(tmp_path)
        decisions = {
            "task": "sft", "format": "alpaca", "max_length": 1024,
            "quantization": "4bit",
            "lora": {"r": 8, "alpha": 16, "use_dora": False},
            "lr": 2e-4, "epochs": 1, "batch_size": 1, "grad_accum": 1,
            "perf": {
                "use_flash_attn": False, "use_liger": False,
                "gradient_checkpointing": False,
            },
            "output": "/tmp/escape",
        }
        with pytest.raises(ValueError, match="under cwd"):
            generate_config(
                base="meta-llama/Llama-3-8B",
                data_path="data.jsonl",
                decisions=decisions,
                output_path=Path("soup.yaml"),
            )


# --------------------------------------------------------------------------- #
# Integration: lr-finder CLI smoke (offline-safe)                              #
# --------------------------------------------------------------------------- #

class TestLRFinderRunner:
    """save_lr_finder_report writes a JSON report users can plot."""

    def test_save_report(self, tmp_path, monkeypatch):
        from soup_cli.utils.lr_finder import save_lr_finder_report

        monkeypatch.chdir(tmp_path)
        lrs = [1e-6, 1e-5, 1e-4, 1e-3]
        losses = [3.0, 2.8, 2.4, 2.6]
        out = Path("lr_report.json")
        save_lr_finder_report(lrs, losses, out)

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["lrs"] == lrs
        assert data["losses"] == losses
        assert "recommended_lr" in data

    def test_save_report_rejects_nan(self, tmp_path, monkeypatch):
        from soup_cli.utils.lr_finder import save_lr_finder_report

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="non-finite"):
            save_lr_finder_report(
                [1e-6, 1e-5, 1e-4, 1e-3],
                [3.0, 2.8, float("nan"), 2.4],
                Path("report.json"),
            )

    def test_save_report_rejects_infinity(self, tmp_path, monkeypatch):
        from soup_cli.utils.lr_finder import save_lr_finder_report

        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="non-finite"):
            save_lr_finder_report(
                [1e-6, 1e-5, 1e-4, 1e-3],
                [3.0, 2.8, 2.4, float("inf")],
                Path("report.json"),
            )

    def test_save_report_path_must_stay_under_cwd(self, tmp_path, monkeypatch):
        from soup_cli.utils.lr_finder import save_lr_finder_report

        # tmp_path is the inner; we chdir to a subdir so tmp_path itself
        # is outside cwd and thus cannot be the target.
        inner = tmp_path / "inner"
        inner.mkdir()
        monkeypatch.chdir(inner)
        outside = (tmp_path / "lr_report.json").resolve()
        with pytest.raises(ValueError, match="under cwd"):
            save_lr_finder_report([1e-6, 1e-5], [3.0, 2.8], outside)


# --------------------------------------------------------------------------- #
# Cross-cutting: SoupConfig top-level still serializes                         #
# --------------------------------------------------------------------------- #

class TestNewFieldsRoundTrip:
    def test_roundtrip_yaml(self):
        from soup_cli.config.loader import load_config_from_string

        yaml_text = """
base: meta-llama/Llama-3-8B
task: sft
data:
  train: data.jsonl
  format: alpaca
training:
  epochs: 3
  lr: 2e-4
  batch_size: 4
  warmup_auto: true
  loss_watchdog: true
  loss_spike_recovery: true
  loss_spike_recovery_max_attempts: 2
  convergence_detection: true
output: ./out
"""
        cfg = load_config_from_string(yaml_text)
        assert cfg.training.warmup_auto is True
        assert cfg.training.loss_spike_recovery is True
        assert cfg.training.loss_spike_recovery_max_attempts == 2
        assert cfg.training.convergence_detection is True
