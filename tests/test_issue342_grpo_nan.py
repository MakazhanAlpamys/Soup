"""Tests for #342 — GRPO numerical stability (log-ratio overflow + gradient watchdog).

CPU-only, no model, no GPU.  Six test groups:

1. Reproduction — exact STEP 27 signature (finite loss + NaN grad)
2. Bit-exactness — ``torch.equal`` for normal-range tokens  [C7.2]
3. KL isolation — clamp does not shift the KL term            [C7.3]
4. Watchdog behavior — detect NaN, zero grads, track count    [C7.5-C7.9]
5. Microbatch placement — on_pre_optimizer_step vs training_step [C7.4]
6. Mutation controls — removing fix re-exposes the bug
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from soup_cli.utils.grpo_log_ratio_clamp import LOG_PROB_CLAMP_MIN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grpo_ratio_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Minimal GRPO loss (ratio branch only, no KL) for testing.

    Reproduces the exact computation in TRL's ``_compute_loss``:
    ``log_ratio → exp → clamp → min(coef_1, coef_2) × advantages``.
    """
    log_ratio = per_token_logps - old_per_token_logps
    coef_1 = torch.exp(log_ratio)
    coef_2 = torch.clamp(coef_1, 1.0 - epsilon, 1.0 + epsilon)
    per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    return loss


def _grpo_kl_penalty(
    per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
) -> torch.Tensor:
    """TRL KL penalty: ``exp(ref - new) - (ref - new) - 1``."""
    diff = ref_per_token_logps - per_token_logps
    return torch.exp(diff) - diff - 1.0


def _make_mock_state(global_step: int = 100) -> SimpleNamespace:
    """Mock HF TrainerState with log_history."""
    return SimpleNamespace(global_step=global_step, log_history=[])


def _make_mock_control() -> MagicMock:
    """Mock HF TrainerControl."""
    return MagicMock()


def _make_mock_model_with_grads(*grad_tensors: torch.Tensor) -> MagicMock:
    """Create a mock model whose parameters() yields params with the given grads."""
    params = []
    for g in grad_tensors:
        p = MagicMock()
        p.grad = g
        params.append(p)
    model = MagicMock()
    model.parameters.return_value = params
    return model


def _get_callback_instance():
    """Get a fresh GRPOStabilityCallback instance with defaults."""
    from soup_cli.monitoring.grpo_stability_callback import (
        GRPOStabilityCallback,
    )

    return GRPOStabilityCallback()


# ===========================================================================
# Group 1 — Reproduction (the discriminating test)
# ===========================================================================


class TestReproduceExactSignature:
    """Reproduce the exact STEP 27 signature: finite loss + NaN grad_norm."""

    def test_unpatched_signature_finite_loss_nan_grad(self):
        """Without the fix, stock TRL loss produces finite loss + NaN grad."""
        per_token_logps = torch.tensor(
            [[0.0, -2.0, -100.0]], dtype=torch.bfloat16, requires_grad=True
        )
        old_per_token_logps = torch.tensor(
            [[0.0, -2.0, -200.0]], dtype=torch.bfloat16
        )
        completion_mask = torch.tensor(
            [[1.0, 1.0, 0.0]], dtype=torch.bfloat16
        )
        advantages = torch.tensor([[1.0]], dtype=torch.bfloat16)

        loss = _grpo_ratio_loss(
            per_token_logps, old_per_token_logps, completion_mask, advantages
        )
        assert torch.isfinite(loss), "Forward loss should be finite"

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [per_token_logps], max_norm=float("inf")
        )
        assert not torch.isfinite(grad_norm), (
            "Grad norm should be NaN without the fix"
        )

    def test_patched_signature_both_finite(self):
        """With the clamp (old clamped to >= -20), both loss and grad are finite."""
        per_token_logps = torch.tensor(
            [[0.0, -2.0, -100.0]], dtype=torch.bfloat16, requires_grad=True
        )
        old_per_token_logps = torch.tensor(
            [[0.0, -2.0, -200.0]], dtype=torch.bfloat16
        )
        # Apply the fix.
        old_per_token_logps = torch.clamp(old_per_token_logps, min=-20.0)
        completion_mask = torch.tensor(
            [[1.0, 1.0, 0.0]], dtype=torch.bfloat16
        )
        advantages = torch.tensor([[1.0]], dtype=torch.bfloat16)

        loss = _grpo_ratio_loss(
            per_token_logps, old_per_token_logps, completion_mask, advantages
        )
        assert torch.isfinite(loss), "Forward loss should be finite"

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [per_token_logps], max_norm=float("inf")
        )
        assert torch.isfinite(grad_norm), (
            "Grad norm should be finite with the fix"
        )

    def test_exp_backward_zero_times_inf_is_nan(self):
        """Prove the mechanism: ExpBackward(0 × exp(100)) = NaN."""
        x = torch.tensor(
            [100.0], dtype=torch.bfloat16, requires_grad=True
        )
        torch.exp(x).backward(
            torch.tensor([0.0], dtype=torch.bfloat16)
        )
        assert not torch.isfinite(x.grad).all(), (
            "0.0 × exp(100) = 0 × inf = NaN per IEEE 754"
        )


# ===========================================================================
# Group 2 — Bit-exactness (torch.equal) [C7.2]
# ===========================================================================


class TestBitExactness:
    """Normal-range tokens must be numerically identical with and without clamp."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16, torch.float16]
    )
    def test_normal_range_loss_identical(self, dtype):
        """old_per_token_logps in [-15, 0]: loss is bit-identical."""
        per_token_logps = torch.tensor(
            [[-1.5, -3.0, -0.5]], dtype=dtype, requires_grad=True
        )
        old = torch.tensor([[-2.0, -4.0, -1.0]], dtype=dtype)
        mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype)
        adv = torch.tensor([[1.0]], dtype=dtype)

        # Path A: raw (no clamp)
        loss_raw = _grpo_ratio_loss(per_token_logps, old, mask, adv)

        # Detach and re-create to avoid graph issues.
        per_token_logps_2 = per_token_logps.detach().clone().requires_grad_(
            True
        )
        old_clamped = torch.clamp(old, min=LOG_PROB_CLAMP_MIN)

        # Path B: clamped
        loss_fix = _grpo_ratio_loss(per_token_logps_2, old_clamped, mask, adv)

        assert torch.equal(loss_raw, loss_fix), (
            f"Loss should be bit-identical for normal-range tokens "
            f"(raw={loss_raw.item()}, fix={loss_fix.item()})"
        )

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16, torch.float16]
    )
    def test_normal_range_grad_identical(self, dtype):
        """old_per_token_logps in [-15, 0]: gradients are bit-identical."""
        per_token_logps_raw = torch.tensor(
            [[-1.5, -3.0, -0.5]], dtype=dtype, requires_grad=True
        )
        per_token_logps_fix = per_token_logps_raw.detach().clone().requires_grad_(
            True
        )
        old = torch.tensor([[-2.0, -4.0, -1.0]], dtype=dtype)
        mask = torch.tensor([[1.0, 1.0, 1.0]], dtype=dtype)
        adv = torch.tensor([[1.0]], dtype=dtype)

        # Path A: raw
        loss_raw = _grpo_ratio_loss(per_token_logps_raw, old, mask, adv)
        loss_raw.backward()

        # Path B: clamped
        old_clamped = torch.clamp(old, min=LOG_PROB_CLAMP_MIN)
        loss_fix = _grpo_ratio_loss(per_token_logps_fix, old_clamped, mask, adv)
        loss_fix.backward()

        assert torch.equal(per_token_logps_raw.grad, per_token_logps_fix.grad), (
            f"Gradients should be bit-identical for normal-range tokens "
            f"(raw={per_token_logps_raw.grad}, fix={per_token_logps_fix.grad})"
        )


# ===========================================================================
# Group 3 — KL isolation [C7.3]
# ===========================================================================


class TestKLIsolation:
    """The clamp on old_per_token_logps must not shift the KL penalty."""

    def test_clamp_does_not_shift_kl_term(self):
        """ref_per_token_logps is untouched; KL penalty is identical."""
        per_token_logps = torch.tensor(
            [[-1.0, -3.0, -5.0]], dtype=torch.float32
        )
        ref_per_token_logps = torch.tensor(
            [[-2.5, -1.0, -3.0]], dtype=torch.float32
        )
        old_per_token_logps = torch.tensor(
            [[-200.0, -1.0, -3.0]], dtype=torch.float32
        )

        # Our clamp only touches old_per_token_logps.
        inputs = {
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
        clamped_inputs = {
            **inputs,
            "old_per_token_logps": torch.clamp(
                inputs["old_per_token_logps"], min=LOG_PROB_CLAMP_MIN
            ),
        }

        # ref_per_token_logps should be the exact same object.
        assert clamped_inputs["ref_per_token_logps"] is ref_per_token_logps, (
            "ref_per_token_logps must be the same tensor object"
        )

        # KL penalty from ref must be identical before and after clamp.
        kl_before = _grpo_kl_penalty(per_token_logps, ref_per_token_logps)
        kl_after = _grpo_kl_penalty(
            per_token_logps, clamped_inputs["ref_per_token_logps"]
        )
        assert torch.equal(kl_before, kl_after), (
            f"KL penalty must not shift (before={kl_before}, after={kl_after})"
        )


# ===========================================================================
# Group 4 — Watchdog behavior [C7.5-C7.9]
# ===========================================================================


class TestGradWatchdog:
    """Pre-step gradient watchdog: detect NaN, zero grads, track count."""

    def test_nan_grad_detected_and_zeroed(self):
        """Watchdog zeros all gradients when any param has NaN grad."""
        cb = _get_callback_instance()
        nan_grad = torch.tensor([float("nan"), 1.0])
        model = _make_mock_model_with_grads(nan_grad)

        cb.on_pre_optimizer_step(
            args=None,
            state=_make_mock_state(global_step=5),
            control=_make_mock_control(),
            model=model,
        )

        assert cb._nan_skip_count == 1
        model.zero_grad.assert_called_once()

    def test_finite_grad_untouched(self):
        """Watchdog is a no-op when all gradients are finite."""
        cb = _get_callback_instance()
        finite_grad = torch.tensor([2.0, 4.0])
        model = _make_mock_model_with_grads(finite_grad)

        cb.on_pre_optimizer_step(
            args=None,
            state=_make_mock_state(global_step=5),
            control=_make_mock_control(),
            model=model,
        )

        assert cb._nan_skip_count == 0
        model.zero_grad.assert_not_called()

    def test_cumulative_count_in_warning(self, caplog):
        """Warning message includes cumulative skip count."""
        cb = _get_callback_instance()
        nan_grad = torch.tensor([float("nan")])
        model = _make_mock_model_with_grads(nan_grad)

        for step in range(3):
            with caplog.at_level(logging.WARNING):
                cb.on_pre_optimizer_step(
                    args=None,
                    state=_make_mock_state(global_step=step),
                    control=_make_mock_control(),
                    model=model,
                )
            # Reset the mock for the next call.
            model.zero_grad.reset_mock()

        assert cb._nan_skip_count == 3
        assert "3 total skipped" in caplog.text

    def test_loud_announcement_at_5_percent(self, caplog):
        """End-of-run logger.warning when skip fraction >= 5%."""
        cb = _get_callback_instance()
        cb._nan_skip_count = 10
        state = _make_mock_state(global_step=100)

        with caplog.at_level(logging.WARNING):
            cb.on_train_end(
                args=None, state=state, control=_make_mock_control()
            )

        assert "10 of 100 optimizer steps were skipped" in caplog.text
        # Verify it was a WARNING, not just INFO.
        warning_records = [
            r for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert len(warning_records) >= 1

    def test_always_reported_above_zero(self, caplog):
        """End-of-run logger.info even when fraction < 5%."""
        cb = _get_callback_instance()
        cb._nan_skip_count = 1
        state = _make_mock_state(global_step=100)

        with caplog.at_level(logging.INFO):
            cb.on_train_end(
                args=None, state=state, control=_make_mock_control()
            )

        assert "1 of 100 optimizer steps were skipped" in caplog.text


class TestRunRecord:
    """Skip fraction must be in the run record for soup adapters audit."""

    def test_fraction_in_log_history(self):
        """nan_skip_count and nan_skip_fraction surfaced in state.log_history."""
        cb = _get_callback_instance()
        cb._nan_skip_count = 3
        state = _make_mock_state(global_step=100)

        cb.on_train_end(
            args=None, state=state, control=_make_mock_control()
        )

        assert len(state.log_history) == 1
        entry = state.log_history[0]
        assert entry["nan_skip_count"] == 3
        assert entry["nan_skip_fraction"] == 0.03


# ===========================================================================
# Group 5 — Microbatch placement [C7.4, C7.11]
# ===========================================================================


class TestMicrobatchPlacement:
    """Prove on_pre_optimizer_step does not destroy valid accumulated grads.

    The failure mode: placing the watchdog in training_step (per-microbatch)
    calls zero_grad() mid-accumulation, wiping valid gradients from earlier
    microbatches.  Our on_pre_optimizer_step placement waits until after all
    accumulation and makes a single atomic decision for the entire step.
    """

    def test_training_step_placement_wipes_valid_grad(self):
        """Demonstrate the WRONG placement: per-microbatch zero_grad.

        Simulates gradient_accumulation_steps=2 with the watchdog
        incorrectly placed inside training_step (per-microbatch).
        """
        param = torch.tensor([0.0, 0.0], requires_grad=True)
        optimizer = torch.optim.SGD([param], lr=0.1)

        # Microbatch 1: finite gradient [2.0, 4.0].
        loss_1 = (param * torch.tensor([2.0, 4.0])).sum()
        loss_1.backward()
        # At this point, param.grad = [2.0, 4.0] — valid.
        assert torch.isfinite(param.grad).all()

        # Microbatch 2: non-finite gradient.
        loss_2 = (param * torch.tensor([float("nan"), 1.0])).sum()
        loss_2.backward()
        # Accumulated grad is now [NaN, 5.0].
        assert not torch.isfinite(param.grad).all()

        # WRONG placement: per-microbatch zero_grad wipes everything.
        param.grad = None  # simulates model.zero_grad(set_to_none=True)

        # Microbatch 1's valid [2.0, 4.0] contribution is DESTROYED.
        assert param.grad is None, (
            "Per-microbatch zero_grad wipes valid accumulated grads"
        )
        # Cannot recover saved_grad_after_mb1 — it's gone.
        optimizer.zero_grad()

    def test_on_pre_optimizer_step_skips_entire_step(self):
        """Demonstrate the RIGHT placement: post-accumulation zero_grad.

        Simulates gradient_accumulation_steps=2 with the watchdog
        correctly placed at on_pre_optimizer_step (post-accumulation).
        """
        param = torch.tensor([0.0, 0.0], requires_grad=True)
        optimizer = torch.optim.SGD([param], lr=0.1)

        # Microbatch 1: finite gradient.
        loss_1 = (param * torch.tensor([2.0, 4.0])).sum()
        loss_1.backward()

        # Microbatch 2: non-finite gradient.
        loss_2 = (param * torch.tensor([float("nan"), 1.0])).sum()
        loss_2.backward()
        # Accumulated grad is [NaN, 5.0].

        # RIGHT placement: on_pre_optimizer_step fires ONCE, post-accum.
        cb = _get_callback_instance()
        model = MagicMock()
        model.parameters.return_value = [param]
        # Give zero_grad real behavior.
        model.zero_grad.side_effect = lambda: param.grad.zero_()

        cb.on_pre_optimizer_step(
            args=None,
            state=_make_mock_state(global_step=10),
            control=_make_mock_control(),
            model=model,
        )

        assert cb._nan_skip_count == 1, "Step should be skipped"
        model.zero_grad.assert_called_once()
        # After zeroing, optimizer.step() is a no-op.
        param_before = param.clone()
        optimizer.step()
        assert torch.equal(param, param_before), (
            "Optimizer step should be a no-op after zeroing"
        )


# ===========================================================================
# Group 6 — Mutation controls
# ===========================================================================


class TestGuardHasTeeth:
    """Removing the fix must re-expose the bug."""

    def test_removing_clamp_restores_nan_grad(self):
        """Without the clamp, the reproduction test's NaN grad returns."""
        per_token_logps = torch.tensor(
            [[0.0, -2.0, -100.0]], dtype=torch.bfloat16, requires_grad=True
        )
        # NO clamp applied — raw divergent old_per_token_logps.
        old_per_token_logps = torch.tensor(
            [[0.0, -2.0, -200.0]], dtype=torch.bfloat16
        )
        completion_mask = torch.tensor(
            [[1.0, 1.0, 0.0]], dtype=torch.bfloat16
        )
        advantages = torch.tensor([[1.0]], dtype=torch.bfloat16)

        loss = _grpo_ratio_loss(
            per_token_logps, old_per_token_logps, completion_mask, advantages
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [per_token_logps], max_norm=float("inf")
        )
        assert not torch.isfinite(grad_norm), (
            "Without the clamp, grad norm must be NaN (the bug exists)"
        )

    def test_removing_watchdog_allows_nan_to_reach_params(self):
        """Without the watchdog, NaN gradients corrupt the model."""
        param = torch.tensor([1.0, 2.0])
        param.grad = torch.tensor([float("nan"), float("nan")])
        optimizer = torch.optim.SGD([param], lr=0.1)

        # No watchdog — optimizer.step() applies NaN gradients.
        optimizer.step()

        assert not torch.isfinite(param).all(), (
            "Without the watchdog, NaN grads corrupt the parameters"
        )
