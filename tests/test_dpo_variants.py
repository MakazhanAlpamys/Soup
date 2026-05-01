"""Tests for v0.40.0 Part C — KL-controlled DPO variants.

Adds two opt-in DPO controls:
  * ``dpo_beta_schedule``: anneal β over training (linear / cosine / exponential).
  * ``dpo_ref_regen_epochs``: replace the frozen ref model with the current
    student every N epochs.

Both are SFT-trainer-style additive flags; the existing constant-β,
constant-ref-model path is unchanged when both flags are unset.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig

# ─── Schema bounds ──────────────────────────────────────────────────────────


class TestDPOVariantsConfig:
    def _base(self, **training):
        return SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"dpo_beta": 0.1, **training},
        )

    @pytest.mark.parametrize("sched", ["linear", "cosine", "exponential"])
    def test_beta_schedule_accepted(self, sched):
        cfg = self._base(dpo_beta_schedule=sched, dpo_beta_end=0.01)
        assert cfg.training.dpo_beta_schedule == sched
        assert cfg.training.dpo_beta_end == pytest.approx(0.01)

    def test_beta_schedule_unknown_rejected(self):
        with pytest.raises(ValidationError, match="dpo_beta_schedule"):
            self._base(dpo_beta_schedule="random", dpo_beta_end=0.01)

    def test_beta_schedule_requires_end(self):
        with pytest.raises(ValidationError, match="dpo_beta_end"):
            self._base(dpo_beta_schedule="linear")

    def test_beta_end_must_be_positive(self):
        with pytest.raises(ValidationError, match="dpo_beta_end"):
            self._base(dpo_beta_schedule="linear", dpo_beta_end=0)

    def test_beta_end_alone_rejected(self):
        """Setting end without schedule is meaningless."""
        with pytest.raises(ValidationError, match="dpo_beta_schedule"):
            self._base(dpo_beta_end=0.01)

    def test_ref_regen_epochs_positive(self):
        cfg = self._base(dpo_ref_regen_epochs=2)
        assert cfg.training.dpo_ref_regen_epochs == 2

    def test_ref_regen_epochs_zero_rejected(self):
        with pytest.raises(ValidationError, match="dpo_ref_regen_epochs"):
            self._base(dpo_ref_regen_epochs=0)

    def test_ref_regen_epochs_negative_rejected(self):
        with pytest.raises(ValidationError, match="dpo_ref_regen_epochs"):
            self._base(dpo_ref_regen_epochs=-1)

    def test_ref_regen_epochs_too_large_rejected(self):
        """Bound at 1000 — runaway values are almost certainly typos."""
        with pytest.raises(ValidationError, match="dpo_ref_regen_epochs"):
            self._base(dpo_ref_regen_epochs=10_000)

    def test_dpo_variants_only_for_dpo_family(self):
        """β-schedule + ref-regen require a DPO-family trainer."""
        with pytest.raises(ValidationError, match="dpo|ipo|preference"):
            SoupConfig(
                base="some-model",
                task="sft",
                data={"train": "./data.jsonl"},
                training={
                    "dpo_beta_schedule": "linear",
                    "dpo_beta_end": 0.01,
                },
            )

    def test_ref_regen_only_for_dpo_family(self):
        with pytest.raises(ValidationError, match="dpo|ipo|preference"):
            SoupConfig(
                base="some-model",
                task="orpo",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"dpo_ref_regen_epochs": 2},
            )

    def test_dpo_variants_allowed_on_ipo(self):
        cfg = SoupConfig(
            base="some-model",
            task="ipo",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"dpo_beta_schedule": "cosine", "dpo_beta_end": 0.01},
        )
        assert cfg.training.dpo_beta_schedule == "cosine"

    def test_dpo_variants_allowed_on_preference_dpo(self):
        cfg = SoupConfig(
            base="some-model",
            task="preference",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={
                "preference_loss": "dpo",
                "dpo_beta_schedule": "linear",
                "dpo_beta_end": 0.05,
            },
        )
        assert cfg.training.dpo_beta_end == pytest.approx(0.05)

    def test_dpo_variants_rejected_on_preference_orpo(self):
        with pytest.raises(ValidationError, match="dpo|ipo"):
            SoupConfig(
                base="some-model",
                task="preference",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={
                    "preference_loss": "orpo",
                    "dpo_beta_schedule": "linear",
                    "dpo_beta_end": 0.05,
                },
            )

    def test_dpo_variants_rejected_on_mlx(self):
        with pytest.raises(ValidationError, match="mlx"):
            SoupConfig(
                base="some-model",
                task="dpo",
                backend="mlx",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"dpo_beta_schedule": "linear", "dpo_beta_end": 0.01},
            )


# ─── β schedule math ────────────────────────────────────────────────────────


class TestBetaSchedule:
    def test_linear_endpoints(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=0, total_steps=100, schedule="linear",
        ) == pytest.approx(0.1)
        assert compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=100, total_steps=100, schedule="linear",
        ) == pytest.approx(0.01)

    def test_linear_midpoint(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        # Schema requires beta_end > 0; mid of (0.1, 0.02) = 0.06.
        mid = compute_beta_at_step(
            beta_start=0.1, beta_end=0.02, step=50, total_steps=100, schedule="linear",
        )
        assert mid == pytest.approx(0.06)

    def test_cosine_endpoints(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=0, total_steps=100, schedule="cosine",
        ) == pytest.approx(0.1)
        assert compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=100, total_steps=100, schedule="cosine",
        ) == pytest.approx(0.01)

    def test_cosine_midpoint(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        # Cosine: at midpoint we expect (start+end)/2.
        mid = compute_beta_at_step(
            beta_start=0.2, beta_end=0.02, step=50, total_steps=100, schedule="cosine",
        )
        assert mid == pytest.approx(0.11, abs=1e-6)

    def test_exponential_endpoints(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=0, total_steps=100,
            schedule="exponential",
        ) == pytest.approx(0.1)
        end = compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=100, total_steps=100,
            schedule="exponential",
        )
        assert end == pytest.approx(0.01, rel=1e-4)

    def test_clamps_at_total_steps(self):
        """Step beyond total_steps clamps to beta_end (not extrapolation)."""
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=200, total_steps=100, schedule="linear",
        ) == pytest.approx(0.01)

    def test_negative_step_clamps_to_start(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=-5, total_steps=100, schedule="linear",
        ) == pytest.approx(0.1)

    def test_invalid_schedule_raises(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        with pytest.raises(ValueError, match="schedule"):
            compute_beta_at_step(
                beta_start=0.1, beta_end=0.01, step=0, total_steps=100,
                schedule="garbage",
            )

    def test_zero_total_steps_returns_end(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=0, total_steps=0, schedule="linear",
        ) == pytest.approx(0.01)

    def test_negative_total_steps_rejected(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        with pytest.raises(ValueError, match="total_steps"):
            compute_beta_at_step(
                beta_start=0.1, beta_end=0.01, step=0, total_steps=-1,
                schedule="linear",
            )

    def test_non_finite_betas_rejected(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        for bad in (float("nan"), float("inf"), -1.0):
            with pytest.raises(ValueError, match="finite|> 0"):
                compute_beta_at_step(
                    beta_start=bad, beta_end=0.01, step=0, total_steps=100,
                    schedule="linear",
                )

    def test_step_bool_rejected(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        with pytest.raises(ValueError, match="step"):
            compute_beta_at_step(
                beta_start=0.1, beta_end=0.01, step=True, total_steps=100,
                schedule="linear",
            )


# ─── BetaScheduleCallback ──────────────────────────────────────────────────


class TestBetaScheduleCallback:
    def test_on_step_begin_writes_beta_on_trainer(self):
        from soup_cli.utils.dpo_variants import BetaScheduleCallback

        cb = BetaScheduleCallback(
            beta_start=0.1, beta_end=0.01, total_steps=100, schedule="linear",
        )
        trainer = MagicMock()
        trainer.beta = 0.1
        state = MagicMock(global_step=50)
        cb.on_step_begin(args=None, state=state, control=None, model=None)
        cb.attach(trainer)
        cb.on_step_begin(args=None, state=state, control=None, model=None)
        # After attachment, β is updated.
        assert trainer.beta == pytest.approx(0.055, abs=1e-6)

    def test_callback_no_trainer_attached_is_noop(self):
        """Without a trainer, callback updates nothing."""
        from soup_cli.utils.dpo_variants import BetaScheduleCallback

        cb = BetaScheduleCallback(
            beta_start=0.1, beta_end=0.01, total_steps=100, schedule="linear",
        )
        state = MagicMock(global_step=50)
        # Should not raise.
        cb.on_step_begin(args=None, state=state, control=None, model=None)


# ─── RefModelRegenCallback ─────────────────────────────────────────────────


class TestRefModelRegenCallback:
    def test_fires_on_target_epoch(self):
        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        cb = RefModelRegenCallback(every_n_epochs=2)
        trainer = MagicMock()
        cb.attach(trainer)
        state = MagicMock(epoch=2.0)
        cb.on_epoch_end(args=None, state=state, control=None, model=None)
        assert cb.regen_count == 1

    def test_does_not_fire_off_target(self):
        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        cb = RefModelRegenCallback(every_n_epochs=2)
        trainer = MagicMock()
        cb.attach(trainer)
        state = MagicMock(epoch=1.0)
        cb.on_epoch_end(args=None, state=state, control=None, model=None)
        assert cb.regen_count == 0

    def test_skip_at_epoch_zero(self):
        """Regen at epoch 0 would copy untrained student → undesirable."""
        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        cb = RefModelRegenCallback(every_n_epochs=1)
        trainer = MagicMock()
        cb.attach(trainer)
        state = MagicMock(epoch=0.0)
        cb.on_epoch_end(args=None, state=state, control=None, model=None)
        assert cb.regen_count == 0

    def test_invalid_period_int_below_one_rejected(self):
        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        for bad in (0, -1):
            with pytest.raises(ValueError, match="every_n_epochs"):
                RefModelRegenCallback(every_n_epochs=bad)

    def test_invalid_period_non_int_rejected(self):
        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        for bad in (0.5, "2", True):
            with pytest.raises(TypeError, match="every_n_epochs"):
                RefModelRegenCallback(every_n_epochs=bad)

    def test_no_trainer_attached_is_noop(self):
        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        cb = RefModelRegenCallback(every_n_epochs=2)
        state = MagicMock(epoch=2.0)
        cb.on_epoch_end(args=None, state=state, control=None, model=None)
        assert cb.regen_count == 0


# ─── build_dpo_variant_callbacks ────────────────────────────────────────────


class TestBuildDPOVariantCallbacks:
    def test_returns_empty_list_when_no_variants(self):
        from soup_cli.utils.dpo_variants import build_dpo_variant_callbacks

        cbs = build_dpo_variant_callbacks(
            beta_start=0.1, beta_end=None, schedule=None,
            total_steps=0, ref_regen_epochs=None,
        )
        assert cbs == []

    def test_returns_beta_only(self):
        from soup_cli.utils.dpo_variants import (
            BetaScheduleCallback,
            build_dpo_variant_callbacks,
        )

        cbs = build_dpo_variant_callbacks(
            beta_start=0.1, beta_end=0.01, schedule="linear",
            total_steps=0, ref_regen_epochs=None,
        )
        assert len(cbs) == 1
        assert isinstance(cbs[0], BetaScheduleCallback)

    def test_returns_regen_only(self):
        from soup_cli.utils.dpo_variants import (
            RefModelRegenCallback,
            build_dpo_variant_callbacks,
        )

        cbs = build_dpo_variant_callbacks(
            beta_start=0.1, beta_end=None, schedule=None,
            total_steps=0, ref_regen_epochs=2,
        )
        assert len(cbs) == 1
        assert isinstance(cbs[0], RefModelRegenCallback)

    def test_returns_both_callbacks(self):
        from soup_cli.utils.dpo_variants import (
            BetaScheduleCallback,
            RefModelRegenCallback,
            build_dpo_variant_callbacks,
        )

        cbs = build_dpo_variant_callbacks(
            beta_start=0.1, beta_end=0.01, schedule="linear",
            total_steps=0, ref_regen_epochs=2,
        )
        assert len(cbs) == 2
        kinds = {type(cb) for cb in cbs}
        assert kinds == {BetaScheduleCallback, RefModelRegenCallback}


class TestBetaScheduleLazyTotalSteps:
    def test_on_train_begin_resolves_total_steps_from_state(self):
        from soup_cli.utils.dpo_variants import BetaScheduleCallback

        cb = BetaScheduleCallback(
            beta_start=0.1, beta_end=0.01, total_steps=0, schedule="linear",
        )
        # Initially 0 — sentinel.
        assert cb.total_steps == 0
        state = MagicMock(max_steps=200)
        cb.on_train_begin(args=None, state=state, control=None)
        assert cb.total_steps == 200

    def test_on_step_begin_skips_when_total_steps_unresolvable(self):
        """No max_steps available → don't fall through to compute_beta_at_step."""
        from soup_cli.utils.dpo_variants import BetaScheduleCallback

        cb = BetaScheduleCallback(
            beta_start=0.1, beta_end=0.01, total_steps=0, schedule="linear",
        )
        trainer = MagicMock()
        trainer.beta = 0.1
        cb.attach(trainer)
        state = MagicMock(global_step=10)
        cb.on_step_begin(args=None, state=state, control=None)
        # beta unchanged because total_steps still 0.
        assert trainer.beta == 0.1

    def test_math_module_used_in_cosine(self):
        """Sanity: math module imported at top is exercised by cosine schedule."""
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        # Cosine at progress=0 should equal beta_start exactly (cos(0)=1).
        result = compute_beta_at_step(
            beta_start=0.1, beta_end=0.01, step=1, total_steps=10**9,
            schedule="cosine",
        )
        # Very near beta_start since progress ~ 1e-9.
        assert math.isclose(result, 0.1, rel_tol=1e-6)
