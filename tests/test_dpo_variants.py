"""Tests for v0.40.0 Part C — KL-controlled DPO variants.

Adds two opt-in DPO controls:
  * ``dpo_beta_schedule``: anneal β over training (linear / cosine / exponential).
  * ``dpo_ref_regen_epochs``: replace the frozen ref model with the current
    student every N epochs.

Both are SFT-trainer-style additive flags; the existing constant-β,
constant-ref-model path is unchanged when both flags are unset.
"""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig


def _local_base_model_dir() -> str:
    """A tiny GPT-2 + tokenizer built locally (no download), standing in for
    sshleifer/tiny-gpt2 -- #1356 found these real-trainer tests pulling it
    from the Hub for nothing they check depends on real pretrained weights.
    Shared and cached across every file that needs one (tests/_tiny_hf_models.py)."""
    from tests._tiny_hf_models import tiny_model_dir

    return tiny_model_dir("gpt2")


# ─── Schema bounds ──────────────────────────────────────────────────────────


class TestDPOVariantsConfig:
    def _base(self, **training):
        return SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"dpo_beta": 0.1, **training},
        )

    @pytest.mark.parametrize("lora_r", [1, 8, 16])
    def test_ref_regen_epochs_accepted_with_lora(self, lora_r):
        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"dpo_ref_regen_epochs": 2, "lora": {"r": lora_r}},
        )
        assert cfg.training.dpo_ref_regen_epochs == 2

    def test_ref_regen_epochs_accepted_with_default_lora(self):
        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"dpo_ref_regen_epochs": 2},
        )
        assert cfg.training.dpo_ref_regen_epochs == 2

    def test_ref_regen_epochs_refused_on_full_fine_tuning(self):
        with pytest.raises(ValidationError, match="full fine-tuning"):
            SoupConfig(
                base="some-model",
                task="dpo",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"dpo_ref_regen_epochs": 2, "lora": {"r": 0}},
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


@pytest.mark.parametrize(
    ("task", "extra"),
    [
        ("dpo", {}),
        ("ipo", {}),
        ("preference", {"preference_loss": "dpo"}),
        ("preference", {"preference_loss": "ipo"}),
    ],
)
def test_ref_regen_epochs_accepted_on_every_dpo_family_task(task, extra):
    cfg = SoupConfig(
        base="some-model",
        task=task,
        data={"train": "./data.jsonl", "format": "dpo"},
        training={"dpo_ref_regen_epochs": 2, **extra},
    )
    assert cfg.training.dpo_ref_regen_epochs == 2


@pytest.mark.parametrize(
    ("task", "extra"),
    [
        ("dpo", {}),
        ("ipo", {}),
        ("preference", {"preference_loss": "dpo"}),
        ("preference", {"preference_loss": "ipo"}),
    ],
)
def test_ref_regen_epochs_refused_on_full_fine_tuning_every_dpo_family_task(task, extra):
    with pytest.raises(ValidationError, match="full fine-tuning"):
        SoupConfig(
            base="some-model",
            task=task,
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"dpo_ref_regen_epochs": 2, "lora": {"r": 0}, **extra},
        )


class TestBetaSchedule:
    def test_linear_endpoints(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.01,
            step=0,
            total_steps=100,
            schedule="linear",
        ) == pytest.approx(0.1)
        assert compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.01,
            step=100,
            total_steps=100,
            schedule="linear",
        ) == pytest.approx(0.01)

    def test_linear_midpoint(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        # Schema requires beta_end > 0; mid of (0.1, 0.02) = 0.06.
        mid = compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.02,
            step=50,
            total_steps=100,
            schedule="linear",
        )
        assert mid == pytest.approx(0.06)

    def test_cosine_endpoints(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.01,
            step=0,
            total_steps=100,
            schedule="cosine",
        ) == pytest.approx(0.1)
        assert compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.01,
            step=100,
            total_steps=100,
            schedule="cosine",
        ) == pytest.approx(0.01)

    def test_cosine_midpoint(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        # Cosine: at midpoint we expect (start+end)/2.
        mid = compute_beta_at_step(
            beta_start=0.2,
            beta_end=0.02,
            step=50,
            total_steps=100,
            schedule="cosine",
        )
        assert mid == pytest.approx(0.11, abs=1e-6)

    def test_exponential_endpoints(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.01,
            step=0,
            total_steps=100,
            schedule="exponential",
        ) == pytest.approx(0.1)
        end = compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.01,
            step=100,
            total_steps=100,
            schedule="exponential",
        )
        assert end == pytest.approx(0.01, rel=1e-4)

    def test_clamps_at_total_steps(self):
        """Step beyond total_steps clamps to beta_end (not extrapolation)."""
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.01,
            step=200,
            total_steps=100,
            schedule="linear",
        ) == pytest.approx(0.01)

    def test_negative_step_clamps_to_start(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.01,
            step=-5,
            total_steps=100,
            schedule="linear",
        ) == pytest.approx(0.1)

    def test_invalid_schedule_raises(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        with pytest.raises(ValueError, match="schedule"):
            compute_beta_at_step(
                beta_start=0.1,
                beta_end=0.01,
                step=0,
                total_steps=100,
                schedule="garbage",
            )

    def test_zero_total_steps_returns_end(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        assert compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.01,
            step=0,
            total_steps=0,
            schedule="linear",
        ) == pytest.approx(0.01)

    def test_negative_total_steps_rejected(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        with pytest.raises(ValueError, match="total_steps"):
            compute_beta_at_step(
                beta_start=0.1,
                beta_end=0.01,
                step=0,
                total_steps=-1,
                schedule="linear",
            )

    def test_non_finite_betas_rejected(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        for bad in (float("nan"), float("inf"), -1.0):
            with pytest.raises(ValueError, match="finite|> 0"):
                compute_beta_at_step(
                    beta_start=bad,
                    beta_end=0.01,
                    step=0,
                    total_steps=100,
                    schedule="linear",
                )

    def test_step_bool_rejected(self):
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        with pytest.raises(ValueError, match="step"):
            compute_beta_at_step(
                beta_start=0.1,
                beta_end=0.01,
                step=True,
                total_steps=100,
                schedule="linear",
            )


# ─── BetaScheduleCallback ──────────────────────────────────────────────────


class TestBetaScheduleCallback:
    def test_on_step_begin_writes_beta_on_trainer(self):
        from soup_cli.utils.dpo_variants import BetaScheduleCallback

        cb = BetaScheduleCallback(
            beta_start=0.1,
            beta_end=0.01,
            total_steps=100,
            schedule="linear",
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
            beta_start=0.1,
            beta_end=0.01,
            total_steps=100,
            schedule="linear",
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
            beta_start=0.1,
            beta_end=None,
            schedule=None,
            total_steps=0,
            ref_regen_epochs=None,
        )
        assert cbs == []

    def test_returns_beta_only(self):
        from soup_cli.utils.dpo_variants import (
            BetaScheduleCallback,
            build_dpo_variant_callbacks,
        )

        cbs = build_dpo_variant_callbacks(
            beta_start=0.1,
            beta_end=0.01,
            schedule="linear",
            total_steps=0,
            ref_regen_epochs=None,
        )
        assert len(cbs) == 1
        assert isinstance(cbs[0], BetaScheduleCallback)

    def test_returns_regen_only(self):
        from soup_cli.utils.dpo_variants import (
            RefModelRegenCallback,
            build_dpo_variant_callbacks,
        )

        cbs = build_dpo_variant_callbacks(
            beta_start=0.1,
            beta_end=None,
            schedule=None,
            total_steps=0,
            ref_regen_epochs=2,
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
            beta_start=0.1,
            beta_end=0.01,
            schedule="linear",
            total_steps=0,
            ref_regen_epochs=2,
        )
        assert len(cbs) == 2
        kinds = {type(cb) for cb in cbs}
        assert kinds == {BetaScheduleCallback, RefModelRegenCallback}


class TestRefModelRegenCallbackUnit:
    """Unit tests for RefModelRegenCallback adapter copying."""

    def test_ref_model_regen_copies_lora_adapter_weights(self):
        import torch
        from torch import nn

        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Module()
                self.layer.lora_A_default = nn.Parameter(torch.ones(4, 4), requires_grad=True)
                self.layer.lora_B_default = nn.Parameter(
                    torch.full((4, 4), 2.0), requires_grad=True
                )
                self.layer.lora_A_ref = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
                self.layer.lora_B_ref = nn.Parameter(torch.zeros(4, 4), requires_grad=False)

            def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
                yield "base_model.model.layer.lora_A.default.weight", self.layer.lora_A_default
                yield "base_model.model.layer.lora_B.default.weight", self.layer.lora_B_default
                yield "base_model.model.layer.lora_A.ref.weight", self.layer.lora_A_ref
                yield "base_model.model.layer.lora_B.ref.weight", self.layer.lora_B_ref

        model = DummyModel()
        trainer = MagicMock()
        trainer.ref_model = None
        trainer.model = model

        cb = RefModelRegenCallback(every_n_epochs=1)
        cb.attach(trainer)

        state = MagicMock(epoch=1.0)
        cb.on_epoch_end(args=None, state=state, control=None)

        assert cb.regen_count == 1
        assert torch.equal(model.layer.lora_A_ref, model.layer.lora_A_default)
        assert torch.equal(model.layer.lora_B_ref, model.layer.lora_B_default)
        assert model.layer.lora_A_ref.requires_grad is False
        assert model.layer.lora_B_ref.requires_grad is False

    def test_ref_model_regen_skips_non_divisible_epoch(self):
        import torch
        from torch import nn

        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Parameter(torch.zeros(2, 2), requires_grad=False)

            def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
                yield "lora.ref.weight", self.layer

        model = DummyModel()
        trainer = MagicMock(ref_model=None, model=model)
        cb = RefModelRegenCallback(every_n_epochs=2)
        cb.attach(trainer)

        cb.on_epoch_end(args=None, state=MagicMock(epoch=1.0), control=None)
        assert cb.regen_count == 0

    def test_ref_model_regen_skips_epoch_zero(self):
        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        trainer = MagicMock()
        cb = RefModelRegenCallback(every_n_epochs=1)
        cb.attach(trainer)

        cb.on_epoch_end(args=None, state=MagicMock(epoch=0.0), control=None)
        assert cb.regen_count == 0

    def test_ref_model_regen_warns_when_no_ref_adapter_found(self, caplog):
        import logging

        from torch import nn

        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        model = nn.Linear(2, 2)
        trainer = MagicMock(ref_model=None, model=model)
        cb = RefModelRegenCallback(every_n_epochs=1)
        cb.attach(trainer)

        with caplog.at_level(logging.WARNING):
            cb.on_epoch_end(args=None, state=MagicMock(epoch=1.0), control=None)

        assert cb.regen_count == 0
        assert "no '.ref.' adapter parameters found on model" in caplog.text

    def test_ref_model_regen_counts_nothing_when_no_source_matches(self, caplog):
        import logging

        import torch
        from torch import nn

        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        class OnlyRef(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(2, 2), requires_grad=False)

            def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
                yield "base_model.model.layer.lora_A.ref.weight", self.w

        trainer = MagicMock(ref_model=None, model=OnlyRef())
        cb = RefModelRegenCallback(every_n_epochs=1)
        cb.attach(trainer)
        with caplog.at_level(logging.WARNING):
            cb.on_epoch_end(args=None, state=MagicMock(epoch=1.0), control=None)
        assert cb.regen_count == 0
        assert "matching source parameter" in caplog.text

    def test_ref_model_regen_ensures_ref_adapter_is_frozen(self):
        import torch
        from torch import nn

        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        class ModelWithUnfrozenRef(nn.Module):
            def __init__(self):
                super().__init__()
                self.default_w = nn.Parameter(torch.ones(2, 2), requires_grad=True)
                self.ref_w = nn.Parameter(torch.zeros(2, 2), requires_grad=True)

            def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
                yield "base_model.model.layer.lora_A.default.weight", self.default_w
                yield "base_model.model.layer.lora_A.ref.weight", self.ref_w

        trainer = MagicMock(ref_model=None, model=ModelWithUnfrozenRef())
        cb = RefModelRegenCallback(every_n_epochs=1)
        cb.attach(trainer)
        cb.on_epoch_end(args=None, state=MagicMock(epoch=1.0), control=None)
        assert trainer.model.ref_w.requires_grad is False
        assert cb.regen_count == 1

    def test_ref_model_regen_supports_separate_ref_model(self):
        import torch
        from torch import nn

        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        student = nn.Linear(2, 2)
        student.weight.data.fill_(5.0)
        ref_model = nn.Linear(2, 2)
        ref_model.weight.data.fill_(0.0)

        trainer = MagicMock(ref_model=ref_model, model=student)
        cb = RefModelRegenCallback(every_n_epochs=1)
        cb.attach(trainer)

        cb.on_epoch_end(args=None, state=MagicMock(epoch=1.0), control=None)
        assert cb.regen_count == 1
        assert torch.equal(ref_model.weight, student.weight)


class TestBetaScheduleLazyTotalSteps:
    def test_on_train_begin_resolves_total_steps_from_state(self):
        from soup_cli.utils.dpo_variants import BetaScheduleCallback

        cb = BetaScheduleCallback(
            beta_start=0.1,
            beta_end=0.01,
            total_steps=0,
            schedule="linear",
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
            beta_start=0.1,
            beta_end=0.01,
            total_steps=0,
            schedule="linear",
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
            beta_start=0.1,
            beta_end=0.01,
            step=1,
            total_steps=10**9,
            schedule="cosine",
        )
        # Very near beta_start since progress ~ 1e-9.
        assert math.isclose(result, 0.1, rel_tol=1e-6)


# ─── Issue #1229: TrainerCallback Subclassing & Dispatch ────────────────────


class TestCallbacksSubclassTrainerCallback:
    """Pins issue #1229: callbacks must be real TrainerCallback subclasses."""

    def test_beta_schedule_is_trainer_callback_subclass(self):
        from transformers import TrainerCallback

        from soup_cli.utils.dpo_variants import BetaScheduleCallback

        cb = BetaScheduleCallback(0.1, 0.05, 10, "linear")
        assert isinstance(cb, TrainerCallback)

    def test_ref_model_regen_is_trainer_callback_subclass(self):
        from transformers import TrainerCallback

        from soup_cli.utils.dpo_variants import RefModelRegenCallback

        cb = RefModelRegenCallback(1)
        assert isinstance(cb, TrainerCallback)

    def test_inherits_noop_unimplemented_events(self):
        from soup_cli.utils.dpo_variants import BetaScheduleCallback, RefModelRegenCallback

        beta_cb = BetaScheduleCallback(0.1, 0.05, 10, "linear")
        assert callable(beta_cb.on_epoch_begin)
        # Calling inherited no-op stub must not raise AttributeError
        beta_cb.on_epoch_begin(None, None, None)

        regen_cb = RefModelRegenCallback(1)
        assert callable(regen_cb.on_train_begin)
        regen_cb.on_train_begin(None, None, None)

    def test_callback_handler_dispatch_no_attribute_error(self):
        from transformers.trainer_callback import CallbackHandler, TrainerControl, TrainerState

        from soup_cli.utils.dpo_variants import BetaScheduleCallback, RefModelRegenCallback

        for callback, event in (
            (BetaScheduleCallback(0.1, 0.05, 10, "linear"), "on_epoch_begin"),
            (RefModelRegenCallback(1), "on_train_begin"),
        ):
            handler = CallbackHandler([], None, None, None, None)
            handler.add_callback(callback)
            # HF CallbackHandler.call_event dispatches with getattr(cb, event)
            getattr(handler, event)(None, TrainerState(), TrainerControl())


class TestRealTrainerBetaSchedule:
    """Real DPO / IPO / Preference train() runs on tiny CPU model."""

    @pytest.mark.parametrize("sched", ["linear", "cosine", "exponential"])
    def test_dpo_train_beta_schedule_acts(self, tmp_path, sched):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.dpo import DPOTrainerWrapper
        from soup_cli.utils.dpo_variants import compute_beta_at_step

        rows = [
            {"prompt": f"Q{i}?", "chosen": f"A{i} good.", "rejected": f"A{i} bad."}
            for i in range(4)
        ]
        out_dir = (tmp_path / f"out_dpo_{sched}").as_posix()
        base_dir = Path(_local_base_model_dir()).as_posix()
        cfg = load_config_from_string(f"""
base: {base_dir}
task: dpo
data:
  train: x.jsonl
  max_length: 64
training:
  epochs: 2
  batch_size: 2
  quantization: none
  lr: 1e-4
  dpo_beta: 0.1
  dpo_beta_schedule: {sched}
  dpo_beta_end: 0.05
output: {out_dir}
""")
        wrapper = DPOTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": list(rows)})
        res = wrapper.train()
        assert res.get("total_steps", 0) >= 2
        expected_beta = compute_beta_at_step(
            beta_start=0.1,
            beta_end=0.05,
            step=1,
            total_steps=wrapper.trainer.state.max_steps,
            schedule=sched,
        )
        assert wrapper.trainer.beta == pytest.approx(expected_beta, abs=1e-5)


def _run_schedule(tmp_path, task, training):
    from soup_cli.config.loader import load_config_from_string

    rows = [
        {"prompt": f"Q{i}?", "chosen": f"A{i} good.", "rejected": f"A{i} bad."} for i in range(4)
    ]
    cfg = load_config_from_string(
        f"base: {Path(_local_base_model_dir()).as_posix()}\n"
        f"task: {task}\n"
        "data:\n  train: x.jsonl\n  max_length: 64\n"
        "training:\n  epochs: 2\n  batch_size: 2\n  quantization: none\n  lr: 1e-4\n"
        f"{training}"
        f"output: {(tmp_path / 'out').as_posix()}\n"
    )
    if task == "ipo":
        from soup_cli.trainer.ipo import IPOTrainerWrapper as Wrapper
    else:
        from soup_cli.trainer.preference import PreferenceTrainerWrapper as Wrapper
    wrapper = Wrapper(cfg, device="cpu")
    wrapper.setup({"train": list(rows)})
    return wrapper, wrapper.train()


def test_ipo_schedule_anneals_from_ipo_tau(tmp_path):
    from soup_cli.utils.dpo_variants import compute_beta_at_step

    wrapper, result = _run_schedule(
        tmp_path, "ipo", "  ipo_tau: 0.2\n  dpo_beta_schedule: linear\n  dpo_beta_end: 0.05\n"
    )
    assert result.get("total_steps", 0) >= 2
    expected = compute_beta_at_step(
        beta_start=0.2,
        beta_end=0.05,
        step=1,
        total_steps=wrapper.trainer.state.max_steps,
        schedule="linear",
    )
    assert wrapper.trainer.beta == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("loss", ["dpo", "ipo"])
def test_preference_dispatcher_schedule_acts(tmp_path, loss):
    from soup_cli.utils.dpo_variants import compute_beta_at_step

    start, start_key = (0.1, "dpo_beta") if loss == "dpo" else (0.2, "ipo_tau")
    wrapper, result = _run_schedule(
        tmp_path,
        "preference",
        f"  preference_loss: {loss}\n  {start_key}: {start}\n"
        "  dpo_beta_schedule: cosine\n  dpo_beta_end: 0.05\n",
    )
    assert result.get("total_steps", 0) >= 2
    expected = compute_beta_at_step(
        beta_start=start,
        beta_end=0.05,
        step=1,
        total_steps=wrapper.trainer.state.max_steps,
        schedule="cosine",
    )
    assert wrapper.trainer.beta == pytest.approx(expected, abs=1e-6)


class TestRealTrainerRefModelRegen:
    """Real DPO / IPO / Preference train() runs on tiny CPU model testing ref_regen."""

    @pytest.mark.parametrize(
        ("task", "extra"),
        [
            ("dpo", "  dpo_ref_regen_epochs: 1\n"),
            ("ipo", "  dpo_ref_regen_epochs: 1\n"),
            ("preference", "  preference_loss: dpo\n  dpo_ref_regen_epochs: 1\n"),
            ("preference", "  preference_loss: ipo\n  dpo_ref_regen_epochs: 1\n"),
        ],
    )
    def test_dpo_family_ref_regen_acts_after_epoch_1(self, tmp_path, task, extra):
        import torch
        from transformers import TrainerCallback

        class _EpochObserver(TrainerCallback):
            def __init__(self, trainer):
                self.trainer = trainer
                self.epoch1_regen_count = None
                self.epoch1_all_equal = None
                self.epoch1_ref_requires_grad = None

            def on_train_begin(self, args, state, control, **kwargs):
                # Ensure observer runs AFTER RefModelRegenCallback in on_epoch_end
                cbs = self.trainer.callback_handler.callbacks
                if self in cbs:
                    cbs.remove(self)
                    cbs.append(self)

            def on_epoch_end(self, args, state, control, **kwargs):
                if int(round(state.epoch)) == 1:
                    regen_cbs = [
                        cb for cb in self.trainer.callback_handler.callbacks
                        if type(cb).__name__ == "RefModelRegenCallback"
                    ]
                    assert len(regen_cbs) == 1
                    self.epoch1_regen_count = regen_cbs[0].regen_count
                    named = dict(self.trainer.model.named_parameters())
                    ref_params = {k: v for k, v in named.items() if ".ref." in k}
                    self.epoch1_all_equal = bool(ref_params) and all(
                        torch.equal(ref_p, named[ref_name.replace(".ref.", ".default.", 1)])
                        for ref_name, ref_p in ref_params.items()
                    )
                    self.epoch1_ref_requires_grad = any(
                        p.requires_grad for p in ref_params.values()
                    )

        from soup_cli.config.loader import load_config_from_string

        rows = [
            {"prompt": f"Q{i}?", "chosen": f"A{i} good.", "rejected": f"A{i} bad."}
            for i in range(4)
        ]
        cfg = load_config_from_string(
            f"base: {Path(_local_base_model_dir()).as_posix()}\n"
            f"task: {task}\n"
            "data:\n  train: x.jsonl\n  max_length: 64\n"
            "training:\n  epochs: 2\n  batch_size: 2\n  quantization: none\n  lr: 1e-4\n"
            f"{extra}"
            f"output: {(tmp_path / 'out').as_posix()}\n"
        )
        if task == "dpo":
            from soup_cli.trainer.dpo import DPOTrainerWrapper as Wrapper
        elif task == "ipo":
            from soup_cli.trainer.ipo import IPOTrainerWrapper as Wrapper
        else:
            from soup_cli.trainer.preference import PreferenceTrainerWrapper as Wrapper

        wrapper = Wrapper(cfg, device="cpu")
        wrapper.setup({"train": list(rows)})

        observer = _EpochObserver(wrapper.trainer)
        wrapper.trainer.add_callback(observer)

        wrapper.train()

        regen_cbs = [
            cb for cb in wrapper.trainer.callback_handler.callbacks
            if type(cb).__name__ == "RefModelRegenCallback"
        ]
        assert len(regen_cbs) == 1
        regen_cb = regen_cbs[0]

        # Criteria 1 & 2: after epoch 1 each .ref. tensor equals its
        # .default. tensor and regen_count == 1
        assert observer.epoch1_regen_count == 1
        assert observer.epoch1_all_equal is True
        assert observer.epoch1_ref_requires_grad is False

        # After epoch 2 completes: regen_count == 2
        assert regen_cb.regen_count == 2
        named = dict(wrapper.trainer.model.named_parameters())
        ref_params = {k: v for k, v in named.items() if ".ref." in k}
        assert len(ref_params) > 0
        assert all(
            torch.equal(ref_p, named[ref_name.replace(".ref.", ".default.", 1)])
            for ref_name, ref_p in ref_params.items()
        )
        assert not any(p.requires_grad for p in ref_params.values())

    def test_dpo_ref_regen_skips_non_divisible_epochs(self, tmp_path):
        import torch
        from transformers import TrainerCallback

        class _EpochObserver(TrainerCallback):
            def __init__(self, trainer):
                self.trainer = trainer
                self.epoch1_regen_count = None
                self.epoch1_all_equal = None
                self.epoch1_all_999 = None
                self.epoch1_ref_requires_grad = None

            def on_train_begin(self, args, state, control, **kwargs):
                cbs = self.trainer.callback_handler.callbacks
                if self in cbs:
                    cbs.remove(self)
                    cbs.append(self)

            def on_epoch_end(self, args, state, control, **kwargs):
                if int(round(state.epoch)) == 1:
                    regen_cbs = [
                        cb for cb in self.trainer.callback_handler.callbacks
                        if type(cb).__name__ == "RefModelRegenCallback"
                    ]
                    assert len(regen_cbs) == 1
                    self.epoch1_regen_count = regen_cbs[0].regen_count
                    named = dict(self.trainer.model.named_parameters())
                    ref_params = {k: v for k, v in named.items() if ".ref." in k}
                    self.epoch1_all_999 = bool(ref_params) and all(
                        torch.equal(ref_p, torch.full_like(ref_p, 999.0))
                        for ref_p in ref_params.values()
                    )
                    self.epoch1_all_equal = bool(ref_params) and all(
                        torch.equal(ref_p, named[ref_name.replace(".ref.", ".default.", 1)])
                        for ref_name, ref_p in ref_params.items()
                    )
                    self.epoch1_ref_requires_grad = any(
                        p.requires_grad for p in ref_params.values()
                    )

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        rows = [
            {"prompt": f"Q{i}?", "chosen": f"A{i} good.", "rejected": f"A{i} bad."}
            for i in range(4)
        ]
        cfg = load_config_from_string(
            f"base: {Path(_local_base_model_dir()).as_posix()}\n"
            "task: dpo\n"
            "data:\n  train: x.jsonl\n  max_length: 64\n"
            "training:\n  epochs: 2\n  batch_size: 2\n  quantization: none\n  lr: 1e-4\n"
            "  dpo_ref_regen_epochs: 2\n"
            f"output: {(tmp_path / 'out').as_posix()}\n"
        )
        wrapper = DPOTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": list(rows)})

        # Intentionally fill ref parameters with sentinel before training to prove
        # that epoch 1 leaves them completely untouched.
        with torch.no_grad():
            for name, param in wrapper.trainer.model.named_parameters():
                if ".ref." in name:
                    param.fill_(999.0)

        observer = _EpochObserver(wrapper.trainer)
        wrapper.trainer.add_callback(observer)

        wrapper.train()

        regen_cbs = [
            cb for cb in wrapper.trainer.callback_handler.callbacks
            if type(cb).__name__ == "RefModelRegenCallback"
        ]
        assert len(regen_cbs) == 1
        regen_cb = regen_cbs[0]

        # Criteria 3: Epoch 1 is not divisible by 2, so ref is unchanged and receives no gradients
        assert observer.epoch1_regen_count == 0
        assert observer.epoch1_all_999 is True
        assert observer.epoch1_all_equal is False
        assert observer.epoch1_ref_requires_grad is False

        # After epoch 2 (divisible by 2): regen_count == 1 and .ref. equals .default.
        assert regen_cb.regen_count == 1
        named = dict(wrapper.trainer.model.named_parameters())
        ref_params = {k: v for k, v in named.items() if ".ref." in k}
        assert len(ref_params) > 0
        assert all(
            torch.equal(ref_p, named[ref_name.replace(".ref.", ".default.", 1)])
            for ref_name, ref_p in ref_params.items()
        )
        assert not any(
            torch.equal(ref_p, torch.full_like(ref_p, 999.0))
            for ref_p in ref_params.values()
        )
        assert not any(p.requires_grad for p in ref_params.values())


def _lazy_body_names(tree):
    names = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        if any(isinstance(t, ast.Name) and t.id == "_LAZY_CALLBACKS" for t in node.targets):
            names |= {v.id for v in node.value.values if isinstance(v, ast.Name)}
    return names


def test_no_bare_callback_classes():
    from transformers import TrainerCallback

    import soup_cli

    root = Path(soup_cli.__file__).parent
    events = {name for name in dir(TrainerCallback) if name.startswith("on_")}
    bare = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        lazy = _lazy_body_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name in lazy:
                continue
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            if not methods & events:
                continue
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            if not node.bases or bases == ["object"]:
                bare.append(f"{path.relative_to(root).as_posix()}::{node.name}")
    assert not bare, f"callback classes without a TrainerCallback base: {bare}"


def test_every_lazy_callback_builds_a_trainer_callback():
    from transformers import TrainerCallback

    import soup_cli

    root = Path(soup_cli.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        if not _lazy_body_names(tree):
            continue
        dotted = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        module = importlib.import_module(dotted)
        for name in module._LAZY_CALLBACKS:
            cls = getattr(module, name)
            assert issubclass(cls, TrainerCallback), f"{dotted}.{name}"
