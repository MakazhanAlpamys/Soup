"""#1263 -- GRPO variants log the ``kl`` metric trl's stock loss reports.

Since #1249 every non-standard variant adds ``grpo_beta * kl_t`` inside Soup's
variant loss. That loss replaces trl's ``GRPOTrainer._compute_loss``, which is
where trl records the batch-mean per-token KL as the ``kl`` metric, so a variant
run logged no ``kl``. The variant now records it under the same key, with the
same masking and the same cross-process gather as trl 0.29's stock loss; ``rft``
averages over the accepted tokens it applies the KL to.

Helpers and the tiny offline trainer are shared with the #1232 tests.
"""

from __future__ import annotations

import types
from collections import defaultdict

import pytest

from tests.test_issue1232_grpo_variant_kl import (
    VARIANTS,
    _batch,
    _build_real,
    _generated_batch_off_reference,
    _requires_train_extra,
    _StandInGRPOBase,
    _torch,
    _trainer_inputs,
    _trl_stock_loss,
)


def _trl_kl(torch, logp, reference, mask):
    """trl 0.29's ``kl`` metric: masked batch mean of its per-token estimator."""
    kl = torch.exp(reference - logp) - (reference - logp) - 1
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


def _metric(variant, batch, *, beta, mask="batch"):
    from soup_cli.utils.grpo_variants import variant_kl_metric

    return variant_kl_metric(
        variant,
        logp_new=batch["logp_new"],
        advantages=batch["advantages"],
        beta=beta,
        completion_mask=batch["mask"] if mask == "batch" else mask,
        reference_logp=batch["reference"],
    )


class TestTheMetric:
    @pytest.mark.parametrize("variant", [v for v in VARIANTS if v != "rft"])
    def test_equals_trls_masked_batch_mean(self, variant):
        torch = _torch()
        batch = _batch(torch)
        expected = _trl_kl(torch, batch["logp_new"].detach(), batch["reference"], batch["mask"])
        assert _metric(variant, batch, beta=0.1).item() == pytest.approx(expected.item(), rel=1e-12)

    def test_rft_averages_over_its_accepted_tokens_only(self):
        """Rows 0 and 2 have positive advantage; row 1's tokens are not trained on."""
        torch = _torch()
        batch = _batch(torch)
        accepted = batch["mask"] * (batch["advantages"] > 0).to(batch["mask"].dtype).unsqueeze(-1)
        expected = _trl_kl(torch, batch["logp_new"].detach(), batch["reference"], accepted)
        everything = _trl_kl(torch, batch["logp_new"].detach(), batch["reference"], batch["mask"])
        got = _metric("rft", batch, beta=0.1).item()
        assert got == pytest.approx(expected.item(), rel=1e-12)
        assert got != pytest.approx(everything.item(), rel=1e-6)

    def test_rft_without_a_completion_mask_counts_tokens_not_rows(self):
        torch = _torch()
        batch = _batch(torch)
        accepted = (batch["advantages"] > 0).to(batch["mask"].dtype).unsqueeze(-1).expand(3, 5)
        expected = _trl_kl(torch, batch["logp_new"].detach(), batch["reference"], accepted)
        assert _metric("rft", batch, beta=0.1, mask=None).item() == pytest.approx(
            expected.item(), rel=1e-12
        )

    @pytest.mark.parametrize("variant", VARIANTS + ("standard",))
    def test_nothing_to_log_at_beta_zero_or_for_standard(self, variant):
        torch = _torch()
        batch = _batch(torch)
        assert _metric(variant, batch, beta=0.0) is None
        if variant == "standard":
            assert _metric(variant, batch, beta=0.1) is None


class _StandInWithMetrics(_StandInGRPOBase):
    """Adds trl's ``_metrics`` store and an ``accelerator`` with a single-process gather."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.accelerator = types.SimpleNamespace(gather=lambda tensor: tensor.reshape(-1))


def _variant_trainer(variant, **kwargs):
    from soup_cli.trainer.grpo import make_grpo_trainer_variant

    trainer = make_grpo_trainer_variant(_StandInWithMetrics, variant)(**kwargs)
    if variant == "two_sided":
        trainer._soup_grpo_delta = 0.2
    return trainer


class TestTheVariantTrainerLogsIt:
    @pytest.mark.parametrize("variant", VARIANTS)
    def test_a_variant_step_records_kl(self, variant):
        torch = _torch()
        batch = _batch(torch)
        trainer = _variant_trainer(variant, logps=batch["logp_new"], beta=0.1)
        trainer._compute_loss(None, _trainer_inputs(torch, batch))
        expected = _metric(variant, batch, beta=0.1).item()
        assert trainer._metrics["train"]["kl"] == [pytest.approx(expected, rel=1e-12)]

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_beta_zero_records_nothing(self, variant):
        torch = _torch()
        batch = _batch(torch)
        trainer = _variant_trainer(variant, logps=batch["logp_new"], beta=0.0)
        trainer._compute_loss(None, _trainer_inputs(torch, batch, with_reference=False))
        assert "kl" not in trainer._metrics["train"]


class TestTheRealTrainer:
    def test_dapo_logs_the_same_kl_as_trls_stock_loss(self, tmp_path, monkeypatch):
        """On one real generated batch, the variant's ``kl`` equals trl's stock ``kl``."""
        _requires_train_extra()
        import torch

        wrapper = _build_real(tmp_path, monkeypatch, "dapo")
        trainer = wrapper.trainer
        inputs = _generated_batch_off_reference(trainer)
        kl_log = trainer._metrics["train"]["kl"]

        # Same seed before each forward: LoRA dropout is active in train mode.
        torch.manual_seed(3)
        trainer._compute_loss(trainer.model, inputs)
        assert not trainer._soup_fallback_warned, "fell back to trl's stock loss"
        assert len(kl_log) == 1, "the variant loss did not log kl"
        torch.manual_seed(3)
        _trl_stock_loss(trainer, "dapo", inputs)
        assert len(kl_log) == 2
        assert kl_log[0] == pytest.approx(kl_log[1], rel=1e-5, abs=1e-8)
        assert kl_log[0] > 0.0, "the policy was moved off its reference, so kl is non-zero"
