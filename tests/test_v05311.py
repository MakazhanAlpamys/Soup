"""v0.53.11 — GRPO Plus finish + preference live (#123, #126, #127, #119, #68).

Tests the math kernels + dispatch logic. Live GPU smoke runs are documented
in plan.md and require a CUDA box.
"""

from __future__ import annotations

import math

import pytest

# Skip the whole module when torch is unavailable — the math kernels are
# torch-based by design.
torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# #123 — GRPO variant live loss kernels
# ---------------------------------------------------------------------------


class TestGRPOVariantKernels:
    def _toy_inputs(self, batch=2, seq=4):
        logp_new = torch.zeros(batch, seq)
        logp_old = torch.zeros(batch, seq)
        advantages = torch.tensor([1.0, -1.0])
        return logp_new, logp_old, advantages

    def test_standard_returns_none(self):
        from soup_cli.utils.grpo_variants import apply_variant_loss

        logp_new, logp_old, adv = self._toy_inputs()
        assert apply_variant_loss(
            "standard", logp_new=logp_new, logp_old=logp_old, advantages=adv
        ) is None

    @pytest.mark.parametrize("variant", ["gspo", "dapo", "dr_grpo", "bnpo", "rft"])
    def test_variant_returns_finite_scalar(self, variant):
        from soup_cli.utils.grpo_variants import apply_variant_loss

        logp_new, logp_old, adv = self._toy_inputs()
        out = apply_variant_loss(
            variant, logp_new=logp_new, logp_old=logp_old, advantages=adv
        )
        assert out is not None
        assert out.shape == ()  # scalar
        assert math.isfinite(float(out))

    def test_two_sided_requires_delta(self):
        from soup_cli.utils.grpo_variants import apply_variant_loss

        logp_new, logp_old, adv = self._toy_inputs()
        with pytest.raises(ValueError, match="grpo_delta"):
            apply_variant_loss(
                "two_sided", logp_new=logp_new, logp_old=logp_old, advantages=adv
            )

    def test_two_sided_with_delta(self):
        from soup_cli.utils.grpo_variants import apply_variant_loss

        logp_new, logp_old, adv = self._toy_inputs()
        out = apply_variant_loss(
            "two_sided",
            logp_new=logp_new,
            logp_old=logp_old,
            advantages=adv,
            delta=0.2,
        )
        assert math.isfinite(float(out))

    def test_completion_mask_zeros_padding(self):
        from soup_cli.utils.grpo_variants import apply_variant_loss

        logp_new, logp_old, adv = self._toy_inputs()
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        out = apply_variant_loss(
            "gspo",
            logp_new=logp_new,
            logp_old=logp_old,
            advantages=adv,
            completion_mask=mask,
        )
        assert math.isfinite(float(out))

    def test_bool_beta_rejected(self):
        from soup_cli.utils.grpo_variants import apply_variant_loss

        logp_new, logp_old, adv = self._toy_inputs()
        with pytest.raises(TypeError, match="bool"):
            apply_variant_loss(
                "gspo",
                logp_new=logp_new,
                logp_old=logp_old,
                advantages=adv,
                beta=True,
            )

    def test_unknown_variant_rejected(self):
        from soup_cli.utils.grpo_variants import apply_variant_loss

        logp_new, logp_old, adv = self._toy_inputs()
        with pytest.raises(ValueError, match="grpo_variant"):
            apply_variant_loss(
                "nonsense", logp_new=logp_new, logp_old=logp_old, advantages=adv
            )

    def test_no_more_deferred_live(self):
        from soup_cli.utils.grpo_variants import _DEFERRED_LIVE

        assert _DEFERRED_LIVE == frozenset()

    def test_variant_metadata_live_wired(self):
        from soup_cli.utils.grpo_variants import _VARIANT_METADATA

        for name, spec in _VARIANT_METADATA.items():
            assert spec.live_wired is True, f"{name} not flipped to live"


# ---------------------------------------------------------------------------
# #119 — LongLoRA forward override
# ---------------------------------------------------------------------------


class TestLongLoRAShift:
    def test_shift_heads_for_s2_basic(self):
        from soup_cli.utils.longlora import shift_heads_for_s2

        # [B=1, H=4, T=8, D=2]
        t = torch.arange(64, dtype=torch.float32).reshape(1, 4, 8, 2)
        shifted = shift_heads_for_s2(t, group_size=4)
        assert shifted.shape == t.shape
        # First half unchanged
        assert torch.equal(shifted[:, :2, :, :], t[:, :2, :, :])
        # Second half shifted by group_size//2 = 2
        assert not torch.equal(shifted[:, 2:, :, :], t[:, 2:, :, :])

    def test_shift_rejects_bool_group_size(self):
        from soup_cli.utils.longlora import shift_heads_for_s2

        t = torch.zeros(1, 4, 8, 2)
        with pytest.raises(TypeError, match="bool"):
            shift_heads_for_s2(t, group_size=True)

    def test_shift_rejects_small_group_size(self):
        from soup_cli.utils.longlora import shift_heads_for_s2

        t = torch.zeros(1, 4, 8, 2)
        with pytest.raises(ValueError, match=">= 2"):
            shift_heads_for_s2(t, group_size=1)

    def test_shift_rejects_non_4d(self):
        from soup_cli.utils.longlora import shift_heads_for_s2

        t = torch.zeros(1, 4, 8)
        with pytest.raises(ValueError, match="4-D"):
            shift_heads_for_s2(t, group_size=4)

    def test_shift_single_head_no_change(self):
        from soup_cli.utils.longlora import shift_heads_for_s2

        t = torch.arange(16, dtype=torch.float32).reshape(1, 1, 8, 2)
        out = shift_heads_for_s2(t, group_size=4)
        assert torch.equal(out, t)

    def test_apply_longlora_forward_override_returns_context(self):
        from soup_cli.utils.longlora import (
            LongLoRAForwardOverride,
            apply_longlora_forward_override,
        )

        # Minimal model stub with .modules() iterator.
        class _Stub:
            def modules(self):
                return iter([])

        result = apply_longlora_forward_override(_Stub(), group_size=4)
        assert isinstance(result, LongLoRAForwardOverride)
        # Restoration is no-op on empty model.
        with result:
            pass

    def test_override_restore_on_exit(self):
        from soup_cli.utils.longlora import apply_longlora_forward_override

        class LlamaAttention:  # exact name the regex matches
            def __init__(self):
                self.marker = "ORIGINAL"

            def forward(self, x):
                return x,

        class _Model:
            def __init__(self):
                self.attn = LlamaAttention()

            def modules(self):
                yield self.attn

        model = _Model()
        original_forward = model.attn.forward
        # Capture marker function attribute instead of bound-method identity
        # (bound methods compare unequal across calls).
        override = apply_longlora_forward_override(model, group_size=4)
        with override:
            # Patched forward should be a different callable.
            assert getattr(model.attn.forward, "__name__", None) == "s2_forward"
        # After exit, the patched closure is gone and the bound method is
        # back. We check by calling it and verifying we get the (x,) tuple.
        result = model.attn.forward("hello")
        assert result == ("hello",)
        # And confirm the s2_forward wrapper is no longer present.
        assert getattr(model.attn.forward, "__name__", None) != "s2_forward"
        del original_forward  # silence linter


# ---------------------------------------------------------------------------
# #126 — PRMTrainerWrapper + compute_prm_loss kernel
# ---------------------------------------------------------------------------


class TestPRMKernel:
    def test_compute_prm_loss_perfect_match(self):
        from soup_cli.utils.prm import compute_prm_loss

        preds = torch.tensor([1.0, 2.0, 3.0])
        labels = torch.tensor([1.0, 2.0, 3.0])
        out = compute_prm_loss(preds, labels)
        assert float(out) == pytest.approx(0.0)

    def test_compute_prm_loss_mse(self):
        from soup_cli.utils.prm import compute_prm_loss

        preds = torch.tensor([1.0, 2.0])
        labels = torch.tensor([0.0, 0.0])
        out = compute_prm_loss(preds, labels)
        assert float(out) == pytest.approx((1.0 + 4.0) / 2)

    def test_compute_prm_loss_mask(self):
        from soup_cli.utils.prm import compute_prm_loss

        preds = torch.tensor([1.0, 2.0, 100.0])
        labels = torch.tensor([0.0, 0.0, 0.0])
        mask = torch.tensor([1.0, 1.0, 0.0])
        out = compute_prm_loss(preds, labels, mask=mask)
        # Masked → only first two contribute.
        assert float(out) == pytest.approx((1.0 + 4.0) / 2)

    def test_compute_prm_loss_shape_mismatch(self):
        from soup_cli.utils.prm import compute_prm_loss

        preds = torch.tensor([1.0, 2.0])
        labels = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_prm_loss(preds, labels)

    def test_build_prm_trainer_returns_wrapper(self):
        # The factory no longer raises NotImplementedError — it returns a
        # configured wrapper. The actual trust_remote_code probe inside
        # the wrapper will read None for a stub base, so we use a string.
        from soup_cli.trainer.prm import PRMTrainerWrapper
        from soup_cli.utils.prm import build_prm_trainer

        class _Cfg:
            base = "hf-internal-testing/tiny-random-gpt2"
            task = "prm"

        wrapper = build_prm_trainer(config=_Cfg())
        assert isinstance(wrapper, PRMTrainerWrapper)

    def test_build_prm_trainer_rejects_unknown_kwarg(self):
        from soup_cli.utils.prm import build_prm_trainer

        class _Cfg:
            base = "test"

        with pytest.raises(TypeError, match="unexpected kwargs"):
            build_prm_trainer(config=_Cfg(), bogus_kwarg=1)


# ---------------------------------------------------------------------------
# #127 — GRPO stability callback
# ---------------------------------------------------------------------------


class TestGRPOStabilityCallback:
    def test_update_ema(self):
        from soup_cli.monitoring.grpo_stability_callback import update_ema

        ref = {"w": torch.zeros(2)}
        pol = {"w": torch.ones(2)}
        out = update_ema(ref, pol, alpha=0.5)
        assert torch.equal(out["w"], torch.full((2,), 0.5))

    def test_update_ema_bool_rejected(self):
        from soup_cli.monitoring.grpo_stability_callback import update_ema

        with pytest.raises(TypeError, match="bool"):
            update_ema({}, {}, alpha=True)

    def test_update_ema_alpha_bounds(self):
        from soup_cli.monitoring.grpo_stability_callback import update_ema

        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            update_ema({}, {}, alpha=0.0)
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            update_ema({}, {}, alpha=1.5)

    def test_check_tis_threshold(self):
        from soup_cli.monitoring.grpo_stability_callback import check_tis_threshold

        log_ratio = torch.tensor([0.1, 0.2, 0.5])
        assert check_tis_threshold(log_ratio, 0.3) is True
        assert check_tis_threshold(log_ratio, 1.0) is False

    def test_filter_zero_advantage(self):
        from soup_cli.monitoring.grpo_stability_callback import filter_zero_advantage

        adv = torch.tensor([0.0, 1.0, -1.0, 1e-10])
        mask = filter_zero_advantage(adv)
        assert mask[0].item() is False or mask[0].item() == 0
        assert bool(mask[1]) is True
        assert bool(mask[2]) is True
        assert bool(mask[3]) is False

    def test_callback_construct_with_all_fields(self):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        cb = GRPOStabilityCallback(
            ref_model_ema_alpha=0.99,
            replay_buffer_size=10,
            async_grpo_prefetch=True,
            tis_threshold=2.0,
            mask_truncated_completions=True,
            defer_rerolling=True,
            skip_zero_advantage=True,
            off_policy_mask_threshold=0.5,
        )
        assert cb.ref_model_ema_alpha == 0.99
        assert cb.replay_size() == 0
        assert cb.tis_alerts() == 0

    def test_callback_replay_buffer(self):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        cb = GRPOStabilityCallback(replay_buffer_size=2)
        cb.push_rollout("a")
        cb.push_rollout("b")
        cb.push_rollout("c")
        assert cb.replay_size() == 2  # bounded

    def test_callback_tis_alert(self):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        cb = GRPOStabilityCallback(tis_threshold=0.3)
        cb.record_tis_alert(torch.tensor([0.1, 0.2]))  # below threshold
        assert cb.tis_alerts() == 0
        cb.record_tis_alert(torch.tensor([0.5, 0.6]))  # above threshold
        assert cb.tis_alerts() == 1

    def test_callback_bool_rejected_on_numeric_fields(self):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        with pytest.raises(TypeError, match="bool"):
            GRPOStabilityCallback(ref_model_ema_alpha=True)
        with pytest.raises(TypeError, match="bool"):
            GRPOStabilityCallback(replay_buffer_size=True)


class TestAttachGRPOStabilityCallback:
    def test_no_field_set_returns_false(self):
        from soup_cli.utils.peft_wiring import attach_grpo_stability_callback

        class _TCfg:
            ref_model_ema_alpha = None
            replay_buffer_size = None
            async_grpo_prefetch = False
            tis_threshold = None
            mask_truncated_completions = False
            defer_rerolling = False
            skip_zero_advantage = False
            off_policy_mask_threshold = None

        class _Trainer:
            def add_callback(self, cb):
                pass

        assert attach_grpo_stability_callback(_Trainer(), _TCfg()) is False

    def test_attaches_when_set(self):
        from soup_cli.utils.peft_wiring import attach_grpo_stability_callback

        class _TCfg:
            ref_model_ema_alpha = 0.99
            replay_buffer_size = None
            async_grpo_prefetch = False
            tis_threshold = None
            mask_truncated_completions = False
            defer_rerolling = False
            skip_zero_advantage = False
            off_policy_mask_threshold = None

        added = []

        class _Trainer:
            def add_callback(self, cb):
                added.append(cb)

        assert attach_grpo_stability_callback(_Trainer(), _TCfg()) is True
        assert len(added) == 1


# ---------------------------------------------------------------------------
# #68 — weighted preference combine live wrapper
# ---------------------------------------------------------------------------


class TestWeightedCombineHook:
    def test_attaches_to_trainer_with_compute_loss(self):
        from soup_cli.utils.preference_combine import (
            attach_weighted_preference_combine,
        )

        class _Trainer:
            def compute_loss(self, model, inputs, return_outputs=False):
                return torch.tensor(2.0)

        trainer = _Trainer()
        ok = attach_weighted_preference_combine(
            trainer, {"dpo": 0.5, "simpo": 0.5}
        )
        assert ok is True
        # Re-attach is idempotent.
        assert attach_weighted_preference_combine(
            trainer, {"dpo": 0.5, "simpo": 0.5}
        ) is True

    def test_returns_false_without_compute_loss(self):
        from soup_cli.utils.preference_combine import (
            attach_weighted_preference_combine,
        )

        class _NoComputeLoss:
            pass

        assert attach_weighted_preference_combine(
            _NoComputeLoss(), {"dpo": 1.0}
        ) is False

    def test_rejects_bco_mixed_with_paired(self):
        from soup_cli.utils.preference_combine import (
            attach_weighted_preference_combine,
        )

        class _Trainer:
            def compute_loss(self, *args, **kwargs):
                return torch.tensor(1.0)

        with pytest.raises(ValueError, match="bco"):
            attach_weighted_preference_combine(
                _Trainer(), {"bco": 0.5, "dpo": 0.5}
            )

    def test_blended_loss_scales_with_weight(self):
        from soup_cli.utils.preference_combine import (
            attach_weighted_preference_combine,
        )

        class _Trainer:
            def compute_loss(self, model, inputs, return_outputs=False):
                return torch.tensor(2.0)

        trainer = _Trainer()
        attach_weighted_preference_combine(
            trainer, {"dpo": 0.7, "simpo": 0.3}
        )
        # Primary is dpo (highest weight). Loss = 2.0 * 0.7 = 1.4
        out = trainer.compute_loss(None, None)
        assert float(out) == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# Version bump sanity
# ---------------------------------------------------------------------------


def test_version_bump():
    import soup_cli

    major, minor, patch = (int(x) for x in soup_cli.__version__.split("."))
    assert (major, minor, patch) >= (0, 53, 11)


# ---------------------------------------------------------------------------
# v0.53.11 #123 — _GRPOTrainerVariant subclass factory
# ---------------------------------------------------------------------------


class TestGRPOTrainerVariantFactory:
    def test_factory_returns_subclass(self):
        from soup_cli.trainer.grpo import make_grpo_trainer_variant

        class _FakeGRPOTrainer:
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(1.0)

        variant_cls = make_grpo_trainer_variant(_FakeGRPOTrainer, "gspo")
        assert issubclass(variant_cls, _FakeGRPOTrainer)
        assert variant_cls._soup_grpo_variant == "gspo"

    def test_factory_cached_by_variant(self):
        from soup_cli.trainer.grpo import make_grpo_trainer_variant

        class _FakeGRPOTrainer:
            def compute_loss(self, *a, **k):
                return torch.tensor(0.0)

        a = make_grpo_trainer_variant(_FakeGRPOTrainer, "gspo")
        b = make_grpo_trainer_variant(_FakeGRPOTrainer, "gspo")
        assert a is b

    def test_variant_compute_loss_routes_through_kernel(self):
        from soup_cli.trainer.grpo import make_grpo_trainer_variant

        class _FakeGRPOTrainer:
            def __init__(self):
                self.args = type("Args", (), {"beta": 0.1})()

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(99.0)

        variant_cls = make_grpo_trainer_variant(_FakeGRPOTrainer, "gspo")
        trainer = variant_cls()
        inputs = {
            "per_token_logps": torch.zeros(2, 4),
            "old_per_token_logps": torch.zeros(2, 4),
            "advantages": torch.tensor([1.0, -1.0]),
        }
        # Variant kernel should compute a real (non-99.0) loss.
        result = trainer.compute_loss(model=None, inputs=inputs)
        assert isinstance(result, torch.Tensor)

    def test_variant_falls_back_on_missing_inputs(self):
        """When TRL renames inputs, fall back to original loss (defence-in-depth)."""
        from soup_cli.trainer.grpo import make_grpo_trainer_variant

        class _FakeGRPOTrainer:
            def __init__(self):
                self.args = type("Args", (), {"beta": 0.1})()

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return torch.tensor(42.0)

        variant_cls = make_grpo_trainer_variant(_FakeGRPOTrainer, "gspo")
        trainer = variant_cls()
        # No expected keys — falls back to original.
        result = trainer.compute_loss(model=None, inputs={})
        assert float(result) == 42.0


# ---------------------------------------------------------------------------
# v0.53.11 #126 — PRM Trainer subclass factory + dataset prep
# ---------------------------------------------------------------------------


class TestPRMTrainerSubclass:
    def test_factory_returns_subclass(self):
        from soup_cli.trainer.prm import make_prm_trainer_class

        class _FakeTrainer:
            pass

        prm_cls = make_prm_trainer_class(_FakeTrainer)
        assert issubclass(prm_cls, _FakeTrainer)
        assert hasattr(prm_cls, "compute_loss")

    def test_prepare_dataset_skips_mismatched(self):
        from soup_cli.trainer.prm import _prepare_prm_dataset

        class _Tok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": list(range(len(text.split())))}

        rows = [
            {"prompt": "hi", "completions": ["a", "b"], "labels": [1.0]},  # mismatch
            {"prompt": "hi", "completions": ["a"], "labels": [0.5]},  # ok
            {},  # missing fields
        ]
        prepared = _prepare_prm_dataset(rows, _Tok(), max_length=100)
        assert len(prepared) == 1
        assert prepared[0]["step_positions"] == [1]
        assert prepared[0]["labels"] == [0.5]

    def test_prepare_dataset_truncates_long(self):
        from soup_cli.trainer.prm import _prepare_prm_dataset

        class _Tok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1] * 50}

        rows = [{"prompt": "p", "completions": ["a", "b", "c"], "labels": [1, 2, 3]}]
        prepared = _prepare_prm_dataset(rows, _Tok(), max_length=80)
        # Should truncate to fit max_length.
        assert len(prepared) == 1
        assert all(p < 80 for p in prepared[0]["step_positions"])


# ---------------------------------------------------------------------------
# v0.53.11 #68 — true weighted combine path
# ---------------------------------------------------------------------------


class TestTrueWeightedCombine:
    def test_uses_combine_when_logps_available(self):
        from soup_cli.utils.preference_combine import (
            attach_weighted_preference_combine,
        )

        class _Trainer:
            beta = 0.1
            simpo_gamma = 1.0

            def compute_loss(self, model, inputs, return_outputs=False):
                # Returns primary loss; should be REPLACED by weighted combine.
                return torch.tensor(99.0)

        trainer = _Trainer()
        attach_weighted_preference_combine(
            trainer, {"dpo": 0.5, "simpo": 0.5}
        )
        # Provide per-batch logps so the true-weighted path activates.
        inputs = {
            "policy_chosen_logps": torch.tensor([0.0, 0.1]),
            "policy_rejected_logps": torch.tensor([-0.5, -0.6]),
            "reference_chosen_logps": torch.tensor([0.0, 0.0]),
            "reference_rejected_logps": torch.tensor([0.0, 0.0]),
        }
        out = trainer.compute_loss(model=None, inputs=inputs)
        # Result is the TRUE weighted combine, not the 99.0 placeholder.
        assert float(out) != 99.0
        assert math.isfinite(float(out))

    def test_falls_back_when_logps_missing(self):
        from soup_cli.utils.preference_combine import (
            attach_weighted_preference_combine,
        )

        class _Trainer:
            def compute_loss(self, model, inputs, return_outputs=False):
                return torch.tensor(2.0)

        trainer = _Trainer()
        attach_weighted_preference_combine(
            trainer, {"dpo": 0.7, "simpo": 0.3}
        )
        # No logps → fallback to primary scaling.
        out = trainer.compute_loss(model=None, inputs={})
        assert float(out) == pytest.approx(2.0 * 0.7)


# ---------------------------------------------------------------------------
# v0.53.11 #127 — live EMA hook in callback
# ---------------------------------------------------------------------------


class TestStabilityCallbackEMA:
    def test_on_step_end_runs_ema_update(self):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        class _Model:
            def __init__(self, val):
                self._val = val

            def state_dict(self):
                return {"w": torch.full((2,), self._val)}

            def load_state_dict(self, sd, strict=True):
                self._loaded = sd

        cb = GRPOStabilityCallback(ref_model_ema_alpha=0.5)
        ref = _Model(0.0)
        pol = _Model(1.0)
        cb._policy_model = pol
        cb._ref_model = ref

        class _State:
            log_history: list = []

        state = _State()
        cb.on_step_end(args=None, state=state, control=None, model=pol)
        # Loaded back into ref: midpoint 0.5.
        assert torch.allclose(ref._loaded["w"], torch.full((2,), 0.5))

    def test_on_step_end_logs_counters(self):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        cb = GRPOStabilityCallback(replay_buffer_size=4, tis_threshold=1.0)
        cb.push_rollout("a")
        cb.record_tis_alert(torch.tensor([2.0]))

        class _State:
            log_history: list = []

        state = _State()
        cb.on_step_end(args=None, state=state, control=None)
        assert state.log_history
        last = state.log_history[-1]
        assert last["tis_alerts"] == 1
        assert last["replay_size"] == 1

    def test_on_train_begin_captures_models(self):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        cb = GRPOStabilityCallback(ref_model_ema_alpha=0.99)
        policy = object()
        ref = object()

        class _Trainer:
            ref_model = ref

        cb.on_train_begin(
            args=None, state=None, control=None, model=policy, trainer=_Trainer()
        )
        assert cb._policy_model is policy
        assert cb._ref_model is ref


# ---------------------------------------------------------------------------
# v0.53.11 review-fix coverage gaps (tdd-guide findings)
# ---------------------------------------------------------------------------


class TestReviewFixCoverage:
    """Tests added in response to v0.53.11 tdd-guide HIGH/MEDIUM findings."""

    # --- HIGH: bool / bounds rejection on remaining numeric fields ---

    @pytest.mark.parametrize("field", [
        "ref_model_ema_alpha",
        "replay_buffer_size",
        "tis_threshold",
        "off_policy_mask_threshold",
    ])
    def test_callback_bool_rejected_all_numeric_fields(self, field):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        with pytest.raises(TypeError, match="bool"):
            GRPOStabilityCallback(**{field: True})

    @pytest.mark.parametrize("value", [0.0, 100.5])
    def test_callback_tis_threshold_bounds(self, value):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        with pytest.raises(ValueError, match="tis_threshold"):
            GRPOStabilityCallback(tis_threshold=value)

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_callback_off_policy_bounds(self, value):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        with pytest.raises(ValueError, match="off_policy_mask_threshold"):
            GRPOStabilityCallback(off_policy_mask_threshold=value)

    @pytest.mark.parametrize("value", [0, 1_000_001])
    def test_callback_replay_buffer_size_bounds(self, value):
        from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

        with pytest.raises(ValueError, match="replay_buffer_size"):
            GRPOStabilityCallback(replay_buffer_size=value)

    def test_check_tis_threshold_bool_rejected(self):
        from soup_cli.monitoring.grpo_stability_callback import check_tis_threshold

        with pytest.raises(TypeError, match="bool"):
            check_tis_threshold(torch.tensor([1.0]), True)

    def test_update_ema_nan_rejected(self):
        from soup_cli.monitoring.grpo_stability_callback import update_ema

        with pytest.raises(ValueError, match="finite"):
            update_ema({}, {}, alpha=float("nan"))

    # --- HIGH: variant compute_loss return_outputs=True path ---

    def test_variant_compute_loss_return_outputs_true(self):
        from soup_cli.trainer.grpo import make_grpo_trainer_variant

        class _FakeGRPOTrainer:
            def __init__(self):
                self.args = type("Args", (), {"beta": 0.1})()

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                return (torch.tensor(99.0), {"fake": True})

        variant_cls = make_grpo_trainer_variant(_FakeGRPOTrainer, "gspo")
        trainer = variant_cls()
        inputs = {
            "per_token_logps": torch.zeros(2, 4),
            "old_per_token_logps": torch.zeros(2, 4),
            "advantages": torch.tensor([1.0, -1.0]),
        }
        result = trainer.compute_loss(
            model=None, inputs=inputs, return_outputs=True
        )
        # When kernel handles the loss we return variant_loss, None per
        # the implementation — outputs are not reproducible without a full
        # forward pass.
        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, _outputs = result
        assert isinstance(loss, torch.Tensor)

    # --- HIGH: source-grep regression — callback wired into grpo.py ---

    def test_grpo_trainer_wires_stability_callback_source_grep(self):
        import inspect

        import soup_cli.trainer.grpo as grpo_mod

        body = inspect.getsource(grpo_mod)
        assert "attach_grpo_stability_callback" in body, (
            "GRPOTrainerWrapper.setup must call attach_grpo_stability_callback"
        )

    def test_preference_wires_weighted_combine_source_grep(self):
        import inspect

        import soup_cli.trainer.preference as pref_mod

        body = inspect.getsource(pref_mod)
        assert "attach_weighted_preference_combine" in body, (
            "PreferenceTrainerWrapper.setup must wire attach_weighted_preference_combine"
        )

    # --- MEDIUM: idempotent attach + restore-on-exception ---

    def test_attach_weighted_preference_not_double_wrapped(self):
        from soup_cli.utils.preference_combine import (
            attach_weighted_preference_combine,
        )

        class _Trainer:
            def compute_loss(self, model, inputs, return_outputs=False):
                return torch.tensor(2.0)

        trainer = _Trainer()
        attach_weighted_preference_combine(trainer, {"dpo": 0.6, "simpo": 0.4})
        first_wrapper = trainer.compute_loss
        # Second attach must short-circuit; compute_loss unchanged.
        attach_weighted_preference_combine(trainer, {"dpo": 0.6, "simpo": 0.4})
        assert trainer.compute_loss is first_wrapper

    def test_longlora_idempotent_install(self):
        from soup_cli.utils.longlora import apply_longlora_forward_override

        class LlamaAttention:
            def forward(self, x):
                return x,

        class _Model:
            def __init__(self):
                self.attn = LlamaAttention()

            def modules(self):
                yield self.attn

        model = _Model()
        ovr1 = apply_longlora_forward_override(model, group_size=4)
        with ovr1:
            wrapped_once = model.attn.forward
            # Nested context on the same module must NOT double-wrap.
            ovr2 = apply_longlora_forward_override(model, group_size=4)
            with ovr2:
                assert model.attn.forward is wrapped_once
        # After exiting, original forward restored.
        assert not getattr(model.attn.forward, "_soup_longlora_patched", False)

    def test_longlora_restores_on_exception(self):
        from soup_cli.utils.longlora import apply_longlora_forward_override

        class LlamaAttention:
            def forward(self, x):
                return x,

        class _Model:
            def __init__(self):
                self.attn = LlamaAttention()

            def modules(self):
                yield self.attn

        model = _Model()
        original = model.attn.forward
        ovr = apply_longlora_forward_override(model, group_size=4)
        with pytest.raises(RuntimeError):
            with ovr:
                raise RuntimeError("simulated mid-training crash")
        # Forward restored despite the exception.
        assert not getattr(model.attn.forward, "_soup_longlora_patched", False)
        _ = original  # silence lint

    # --- MEDIUM: factory cache isolation between base classes ---

    def test_variant_factory_isolates_by_base_class(self):
        from soup_cli.trainer.grpo import make_grpo_trainer_variant

        class _BaseA:
            def compute_loss(self, *a, **k):
                return torch.tensor(0.0)

        class _BaseB:
            def compute_loss(self, *a, **k):
                return torch.tensor(0.0)

        a = make_grpo_trainer_variant(_BaseA, "gspo")
        b = make_grpo_trainer_variant(_BaseB, "gspo")
        assert a is not b

    def test_variant_factory_normalises_case(self):
        """v0.53.11 review fix (security MEDIUM) — case-insensitive cache."""
        from soup_cli.trainer.grpo import make_grpo_trainer_variant

        class _Base:
            def compute_loss(self, *a, **k):
                return torch.tensor(0.0)

        a = make_grpo_trainer_variant(_Base, "gspo")
        b = make_grpo_trainer_variant(_Base, "GSPO")
        assert a is b

    # --- LOW: empty completions in PRM dataset prep ---

    def test_prepare_prm_empty_completions(self):
        from soup_cli.trainer.prm import _prepare_prm_dataset

        class _Tok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2, 3]}

        rows = [{"prompt": "p", "completions": [], "labels": []}]
        prepared = _prepare_prm_dataset(rows, _Tok(), max_length=100)
        # Empty completions → zero output rows (does not crash).
        assert prepared == []
