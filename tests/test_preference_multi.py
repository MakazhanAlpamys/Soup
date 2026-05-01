"""Tests for v0.40.0 Part D — multi-objective preference loss.

Adds ``training.preference_loss_weights: dict[str, float]`` for blending
preference losses (e.g. ``{"dpo": 0.7, "bco": 0.3}``). The schema-level
surface ships in v0.40.0; the live runtime weighted combination is
deferred to v0.40.1 (mirrors the project's stub-then-live pattern from
v0.27.0 MII / v0.37.0 multipack / v0.38.0 quant menu / v0.39.0 ReLoRA).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import SoupConfig

# ─── Schema bounds ──────────────────────────────────────────────────────────


def _base(**training):
    return SoupConfig(
        base="some-model",
        task="preference",
        data={"train": "./data.jsonl", "format": "dpo"},
        training=training,
    )


class TestPreferenceLossWeightsConfig:
    def test_two_loss_blend_accepted(self):
        cfg = _base(preference_loss_weights={"dpo": 0.7, "bco": 0.3})
        assert cfg.training.preference_loss_weights == {"dpo": 0.7, "bco": 0.3}

    def test_three_loss_blend_accepted(self):
        cfg = _base(
            preference_loss_weights={"dpo": 0.5, "bco": 0.3, "simpo": 0.2},
        )
        assert sum(cfg.training.preference_loss_weights.values()) == pytest.approx(1.0)

    def test_single_entry_rejected_use_scalar_form_instead(self):
        """Single-entry blends are equivalent to scalar preference_loss; reject."""
        with pytest.raises(ValidationError, match="2 and 5"):
            _base(preference_loss_weights={"dpo": 1.0})

    def test_more_than_five_entries_rejected(self):
        with pytest.raises(ValidationError, match="2 and 5"):
            _base(
                preference_loss_weights={
                    "dpo": 0.2, "bco": 0.2, "simpo": 0.2,
                    "orpo": 0.2, "ipo": 0.1, "extra": 0.1,
                },
            )

    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError, match="unknown"):
            _base(preference_loss_weights={"dpo": 0.5, "garbage": 0.5})

    def test_null_byte_in_key_rejected(self):
        with pytest.raises(ValidationError, match="null byte"):
            _base(preference_loss_weights={"dpo": 0.5, "bco\x00": 0.5})

    def test_empty_dict_rejected(self):
        with pytest.raises(ValidationError, match="2 and 5"):
            _base(preference_loss_weights={})

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValidationError, match="sum"):
            _base(preference_loss_weights={"dpo": 0.5, "bco": 0.4})

    def test_weight_zero_rejected(self):
        with pytest.raises(ValidationError, match=r"\(0, 1\]"):
            _base(preference_loss_weights={"dpo": 1.0, "bco": 0.0})

    def test_weight_negative_rejected(self):
        with pytest.raises(ValidationError, match=r"\(0, 1\]"):
            _base(preference_loss_weights={"dpo": 1.5, "bco": -0.5})

    def test_weight_above_one_rejected(self):
        # Two values both > 1 — the per-value bound (0,1] fires before the
        # sum gate.
        with pytest.raises(ValidationError, match=r"\(0, 1\]"):
            _base(preference_loss_weights={"dpo": 1.1, "bco": 1.1})

    def test_weight_bool_coerced_to_zero_then_rejected(self):
        """Pydantic coerces True/False → 1.0/0.0 before model_validator sees
        them, so a False weight is rejected by the (0, 1] gate (not the bool
        guard). Either rejection path is acceptable; this test pins the
        observed behaviour rather than the rejection mechanism."""
        with pytest.raises(ValidationError, match=r"\(0, 1\]|must be a number"):
            _base(preference_loss_weights={"dpo": True, "bco": False})

    def test_requires_preference_task(self):
        with pytest.raises(ValidationError, match="preference"):
            SoupConfig(
                base="some-model",
                task="dpo",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"preference_loss_weights": {"dpo": 1.0}},
            )

    def test_mutually_exclusive_with_scalar_loss(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            _base(
                preference_loss="dpo",
                preference_loss_weights={"dpo": 0.7, "bco": 0.3},
            )

    def test_rejected_on_mlx(self):
        with pytest.raises(ValidationError, match="mlx"):
            SoupConfig(
                base="some-model",
                task="preference",
                backend="mlx",
                data={"train": "./data.jsonl", "format": "dpo"},
                training={"preference_loss_weights": {"dpo": 0.7, "bco": 0.3}},
            )


# ─── Helper API ─────────────────────────────────────────────────────────────


class TestMultiObjectiveHelpers:
    def test_is_multi_objective_true(self):
        from soup_cli.trainer.preference import is_multi_objective_preference

        cfg = _base(preference_loss_weights={"dpo": 0.7, "bco": 0.3})
        assert is_multi_objective_preference(cfg) is True

    def test_is_multi_objective_false_scalar(self):
        from soup_cli.trainer.preference import is_multi_objective_preference

        cfg = _base(preference_loss="dpo")
        assert is_multi_objective_preference(cfg) is False

    def test_is_multi_objective_false_legacy(self):
        from soup_cli.trainer.preference import is_multi_objective_preference

        cfg = SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl", "format": "dpo"},
        )
        assert is_multi_objective_preference(cfg) is False

    def test_get_loss_weights_returns_copy(self):
        """Defensive copy so caller mutation cannot affect cfg."""
        from soup_cli.trainer.preference import get_loss_weights

        cfg = _base(preference_loss_weights={"dpo": 0.7, "bco": 0.3})
        weights = get_loss_weights(cfg)
        assert weights == {"dpo": 0.7, "bco": 0.3}
        weights["dpo"] = 999.0
        # Re-fetch — original cfg unchanged.
        assert get_loss_weights(cfg) == {"dpo": 0.7, "bco": 0.3}

    def test_get_loss_weights_none_when_not_set(self):
        from soup_cli.trainer.preference import get_loss_weights

        cfg = _base(preference_loss="dpo")
        assert get_loss_weights(cfg) is None


# ─── Live wiring stub-then-live ─────────────────────────────────────────────


class TestMultiObjectiveDeferred:
    def test_setup_raises_with_actionable_message(self):
        """v0.40.0 ships schema only; live runtime wiring deferred to v0.40.1.

        ``setup`` must raise a ``NotImplementedError`` that names the
        deferred-version follow-up so users know whether to wait or
        switch to the scalar form.
        """
        from soup_cli.trainer.preference import PreferenceTrainerWrapper

        cfg = _base(preference_loss_weights={"dpo": 0.7, "bco": 0.3})
        wrapper = PreferenceTrainerWrapper(cfg, device="cpu")
        with pytest.raises(NotImplementedError, match="v0.40.1"):
            wrapper.setup({"train": [{"prompt": "p", "chosen": "c", "rejected": "r"}]})
