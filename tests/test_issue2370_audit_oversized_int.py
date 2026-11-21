"""#2370: an int past the double range is not a number, not an OverflowError.

``_is_number`` in ``src/soup_cli/utils/adapter_audit.py`` is the guard in front
of the audit arithmetic.  It excludes ``bool`` and, since #763, refuses NaN and
infinity with ``math.isfinite``.  ``math.isfinite`` converts its argument to a
float first, so an integer that does not fit in a double raised
``OverflowError`` there -- the very error the guard was written to prevent,
for a different input.

The guard now bounds the integer before the conversion, so an oversized one is
reported as "not a number" and the command answers with a ``DIVERGED`` row
instead of a traceback.
"""

from __future__ import annotations

import math

import pytest

from soup_cli.utils.adapter_audit import DIVERGED, _is_number, audit_adapter

pytestmark = pytest.mark.unit

HUGE = 10**400


def _record(**overrides):
    """A record in the shape mlx_sft.py writes, with the named fields changed."""
    record = {
        "fine_tune_type": "lora",
        "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "peak_lr": 1e-4,
        "warmup_updates": 3,
        "total_updates": 12,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "grad_checkpoint": False,
        "grad_accumulation_steps": 4,
        "mask_prompt": False,
        "response_token_mask": True,
        "train_on_responses_only": True,
        "lora_parameters": {"rank": 8, "scale": 2.0, "dropout": 0.0},
    }
    record.update(overrides)
    return record


def _row(result, setting: str):
    return next(row for row in result.rows if row.setting == setting)


@pytest.mark.parametrize("value", [HUGE, -HUGE])
def test_an_int_past_the_double_range_is_not_a_number(value):
    assert _is_number(value) is False


@pytest.mark.parametrize("value", [10**15, 2e-5, 0, 1.5, -10**15])
def test_the_numbers_the_arithmetic_can_use_still_count(value):
    assert _is_number(value) is True


@pytest.mark.parametrize("value", [True, False, None, "2e-5", float("nan"), float("inf")])
def test_the_other_rejections_are_unchanged(value):
    assert _is_number(value) is False


def test_the_guard_does_not_raise_on_an_oversized_int():
    # The error the guard existed to prevent, for the input that reached it.
    with pytest.raises(OverflowError):
        math.isfinite(HUGE)


def test_an_oversized_recorded_peak_lr_is_a_diverged_row():
    result = audit_adapter({"training": {"lr": 2e-5}}, _record(peak_lr=HUGE))

    row = _row(result, "learning_rate")
    assert row.status == DIVERGED
    assert "must both be finite numbers" in row.detail


def test_an_oversized_configured_lr_is_a_diverged_row():
    result = audit_adapter({"training": {"lr": HUGE}}, _record(peak_lr=2e-5))

    row = _row(result, "learning_rate")
    assert row.status == DIVERGED
    assert "must both be finite numbers" in row.detail


def test_an_oversized_warmup_total_is_a_diverged_row():
    result = audit_adapter({"training": {"warmup_ratio": 0.03}}, _record(total_updates=HUGE))

    row = _row(result, "warmup_ratio")
    assert row.status == DIVERGED
    assert "both must be numbers" in row.detail


def test_an_oversized_warmup_update_count_is_a_diverged_row():
    result = audit_adapter({"training": {"warmup_ratio": 0.03}}, _record(warmup_updates=HUGE))

    row = _row(result, "warmup_ratio")
    assert row.status == DIVERGED
    assert "both must be numbers" in row.detail


def test_an_oversized_recorded_rank_leaves_the_alpha_unreadable():
    # The rank is read next to the scale in `_ran_alpha`, which is the one
    # place a LoRA shape goes through the number guard.
    result = audit_adapter(
        {"training": {"lora": {"alpha": 16}}},
        _record(lora_parameters={"rank": HUGE, "scale": 2.0}),
    )

    row = _row(result, "lora.alpha")
    assert row.status == "unknown"


def test_the_recorded_rank_row_compares_the_values_it_has():
    # `lora.r` is compared, not guarded, so an oversized rank reads as a
    # divergence rather than as an error.
    result = audit_adapter(
        {"training": {"lora": {"r": 8}}},
        _record(lora_parameters={"rank": HUGE, "scale": 2.0}),
    )

    row = _row(result, "lora.r")
    assert row.status == DIVERGED


def test_an_oversized_recorded_scale_leaves_the_alpha_unreadable():
    # The scale is the only source of the alpha in this record shape, so an
    # unusable one leaves the row unknown -- never agreement, never a raise.
    result = audit_adapter(
        {"training": {"lora": {"alpha": 16}}}, _record(lora_parameters={"rank": 8, "scale": HUGE})
    )

    row = _row(result, "lora.alpha")
    assert row.status == "unknown"


def test_a_usable_record_still_agrees():
    result = audit_adapter(
        {
            "training": {"lr": 1e-4, "warmup_ratio": 0.25, "lora": {"r": 8, "alpha": 16}},
        },
        _record(),
    )

    assert _row(result, "learning_rate").status == "ok"
    assert _row(result, "lora.r").status == "ok"
    assert result.diverged_count == 0
