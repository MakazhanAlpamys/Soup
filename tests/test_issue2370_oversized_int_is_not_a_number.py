"""``_is_number`` refuses an int too large for a float instead of raising ``OverflowError``.

``math.isfinite`` converts its argument to a float first, so an integer that does not fit
in a double raises there. Since #763 the guard calls ``math.isfinite`` to keep NaN and inf
away from the arithmetic that follows, which means an oversized int made the guard raise
the very error it exists to prevent -- raised out of ``soup adapters audit`` instead of
becoming a ``DIVERGED`` row.

An int beyond the double range is not a number the audit can compare, so it takes the same
path NaN and inf take: ``False`` from the guard, which every caller already turns into a
``DIVERGED`` or ``unknown`` row.
"""

from __future__ import annotations

import math

import pytest

from soup_cli.utils.adapter_audit import _is_number, audit_adapter

# Larger than sys.float_info.max (~1.8e308), so float() raises OverflowError.
HUGE = 10**400


@pytest.mark.unit
@pytest.mark.parametrize("value", [HUGE, -HUGE])
def test_oversized_int_is_not_a_number(value: int) -> None:
    """Both signs are refused rather than raising."""
    assert _is_number(value) is False


@pytest.mark.unit
@pytest.mark.parametrize("value", [10**15, 2e-5, 0, -0.0, 1.5, 10**308])
def test_in_range_numbers_stay_numbers(value: float) -> None:
    """The values the audit does compare keep returning True."""
    assert _is_number(value) is True


@pytest.mark.unit
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_stay_refused(value: float) -> None:
    """#763's behaviour is unchanged: NaN and inf are not numbers here either."""
    assert _is_number(value) is False


@pytest.mark.unit
def test_bool_is_still_excluded() -> None:
    """``bool`` remains a non-number even though it is an ``int`` subclass."""
    assert _is_number(True) is False
    assert _is_number(False) is False


@pytest.mark.unit
def test_audit_reports_diverged_when_the_record_holds_an_oversized_peak_lr() -> None:
    """An oversized recorded ``peak_lr`` becomes a DIVERGED row, not an exception."""
    result = audit_adapter({"training": {"lr": 2e-5}}, {"peak_lr": HUGE})
    row = next(r for r in result.rows if r.setting == "learning_rate")
    assert row.status == "diverged", row
    assert "must both be finite numbers" in row.detail, row.detail


@pytest.mark.unit
def test_audit_reports_diverged_when_the_config_asks_for_an_oversized_lr() -> None:
    """The same with the oversized value on the config side."""
    result = audit_adapter({"training": {"lr": HUGE}}, {"peak_lr": 2e-5})
    row = next(r for r in result.rows if r.setting == "learning_rate")
    assert row.status == "diverged", row
    assert "must both be finite numbers" in row.detail, row.detail


@pytest.mark.unit
def test_warmup_fields_with_an_oversized_total_do_not_raise() -> None:
    """The warmup comparison at the second ``_is_number`` call site also survives."""
    result = audit_adapter(
        {"training": {"warmup_ratio": 0.03}},
        {"warmup_updates": 10, "total_updates": HUGE},
    )
    row = next(r for r in result.rows if r.setting == "warmup_ratio")
    assert row.status == "diverged", row


@pytest.mark.unit
def test_lora_rank_and_scale_with_an_oversized_value_do_not_raise() -> None:
    """The LoRA scale/rank call site reports ``unknown`` rather than raising."""
    result = audit_adapter(
        {"training": {"lora": {"r": 8, "alpha": 16, "use_rslora": True}}},
        {"lora_parameters": {"rank": 8, "scale": HUGE, "use_rslora": True}},
    )
    row = next(r for r in result.rows if r.setting == "lora.alpha")
    assert row.status == "unknown", row
