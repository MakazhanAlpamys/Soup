"""SDK recheck limits must be finite without changing report-error priority."""

from __future__ import annotations

import pytest

from soup_cli.eval.calibrate import JudgeCalibrationReport, ensure_judge_calibrated


def report(acceptable: bool = True, calibrated: bool = True) -> JudgeCalibrationReport:
    return JudgeCalibrationReport(
        position_bias=0.1 if acceptable else 0.5, conformal_threshold=0.3,
        agreement_rate=0.9 if acceptable else 0.5, num_pairs=20, calibrated=calibrated,
    )


@pytest.mark.parametrize("field", ["min_agreement", "max_bias"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")],
                         ids=["nan", "positive-infinity", "negative-infinity"])
@pytest.mark.parametrize("acceptable", [True, False], ids=["accepted-report", "refused-report"])
def test_nonfinite_limit_is_refused_before_comparisons(
    field: str, value: float, acceptable: bool,
) -> None:
    with pytest.raises(ValueError) as caught:
        ensure_judge_calibrated(report(acceptable), **{field: value})
    assert str(caught.value) == f"{field} must be finite"


@pytest.mark.parametrize("field", ["min_agreement", "max_bias"])
@pytest.mark.parametrize("acceptable", [True, False])
def test_integer_limit_beyond_float_range_is_named(field: str, acceptable: bool) -> None:
    with pytest.raises(ValueError) as caught:
        ensure_judge_calibrated(report(acceptable), **{field: 10**400})
    assert str(caught.value) == f"{field} must be finite"


@pytest.mark.parametrize("field", ["min_agreement", "max_bias"])
@pytest.mark.parametrize("invalid,error,message", [
    (None, RuntimeError, "not calibrated"),
    ("not-a-report", TypeError, "report must be JudgeCalibrationReport"),
    (report(calibrated=False), RuntimeError, "calibration failed"),
])
def test_report_errors_keep_priority(
    field: str, invalid: JudgeCalibrationReport | str | None,
    error: type[Exception], message: str,
) -> None:
    with pytest.raises(error, match=message):
        ensure_judge_calibrated(invalid, **{field: float("nan")})


@pytest.mark.parametrize("agreement,bias,floor,ceiling", [
    (0.7, 0.3, 0.7, 0.3),
    (0.5, 0.5, 0.0, 1.0),
    (0.0, -1.0, -1.0, 2.0),
    (0.5, 0.5, -(10**300), 10**300),
])
def test_finite_custom_limits_and_equal_boundaries_are_preserved(
    agreement: float, bias: float, floor: float, ceiling: float,
) -> None:
    candidate = JudgeCalibrationReport(bias, 0.3, agreement, 20, True)
    ensure_judge_calibrated(candidate, min_agreement=floor, max_bias=ceiling)


@pytest.mark.parametrize("field,value,message", [
    ("min_agreement", 2.0, "agreement"),
    ("max_bias", -1.0, "bias"),
])
def test_finite_limits_still_apply_existing_comparison(
    field: str, value: float, message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        ensure_judge_calibrated(report(), **{field: value})
