"""Calibration SDK input validation names non-finite and unrepresentable numbers."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from soup_cli.eval.calibrate import (
    JudgeCalibrationReport,
    classify_kl_delta,
    conformal_threshold,
    kl_divergence,
    load_judge_calibration,
    run_calibration,
)


def _report_values():
    return dict(position_bias=0.1, conformal_threshold=0.5, agreement_rate=0.9,
                num_pairs=10, calibrated=True)


def _report_with(field, value):
    values = _report_values()
    values[field] = value
    return JudgeCalibrationReport(**values)


@pytest.mark.parametrize("call,reason", [
    pytest.param(lambda v: kl_divergence([v, 0], [0, 0]),
                 "p_logits must be finite", id="kl-p"),
    pytest.param(lambda v: kl_divergence([0, 0], [v, 0]),
                 "q_logits must be finite", id="kl-q"),
    pytest.param(lambda v: run_calibration([[v, 0]], [[0, 0]]),
                 "p_logits must be finite", id="calibration-base"),
    pytest.param(lambda v: run_calibration([[0, 0]], [[v, 0]]),
                 "q_logits must be finite", id="calibration-quantized"),
    pytest.param(classify_kl_delta,
                 "mean_kl must be a finite non-negative number", id="mean-kl"),
    pytest.param(lambda v: conformal_threshold([0.5], alpha=v),
                 "alpha must be finite", id="alpha"),
    pytest.param(lambda v: conformal_threshold([v], alpha=0.5),
                 "scores must be finite", id="scores"),
    pytest.param(lambda v: _report_with("position_bias", v),
                 "position_bias must be finite", id="report-position-bias"),
    pytest.param(lambda v: _report_with("conformal_threshold", v),
                 "conformal_threshold must be finite", id="report-threshold"),
    pytest.param(lambda v: _report_with("agreement_rate", v),
                 "agreement_rate must be finite", id="report-agreement"),
])
@pytest.mark.parametrize("value", [10**400, -(10**400), float("nan"),
                                 float("inf"), -float("inf")],
                         ids=["huge-positive-int", "huge-negative-int", "nan", "inf", "-inf"])
def test_invalid_numeric_input_has_its_existing_field_reason(call, reason, value):
    with pytest.raises(ValueError) as error:
        call(value)
    assert str(error.value) == reason


@pytest.mark.parametrize("field", ["position_bias", "conformal_threshold", "agreement_rate"])
@pytest.mark.parametrize("value", [10**400, -(10**400)], ids=["positive", "negative"])
def test_persisted_report_revalidates_unrepresentable_float_fields(
    tmp_path, monkeypatch, field, value,
):
    monkeypatch.chdir(tmp_path)
    values = _report_values()
    values[field] = value
    Path("calibration.json").write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_judge_calibration("calibration.json")
    assert str(error.value) == f"{field} must be finite"


@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_oversized_fraction_has_the_kl_input_reason(sign):
    with pytest.raises(ValueError) as error:
        kl_divergence([Fraction(sign * 10**400), 0], [0, 0])
    assert str(error.value) == "p_logits must be finite"


def test_representable_numeric_logits_and_calibration_keep_their_results():
    for large in (1, 10**300, -(10**300)):
        assert kl_divergence([large, 0], [large, 0]) == 0.0
    assert kl_divergence([Fraction(1, 2), 0], [0.5, 0]) == 0.0
    result = run_calibration([[0, 1]], [[0, 1]])
    assert result.mean_kl == 0.0
    assert result.per_prompt_kl == (0.0,)
    assert result.delta_status == "OK"


@pytest.mark.parametrize("value,status", [(0, "OK"), (0.05, "MINOR"),
                                          (0.2, "MAJOR"), (10**300, "MAJOR")])
def test_representable_kl_values_preserve_classification(value, status):
    assert classify_kl_delta(value) == status


@pytest.mark.parametrize("alpha,expected", [(0, 0.0), (0.5, 0.5), (1, 1.0)])
def test_integer_scores_and_alpha_preserve_conformal_threshold(alpha, expected):
    assert conformal_threshold([0, 0.5, 1], alpha=alpha) == expected


def test_report_keeps_valid_integer_fields():
    values = dict(position_bias=-1, conformal_threshold=0, agreement_rate=1,
                  num_pairs=10, calibrated=True)
    assert JudgeCalibrationReport(**values).to_dict() == values


@pytest.mark.parametrize("call,reason", [
    (lambda: conformal_threshold([10**300], alpha=0.5), "scores must be in range [0.0, 1.0]"),
    (lambda: conformal_threshold([0.5], alpha=10**300), "alpha must be in [0.0, 1.0]"),
    (lambda: _report_with("position_bias", 10**300), "position_bias must be in [-1.0, 1.0]"),
    (lambda: _report_with("conformal_threshold", 10**300),
     "conformal_threshold must be in [0.0, 1.0]"),
    (lambda: _report_with("agreement_rate", 10**300), "agreement_rate must be in [0.0, 1.0]"),
])
def test_representable_out_of_range_values_keep_range_errors(call, reason):
    with pytest.raises(ValueError) as error:
        call()
    assert str(error.value) == reason


def test_logit_length_error_keeps_priority_over_element_validation():
    with pytest.raises(ValueError) as error:
        kl_divergence([10**400, 0], [0])
    assert str(error.value) == "p_logits (2) and q_logits (1) must have the same length"


def test_alpha_error_keeps_priority_over_empty_scores():
    with pytest.raises(ValueError) as error:
        conformal_threshold([], alpha=10**400)
    assert str(error.value) == "alpha must be finite"
