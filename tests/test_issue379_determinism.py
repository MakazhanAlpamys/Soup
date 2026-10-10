"""CPU-testable verdict guards for the determinism reconstruction."""

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "determinism.py"
SPEC = importlib.util.spec_from_file_location("determinism_issue379", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def record():
    return {
        "streamed_forward_equal": True,
        "resident_forward_equal": True,
        "streamed_gradient_equal": False,
        "resident_gradient_equal": True,
        "streamed_curve_equal": False,
        "resident_curve_equal": True,
        "streamed_losses": [1.0, 0.9],
        "resident_losses": [1.0, 0.9],
    }


def test_expected_measurement_pattern_passes():
    assert module.determinism_verdict(record())


@pytest.mark.parametrize("field", [
    "streamed_forward_equal",
    "resident_forward_equal",
    "streamed_gradient_equal",
    "resident_gradient_equal",
    "streamed_curve_equal",
    "resident_curve_equal",
])
def test_each_verdict_clause_is_load_bearing(field):
    item = record()
    item[field] = not item[field]
    assert not module.determinism_verdict(item)


def test_non_boolean_flags_are_not_truthy_measurements():
    item = record()
    item["streamed_curve_equal"] = "false"
    with pytest.raises(module.MeasurementInvalidError, match="booleans"):
        module.determinism_verdict(item)


def test_empty_and_nonfinite_measurements_fail():
    item = record()
    item["streamed_losses"] = []
    with pytest.raises(module.MeasurementInvalidError):
        module.determinism_verdict(item)
    item = record()
    item["resident_losses"] = [float("nan")]
    with pytest.raises(module.MeasurementInvalidError):
        module.determinism_verdict(item)


def test_curve_diff_rejects_degenerate_or_nonfinite_data():
    with pytest.raises(module.MeasurementInvalidError):
        module.curve_diff([], [])
    with pytest.raises(module.MeasurementInvalidError):
        module.curve_diff([1.0], [float("inf")])
