import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "determinism.py"
SPEC = importlib.util.spec_from_file_location("determinism", MODULE_PATH)
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


def test_expected_pattern_passes():
    assert module.determinism_verdict(record())


def test_stable_streamed_control_fails():
    item = record()
    item["streamed_curve_equal"] = True
    assert not module.determinism_verdict(item)


def test_missing_comparison_fails():
    item = record()
    del item["resident_gradient_equal"]
    with pytest.raises(module.MeasurementInvalidError):
        module.determinism_verdict(item)


def test_nonfinite_losses_fail():
    item = record()
    item["streamed_losses"] = [float("nan")]
    with pytest.raises(module.MeasurementInvalidError):
        module.determinism_verdict(item)
