import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "repeat_backward.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("repeat_backward", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_first_exact_then_later_corruption_is_required():
    assert module.repeated_backward_verdict([256, 8, 8, 8], 256)


def test_healthy_repeated_backward_fails_verdict():
    assert not module.repeated_backward_verdict([256, 256, 256], 256)


def test_empty_or_single_repeat_fails_verdict():
    assert not module.repeated_backward_verdict([], 256)
    assert not module.repeated_backward_verdict([256], 256)


def test_invalid_counts_are_rejected():
    with pytest.raises(module.MeasurementInvalidError):
        module.repeated_backward_verdict([256, 257], 256)
