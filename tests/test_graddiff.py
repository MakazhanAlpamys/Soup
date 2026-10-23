import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "harness" / "graddiff.py"
HARNESS_DIR = MODULE_PATH.parent
sys.path.insert(0, str(HARNESS_DIR))

SPEC = importlib.util.spec_from_file_location("graddiff_harness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None

module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

historical_pattern_reproduced = module.historical_pattern_reproduced
historical_pattern_exit_code = module.historical_pattern_exit_code


@pytest.mark.parametrize(
    ("exact", "total", "streamed_equal", "resident_equal", "expected"),
    [
        (3, 3, False, True, True),
        (2, 3, False, True, False),
        (0, 0, False, True, False),
        (3, 3, True, True, False),
        (3, 3, False, False, False),
    ],
)
def test_historical_pattern_reproduced_truth_table(
    exact: int,
    total: int,
    streamed_equal: bool,
    resident_equal: bool,
    expected: bool,
) -> None:
    assert (
        historical_pattern_reproduced(
            exact,
            total,
            streamed_equal,
            resident_equal,
        )
        is expected
    )


def test_historical_pattern_reproduced_returns_real_bool() -> None:
    result = historical_pattern_reproduced(3, 3, False, True)
    assert type(result) is bool
    assert result is True


@pytest.mark.parametrize(
    ("historical_pattern", "expected"),
    [
        (True, 0),
        (False, 1),
    ],
)
def test_historical_pattern_exit_code(
    historical_pattern: bool,
    expected: int,
) -> None:
    assert historical_pattern_exit_code(historical_pattern) == expected


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_curve_diff_rejects_non_finite_values(bad_value: float) -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        module.curve_diff([bad_value, 1.0], [1.0, 1.0])
