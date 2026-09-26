import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "layercount.py"
SPEC = importlib.util.spec_from_file_location("layercount", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_exact_multi_point_sweep_passes():
    assert module.layercount_verdict([
        {"layers": 32, "exact": 128, "total": 128},
        {"layers": 48, "exact": 192, "total": 192},
        {"layers": 64, "exact": 256, "total": 256},
    ])


def test_wrong_relationship_fails():
    assert not module.layercount_verdict([
        {"layers": 32, "exact": 128, "total": 128},
        {"layers": 64, "exact": 8, "total": 128},
    ])


def test_degenerate_and_nonfinite_sweeps_fail():
    assert not module.layercount_verdict([
        {"layers": 32, "exact": 128, "total": 128},
    ])
    with pytest.raises(module.MeasurementInvalidError):
        module.layercount_verdict([
            {"layers": 32, "exact": 128, "total": 128, "layer_mib": float("nan")},
            {"layers": 64, "exact": 256, "total": 256, "layer_mib": 1.0},
        ])
