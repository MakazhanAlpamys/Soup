import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "depth_vs_bytes.py"
SPEC = importlib.util.spec_from_file_location("depth_vs_bytes", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_recorded_threshold_pattern_passes():
    assert module.depth_vs_bytes_verdict([
        {"quant": "nf4", "layer_mib": 150.0, "exact": 192, "total": 192},
        {"quant": "nf4", "layer_mib": 180.0, "exact": 8, "total": 192},
        {"quant": "bf16", "layer_mib": 480.0, "exact": 128, "total": 128},
    ])


def test_wrong_relationship_and_degenerate_data_fail():
    assert not module.depth_vs_bytes_verdict([
        {"quant": "nf4", "layer_mib": 150.0, "exact": 192, "total": 192},
        {"quant": "nf4", "layer_mib": 180.0, "exact": 192, "total": 192},
        {"quant": "bf16", "layer_mib": 480.0, "exact": 128, "total": 128},
    ])
    assert not module.depth_vs_bytes_verdict([])


def test_nonfinite_and_malformed_data_fail():
    with pytest.raises(module.MeasurementInvalidError):
        module.depth_vs_bytes_verdict([
            {"quant": "nf4", "layer_mib": float("inf"), "exact": 1, "total": 1},
            {"quant": "nf4", "layer_mib": 180.0, "exact": 0, "total": 1},
            {"quant": "bf16", "layer_mib": 480.0, "exact": 1, "total": 1},
        ])
