import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "pincost.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("pincost_harness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_pinning_verdict_requires_a_real_advantage():
    reproduced, pinned, pageable, ratio, spread = module.pinning_verdict(
        [101.0, 100.0, 99.0],
        [90.0, 89.0, 91.0],
    )

    assert reproduced is False
    assert pinned == pytest.approx(100.0)
    assert pageable == pytest.approx(90.0)
    assert ratio == pytest.approx(100 / 90)
    assert spread > 0


def test_pinning_verdict_accepts_historical_scale_relationship():
    reproduced, *_ = module.pinning_verdict(
        [425.0, 426.0, 424.0],
        [64.0, 65.0, 65.0],
    )

    assert reproduced is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_pinning_verdict_rejects_non_finite_rates(bad):
    with pytest.raises(module.MeasurementInvalidError, match="finite"):
        module.pinning_verdict([bad], [1.0])
