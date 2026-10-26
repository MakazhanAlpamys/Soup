"""The census must refuse unhealthy or occupied lab hardware before CUDA loads."""

import importlib.util
from pathlib import Path

import pytest


def _check():
    path = Path(__file__).parents[1] / "benchmarks/harness/stream_projection_census.py"
    spec = importlib.util.spec_from_file_location("projection_census", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.check_inventory


def test_admits_healthy_idle_gpu_with_headroom():
    row = "1, GPU-test, 24576, 9604, 0, 35, 350\n"
    assert _check()(row, "", 0, "GPU-test", 8 * 2**30)["power_limit"] == 350


def test_driver_lost_error_is_not_hidden_by_successful_exit_code():
    row = "1, GPU-test, 24576, 9604, 0, 35, 350\n"
    with pytest.raises(RuntimeError, match="unhealthy"):
        _check()(row, "GPU0: GPU is lost", 0, "GPU-test", 8 * 2**30)


@pytest.mark.parametrize("used,util,temp", [(17000, 0, 35), (0, 80, 35), (0, 0, 80)])
def test_headroom_refusals(used, util, temp):
    row = f"1, GPU-test, 24576, {used}, {util}, {temp}, 350\n"
    with pytest.raises(RuntimeError, match="headroom"):
        _check()(row, "", 0, "GPU-test", 8 * 2**30)


def test_missing_gpu_is_refused():
    with pytest.raises(RuntimeError, match="missing"):
        _check()("", "", 0, "GPU-test", 8 * 2**30)
