"""The census must refuse unhealthy or occupied lab hardware before CUDA loads."""

import importlib.util
from pathlib import Path

import pytest


def _helper():
    path = Path(__file__).parents[1] / "benchmarks/harness/stream_projection_census.py"
    spec = importlib.util.spec_from_file_location("projection_census", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _check():
    return _helper().check_inventory


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


def _native(cap):
    return dict(
        measured=True, cap_bound="override", cap_bytes=cap, k=10,
        budget_inputs={"override_bytes": cap},
    )


def test_native_override_and_message_bind_the_census_to_eight_gib():
    cap = 8 * 2**30
    receipt = _helper().check_cap_binding(
        _native(cap), ["cap: training.stream_vram_override 8.59 GB"], cap,
    )
    assert receipt == {
        "verified": True, "field": "training.stream_vram_override", "bytes": 8589934592,
        "native_message_names_override": True,
    }


@pytest.mark.parametrize("change", [
    {"cap_bound": "card"}, {"cap_bound": "free"}, {"cap_bytes": 24 * 2**30},
    {"budget_inputs": {}}, {"budget_inputs": {"override_bytes": 8_000_000_000}},
    {"cap_bytes": float(8 * 2**30)},
])
def test_census_refuses_a_different_or_unproven_native_cap(change):
    cap = 8 * 2**30
    native = _native(cap) | change
    with pytest.raises(RuntimeError, match="override"):
        _helper().check_cap_binding(native, ["training.stream_vram_override"], cap)


def test_positive_retention_requires_the_original_native_cap_message():
    with pytest.raises(RuntimeError, match="message"):
        _helper().check_cap_binding(_native(8 * 2**30), ["card cap"], 8 * 2**30)


def test_unavailable_probe_stays_an_explicit_unverified_outcome():
    receipt = _helper().check_cap_binding({"measured": False, "k": 0}, [], 8 * 2**30)
    assert receipt["verified"] is False
    assert receipt["reason"] == "native auto probe unavailable"


def test_zero_retention_keeps_native_no_fit_message_without_inventing_cap_text():
    native = _native(8 * 2**30) | {"k": 0}
    receipt = _helper().check_cap_binding(native, ["one layer does not fit"], 8 * 2**30)
    assert receipt["verified"] is True
    assert receipt["native_message_names_override"] is False
