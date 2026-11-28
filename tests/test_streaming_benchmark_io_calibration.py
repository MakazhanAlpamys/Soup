"""Fail-closed calibration controls; these fixtures are not hardware evidence."""

import copy

import pytest

from benchmarks.harness.streaming_controls.io_calibration import (
    FILE_BYTES,
    TOLERANCE_BYTES,
    assess,
    difference,
    direct_read,
)


def samples():
    return [
        dict(
            application_bytes=FILE_BYTES,
            device_bytes=FILE_BYTES,
            process_bytes=FILE_BYTES,
            content_sha256="a" * 64,
            expected_sha256="a" * 64,
        )
        for _ in range(2)
    ]


def test_calibration_does_not_approve_performance():
    result = assess(samples())
    assert result["status"] == "PASS_COUNTER_CALIBRATION"
    assert result["performance_verdict"] == "NO_VERDICT"


@pytest.mark.parametrize("value", [None, -512, 0, True, FILE_BYTES * 2, float("nan")])
@pytest.mark.parametrize("field", ["device_bytes", "process_bytes"])
def test_unknown_reset_cached_contaminated_counters_block(value, field):
    rows = samples()
    rows[1][field] = value
    assert assess(rows)["status"] == "BLOCKED"


@pytest.mark.parametrize(
    "value", [FILE_BYTES - TOLERANCE_BYTES - 1, FILE_BYTES + TOLERANCE_BYTES + 1]
)
def test_fixed_tolerance_cannot_accept_bad_device_mapping(value):
    rows = samples()
    rows[0]["device_bytes"] = value
    assert assess(rows)["status"] == "BLOCKED"


def test_incomplete_samples_or_wrong_content_block():
    rows = samples()
    assert assess(rows[:1])["status"] == "BLOCKED"
    bad = copy.deepcopy(rows)
    bad[0]["content_sha256"] = "b" * 64
    assert assess(bad)["status"] == "BLOCKED"
    rows[0].pop("content_sha256")
    rows[0].pop("expected_sha256")
    assert assess(rows)["status"] == "BLOCKED"


def test_non_linux_read_is_refused_without_opening_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "benchmarks.harness.streaming_controls.io_calibration.sys.platform", "darwin"
    )
    with pytest.raises(RuntimeError, match="^Linux O_DIRECT is required$"):
        direct_read(tmp_path / "absent", 0)


def test_physical_delta_never_hides_reset_or_device_change():
    before = dict(device="nvme0n1", device_bytes=1024, process_bytes=512)
    after = dict(device="nvme0n1", device_bytes=2048, process_bytes=1024)
    assert difference(before, after) == dict(device_bytes=1024, process_bytes=512)
    with pytest.raises(RuntimeError, match="^physical device changed$"):
        difference(before, dict(after, device="nvme1n1"))
    with pytest.raises(RuntimeError, match="^counter reset$"):
        difference(before, dict(after, device_bytes=0))
