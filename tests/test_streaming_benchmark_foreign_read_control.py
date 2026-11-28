"""Foreign-read accounting and strict acceptance; synthetic evidence only."""

import pytest

from benchmarks.harness.arm_validity import DiskReadings, foreign_reads_check_result
from benchmarks.harness.streaming_controls.foreign_read_control import assess
from benchmarks.harness.streaming_controls.io_calibration import FILE_BYTES


def test_foreign_parent_reads_are_rejected_by_arm_validity():
    def reading(device, reader):
        return DiskReadings(
            device,
            0,
            {
                "own": 0,
                "uninspectable": 0,
                "others": {99: {"name": "owned-reader", "read_bytes": reader}},
            },
            [],
        )

    count = FILE_BYTES * 16
    validity = foreign_reads_check_result(
        "/data/control", "nvme0n1", None, reading(100, 0), reading(100 + count, count)
    )
    result = assess(validity, count, True)
    assert result["status"] == "PASS_FOREIGN_READER_DETECTED"
    assert result["performance_verdict"] == "NO_VERDICT"


@pytest.mark.parametrize("outcome", ["ok", "unknown", None])
def test_unproven_detector_blocks(outcome):
    assert assess({"outcome": outcome}, FILE_BYTES * 16, True)["status"] == "BLOCKED"


def test_incomplete_or_corrupted_injection_blocks():
    assert assess({"outcome": "void"}, FILE_BYTES, True)["status"] == "BLOCKED"
    assert assess({"outcome": "void"}, FILE_BYTES * 16, False)["status"] == "BLOCKED"
