"""Replay the recorded benchmark outcomes and detect tampered evidence."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from benchmarks.harness.streaming_controls.replay_evidence import replay

ROOT = Path(__file__).parent / "fixtures" / "streaming_benchmark_evidence"


def test_complete_series_replays_numerical_pass_without_speed_verdict():
    report = replay(ROOT / "numerical_match.zip")
    assert report["failed_checks"] == []
    assert report["status"] == "NUMERICAL_PASS_GATES_PASS"
    assert report["performance_verdict"] == "NO_VERDICT_INSUFFICIENT_REPETITIONS"
    assert report["completed_arms"] == 20 and report["cuda_controls_passed"] == 5
    assert report["repetitions"]["blocks_to_add"] == 15
    assert report["descriptive_process_read_reduction_pct"] == pytest.approx(25.0)


def test_stopped_series_replays_recorded_foreign_read_stop():
    report = replay(ROOT / "foreign_reads.zip")
    assert report["failed_checks"] == []
    assert report["status"] == "BLOCKED_FOREIGN_READS"
    assert report["completed_arms"] == 22 and report["cuda_controls_passed"] == 6
    assert [row["index"] for row in report["validity_failures"]] == [14, 19]
    assert report["partial_arms"] == [{"index": 22, "arm": "B", "status": "RUNNING", "records": 0}]
    assert report["numerical_completed_arms"]["status"] == "NUMERICAL_PASS"


def test_modified_member_is_detected(tmp_path):
    packet = tmp_path / "packet.zip"
    arm = "comparison/arm_01_B/result.json"
    with ZipFile(ROOT / "numerical_match.zip") as original, ZipFile(packet, "w") as changed:
        for name in original.namelist():
            content = original.read(name)
            if name == arm:
                data = json.loads(content)
                data["records"][0]["updates"][0]["loss"] += 1e-6
                content = json.dumps(data).encode()
            changed.writestr(name, content)
    report = replay(packet)
    assert "member:comparison/arm_01_B/result.json" in report["failed_checks"]
    assert report["status"] == "REPLAY_CHECKS_FAILED"
