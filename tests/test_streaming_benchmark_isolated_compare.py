"""Real CPU process isolation and rejection tests; no CUDA inference from CPU passes."""

import copy
import json
import os
import time
from pathlib import Path

import pytest

from benchmarks.harness.streaming_controls.isolated_compare import compare_results, run


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
def test_fresh_processes_preserve_exact_training(tmp_path, monkeypatch):
    from benchmarks.harness.streaming_controls import isolated_compare

    original = isolated_compare.supervise
    deadlines = []

    def record_deadline(script, sha, output, bound, deadline):
        deadlines.append(deadline)
        return original(script, sha, output, bound, deadline)

    monkeypatch.setattr(isolated_compare, "supervise", record_deadline)
    outer_deadline = time.time() + 180
    result = run(tmp_path / "isolated", "cpu-smoke", outer_deadline)
    assert result["status"] == "NUMERICAL_PASS", result
    assert deadlines == [outer_deadline - 40] * 2
    assert result["outer_deadline_unix_s"] == outer_deadline
    assert result["work_deadline_unix_s"] == outer_deadline - 40
    assert result["performance_verdict"] == "NO_VERDICT"
    first, second = result["arms"]
    assert first["pid"] != second["pid"] != os.getpid()
    assert first["order"] == ["A"] and second["order"] == ["B"]
    assert second["records"][0]["resident_indices"]
    assert (
        second["records"][0]["range_reads_total"]["bytes"]
        < (first["records"][0]["range_reads_total"]["bytes"])
    )
    bad = copy.deepcopy(result["arms"])
    bad[1]["records"][0]["updates"][0]["loss"] += 0.01
    assert compare_results(bad, ["A", "B"])["status"] == "BLOCKED"
    bad = copy.deepcopy(result["arms"])
    bad[1]["weight_sha256"] = {}
    assert compare_results(bad, ["A", "B"])["status"] == "BLOCKED"
    assert compare_results(result["arms"][:1], ["A", "B"])["status"] == "BLOCKED"
    assert compare_results(result["arms"], ["B", "A"])["status"] == "BLOCKED"


def test_empty_comparison_is_not_a_pass():
    assert compare_results([], ["A", "B"])["status"] == "BLOCKED"


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
def test_existing_evidence_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path, "cpu-smoke")


def test_non_posix_supervisor_is_explicitly_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.harness.streaming_controls.isolated_compare.os.name", "nt")
    with pytest.raises(RuntimeError, match="^POSIX supervisor required$"):
        run(tmp_path / "unused", "cpu-smoke")


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
@pytest.mark.parametrize(
    "deadline", [None, True, "2000", float("nan"), float("inf"), -1, 1000, 1040]
)
def test_invalid_cuda_deadline_refused_before_output(tmp_path, monkeypatch, deadline):
    monkeypatch.setattr(
        "benchmarks.harness.streaming_controls.isolated_compare.time.time", lambda: 1000
    )
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="deadline"):
        run(output, "cuda-pilot", deadline)
    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
def test_calibration_cannot_extend_deadline_or_start_late_arm(tmp_path, monkeypatch):
    from benchmarks.harness.streaming_controls import (
        foreign_read_control,
        io_calibration,
        isolated_compare,
    )

    clock = [1000]
    monkeypatch.setattr(isolated_compare.time, "time", lambda: clock[0])
    monkeypatch.setattr(io_calibration, "run", lambda _: {"status": "PASS_COUNTER_CALIBRATION"})

    def finish_control(*_):
        clock[0] = 1060
        return {"status": "PASS_FOREIGN_READER_DETECTED"}

    monkeypatch.setattr(foreign_read_control, "run", finish_control)
    monkeypatch.setattr(
        isolated_compare,
        "supervise",
        lambda *_: pytest.fail("expired deadline must not spawn an arm"),
    )
    output = tmp_path / "expired"
    result = run(output, "cuda-pilot", 1100)
    assert result["status"] == "BLOCKED"
    assert result["arms"] == []
    assert "deadline expired" in result["error"]
    assert result["work_deadline_unix_s"] == 1060
    assert not list(output.glob("arm_*.sh"))


@pytest.mark.parametrize("blocks", [True, 0, 1, 19, 21, 20.0, "20", None])
def test_unregistered_block_count_refused_before_output(tmp_path, blocks):
    output = tmp_path / "unused"
    with pytest.raises(
        ValueError, match=r"^abba_blocks must be the frozen pilot \(5\) or confirmation \(20\)$"
    ):
        run(
            output,
            "cuda-representative",
            time.time() + 7200,
            tmp_path / "prepared.json",
            abba_blocks=blocks,
        )
    assert not output.exists()


@pytest.mark.parametrize("mode", ["cpu-smoke", "cuda-pilot"])
def test_confirmation_count_cannot_change_old_modes(tmp_path, mode):
    with pytest.raises(ValueError, match="^20 ABBA blocks require cuda-representative mode$"):
        run(tmp_path / "unused", mode, time.time() + 7200, abba_blocks=20)
    assert not (tmp_path / "unused").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor required")
@pytest.mark.parametrize(
    "failure", [None, "process", "void", "unknown", "missing_check", "clock", "clock_samples"]
)
def test_confirmation_runs_frozen_order_without_retry(tmp_path, monkeypatch, failure):
    """Exercise scheduling/supervision; the fake arms are NOT numerical/GPU evidence."""
    import shlex

    from benchmarks.harness.arm_validity import CHECKS
    from benchmarks.harness.streaming_controls import (
        foreign_read_control,
        io_calibration,
        isolated_compare,
        representative_workload,
    )

    fail_at = 7 if failure else None

    monkeypatch.setattr(isolated_compare.time, "time", lambda: 1000)
    monkeypatch.setattr(io_calibration, "run", lambda _: {"status": "PASS_COUNTER_CALIBRATION"})
    monkeypatch.setattr(
        foreign_read_control, "run", lambda *_: {"status": "PASS_FOREIGN_READER_DETECTED"}
    )
    monkeypatch.setattr(representative_workload, "load_prepared", lambda *_a, **_kw: {})
    prepared = tmp_path / "prepared.json"
    prepared.write_text("{}")
    workload_sha = representative_workload.file_sha(prepared)
    launched = []

    def supervise(script, sha, output, bound, deadline):
        command = shlex.split(script.read_text().split("exec ", 1)[1])
        arm = command[command.index("--arm") + 1]
        directory = Path(command[command.index("--output") + 1])
        assert bound == 300
        assert deadline == 6360  # min(outer=8200, now+5400) minus cleanup reserve
        launched.append(arm)
        if failure == "process" and len(launched) - 1 == fail_at:
            return {"status": "FAILED", "group_cleanup_confirmed": True}
        directory.mkdir()
        item = {
            "status": "ARM_COMPLETE",
            "order": [arm],
            "mode": "cuda-representative",
            "workload_sha256": workload_sha,
            "effective_tokens_per_step": 255,
            "steps_per_arm": 1,
            "warmup_steps": 0,
            "torch": "test-only",
            "config": {},
            "weight_sha256": {"test": "a" * 64},
            "records": [
                {
                    "status": "COMPLETED",
                    "changed_adapter": True,
                    "initial_adapter_sha256": "a" * 64,
                    "final_adapter_sha256": "b" * 64,
                    "updates": [{"loss": 1.0, "step": 0, "gradient_sha256": "c" * 64}],
                    "validity": {
                        "required": list(CHECKS),
                        "checks": {name: {"outcome": "ok"} for name in CHECKS},
                        "gpu_clock": {"outcome": "ok", "samples": 2, "unknown_samples": 0},
                    },
                }
            ],
        }
        if len(launched) - 1 == fail_at:
            validity = item["records"][0]["validity"]
            if failure in ("void", "unknown"):
                validity["checks"]["foreign_reads"]["outcome"] = failure
            elif failure == "missing_check":
                del validity["checks"]["foreign_reads"]
            elif failure == "clock":
                validity["gpu_clock"]["outcome"] = "void"
            elif failure == "clock_samples":
                validity["gpu_clock"]["unknown_samples"] = 1
        (directory / "result.json").write_text(json.dumps(item))
        return {"status": "PASS", "group_cleanup_confirmed": True}

    monkeypatch.setattr(isolated_compare, "supervise", supervise)
    result = run(tmp_path / "confirmation", "cuda-representative", 8200, prepared, abba_blocks=20)
    expected = list("ABBA") * 20
    assert result["order"] == expected
    assert result["planned_abba_blocks"] == 20
    assert result["performance_verdict"] == "NO_VERDICT"
    if fail_at is None:
        assert launched == expected
        assert len(result["arms"]) == 80
        assert result["status"] == "NUMERICAL_PASS"
    else:
        assert launched == expected[: fail_at + 1]
        assert len(result["arms"]) == fail_at + (failure != "process")
        assert result["status"] == "BLOCKED"
