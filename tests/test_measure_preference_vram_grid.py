import pytest

from benchmarks.measure_preference_vram_grid import (
    _assert_full_batch,
    _failed_worker_row,
    _paired_verdict,
    _render_report,
    _safe_error_message,
    _validate_measured_forward,
    _validate_probe_tree,
    classify_ratio,
    parse_nvidia_smi_sample,
)


def test_grid_calls_a_probe_under_read_a_refusal():
    assert classify_ratio(999, 1000) == "under-read"


def test_grid_identifies_the_sft_over_read_band():
    assert classify_ratio(1130, 1000) == "over-read-in-SFT-band"


def test_grid_keeps_a_larger_over_read_visible():
    assert classify_ratio(1200, 1000) == "over-read-above-SFT-band"


def test_grid_does_not_draw_a_conclusion_from_a_missing_peak():
    assert classify_ratio(None, 1000) == "incomplete"


def test_nvidia_smi_sample_records_device_memory_temperature_and_utilization():
    sample = parse_nvidia_smi_sample("2026/11/15 17:30:01.250, 6123, 2069, 58, 91")

    assert sample == {
        "timestamp": "2026/11/15 17:30:01.250",
        "memory_used_mib": 6123,
        "memory_free_mib": 2069,
        "temperature_c": 58,
        "utilization_percent": 91,
    }


def test_assert_full_batch_refuses_a_short_batch():
    with pytest.raises(ValueError, match="under-read refused"):
        _assert_full_batch("dpo", {"input_ids": FakeTensor((2, 1024))}, 2048, 1)


def test_paired_verdict_marks_failed_or_incomplete_rows_incomplete():
    assert _paired_verdict({"status": "failed"}, {"status": "success"}) == (
        "incomplete",
        None,
    )


def test_parent_failure_row_keeps_resolved_commit_identifiers():
    row = _failed_worker_row(
        task="dpo",
        seq_len=2048,
        mode="probe",
        detail="worker timed out",
        main_sha="main-resolved",
        probe_sha="probe-resolved",
        tree_sha="tree-resolved",
    )

    assert row["main_commit"] == "main-resolved"
    assert row["probe_commit"] == "probe-resolved"
    assert row["measurement_tree"] == "tree-resolved"
    assert _paired_verdict({"status": "success"}, {"status": "success"}) == (
        "incomplete",
        None,
    )


def test_forward_shape_mismatch_is_rejected():
    row = {"status": "success"}
    with pytest.raises(ValueError, match="measured forward shape"):
        _validate_measured_forward(row, "dpo", [[2, 1024]], 2048, 1)
    assert row["status"] == "failed"
    assert row["error_type"] == "ValueError"


def test_missing_probe_api_is_reported_before_measurement(tmp_path):
    runtime = tmp_path / "src/soup_cli/utils/layer_stream_runtime.py"
    setup = tmp_path / "src/soup_cli/trainer/stream_setup.py"
    runtime.parent.mkdir(parents=True)
    setup.parent.mkdir(parents=True)
    runtime.write_text("def existing_helper(): pass\n", encoding="utf-8")
    setup.write_text("class StreamingSetupMixin: pass\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires the named #1141 measurement tree"):
        _validate_probe_tree(tmp_path)


def test_error_message_drops_unrecognized_absolute_paths():
    message = _safe_error_message(RuntimeError(r"could not read C:\Users\user\private.json"))

    assert "C:\\Users" not in message
    assert "local path" in message


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)


def test_report_calls_out_measured_under_read(tmp_path):
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    (rows_dir / "dpo-seq2048-probe.json").write_text(
        '{"task":"dpo","sequence_length":2048,"mode":"probe",'
        '"status":"success","probe_peak_bytes":900}',
        encoding="utf-8",
    )
    (rows_dir / "dpo-seq2048-step.json").write_text(
        '{"task":"dpo","sequence_length":2048,"mode":"step",'
        '"status":"success","real_step_peak_bytes":1000}',
        encoding="utf-8",
    )
    report = tmp_path / "report.md"

    _render_report(
        tmp_path,
        report,
        main_sha="main",
        probe_sha="probe",
        tree_sha="tree",
    )

    text = report.read_text(encoding="utf-8")
    assert "under-read" in text
    assert "keep this loss refused" in text
