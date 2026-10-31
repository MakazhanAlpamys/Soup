"""The compact MLP record must preserve failures, exact bytes and acceptance gates."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RECORD = ROOT / "benchmarks/results/issue2162-mlp-v1"


def helper():
    path = ROOT / "benchmarks/harness/issue2162_mlp_record.py"
    spec = importlib.util.spec_from_file_location("mlp_record", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("newlines", ["\n", "\r\n"])
def test_receipt_round_trip_preserves_original_bytes(newlines):
    module = helper()
    steps = [{"logits": {"sha256": "a" * 64}}]
    value = {"status": "ok", "steps": steps, "optimizer_updates": 2}
    raw = (json.dumps(value, indent=2) + "\n").replace("\n", newlines).encode()
    encoded = module.encode_file("gate.json", raw, steps)
    assert encoded["shared_steps"]
    assert module.decode_file(encoded, steps) == raw
    with pytest.raises(AssertionError, match="original bytes"):
        module.decode_file(encoded, [{"logits": {"sha256": "b" * 64}}])


def test_operator_dictionary_preserves_every_event_and_zero_device_boundary():
    module = helper()
    events = [
        dict(
            name="aten::mm",
            ancestors=["soup2162|backward|0|down_proj"],
            children=[],
            device_us=0.0,
            self_cpu_us=1.25,
        )
    ] * 2
    raw = json.dumps({"events": events}).encode()
    entry = module.encode_file("profile.operator-events.json", raw, [])
    assert entry["event_dictionary"]
    assert module.decode_file(entry, []) == raw


def test_telemetry_round_trip_preserves_csv_strings_and_original_rows():
    module = helper()
    rows = [dict(unix_s=123.125, gpus=[["0", " UUID", " 420.00"]], rss=10)] * 3
    raw = "".join(json.dumps(row) + "\r\n" for row in rows).encode()
    entry = module.encode_file("worker-resources.jsonl", raw, [])
    assert entry["kind"] == "jsonl"
    assert module.decode_file(entry, []) == raw


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "a\\..\\escape"])
def test_archive_entry_rejects_path_escape(tmp_path, name):
    with pytest.raises(ValueError, match="path"):
        helper().destination(tmp_path, name)


def test_complete_archive_reproduces_both_original_audits():
    module = helper()
    result = module.audit_record(RECORD)
    assert result["verified"]
    assert result["diagnostic"]["total_updates"] == 14
    assert not result["diagnostic"]["eligible_to_prepare_timing_protocol"]
    assert result["budget"]["primary_passes"]
    assert result["budget"]["gate_processes"] == 6
    assert result["budget"]["timing_processes"] == 30
    assert result["budget"]["exact_update_replays"] == 72
    assert result["budget"]["total_optimizer_updates"] == 1812
    assert result["budget"]["confirmation"]["median_reduction_percent"] == 3.4220144942474704
    assert result["budget"]["family"] == 12
    assert len(result["budget"]["history"]) == 8


def test_no_gpu_library_import_needed_for_record_reader(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert helper().FORMAT == 1


def test_report_reproduces_published_tables_from_evidence(monkeypatch):
    folder = ROOT / "benchmarks/harness"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("mlp_report", folder / "issue2162_mlp_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = module.render(RECORD, helper().audit_record(RECORD))
    assert actual == (ROOT / "benchmarks/gate-2162-mlp-retention.md").read_text(encoding="utf-8")
