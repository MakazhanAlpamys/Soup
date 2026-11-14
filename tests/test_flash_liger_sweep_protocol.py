"""The FA/Liger sweep must not silently mix instrumentation or input workloads."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "issue378_sweep.py"
_SPEC = importlib.util.spec_from_file_location("issue378_sweep", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _records():
    protocol = {"warmup": 5, "steps": 80, "worker_sha256": "same", "source_commit": "fixed"}
    common = {"harness_sha256": "same", "source_commit": "fixed",
              "first_batch_sha256": "tokens-and-labels", "runtime": {"torch": "same"},
              "attention_implementation": "sdpa", "module_counts": {"LlamaRMSNorm": 65}}
    probe = {**common, "activation_counts": {"sdpa": 256}}
    record = {**common, "probe": False, "warmup_steps": 5, "measured_steps": 80,
              "realized_batch_shape": [4, 1024], "sequence": 1024, "arm": "sdpa",
              "activation_counts": None, "profile_top_events": []}
    return copy.deepcopy(record), protocol, copy.deepcopy(probe)


def test_accepts_verified_uninstrumented_timing():
    _MODULE.validate_timing(*_records())


@pytest.mark.parametrize("key,changed", [
    ("measured_steps", 79), ("warmup_steps", 0), ("realized_batch_shape", [4, 512]),
    ("first_batch_sha256", "different-label-mask"), ("source_commit", "different-source"),
    ("runtime", {"torch": "different"}),
])
def test_refuses_hidden_changes_to_the_workload_or_stack(key, changed):
    record, protocol, probe = _records()
    record[key] = changed
    with pytest.raises(ValueError):
        _MODULE.validate_timing(record, protocol, probe)


def test_does_not_count_a_profiler_probe_as_a_fast_timing_repetition():
    record, protocol, probe = _records()
    record["profile_top_events"] = [{"name": "a real profiler event"}]
    with pytest.raises(ValueError, match="profiler disabled"):
        _MODULE.validate_timing(record, protocol, probe)


def test_flash_flag_without_flash_execution_cannot_pass():
    record, protocol, probe = _records()
    record["arm"] = "fa2"
    with pytest.raises(ValueError, match="FA2 activation"):
        _MODULE.validate_timing(record, protocol, probe)


def test_empty_or_interrupted_matrix_is_incomplete(tmp_path):
    result = _MODULE.summarize(tmp_path, {"repeats": 5})
    assert result["complete_grid"] is False
    assert len(result["missing"]) == 32
    assert result["rows"] == []
