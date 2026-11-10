"""Refuse benchmark comparisons that accidentally vary the measured workload."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "issue722_compare.py"
_SPEC = importlib.util.spec_from_file_location("issue722_compare", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _pair():
    common = dict.fromkeys((
        "source_commit", "harness_sha256", "student_snapshot", "teacher_snapshot",
        "tokenized_data_sha256", "initial_adapter_sha256", "shape", "runtime", "dtype",
        "warmup_steps", "measured_steps", "timing_scope",
    ), "identical")
    dense = {**common, "arm": "dense", "config": {
        "output": "/dense", "base": "same-pinned-model", "training": {
            "distill_chunk_size": None, "distill_checkpoint": False, "lr": 2e-5,
        },
    }}
    chunked = copy.deepcopy(dense)
    chunked["arm"] = "chunked"
    chunked["config"]["output"] = "/chunked"
    chunked["config"]["training"].update(distill_chunk_size=128, distill_checkpoint=True)
    return dense, chunked


def test_accepts_only_the_intended_controls_and_separate_output_paths():
    _MODULE.validate_pair(*_pair())


@pytest.mark.parametrize("field", [
    "tokenized_data_sha256", "initial_adapter_sha256", "shape", "dtype", "harness_sha256",
])
def test_refuses_apparent_memory_savings_from_a_different_workload(field):
    dense, chunked = _pair()
    chunked[field] = "different"
    with pytest.raises(ValueError, match=field):
        _MODULE.validate_pair(dense, chunked)


def test_refuses_hidden_training_hyperparameter_changes():
    dense, chunked = _pair()
    chunked["config"]["training"]["lr"] = 1e-5
    with pytest.raises(ValueError, match="training config"):
        _MODULE.validate_pair(dense, chunked)


def test_refuses_a_dense_control_that_already_uses_the_optimization():
    dense, chunked = _pair()
    dense["config"]["training"].update(distill_chunk_size=128, distill_checkpoint=True)
    with pytest.raises(ValueError, match="shipped defaults"):
        _MODULE.validate_pair(dense, chunked)


def test_refuses_two_runs_of_the_same_arm():
    dense, _ = _pair()
    with pytest.raises(ValueError, match="one dense and one chunked"):
        _MODULE.validate_pair(dense, dense)
