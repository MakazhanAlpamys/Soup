"""The fixed B allocation, thread-safe labels and historical correction family."""
import contextvars
import importlib
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/harness"))


def helper():
    return importlib.import_module("stream_mlp_budget")


def census():
    return dict(measured=True, k=11, full_layer_charge=482015738,
                budget_bytes=5756682240, peak_reserved=2833252352,
                peak_allocated=2468564480, retained_indices=list(range(21, 32)))


def test_fixed_b_allocation_charges_every_output_once():
    plan = helper().candidate_plan(census())
    assert plan["retained_indices"] == list(range(26, 32))
    assert plan["qkvo_indices"] == list(range(26))
    assert plan["mlp_indices"] == list(range(15, 26))
    assert plan["effective_k"] == 6 and plan["cache_tensors"] == 126
    assert plan["raw_bytes"] == 2382364672
    assert plan["charged_bytes"] == 2734318038
    assert plan["slack_bytes"] == 130269774
    control = helper().candidate_plan(census(), include_mlp=False)
    assert control["raw_bytes"] == 1090519040 and control["cache_tensors"] == 104
    assert control["retained_indices"] == plan["retained_indices"]


@pytest.mark.parametrize("change", [dict(k=10), dict(budget_bytes=5756682241),
                                   dict(peak_allocated=2468564481)])
def test_changed_census_refuses_b_instead_of_choosing_another_layout(change):
    with pytest.raises(ValueError, match="census"):
        helper().candidate_plan(census() | change)


def test_observer_phase_is_visible_in_autograd_thread_and_restores():
    import torch

    observer = object.__new__(helper().ProjectionObserver)
    observer._phase = contextvars.ContextVar("test_phase", default="outside")
    names = []
    callback = observer._wrap(lambda value: value + 1, 19, "gate_proj")
    values = []
    with patch.object(torch.profiler, "record_function",
                      lambda name: names.append(name) or nullcontext()):
        with observer.phase("backward"):
            with pytest.raises(ValueError, match="concurrent"):
                with observer.phase("forward"):
                    pass
            thread = threading.Thread(target=lambda: values.append(callback(None, 4)))
            thread.start()
            thread.join()
    assert names == ["soup2162|backward|19|gate_proj"] and values == [5]
    assert observer._shared_phase == "outside"


def test_bootstrap_counts_all_eight_old_contrasts_and_four_new():
    stats = importlib.import_module("stream_mlp_budget_statistics")
    row = stats.summarize([1.7, 1.9, 2.0, 2.1, 2.4, 2.8])
    assert row["correction_family"] == 12
    assert row["adjusted_quantiles"] == [0.05 / 24, 1 - 0.05 / 24]
    assert row["median_reduction_percent"] == pytest.approx(2.05)
    assert row["family_adjusted_ci"][0] <= row["ordinary_95"][0]


def test_failed_primary_or_increment_cannot_launch_confirmation():
    stats = importlib.import_module("stream_mlp_budget_statistics")
    make = stats.summarize
    common = {"qkvo_vs_native": make([0.2] * 6), "mixed_vs_qkvo": make([3] * 6)}
    assert not stats.stage_a_decision(common | {"mixed_vs_native": make([1.99] * 6)})
    assert not stats.stage_a_decision(common | {"mixed_vs_native": make([3] * 6),
                                                "mixed_vs_qkvo": make([-0.1] * 6)})
    assert stats.stage_a_decision(common | {"mixed_vs_native": make([3] * 6)})


def test_nonfinite_or_incomplete_timing_family_is_refused():
    stats = importlib.import_module("stream_mlp_budget_statistics")
    for values in ([3] * 4, [3] * 7, [float("nan")] * 6):
        with pytest.raises(ValueError):
            stats.summarize(values)
    with pytest.raises(ValueError):
        stats.stage_a_decision({})


def test_memory_allowance_does_not_hide_one_byte_over_budget():
    checks = importlib.import_module("stream_mlp_budget_checks")
    baseline = 4 * 2**30
    limit = 1291845632 + 64 * 2**20
    for extra, expected in ((0, True), (1, False)):
        rows = {"qkvo": {"peak_allocated": baseline},
                "mixed": {"peak_allocated": baseline + limit + extra}}
        result = checks.memory_comparisons(rows)
        assert len(result) == 1 and result[0]["passes"] is expected
    with pytest.raises(ValueError, match="six complete"):
        checks.gate_decision({}, {})
