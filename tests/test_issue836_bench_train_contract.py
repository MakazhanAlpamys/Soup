"""#836 — a throughput number carries no evidence that the model was training.

The report builder here is deliberately pure: step times, token counts and the
three hard failure checks are plain Python, so every case below runs on CPU with
no trainer, no GPU and no model. The callback that feeds it only collects.

The three checks exist because each catches something the others cannot:

* ``grad_norm == 0`` catches the published-benchmark failure this issue cites,
  but only when the backend reports a norm at all -- MLX deliberately does not,
  and ``monitoring/callback.py:115`` initialises ``_last_grad_norm = 0.0``, so
  "absent" and "genuinely zero" are the same value there.
* The parameter fingerprint check is backend-independent and catches the MLX and
  DeepSpeed cases, where no norm is logged.
* The trainable-parameter check catches the case where neither of the others can
  fire usefully, because nothing was ever going to move.
"""

from __future__ import annotations

import math

import pytest

from soup_cli.bench.train_report import (
    ParamSnapshot,
    StepRecord,
    build_train_report,
    summarize_step_times,
    token_utilisation,
)


def _step(index, seconds=1.0, grad_norm=1.0, useful=8, total=10):
    return StepRecord(
        index=index,
        wall_seconds=seconds,
        grad_norm=grad_norm,
        useful_tokens=useful,
        total_tokens=total,
    )


def _report(steps, *, warmup=0, trainable=3, snapshots=("before", "after")):
    return build_train_report(
        steps=steps,
        warmup_steps=warmup,
        trainable=ParamSnapshot(count=trainable, fingerprint_first=snapshots[0],
                                fingerprint_last=snapshots[1]),
        provenance={"torch": "2.6.0"},
        memory={"max_memory_allocated": 1, "max_memory_reserved": 2},
    )


class TestStepTiming:
    def test_median_and_p95_on_known_inputs(self):
        """Ten known steps; nearest-rank p95 so the value is one that was
        actually measured rather than an interpolation between two."""
        times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
        summary = summarize_step_times(times, warmup_steps=0)
        assert summary["median_seconds"] == 5.5
        assert summary["p95_seconds"] == 100.0
        assert summary["counted_steps"] == 10
        assert summary["warmup_steps_discarded"] == 0

    def test_p95_is_the_measured_step_not_the_maximum(self):
        """20 steps: nearest-rank p95 is the 19th, which is NOT the max. The
        first version of this test used 10 steps, where rank 10 and the maximum
        are the same value, so `return max(...)` would have passed it."""
        times = [float(i) for i in range(1, 20)] + [999.0]
        summary = summarize_step_times(times, warmup_steps=0)
        assert summary["p95_seconds"] == 19.0
        assert max(times) == 999.0

    def test_warmup_steps_are_discarded_and_counted(self):
        """The first steps carry compile and allocator noise. They must not
        reach the median, and the report must say how many went."""
        times = [100.0, 100.0, 1.0, 2.0, 3.0]
        summary = summarize_step_times(times, warmup_steps=2)
        assert summary["median_seconds"] == 2.0
        assert summary["counted_steps"] == 3
        assert summary["warmup_steps_discarded"] == 2

    def test_discarding_every_step_is_an_error_not_an_empty_median(self):
        with pytest.raises(ValueError, match="warmup"):
            summarize_step_times([1.0, 2.0], warmup_steps=2)


class TestTokenUtilisation:
    def test_useful_over_total(self):
        assert token_utilisation(useful=80, total=100) == 0.8

    def test_no_tokens_is_none_not_a_division(self):
        assert token_utilisation(useful=0, total=0) is None

    def test_useful_tokens_drive_throughput_not_padded_ones(self):
        """tok/s must be computed from supervised tokens. Padding a batch to
        twice the length would otherwise double the reported number."""
        steps = [_step(i, seconds=1.0, useful=10, total=100) for i in range(4)]
        report = _report(steps)
        assert report["throughput"]["useful_tokens_per_second"] == 10.0
        assert report["throughput"]["total_tokens_per_second"] == 100.0
        assert report["tokens"]["utilisation"] == 0.1


class TestZeroTrainableParameters:
    def test_zero_trainable_parameters_fails_and_names_the_cause(self):
        report = _report([_step(0)], trainable=0)
        assert report["valid"] is False
        assert any("trainable" in f["check"] for f in report["failures"])
        message = " ".join(f["message"] for f in report["failures"])
        assert "0 trainable" in message

    def test_a_real_count_passes_that_check(self):
        report = _report([_step(0)], trainable=5)
        assert [f for f in report["failures"] if "trainable" in f["check"]] == []


class TestGradNorms:
    def test_a_zero_grad_norm_on_any_step_fails(self):
        """The failure this issue exists for: a throughput number measured on a
        run that was not training."""
        steps = [_step(0), _step(1, grad_norm=0.0), _step(2)]
        report = _report(steps)
        assert report["valid"] is False
        failure = [f for f in report["failures"] if f["check"] == "grad_norm"][0]
        assert "step 1" in failure["message"]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_grad_norm_fails(self, value):
        report = _report([_step(0), _step(1, grad_norm=value)])
        assert report["valid"] is False
        assert any(f["check"] == "grad_norm" for f in report["failures"])

    def test_absent_grad_norms_are_not_treated_as_zero(self):
        """``None`` means the backend does not report one (MLX). That is not a
        failure by itself -- it is why the fingerprint check exists."""
        steps = [_step(i, grad_norm=None) for i in range(3)]
        report = _report(steps)
        assert [f for f in report["failures"] if f["check"] == "grad_norm"] == []
        assert report["checks"]["grad_norm"] == "not reported by this backend"
        assert report["valid"] is True


class TestParametersChanged:
    def test_identical_first_and_last_parameters_fail(self):
        report = _report([_step(0)], snapshots=("same", "same"))
        assert report["valid"] is False
        assert any(f["check"] == "parameters_changed" for f in report["failures"])

    def test_it_fires_even_when_no_grad_norm_was_reported(self):
        """The backend-independent one: MLX and DeepSpeed runs reach this."""
        steps = [_step(i, grad_norm=None) for i in range(3)]
        report = _report(steps, snapshots=("same", "same"))
        assert report["valid"] is False
        checks = {f["check"] for f in report["failures"]}
        assert checks == {"parameters_changed"}

    def test_a_changed_fingerprint_passes(self):
        report = _report([_step(0)], snapshots=("before", "after"))
        assert [f for f in report["failures"] if f["check"] == "parameters_changed"] == []


class TestTheControl:
    def test_a_healthy_run_is_valid_with_no_failures(self):
        """The discrimination control the issue asks for: the same shape of
        report, with real gradients and moving parameters, passes."""
        steps = [_step(i, seconds=0.5, grad_norm=0.7 + i) for i in range(4)]
        report = _report(steps, warmup=1)
        assert report["valid"] is True
        assert report["failures"] == []
        assert report["timing"]["counted_steps"] == 3
        assert math.isclose(report["timing"]["median_seconds"], 0.5)

    def test_memory_peaks_are_separate_fields(self):
        """`gate-h100-validation.md` reported peak VRAM two different ways in one
        file. Allocated and reserved are different quantities and stay apart."""
        report = _report([_step(0)])
        assert report["memory"]["max_memory_allocated"] == 1
        assert report["memory"]["max_memory_reserved"] == 2

    def test_the_report_never_asks_nvidia_smi_for_memory(self):
        """Explicitly out of scope in the issue: nvidia-smi samples a different
        quantity than torch's allocator counters. Provenance does ask it for the
        driver and SM clock, so the guard is on what is queried, and it proves it
        found the query it inspects rather than passing on an empty scan."""
        import re
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "src" / "soup_cli" / "bench"
        queries = [
            query
            for path in source.rglob("*.py")
            for query in re.findall(r"--query-gpu=([\w.,]+)", path.read_text(encoding="utf-8"))
        ]
        assert queries, "no nvidia-smi query found -- the scan is not looking at bench/"
        assert not [q for q in queries if "memory" in q], queries


class TestStepCount:
    """Matched work: a run labelled N steps that ran fewer is not that run."""

    def _report(self, measured, requested):
        return build_train_report(
            steps=[_step(i) for i in range(measured)],
            warmup_steps=0,
            trainable=ParamSnapshot(count=3, fingerprint_first="a", fingerprint_last="b"),
            provenance={},
            memory={},
            steps_requested=requested,
        )

    def test_fewer_steps_than_requested_fails(self):
        report = self._report(3, 5)
        assert report["valid"] is False
        assert [f["check"] for f in report["failures"]] == ["step_count"]
        assert report["steps_measured"] == 3 and report["steps_requested"] == 5

    def test_the_requested_count_passes(self):
        assert self._report(5, 5)["valid"] is True
