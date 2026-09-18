"""#836 — the collecting half: it measures, it does not judge.

The collector only records; every verdict lives in ``train_report``. These tests
drive it with real torch tensors and a fake trainer loop, so no GPU and no model
are needed, and they pin the two things that are easy to get wrong:

* supervised tokens are ``labels != -100`` -- what the loss sees -- so padding a
  batch cannot improve the reported tok/s;
* a trainable parameter on ``meta`` has no storage and cannot train, so it is
  counted separately rather than inflating the trainable count
  (``utils/layer_stream_runtime.py:2161`` exists for that case).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from soup_cli.bench.collector import (  # noqa: E402
    BenchCollector,
    count_supervised_tokens,
    fingerprint_trainable,
    summarize_trainable,
)


def _model(trainable=2, frozen=1, meta=0):
    model = torch.nn.Module()
    for i in range(trainable):
        model.register_parameter(f"t{i}", torch.nn.Parameter(torch.ones(2, 2)))
    for i in range(frozen):
        param = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        model.register_parameter(f"f{i}", param)
    for i in range(meta):
        param = torch.nn.Parameter(torch.ones(2, 2, device="meta"))
        model.register_parameter(f"m{i}", param)
    return model


class TestSupervisedTokens:
    def test_only_unmasked_labels_count(self):
        labels = torch.tensor([[1, 2, -100, -100], [3, -100, -100, -100]])
        useful, total = count_supervised_tokens({"labels": labels})
        assert (useful, total) == (3, 8)

    def test_a_batch_without_labels_counts_no_useful_tokens(self):
        """Not an error: some collators do not build labels until the model
        does. Reporting 0 useful tokens is honest; guessing is not."""
        batch = {"input_ids": torch.ones(2, 4, dtype=torch.long)}
        useful, total = count_supervised_tokens(batch)
        assert (useful, total) == (0, 8)

    def test_padding_the_batch_does_not_raise_the_useful_count(self):
        """The whole point: a longer pad must not look like more work."""
        short = {"labels": torch.tensor([[1, 2, -100]])}
        padded = {"labels": torch.tensor([[1, 2, -100, -100, -100, -100]])}
        assert count_supervised_tokens(short)[0] == count_supervised_tokens(padded)[0]
        assert count_supervised_tokens(padded)[1] > count_supervised_tokens(short)[1]


class TestTrainableSummary:
    def test_counts_trainable_tensors_with_storage(self):
        snapshot = summarize_trainable(_model(trainable=3, frozen=2))
        assert snapshot.count == 3
        assert snapshot.meta_excluded == 0

    def test_meta_parameters_are_excluded_and_reported(self):
        """A trainable adapter stranded on ``meta`` cannot train. Counting it
        would let the zero-trainable check pass on a run that trains nothing."""
        snapshot = summarize_trainable(_model(trainable=1, meta=2))
        assert snapshot.count == 1
        assert snapshot.meta_excluded == 2

    def test_a_frozen_model_has_no_trainable_parameters(self):
        snapshot = summarize_trainable(_model(trainable=0, frozen=3))
        assert snapshot.count == 0


class TestFingerprint:
    def test_the_same_weights_give_the_same_fingerprint(self):
        model = _model()
        assert fingerprint_trainable(model) == fingerprint_trainable(model)

    def test_a_changed_weight_changes_it(self):
        model = _model()
        before = fingerprint_trainable(model)
        with torch.no_grad():
            model.t0 += 1e-6
        assert fingerprint_trainable(model) != before

    def test_frozen_weights_are_not_part_of_it(self):
        """The check is about what was meant to train. A frozen tensor moving
        (a buffer update, say) must not be mistaken for training."""
        model = _model(trainable=1, frozen=1)
        before = fingerprint_trainable(model)
        with torch.no_grad():
            model.f0 += 1.0
        assert fingerprint_trainable(model) == before


class TestTheCollector:
    def _run(self, collector, model, *, steps=3, grad_norms=None, seconds=None):
        clock = iter(seconds or [float(i) for i in range(steps * 2 + 2)])
        collector._now = lambda: next(clock)
        collector.on_train_begin(None, None, None, model=model)
        for index in range(steps):
            collector.on_step_begin(None, None, None)
            if grad_norms is not None and grad_norms[index] is not None:
                collector.on_log(None, None, None, logs={"grad_norm": grad_norms[index]})
            collector.on_step_end(None, None, None)
        collector.on_train_end(None, None, None, model=model)
        return collector

    def test_it_records_one_step_per_step_end(self):
        collector = self._run(BenchCollector(), _model(), steps=3)
        assert len(collector.steps) == 3
        assert [s.index for s in collector.steps] == [0, 1, 2]

    def test_an_absent_grad_norm_stays_none(self):
        """Never 0.0 by default: that is the bug in monitoring/callback.py:115
        this contract exists to avoid repeating."""
        collector = self._run(BenchCollector(), _model(), steps=2, grad_norms=[None, None])
        assert [s.grad_norm for s in collector.steps] == [None, None]

    def test_a_logged_grad_norm_is_attached_to_that_step(self):
        collector = self._run(BenchCollector(), _model(), steps=2, grad_norms=[0.5, 0.0])
        assert [s.grad_norm for s in collector.steps] == [0.5, 0.0]

    def test_step_time_is_the_gap_between_begin_and_end(self):
        # The clock is read twice per step (begin, end) and not at all in
        # on_train_begin, so these four ticks are two steps.
        collector = self._run(
            BenchCollector(), _model(), steps=2, seconds=[0.0, 2.5, 100.0, 103.0]
        )
        assert [s.wall_seconds for s in collector.steps] == [2.5, 3.0]

    def test_it_fingerprints_before_the_first_step_and_after_the_last(self):
        model = _model()
        collector = BenchCollector()
        clock = iter([float(i) for i in range(20)])
        collector._now = lambda: next(clock)
        collector.on_train_begin(None, None, None, model=model)
        collector.on_step_begin(None, None, None)
        with torch.no_grad():
            model.t0 += 1.0
        collector.on_step_end(None, None, None)
        collector.on_train_end(None, None, None, model=model)
        snapshot = collector.param_snapshot()
        assert snapshot.fingerprint_first != snapshot.fingerprint_last

    def test_a_model_that_never_moves_gives_identical_fingerprints(self):
        """The control for the check that has to work without grad_norm."""
        collector = self._run(BenchCollector(), _model(), steps=2)
        snapshot = collector.param_snapshot()
        assert snapshot.fingerprint_first == snapshot.fingerprint_last

    def test_tokens_come_from_the_batch_the_loss_saw(self):
        collector = BenchCollector()
        collector.observe_batch({"labels": torch.tensor([[1, -100, 3, -100]])})
        collector.observe_batch({"labels": torch.tensor([[1, 2, 3, 4]])})
        collector.on_train_begin(None, None, None, model=_model())
        collector._now = lambda: 0.0
        collector.on_step_begin(None, None, None)
        collector.on_step_end(None, None, None)
        assert collector.steps[0].useful_tokens == 6
        assert collector.steps[0].total_tokens == 8

    def test_the_report_it_produces_is_the_pure_builders(self):
        """End to end: collected records go through the same builder the unit
        tests cover, so there is one set of verdict rules, not two."""
        collector = self._run(BenchCollector(), _model(), steps=2, grad_norms=[1.0, 1.0])
        report = collector.build_report(warmup_steps=0, provenance={}, memory={})
        assert report["valid"] is False  # nothing moved: the fingerprint check
        assert {f["check"] for f in report["failures"]} == {"parameters_changed"}
