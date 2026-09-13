"""#23 — the MLX `on_val_loss_report` producer.

#713 built the sinks backend-neutrally: a `val_loss` field on `TrainEvent`, a
`val_loss` column on the `metrics` table with a migration, and a `val_loss`
series on `TrainingDisplay`. It wired the transformers producer. This is the
other producer: mlx-lm's `on_val_loss_report`, the item #23's checklist names.

The discipline these tests exist to enforce is the one that blocked #713 in
review: **the sticky value is for the panel only.** A validation pass happens
every N steps, so the last measured value must stay on screen between passes --
but writing that carried-forward value to the database or the SSE wire on every
training step fabricates rows for measurements that never happened. #713's first
version persisted 9 rows for 2 measurements. `TestOnlyRealMeasurementsAreRecorded`
is the guard against a repeat, on this backend.

mlx-lm's payload is built at `mlx_lm/tuner/trainer.py:310-315`:
    val_info = {"iteration": it - 1, "val_loss": <float>, "val_time": <float>}
Note `it - 1`, which differs from the train report's `iteration`. That is
upstream's choice and it is pinned rather than corrected.
"""

import sys

import pytest

from tests.test_issue634_mlx_resume import (
    _FakeMlxModel,
    _install_fake_mlx,
    _mlx_wrapper,
)


class _RecordingDisplay:
    def __init__(self):
        self.updates = []
        self.started_with = None

    def start(self, total):
        self.started_with = total

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def stop(self):
        pass


class _RecordingTracker:
    def __init__(self):
        self.metrics = []

    def log_metrics(self, **kwargs):
        self.metrics.append(kwargs)


def _run(monkeypatch, tmp_path, *, train_reports=(), val_reports=(),
         sequence=None, display=None, tracker=None, run_id="run-1"):
    """Drive the real train() body with a fake mlx-lm emitting both report kinds.

    ``sequence`` takes explicit ``("val"|"train", payload)`` pairs for tests
    that depend on the ORDER of the two hooks -- anything about a value carried
    from one to the other. The default val-then-train shape below is mlx-lm's
    initial evaluation at ``it = 0``, which is a real sequence but the one in
    which nothing has been measured yet.
    """
    _install_fake_mlx(monkeypatch)

    def _fake_train(**kwargs):
        cb = kwargs.get("training_callback")
        if cb is None:
            return
        if sequence is not None:
            for kind, payload in sequence:
                if kind == "val":
                    cb.on_val_loss_report(payload)
                else:
                    cb.on_train_loss_report(payload)
            return
        # Interleave the way mlx-lm does: an eval fires, then training steps.
        for v in val_reports:
            cb.on_val_loss_report(v)
        for t in train_reports:
            cb.on_train_loss_report(t)

    sys.modules["mlx_lm.tuner.trainer"].train = _fake_train

    w = _mlx_wrapper(tmp_path, epochs=1, lr=1e-4, batch_size=1)
    w.model = _FakeMlxModel()
    w.tokenizer = object()
    w._dataset = {"train": [{"text": "hi"}] * 48, "val": [{"text": "hi"}] * 8}
    w.train(display=display, tracker=tracker, run_id=run_id)
    return w


_VAL_1 = {"iteration": 9, "val_loss": 2.5, "val_time": 0.4}
_VAL_2 = {"iteration": 19, "val_loss": 1.25, "val_time": 0.4}
_TRAIN_1 = {"iteration": 5, "train_loss": 3.0, "learning_rate": 1e-4,
            "iterations_per_second": 5.0, "peak_memory": 0.5}
_TRAIN_2 = {"iteration": 10, "train_loss": 2.0, "learning_rate": 1e-4,
            "iterations_per_second": 5.0, "peak_memory": 0.5}


class TestTheValueReachesAllThreeSinks:
    """The failure #713 was blocked on was reaching two of three."""

    def test_it_reaches_the_display(self, tmp_path, monkeypatch):
        d = _RecordingDisplay()
        _run(monkeypatch, tmp_path, val_reports=[_VAL_1], display=d)
        vals = [u.get("val_loss") for u in d.updates if u.get("val_loss") is not None]
        assert vals and vals[0] == pytest.approx(2.5), (
            "on_val_loss_report did not reach TrainingDisplay"
        )

    def test_it_reaches_the_tracker(self, tmp_path, monkeypatch):
        t = _RecordingTracker()
        _run(monkeypatch, tmp_path, val_reports=[_VAL_1], tracker=t)
        rows = [m for m in t.metrics if m.get("val_loss") is not None]
        assert len(rows) == 1 and rows[0]["val_loss"] == pytest.approx(2.5), (
            "on_val_loss_report did not persist through ExperimentTracker"
        )

    def test_it_reaches_the_sse_wire(self, tmp_path, monkeypatch):
        """Not the dataclass -- the serialized payload a client receives."""
        import soup_cli.utils.train_event_buffer as buf

        seen = []
        monkeypatch.setattr(buf, "push_train_event", lambda e: seen.append(e))
        from soup_cli.utils.sse_train_stream import format_sse_frame, to_payload

        _run(monkeypatch, tmp_path, val_reports=[_VAL_1])
        payloads = [to_payload(e) for e in seen]
        with_val = [p for p in payloads if p.get("val_loss") is not None]
        assert with_val and with_val[0]["val_loss"] == pytest.approx(2.5), (
            "val_loss reached TrainEvent but not the serialized payload; a "
            "field on the dataclass is not a field on the wire -- to_payload() "
            "filters through _ALLOWED_KEYS, so a field absent from that set is "
            "dropped silently"
        )
        # And the frame a client actually receives, not just the dict.
        frame = format_sse_frame([e for e in seen if e.val_loss is not None][0])
        assert "val_loss" in frame and "2.5" in frame, (
            f"val_loss survived to_payload() but not the SSE frame: {frame!r}"
        )


class TestOnlyRealMeasurementsAreRecorded:
    """The blocking finding from #713, guarded on this backend."""

    def test_training_steps_do_not_persist_a_carried_forward_value(
        self, tmp_path, monkeypatch
    ):
        """2 evaluations and 2 training steps must give 2 val rows, not 4."""
        t = _RecordingTracker()
        # A display is REQUIRED here, not incidental: `on_train_loss_report`
        # returns early when `display is None`, so without one the tracker is
        # never reached on a training step and this test cannot observe the
        # defect it exists for. Found by mutation -- writing the sticky value
        # to the tracker on every training step survived the first version.
        d = _RecordingDisplay()
        _run(monkeypatch, tmp_path, val_reports=[_VAL_1, _VAL_2],
             train_reports=[_TRAIN_1, _TRAIN_2], tracker=t, display=d)
        assert len(t.metrics) > 2, (
            "sanity: training steps must reach the tracker at all, or this "
            "test would pass for the wrong reason"
        )
        rows = [m["val_loss"] for m in t.metrics if m.get("val_loss") is not None]
        assert rows == [pytest.approx(2.5), pytest.approx(1.25)], (
            f"expected one row per evaluation, got {rows} — a sticky value is "
            "being written on training steps, fabricating measurements"
        )

    def test_training_steps_do_not_put_a_stale_value_on_the_wire(
        self, tmp_path, monkeypatch
    ):
        import soup_cli.utils.train_event_buffer as buf

        seen = []
        monkeypatch.setattr(buf, "push_train_event", lambda e: seen.append(e))
        # A display is REQUIRED here, exactly as it is for the tracker sibling
        # above. `on_train_loss_report` returns at the `display is None` guard
        # before the SSE push, so without one a training step pushes nothing at
        # all and "exactly one event carries val_loss" is true for every
        # possible implementation -- including one that puts the sticky value
        # on the wire. Found by @MakazhanAlpamys, who mutated the push to
        # `val_loss=type(self)._sticky_val_loss` and watched this test pass.
        # I had already written this reasoning down for the tracker and did not
        # carry it here.
        _run(monkeypatch, tmp_path, val_reports=[_VAL_1],
             train_reports=[_TRAIN_1, _TRAIN_2], display=_RecordingDisplay())
        assert len(seen) > 1, (
            "sanity: training steps must reach the wire at all, or the "
            "assertion below is vacuous"
        )
        with_val = [e for e in seen if getattr(e, "val_loss", None) is not None]
        assert len(with_val) == 1, (
            f"{len(with_val)} events carry val_loss for 1 evaluation"
        )

    def test_but_the_panel_does_keep_showing_it(self, tmp_path, monkeypatch):
        """Sticky on screen is the point -- it must not blink out between evals."""
        d = _RecordingDisplay()
        _run(monkeypatch, tmp_path, val_reports=[_VAL_1],
             train_reports=[_TRAIN_1, _TRAIN_2], display=d)
        train_updates = [u for u in d.updates if u.get("loss") in (3.0, 2.0)]
        assert train_updates, "sanity: training updates reached the display"
        assert all(u.get("val_loss") == pytest.approx(2.5) for u in train_updates), (
            "the panel lost the last validation loss between evaluations"
        )


class TestTheUpstreamPayloadContract:
    def test_the_iteration_key_is_upstreams_off_by_one(self, tmp_path, monkeypatch):
        """mlx-lm sends `it - 1` for val and `it` for train. Pinned, not fixed."""
        t = _RecordingTracker()
        _run(monkeypatch, tmp_path, val_reports=[_VAL_1], tracker=t)
        row = [m for m in t.metrics if m.get("val_loss") is not None][0]
        assert row["step"] == 9, (
            "the step recorded for a validation row must be the iteration "
            "mlx-lm reported, not a recomputed one"
        )

    def test_upstream_still_sends_the_keys_this_reads(self):
        """If mlx-lm renames these, this fails here rather than silently."""
        pytest.importorskip("mlx_lm")
        import inspect

        from mlx_lm.tuner import trainer as t

        src = inspect.getsource(t.train)
        for key in ('"iteration"', '"val_loss"', '"val_time"'):
            assert key in src, f"mlx-lm no longer builds val_info with {key}"

    def test_a_missing_val_loss_key_writes_no_row_at_all(
        self, tmp_path, monkeypatch
    ):
        """Not merely "no non-None value" -- no row and no event whatsoever.

        Asserting only that nothing non-None was written passes for an
        implementation that logs a `val_loss=None` row, which is a spurious
        metrics row and clears the panel. Found by mutation: deleting the
        guard survived the first version of this test.
        """
        import soup_cli.utils.train_event_buffer as buf

        seen = []
        monkeypatch.setattr(buf, "push_train_event", lambda e: seen.append(e))
        t = _RecordingTracker()
        _run(monkeypatch, tmp_path, val_reports=[{"iteration": 3}], tracker=t)
        assert t.metrics == [], f"a payload with no val_loss wrote {t.metrics}"
        assert seen == [], f"a payload with no val_loss emitted {seen}"

    def test_a_missing_key_does_not_clear_an_earlier_measurement(
        self, tmp_path, monkeypatch
    ):
        """The sticky panel value must survive a malformed later payload."""
        d = _RecordingDisplay()
        _run(monkeypatch, tmp_path,
             val_reports=[_VAL_1, {"iteration": 12}],
             train_reports=[_TRAIN_1], display=d)
        train_updates = [u for u in d.updates if u.get("loss") == 3.0]
        assert train_updates and all(
            u.get("val_loss") == pytest.approx(2.5) for u in train_updates
        ), "a payload without val_loss wiped the last real measurement"


class TestTelemetryNeverKillsTheRun:
    def test_a_failing_sse_push_does_not_abort_training(self, tmp_path, monkeypatch):
        import soup_cli.utils.train_event_buffer as buf

        def _boom(_e):
            raise RuntimeError("buffer exploded")

        monkeypatch.setattr(buf, "push_train_event", _boom)
        d = _RecordingDisplay()
        _run(monkeypatch, tmp_path, val_reports=[_VAL_1], display=d)
        assert any(u.get("val_loss") is not None for u in d.updates), (
            "an SSE failure took out the display update alongside it"
        )

    def test_a_failing_tracker_does_not_abort_training(self, tmp_path, monkeypatch):
        class _Boom:
            def log_metrics(self, **kwargs):
                raise RuntimeError("db is gone")

        d = _RecordingDisplay()
        _run(monkeypatch, tmp_path, val_reports=[_VAL_1], display=d, tracker=_Boom())
        assert any(u.get("val_loss") is not None for u in d.updates)


class TestTheValidationRowInventsNoTrainingNumbers:
    """@MakazhanAlpamys's second blocker on #739.

    `on_val_loss_report` issues its own `tracker.log_metrics(...)`, and
    `log_metrics` defaults `loss`, `lr` and `speed` to ``0.0`` -- not ``None``.
    So every evaluation appended a row whose three training columns were
    invented, and at a realistic cadence over half the `loss` series a consumer
    reads back was zeros no step ever measured:

        post-PR       [0.0, 0.0, 3.0, 0.0, 2.4, 0.0, 2.0, 0.0, 1.8]
        training truth      [3.0,      2.4,      2.0,      1.8]

    That is the same defect class #713 was blocked on, one column to the left,
    and `log_metrics`'s own docstring makes the argument: a zero there is
    indistinguishable from a genuinely measured zero.

    The fix mirrors the merged transformers producer, which carries
    `_last_loss` / `_last_lr` into the row an evaluation creates
    (`callback.py:205-207`). Measured rather than assumed -- driving the real
    `SoupTrainerCallback` against a real tracker with three training logs and
    one eval log writes **four** rows, the eval row carrying `loss=2.4`,
    `lr=1e-4` and its own `val_loss`. So an evaluation there does create a
    row; what it does not do is fill it with zeros.
    """

    def test_the_val_row_carries_the_last_measured_training_values(
        self, tmp_path, monkeypatch
    ):
        tracker = _RecordingTracker()
        # Training first, THEN the evaluation -- the sequence in which there is
        # a measured value to carry. (The default harness order is mlx-lm's
        # initial eval at it=0, where 0.0 is the honest answer because nothing
        # has been measured; that case is pinned below.)
        _run(monkeypatch, tmp_path,
             sequence=[("train", _TRAIN_1), ("val", _VAL_1)],
             display=_RecordingDisplay(), tracker=tracker)

        val_rows = [m for m in tracker.metrics if m.get("val_loss") is not None]
        assert len(val_rows) == 1, "expected exactly one row per evaluation"
        row = val_rows[0]
        assert row["loss"] == pytest.approx(3.0), (
            f"the validation row reports loss={row['loss']}, which no step "
            "measured; _TRAIN_1's 3.0 is the last real value"
        )
        assert row["lr"] == pytest.approx(1e-4)

    def test_a_training_row_still_reports_its_own_loss_not_a_carried_one(
        self, tmp_path, monkeypatch
    ):
        """Control: carrying must apply to the validation row only. If the
        carried value leaked onto training rows they would all report the same
        loss, which is the mirror-image fabrication."""
        tracker = _RecordingTracker()
        _run(monkeypatch, tmp_path, train_reports=[_TRAIN_1, _TRAIN_2],
             val_reports=[], display=_RecordingDisplay(), tracker=tracker)

        losses = [m["loss"] for m in tracker.metrics]
        assert losses == [pytest.approx(3.0), pytest.approx(2.0)], (
            f"training rows report {losses}; each must carry its own measurement"
        )

    def test_an_evaluation_before_any_training_step_carries_nothing(
        self, tmp_path, monkeypatch
    ):
        """The honest edge: mlx-lm evaluates at it=0, before a single training
        loss exists. 0.0 there is not a fabrication, it is the initial value,
        and the transformers callback reports the same at the same point
        (`callback.py:61-63`). Pinned so the carrying above cannot quietly
        start inventing a number for this row instead."""
        tracker = _RecordingTracker()
        _run(monkeypatch, tmp_path,
             sequence=[("val", _VAL_1), ("train", _TRAIN_1)],
             display=_RecordingDisplay(), tracker=tracker)

        val_rows = [m for m in tracker.metrics if m.get("val_loss") is not None]
        assert len(val_rows) == 1
        assert val_rows[0]["loss"] == pytest.approx(0.0)

    def test_the_carried_values_survive_a_run_with_no_display(
        self, tmp_path, monkeypatch
    ):
        """The third instance of the same trap, caught by mutating rather than
        by reading.

        `on_train_loss_report` returns at the `display is None` guard, so
        reading the training numbers after that line leaves them at 0.0 for
        exactly the configuration with no panel to notice -- a `soup train`
        without a live display would write a fabricated `loss` on every
        validation row while every test above stayed green.

        Moving the three assignments below the guard survives all fourteen
        other tests in this file; it fails here.
        """
        tracker = _RecordingTracker()
        _run(monkeypatch, tmp_path,
             sequence=[("train", _TRAIN_1), ("val", _VAL_1)],
             display=None, tracker=tracker)

        val_rows = [m for m in tracker.metrics if m.get("val_loss") is not None]
        assert len(val_rows) == 1, "the evaluation must still be recorded"
        assert val_rows[0]["loss"] == pytest.approx(3.0), (
            f"with no display the validation row reports "
            f"loss={val_rows[0]['loss']}; the carried value was never set"
        )


class TestTheFloatCoercionIsNotDecorative:
    """@MakazhanAlpamys's LOW item on #739: removing `_as_float` failed nothing.

    Today's mlx-lm returns a real float via `.item()`, so this is not a live
    bug -- but the guard's stated job is to keep an unserialisable object off
    the SSE wire, and nothing checked that it does. A value that is
    `float()`-able but not JSON-serialisable reaches `to_payload()` intact and
    `json.dumps` raises inside the training loop.
    """

    class _MxScalar:
        """Shaped like an mx scalar: converts to float, serialises to nothing."""

        def __init__(self, v):
            self._v = v

        def __float__(self):
            return float(self._v)

    def test_a_float_able_object_reaches_the_wire_as_a_real_float(
        self, tmp_path, monkeypatch
    ):
        import json

        import soup_cli.utils.train_event_buffer as buf
        from soup_cli.utils.sse_train_stream import format_sse_frame

        seen = []
        monkeypatch.setattr(buf, "push_train_event", lambda e: seen.append(e))
        _run(monkeypatch, tmp_path,
             sequence=[("val", {"iteration": 9, "val_loss": self._MxScalar(2.5),
                                "val_time": 0.4})],
             display=_RecordingDisplay())

        with_val = [e for e in seen if getattr(e, "val_loss", None) is not None]
        assert len(with_val) == 1, "the evaluation reached the wire"
        assert isinstance(with_val[0].val_loss, float), (
            f"val_loss is {type(with_val[0].val_loss).__name__}, not float; "
            "json.dumps will raise inside the training loop"
        )
        # The end of the pipe, not the middle: the frame must actually
        # serialise, which is the failure the guard exists to prevent.
        assert '"val_loss":2.5' in format_sse_frame(with_val[0]).replace(" ", "")
        json.dumps({"val_loss": with_val[0].val_loss})

    def test_an_unconvertible_value_is_dropped_rather_than_recorded(
        self, tmp_path, monkeypatch
    ):
        """Control: coercion must not turn junk into a measurement."""
        tracker = _RecordingTracker()
        _run(monkeypatch, tmp_path,
             sequence=[("val", {"iteration": 9, "val_loss": "not a number"})],
             display=_RecordingDisplay(), tracker=tracker)

        assert [m for m in tracker.metrics if m.get("val_loss") is not None] == []
