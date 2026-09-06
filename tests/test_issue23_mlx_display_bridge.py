"""#23 acceptance item 4 — the Rich callback bridge on the MLX path.

``MLXSFTTrainerWrapper.train()`` accepted ``display`` / ``tracker`` / ``run_id``
and opened with ``del display, tracker, run_id  # accepted for CLI-contract
parity``, so an MLX run showed mlx-lm's raw stdout while the identical config on
the transformers path got Soup's live dashboard.

This is an **adapter**, not a subclass, and the tests are written that way on
purpose: ``SoupTrainerCallback`` is a HuggingFace ``TrainerCallback`` that wants
``args, state, control``, while mlx-lm offers only
``on_train_loss_report(train_info)`` / ``on_val_loss_report(val_info)``.
Manufacturing fake HF state to reuse it would couple the MLX path to HF trainer
internals; the bridge drives ``TrainingDisplay`` directly instead.

Everything here runs the real ``train()`` body against the fake ``mlx`` /
``mlx_lm`` modules from ``test_issue634_mlx_resume`` — real control flow and
real call ordering, not source-text inspection. What that cannot cover, stated
rather than implied: real mlx-lm emitting these dicts at real cadence. That half
was checked by hand on an 8 GB M1 (see ``benchmarks/run-m1-8gb-mlx-sft.md``).
"""

from __future__ import annotations

import sys

import pytest

from tests.test_issue634_mlx_resume import (
    _FakeMlxModel,
    _install_fake_mlx,
    _mlx_wrapper,
)

# The exact keys mlx-lm builds at mlx_lm/tuner/trainer.py:354-362. Copied
# verbatim so a drift in the upstream dict shows up here as a failure rather
# than as a dashboard that silently stops updating.
_REPORT_1 = {
    "iteration": 5,
    "train_loss": 3.639,
    "learning_rate": 1e-4,
    "iterations_per_second": 0.432,
    "tokens_per_second": 19.192,
    "trained_tokens": 222,
    "peak_memory": 0.497,
}
_REPORT_2 = {
    "iteration": 10,
    "train_loss": 1.976,
    "learning_rate": 1e-4,
    "iterations_per_second": 5.481,
    "tokens_per_second": 254.313,
    "trained_tokens": 454,
    "peak_memory": 0.497,
}


class _RecordingDisplay:
    """Duck-types TrainingDisplay: start / update / stop."""

    def __init__(self):
        self.started_with = None
        self.updates = []
        self.stops = 0

    def start(self, total_steps):
        self.started_with = total_steps

    def update(self, step, epoch, loss, lr, **kwargs):
        self.updates.append(
            {"step": step, "epoch": epoch, "loss": loss, "lr": lr, **kwargs}
        )

    def stop(self):
        self.stops += 1


class _RecordingTracker:
    def __init__(self):
        self.metrics = []

    def log_metrics(self, **kwargs):
        self.metrics.append(kwargs)


def _train_with(monkeypatch, tmp_path, reports, *, display=None, tracker=None,
                run_id=None, raise_in_train=False):
    """Run the real train() body with a fake mlx-lm that emits ``reports``."""
    _install_fake_mlx(monkeypatch)

    def _fake_train(**kwargs):
        callback = kwargs.get("training_callback")
        if callback is not None:
            for report in reports:
                callback.on_train_loss_report(report)
        if raise_in_train:
            raise RuntimeError("mlx-lm blew up mid-training")

    sys.modules["mlx_lm.tuner.trainer"].train = _fake_train

    wrapper = _mlx_wrapper(tmp_path, epochs=1, lr=1e-4, batch_size=1)
    wrapper.model = _FakeMlxModel()
    wrapper.tokenizer = object()
    wrapper._dataset = {"train": [{"text": "hi"}] * 48, "val": []}
    return wrapper


class TestTheDisplayIsDriven:
    def test_start_receives_the_total_iteration_count(self, tmp_path, monkeypatch):
        """The progress bar needs the denominator train() already computes."""
        display = _RecordingDisplay()
        wrapper = _train_with(monkeypatch, tmp_path, [_REPORT_1], display=display)
        wrapper.train(display=display)

        assert display.started_with == 48, (
            "display.start() must get the iteration count train() derived from "
            "epochs x ceil(rows / batch_size), or the progress bar has no scale"
        )

    def test_update_maps_every_mlx_field_it_can(self, tmp_path, monkeypatch):
        """The mapping from mlx-lm's dict to the display's arguments."""
        display = _RecordingDisplay()
        wrapper = _train_with(
            monkeypatch, tmp_path, [_REPORT_1, _REPORT_2], display=display
        )
        wrapper.train(display=display)

        assert len(display.updates) == 2, "one update per mlx-lm loss report"
        first = display.updates[0]
        assert first["step"] == 5
        assert first["loss"] == pytest.approx(3.639)
        assert first["lr"] == pytest.approx(1e-4)
        assert first["speed"] == pytest.approx(0.432)  # it/s, NOT tokens/s
        assert "0.497" in str(first["gpu_mem"]), (
            f"peak_memory must reach the display; got {first['gpu_mem']!r}"
        )
        # 48 iterations over 1 epoch -> iteration 5 is ~10% of the way in.
        assert 0.0 < first["epoch"] <= 1.0

    def test_speed_is_iterations_per_second_not_tokens_per_second(
        self, tmp_path, monkeypatch
    ):
        """The display hard-labels this field ``it/s`` (``display.py:116``).

        Caught by rendering a real run, not by a stub: every other test in this
        file passed while the bridge fed ``tokens_per_second`` into a field the
        panel prints as ``it/s``, so the dashboard showed 19.19 it/s for a run
        doing 0.43. The transformers path feeds ``train_steps_per_second``, and
        this must match it or the two backends label the same row differently.
        """
        display = _RecordingDisplay()
        wrapper = _train_with(monkeypatch, tmp_path, [_REPORT_1], display=display)
        wrapper.train(display=display)

        assert display.updates[0]["speed"] == pytest.approx(
            _REPORT_1["iterations_per_second"]
        )
        assert display.updates[0]["speed"] != pytest.approx(
            _REPORT_1["tokens_per_second"]
        ), "tokens/s under an it/s label is a ~44x overstatement here"

    def test_stop_is_called_when_training_finishes(self, tmp_path, monkeypatch):
        display = _RecordingDisplay()
        wrapper = _train_with(monkeypatch, tmp_path, [_REPORT_1], display=display)
        wrapper.train(display=display)

        assert display.stops == 1

    def test_stop_is_called_even_when_training_raises(self, tmp_path, monkeypatch):
        """A live Rich display left running would corrupt the terminal.

        This is the discriminating test: a bridge that calls stop() on the
        happy path only passes every other test in this file.
        """
        display = _RecordingDisplay()
        wrapper = _train_with(
            monkeypatch, tmp_path, [_REPORT_1], display=display, raise_in_train=True
        )
        with pytest.raises(RuntimeError, match="blew up"):
            wrapper.train(display=display)

        assert display.stops == 1, (
            "stop() must run in a finally: an exception mid-train would otherwise "
            "leave the Live display attached to the terminal"
        )


class TestTheTrackerIsDriven:
    def test_tracker_receives_the_metrics(self, tmp_path, monkeypatch):
        display = _RecordingDisplay()
        tracker = _RecordingTracker()
        wrapper = _train_with(
            monkeypatch, tmp_path, [_REPORT_1, _REPORT_2],
            display=display, tracker=tracker, run_id="run-1",
        )
        wrapper.train(display=display, tracker=tracker, run_id="run-1")

        assert len(tracker.metrics) == 2
        assert tracker.metrics[0]["run_id"] == "run-1"
        assert tracker.metrics[0]["step"] == 5
        assert tracker.metrics[0]["loss"] == pytest.approx(3.639)

    def test_tracker_without_run_id_is_not_called(self, tmp_path, monkeypatch):
        """Mirrors the HF path, which gates on ``self.tracker and self.run_id``."""
        display = _RecordingDisplay()
        tracker = _RecordingTracker()
        wrapper = _train_with(
            monkeypatch, tmp_path, [_REPORT_1], display=display, tracker=tracker
        )
        wrapper.train(display=display, tracker=tracker, run_id=None)

        assert tracker.metrics == []


class TestTheWebUiBufferIsFed:
    """`soup ui` / GET /api/train/stream must see an MLX run too.

    Review finding on #665: the terminal panel worked while the Web UI stayed
    blank, because `push_train_event` is called from exactly one place in the
    tree (`monitoring/callback.py`) and the MLX bridge was not one of them.
    """

    def _capture(self, monkeypatch):
        pushed = []
        import soup_cli.utils.train_event_buffer as buf

        monkeypatch.setattr(buf, "push_train_event", lambda event: pushed.append(event))
        return pushed

    def test_each_report_pushes_a_metric_event(self, tmp_path, monkeypatch):
        pushed = self._capture(monkeypatch)
        display = _RecordingDisplay()
        wrapper = _train_with(
            monkeypatch, tmp_path, [_REPORT_1, _REPORT_2], display=display
        )
        wrapper.train(display=display)

        assert len(pushed) == 2, "one SSE event per mlx-lm loss report"
        assert pushed[0].type == "metric"
        assert pushed[0].step == 5
        assert pushed[0].loss == pytest.approx(3.639)
        assert pushed[0].lr == pytest.approx(1e-4)
        assert pushed[0].grad_norm is None, (
            "mlx-lm computes no gradient norm; the event must carry None rather "
            "than a plausible 0.0 — same contract the display holds"
        )

    def test_a_failing_push_does_not_kill_training(self, tmp_path, monkeypatch):
        """Best-effort, exactly as the transformers path treats it.

        This is the discriminating case: a bridge that pushes without guarding
        passes every other test here and then takes down a real run the first
        time the buffer is unavailable.
        """
        import soup_cli.utils.train_event_buffer as buf

        def _boom(event):
            raise RuntimeError("SSE buffer unavailable")

        monkeypatch.setattr(buf, "push_train_event", _boom)
        display = _RecordingDisplay()
        wrapper = _train_with(monkeypatch, tmp_path, [_REPORT_1], display=display)

        result = wrapper.train(display=display)

        assert result["total_steps"] == 48
        assert display.stops == 1, "the Live must still be stopped"

    def test_no_display_means_no_push__matching_the_transformers_path(
        self, tmp_path, monkeypatch
    ):
        """Deliberately NOT pushing when no display is attached.

        I first wrote this test asserting the opposite — that the Web UI should
        get events regardless — and it failed, correctly. `trainer/sft.py:1865`
        attaches `SoupTrainerCallback` only `if display:`, so the transformers
        path pushes nothing without one either. Matching that is the point: two
        backends should not disagree about when `/api/train/stream` goes quiet.

        If the project later decides the Web UI should be fed headlessly, that
        is one change in both paths, and this test is where it gets noticed.
        """
        pushed = self._capture(monkeypatch)
        wrapper = _train_with(monkeypatch, tmp_path, [_REPORT_1])
        wrapper.train()

        assert pushed == []


class TestTheControls:
    def test_training_without_a_display_still_works(self, tmp_path, monkeypatch):
        """The reject-everything control: display=None is the CLI's default."""
        wrapper = _train_with(monkeypatch, tmp_path, [_REPORT_1])
        result = wrapper.train()

        assert result["total_steps"] == 48
        assert result["final_loss"] == pytest.approx(3.639)

    def test_losses_are_still_captured_in_the_result(self, tmp_path, monkeypatch):
        """The bridge must not displace the existing loss capture."""
        display = _RecordingDisplay()
        wrapper = _train_with(
            monkeypatch, tmp_path, [_REPORT_1, _REPORT_2], display=display
        )
        result = wrapper.train(display=display)

        assert result["initial_loss"] == pytest.approx(3.639)
        assert result["final_loss"] == pytest.approx(1.976)

    def test_grad_norm_is_not_fabricated(self, tmp_path, monkeypatch):
        """mlx-lm reports no grad_norm; the bridge must not invent one.

        A dashboard field reading a plausible 0.0 on one backend and a real
        value on another is worse than one that is obviously absent.
        """
        display = _RecordingDisplay()
        wrapper = _train_with(monkeypatch, tmp_path, [_REPORT_1], display=display)
        wrapper.train(display=display)

        assert "grad_norm" not in _REPORT_1, "upstream shape assumption"
        # ABSENT, not 0.0. An earlier version of this assertion allowed
        # `in (None, 0.0)` while the docstring above called a plausible 0.0
        # worse than an absent field -- so the contract this test is most
        # vocal about was the one thing it did not pin. Caught in review by
        # @MakazhanAlpamys, who killed it by injecting grad_norm=0.0 and
        # watching all ten tests still pass.
        assert "grad_norm" not in display.updates[0], (
            "grad_norm must be absent, not a plausible 0.0: mlx-lm computes no "
            "gradient norm, and a field reading 0.0 on this backend while "
            "carrying a real value on another is indistinguishable from a "
            "measurement"
        )
