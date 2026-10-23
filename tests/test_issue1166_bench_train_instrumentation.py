"""#1166 — three instrumentation gaps the gate-836 record found in ``soup bench train``.

1. The SM clock was one ``nvidia-smi`` read after training returned, on an idle
   card; it is now sampled during the counted steps.
2. ``timing`` carried only median and p95; it now carries every counted step.
3. The docs said the report is written for every failure. Pre-flight refusals
   write nothing; only post-run checks leave a ``valid: false`` report.

No GPU in CI starts the default sampler, so the real run is driven on CPU with
an injected one: ``run_bench_train(clock_sampler=...)``.
"""

from __future__ import annotations

import time

import pytest

from soup_cli.bench.collector import BenchCollector
from soup_cli.bench.train_report import summarize_clock_samples, summarize_step_times
from tests import test_issue836_bench_train_cli as cli_tests
from tests.conftest import strip_ansi

workdir = cli_tests.workdir
runner = cli_tests.runner


def _run(clock_sampler, *, steps=6, warmup=3, before_train=None):
    from soup_cli.bench.train_run import run_bench_train
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset

    return run_bench_train(
        load_config("soup.yaml"), steps=steps, warmup=warmup, device="cpu",
        load_dataset=lambda c: load_dataset(c.data), before_train=before_train,
        clock_sampler=clock_sampler,
    )


def _busy(report):
    return report["provenance"]["sm_clock_mhz_busy"]


class _Canned:
    """A sampler that ran and returns fixed samples."""

    available = True

    def __init__(self, samples):
        self._samples = samples

    def start(self):
        return self

    def stop(self):
        return list(self._samples)


class TestTheClockIsReadWhileBusy:
    # Warm-up ramps at t<10, the counted steps run 10..20, the card idles after.
    SAMPLES = [(1.0, 300), (5.0, 1400), (10.0, 2400), (12.0, 2550), (15.0, 2370),
               (20.0, 2500), (21.0, 285), (25.0, 180)]

    def test_only_the_counted_window_is_summarised(self):
        summary = summarize_clock_samples(self.SAMPLES, 10.0, 20.0)
        assert summary == {
            "min": 2370, "median": 2450.0, "max": 2550, "sample_count": 4,
            "unavailable_reason": None,
        }

    def test_a_busy_reading_is_not_the_idle_one(self):
        """The old field was the last sample; the gate record measured it at
        mostly 180 MHz and never above 1417, against a busy median of 2370-2557."""
        idle_after_run = self.SAMPLES[-1][1]
        busy = summarize_clock_samples(self.SAMPLES, 10.0, 20.0)
        assert busy["min"] > idle_after_run * 5

    def test_samples_all_outside_the_window_say_so(self):
        summary = summarize_clock_samples(self.SAMPLES, 100.0, 200.0)
        assert (summary["min"], summary["median"], summary["max"]) == (None, None, None)
        assert summary["sample_count"] == 0
        assert summary["unavailable_reason"] == "no sample fell inside the counted steps"
        # No window start at all is no sample, not a TypeError.
        assert summarize_clock_samples(self.SAMPLES, None)["sample_count"] == 0

    def test_no_readable_sample_says_so(self):
        summary = summarize_clock_samples([], 10.0, 20.0)
        assert summary["unavailable_reason"] == "nvidia-smi returned no readable SM clock"

    def test_the_callers_reason_wins(self):
        summary = summarize_clock_samples([], None, unavailable_reason="no nvidia-smi tool found")
        assert summary["unavailable_reason"] == "no nvidia-smi tool found"

    def test_the_collector_marks_the_counted_window(self):
        clock = iter(float(i) for i in range(10))
        collector = BenchCollector(warmup_steps=2)
        collector._now = lambda: next(clock)
        for _ in range(4):
            collector.on_step_begin(None, None, None)
            collector.on_step_end(None, None, None)
        # Steps end at 1, 2, 3, 4: warm-up is 0..2, the counted window 2..4.
        assert (collector.counted_window_started, collector.counted_window_ended) == (2.0, 4.0)

    def test_without_warm_up_the_window_opens_at_the_first_step(self):
        clock = iter(float(i) for i in range(10))
        collector = BenchCollector(warmup_steps=0)
        collector._now = lambda: next(clock)
        for _ in range(2):
            collector.on_step_begin(None, None, None)
            collector.on_step_end(None, None, None)
        assert (collector.counted_window_started, collector.counted_window_ended) == (0.0, 2.0)


class TestTheRunCutsSamplesToTheCountedSteps:
    def test_the_window_is_exactly_the_counted_steps(self, workdir):
        """The stub reads 1000 + steps finished so far, so every sample says
        which step it was taken in. Steps are slowed so each is sampled several
        times; the report must count exactly the samples from the counted steps.
        Fails if the warm-up is shifted by one step either way, or if the run
        passes anything but the collector's window."""
        from soup_cli.bench.train_run import ClockSampler

        seen, made = {}, {}

        def slow_steps_and_grab_the_collector(wrapper):
            seen["collector"] = next(
                cb for cb in wrapper.trainer.callback_handler.callbacks
                if type(cb).__name__ == "BenchCollector"
            )
            step = wrapper.trainer.training_step

            def slow(*args, **kwargs):
                time.sleep(0.05)  # releases the GIL: several reads land in every step
                return step(*args, **kwargs)

            wrapper.trainer.training_step = slow

        def read():
            collector = seen.get("collector")
            return str(1000 + len(collector.steps)) if collector else None

        def factory():
            made["sampler"] = ClockSampler(interval=0.001, read=read)
            return made["sampler"]

        report = _run(factory, before_train=slow_steps_and_grab_the_collector)
        values = [clock for _, clock in made["sampler"].samples]
        # 1000+k is read during step k+1; steps=6, warmup=3 -> counted steps read 1003..1005.
        expected = sum(1 for clock in values if 1003 <= clock <= 1005)
        busy = _busy(report)
        assert expected >= 6, "guard: every counted step was sampled more than once"
        # A read can land between a step's end stamp and its append: one per edge.
        assert abs(busy["sample_count"] - expected) <= 2, (busy, expected)
        assert busy["unavailable_reason"] is None

    def test_no_tool_says_so(self, workdir, monkeypatch):
        from soup_cli.bench.train_run import ClockSampler

        monkeypatch.setattr("soup_cli.utils.layer_stream._resolve_tool", lambda *_a: None)
        assert _busy(_run(ClockSampler))["unavailable_reason"] == "no nvidia-smi tool found"

    def test_a_tool_that_never_reads_a_clock_says_so(self, workdir):
        from soup_cli.bench.train_run import ClockSampler

        busy = _busy(_run(lambda: ClockSampler(interval=0.001, read=lambda: "[N/A]")))
        assert busy["sample_count"] == 0
        assert busy["unavailable_reason"] == "nvidia-smi returned no readable SM clock"

    def test_samples_that_miss_the_window_say_so(self, workdir):
        busy = _busy(_run(lambda: _Canned([(0.0, 1500)])))
        assert busy["sample_count"] == 0
        assert busy["unavailable_reason"] == "no sample fell inside the counted steps"

    def test_a_training_error_still_stops_the_sampler(self, workdir):
        from soup_cli.bench.train_run import ClockSampler

        sampler = ClockSampler(interval=0.001, read=lambda: "1500")

        def break_training(wrapper):
            def boom():
                raise RuntimeError("training broke")

            wrapper.trainer.train = boom

        with pytest.raises(RuntimeError, match="training broke"):
            _run(lambda: sampler, before_train=break_training)
        assert not sampler._thread.is_alive()


class TestTheSampler:
    def test_it_reads_the_clock_through_the_resolved_tool(self, tmp_path, monkeypatch):
        from soup_cli.bench.train_run import ClockSampler

        script = cli_tests.TestDriverAndClockProvenance._fake_tool(
            tmp_path, monkeypatch, "2505\n"
        )
        # stop() waits for the tick in flight, so at least one always lands.
        samples = ClockSampler(interval=0.01, gpu="00000000:01:00.0").start().stop()
        assert samples and {clock for _, clock in samples} == {2505}
        asked = (tmp_path / (script.name + ".args")).read_text(encoding="utf-8")
        assert "clocks.sm" in asked and "memory" not in asked
        assert "--id=00000000:01:00.0" in asked

    def test_a_failing_tool_is_no_sample(self, tmp_path, monkeypatch):
        from soup_cli.bench.train_run import ClockSampler

        cli_tests.TestDriverAndClockProvenance._fake_tool(tmp_path, monkeypatch, "2505\n", 9)
        assert ClockSampler(interval=0.01).start().stop() == []

    def test_an_unreadable_tick_is_skipped_and_sampling_goes_on(self):
        from soup_cli.bench.train_run import ClockSampler

        values = iter(["[N/A]", "[N/A]", "[N/A]"])
        sampler = ClockSampler(interval=0.001, read=lambda: next(values, "2505")).start()
        deadline = time.monotonic() + 5
        while not sampler.samples and time.monotonic() < deadline:
            time.sleep(0.01)
        assert {clock for _, clock in sampler.stop()} == {2505}

    def test_a_sample_is_stamped_mid_query(self):
        """Stamped after the query returned, a slow read would be attributed
        across a window edge."""
        from soup_cli.bench.train_run import ClockSampler

        inside = []

        def slow_read():
            entered = time.perf_counter()
            time.sleep(0.1)
            inside.append((entered, time.perf_counter()))
            return "2505"

        (stamp, _), *_ = ClockSampler(interval=1.0, read=slow_read).start().stop()
        # Bounded by the read's own entry and exit, not by the sleep's length:
        # a start stamp lands before ``entered``, an end stamp after ``left``.
        # (A fixed 90 ms bound failed on a macOS runner whose sleep took 114 ms.)
        entered, left = inside[0]
        assert entered < stamp < left

    def test_without_a_tool_nothing_starts(self, monkeypatch):
        from soup_cli.bench.train_run import ClockSampler

        monkeypatch.setattr("soup_cli.utils.layer_stream._resolve_tool", lambda *_a: None)
        sampler = ClockSampler()
        assert sampler.start().stop() == [] and not sampler.available

    def test_torchs_device_is_named_by_pci_bus_id(self):
        from types import SimpleNamespace

        from soup_cli.bench.train_run import _pci_bus_id

        props = SimpleNamespace(pci_domain_id=0, pci_bus_id=1, pci_device_id=0)
        assert _pci_bus_id(props) == "00000000:01:00.0"
        assert _pci_bus_id(SimpleNamespace()) is None


class TestEveryCountedStepIsKept:
    def test_step_seconds_are_the_counted_steps_in_run_order(self):
        """Slow mode first, so a sorted copy would not pass: median and p95
        show a mode change happened, only the sequence shows where."""
        times = [9.0, 9.0] + [2.0] * 5 + [1.0] * 5
        summary = summarize_step_times(times, warmup_steps=2)
        assert summary["step_seconds"] == [2.0] * 5 + [1.0] * 5
        assert len(summary["step_seconds"]) == summary["counted_steps"]


class TestAPreFlightRefusalWritesNoReport:
    def test_zero_trainable_parameters(self, workdir, monkeypatch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        setup = SFTTrainerWrapper.setup

        def setup_then_freeze(self, dataset):
            setup(self, dataset)
            for param in self.trainer.model.parameters():
                param.requires_grad_(False)

        monkeypatch.setattr(SFTTrainerWrapper, "setup", setup_then_freeze)
        result = runner.invoke(
            cli_tests.app, ["bench", "train", "--config", "soup.yaml", "--steps", "3",
                            "--warmup", "1", "-o", "r.json"],
        )
        assert result.exit_code == 1, result.output
        assert "0 trainable parameter tensors" in strip_ansi(result.output)
        assert not (workdir / "r.json").exists()

    def test_warmup_that_eats_every_step(self, workdir):
        result = runner.invoke(
            cli_tests.app, ["bench", "train", "--config", "soup.yaml", "--steps", "2",
                            "--warmup", "2", "-o", "r.json"],
        )
        assert result.exit_code == 1, result.output
        assert "leaves nothing after --warmup" in strip_ansi(result.output)
        assert not (workdir / "r.json").exists()

    def test_the_control_trains_and_writes_a_report(self, workdir):
        """The control for the two refusals above: the same command with enough
        steps trains and writes a report. On CPU its clock says why it is empty."""
        import json

        result = runner.invoke(
            cli_tests.app, ["bench", "train", "--config", "soup.yaml", "--steps", "2",
                            "--warmup", "1", "-o", "r.json"],
        )
        assert result.exit_code == 0, result.output
        report = json.loads((workdir / "r.json").read_text(encoding="utf-8"))
        assert _busy(report)["unavailable_reason"] == "not a CUDA run"
