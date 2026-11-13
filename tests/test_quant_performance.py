"""Quality remains the gate when quant-check also measures latency and memory."""

from __future__ import annotations

import json
from io import StringIO

import pytest

from tests.conftest import strip_ansi


def _fake_factories(monkeypatch, *, after_wrong=0, variable=False, fail=False):
    from soup_cli.eval import quant_performance

    clock = [0.0]
    events = []
    monkeypatch.setattr(quant_performance.time, "perf_counter", lambda: clock[0])

    class Generator:
        device = "cpu"
        dtype = "float32"
        last_generated_tokens = 2

        def __init__(self, label):
            self.label = label
            self.calls = 0
            events.append(f"load:{label}")

        def __call__(self, prompt):
            self.calls += 1
            events.append(f"generate:{self.label}")
            seconds = 0.010 if self.label == "before" else 0.005
            if variable:
                seconds = (0.009, 0.011, 0.010)[(self.calls - 1) % 3]
            clock[0] += seconds
            if fail:
                raise RuntimeError("generation failed")
            return "wrong" if self.label == "after" and int(prompt) < after_wrong else "ok"

        def close(self):
            events.append(f"close:{self.label}")

    return lambda: Generator("before"), lambda: Generator("after"), events


def _tasks(count=10):
    from soup_cli.eval.custom import EvalTask

    return [EvalTask(prompt=str(i), expected="ok") for i in range(count)]


def test_ten_vs_five_ms_reports_half_latency_and_preserves_quality_gate(monkeypatch):
    from soup_cli.eval.quant_check import run_performance_quant_check

    before, after, events = _fake_factories(monkeypatch, after_wrong=1)
    result = run_performance_quant_check(
        before_factory=before, after_factory=after, tasks=_tasks(),
        device="cpu", warmup=1, repeats=3,
    )
    payload = json.loads(result.to_json())
    performance = payload["performance"]
    assert performance["latency_change_pct"] == pytest.approx(-50.0)
    assert performance["tokens_per_second_change_pct"] == pytest.approx(100.0)
    assert performance["before"]["latency_ms"] == pytest.approx(10.0)
    assert performance["after"]["latency_ms"] == pytest.approx(5.0)
    assert performance["before"]["generated_tokens"] == 60  # No warm-up tokens.
    assert performance["before"]["peak_gpu_memory_bytes"] is None
    assert performance["peak_gpu_memory_change_pct"] is None
    assert performance["latency_ranges_overlap"] is False
    assert result.rows[0].before == 1.0
    assert result.rows[0].after == 0.9
    assert result.rows[0].verdict == "MAJOR"
    assert events.index("close:before") < events.index("load:after")


def test_model_is_released_after_generation_failure(monkeypatch):
    from soup_cli.eval.quant_check import run_performance_quant_check

    before, after, events = _fake_factories(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="generation failed"):
        run_performance_quant_check(
            before_factory=before, after_factory=after, tasks=_tasks(),
            device="cpu", warmup=1, repeats=3,
        )
    assert events == ["load:before", "generate:before", "close:before"]


def test_no_generator_reference_survives_into_the_second_load(monkeypatch):
    import weakref

    from soup_cli.eval.quant_check import run_performance_quant_check

    before, after, _ = _fake_factories(monkeypatch)
    references = []

    def first():
        generator = before()
        references.append(weakref.ref(generator))
        return generator

    def second():
        assert references[0]() is None
        return after()

    run_performance_quant_check(
        before_factory=first, after_factory=second, tasks=_tasks(1), device="cpu",
    )


def test_cuda_measurement_resets_after_warmup_and_releases_before_next_load(monkeypatch):
    from contextlib import nullcontext

    import torch

    from soup_cli.eval.quant_check import run_performance_quant_check

    before, after, events = _fake_factories(monkeypatch)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(f"sync:{device}"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: events.append(f"reset:{device}"),
    )

    def peak(device):
        events.append(f"peak:{device}")
        return 8 * 1024 ** 2

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("release"))

    def factory(create):
        generator = create()
        generator.device = "cuda:1"
        return generator

    result = run_performance_quant_check(
        before_factory=lambda: factory(before), after_factory=lambda: factory(after),
        tasks=_tasks(1), device="cuda:1", warmup=1, repeats=1,
    )
    assert result.performance.before.peak_gpu_memory_bytes == 8 * 1024 ** 2
    assert result.performance.after.peak_gpu_memory_bytes == 8 * 1024 ** 2
    assert result.performance.before.generated_tokens == 2
    assert events == [
        "load:before", "generate:before", "sync:cuda:1", "reset:cuda:1",
        "sync:cuda:1", "generate:before", "sync:cuda:1", "peak:cuda:1",
        "close:before", "release",
        "load:after", "generate:after", "sync:cuda:1", "reset:cuda:1",
        "sync:cuda:1", "generate:after", "sync:cuda:1", "peak:cuda:1",
        "close:after", "release",
    ]


def test_failed_model_load_still_releases_memory_and_never_loads_second_model(monkeypatch):
    from soup_cli.eval import quant_performance
    from soup_cli.eval.quant_check import run_performance_quant_check

    events = []
    monkeypatch.setattr(
        quant_performance._MemoryProbe, "release", lambda self: events.append("free"),
    )

    def fail():
        raise OSError("bad checkpoint")

    with pytest.raises(OSError, match="bad checkpoint"):
        run_performance_quant_check(
            before_factory=fail, after_factory=lambda: pytest.fail("Must not load second model"),
            tasks=_tasks(1), device="cpu",
        )
    assert events == ["free"]


def test_only_first_measured_answer_is_scored(monkeypatch):
    from soup_cli.eval.quant_check import run_performance_quant_check

    before, _, _ = _fake_factories(monkeypatch)

    def factory():
        generator = before()

        class Alternating:
            device = "cpu"
            dtype = "float32"
            last_generated_tokens = 2

            def __call__(self, prompt):
                generator(prompt)
                # Call 1 is warm-up; call 2 is the first measured answer.
                return "ok" if generator.calls == 2 else "wrong"

            def close(self):
                generator.close()

        return Alternating()

    result = run_performance_quant_check(
        before_factory=factory, after_factory=factory, tasks=_tasks(1),
        device="cpu", warmup=1, repeats=3,
    )
    assert result.rows[0].before == result.rows[0].after == 1.0


def test_performance_renderers_include_units_and_noise_information(monkeypatch):
    from soup_cli.eval.quant_check import render_markdown, render_table, run_performance_quant_check

    before, after, _ = _fake_factories(monkeypatch, variable=True)
    result = run_performance_quant_check(
        before_factory=before, after_factory=after, tasks=_tasks(1),
        device="cpu", warmup=0, repeats=3,
    )
    markdown = render_markdown(result)
    assert "latency (ms)" in markdown
    assert "tokens/s" in markdown
    assert "CUDA allocated peak (MiB)" in markdown
    assert "N/A" in markdown
    assert "ranges overlap" in markdown
    assert "Quality alone" in render_table(result).caption


def test_overlapping_repeat_ranges_are_not_presented_as_a_clear_speed_change(monkeypatch):
    from soup_cli.eval.quant_check import run_performance_quant_check

    before, after, _ = _fake_factories(monkeypatch, variable=True)
    result = run_performance_quant_check(
        before_factory=before, after_factory=after, tasks=_tasks(1),
        device="cpu", warmup=0, repeats=3,
    )
    assert result.performance.latency_ranges_overlap is True


def test_single_repeat_cannot_estimate_noise(monkeypatch):
    from soup_cli.eval.quant_check import run_performance_quant_check

    before, after, _ = _fake_factories(monkeypatch)
    result = run_performance_quant_check(
        before_factory=before, after_factory=after, tasks=_tasks(1),
        device="cpu", warmup=0, repeats=1,
    )
    assert result.performance.latency_ranges_overlap is None


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"warmup": -1}, {"repeats": True}])
def test_bad_budgets_are_refused_before_model_loading(monkeypatch, kwargs):
    from soup_cli.eval.quant_check import run_performance_quant_check

    before, after, events = _fake_factories(monkeypatch)
    with pytest.raises(ValueError):
        run_performance_quant_check(
            before_factory=before, after_factory=after, tasks=_tasks(), device="cpu", **kwargs,
        )
    assert events == []


def test_empty_suite_is_refused_before_model_loading(monkeypatch):
    from soup_cli.eval.quant_check import run_performance_quant_check

    before, after, events = _fake_factories(monkeypatch)
    with pytest.raises(ValueError, match="No eval tasks"):
        run_performance_quant_check(
            before_factory=before, after_factory=after, tasks=[], device="cpu",
        )
    assert events == []


def test_cli_quality_drop_still_exits_two_with_performance(tmp_path, monkeypatch):
    from soup_cli.cli import app
    from soup_cli.eval import quant_check
    from tests.conftest import make_cli_runner

    monkeypatch.chdir(tmp_path)
    before, after, _ = _fake_factories(monkeypatch, after_wrong=1)
    monkeypatch.setattr(
        quant_check, "make_model_generator",
        lambda path, **kwargs: before() if path.endswith("before") else after(),
    )
    for side in ("before", "after"):
        (tmp_path / side).mkdir()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps({"prompt": str(i), "expected": "ok"}) + "\n"
                            for i in range(10)))
    result = make_cli_runner().invoke(app, [
        "eval", "quant-check", "--before", "before", "--after", "after",
        "--tasks", str(tasks), "--performance", "--device", "cpu", "--format", "json",
    ])
    assert result.exit_code == 2, (result.output, result.exception)
    payload = json.loads(result.output)
    assert payload["rows"][0]["verdict"] == "MAJOR"
    assert payload["performance"]["latency_change_pct"] == pytest.approx(-50.0)


def test_performance_refuses_stub_measurements_before_model_loading(monkeypatch):
    from soup_cli.cli import app
    from soup_cli.eval import quant_check
    from tests.conftest import make_cli_runner

    monkeypatch.setattr(
        quant_check, "make_model_generator", lambda *a, **kw: pytest.fail("Must not load a model"),
    )
    result = make_cli_runner().invoke(app, [
        "eval", "quant-check", "--before", "before", "--after", "after",
        "--tasks", "missing.jsonl", "--performance", "--allow-stub",
    ])
    assert result.exit_code == 3, (result.output, result.exception)
    output = " ".join(strip_ansi(result.output).split())
    assert "--performance cannot measure deterministic --allow-stub generators." in output


@pytest.mark.parametrize("device", ["tpu", "cuda:x", "cuda:"])
def test_unknown_performance_device_is_named_before_model_loading(tmp_path, monkeypatch, device):
    from soup_cli.cli import app
    from soup_cli.eval import quant_check
    from tests.conftest import make_cli_runner

    monkeypatch.chdir(tmp_path)
    for side in ("before", "after"):
        (tmp_path / side).mkdir()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"prompt": "question", "expected": "ok"}\n', encoding="utf-8")
    monkeypatch.setattr(
        quant_check, "make_model_generator", lambda *a, **kw: pytest.fail("Must not load a model"),
    )
    result = make_cli_runner().invoke(app, [
        "eval", "quant-check", "--before", "before", "--after", "after",
        "--tasks", str(tasks), "--performance", "--device", device,
    ])
    assert result.exit_code == 3, (result.output, result.exception)
    output = " ".join(strip_ansi(result.output).split())
    assert "--device must be auto, cpu, cuda, cuda:<index> or mps" in output


def test_performance_runtime_failure_exits_one_and_releases_model(tmp_path, monkeypatch):
    from soup_cli.cli import app
    from soup_cli.eval import quant_check
    from tests.conftest import make_cli_runner

    monkeypatch.chdir(tmp_path)
    before, after, events = _fake_factories(monkeypatch, fail=True)
    monkeypatch.setattr(
        quant_check, "make_model_generator",
        lambda path, **kwargs: before() if path.endswith("before") else after(),
    )
    for side in ("before", "after"):
        (tmp_path / side).mkdir()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"prompt": "0", "expected": "ok"}\n', encoding="utf-8")
    result = make_cli_runner().invoke(app, [
        "eval", "quant-check", "--before", "before", "--after", "after",
        "--tasks", str(tasks), "--performance", "--device", "cpu", "--format", "json",
    ])
    assert result.exit_code == 1, (result.output, result.exception)
    assert "generation failed" in strip_ansi(result.output)
    assert events == ["load:before", "generate:before", "close:before"]


@pytest.mark.parametrize("fmt", ["json", "markdown"])
def test_without_performance_preserves_legacy_long_line_wrapping(tmp_path, monkeypatch, fmt):
    from rich.console import Console

    from soup_cli.cli import app
    from soup_cli.commands import eval as eval_command
    from soup_cli.eval import quant_check
    from tests.conftest import make_cli_runner

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(eval_command, "console", Console(width=24, color_system=None))
    monkeypatch.setattr(quant_check, "make_model_generator", lambda *a, **kw: lambda _: "ok")
    for side in ("before", "after"):
        (tmp_path / side).mkdir()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"prompt": "question", "expected": "ok"}\n', encoding="utf-8")
    long_name = "long_model_name_" * 8
    rendered = (json.dumps({"model": long_name}) if fmt == "json"
                else f"| model | {long_name} |")
    monkeypatch.setattr(quant_check, "render", lambda result, **kw: rendered)
    expected = StringIO()
    # This is the pre-feature rendering contract, including its narrow-terminal wrap.
    Console(file=expected, width=24, color_system=None).print(
        rendered, markup=False, highlight=False,
    )
    assert expected.getvalue() != rendered + "\n"
    result = make_cli_runner().invoke(app, [
        "eval", "quant-check", "--before", "before", "--after", "after",
        "--tasks", str(tasks), "--format", fmt,
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    assert result.output == expected.getvalue()


@pytest.mark.parametrize("options", [
    ["--performance-warmup", "1"], ["--performance-repeats", "3"], ["--device", "cpu"],
])
def test_explicit_performance_options_require_the_flag_even_at_default_values(
    tmp_path, monkeypatch, options,
):
    from soup_cli.cli import app
    from soup_cli.eval import quant_check
    from tests.conftest import make_cli_runner

    monkeypatch.chdir(tmp_path)
    for side in ("before", "after"):
        (tmp_path / side).mkdir()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"prompt": "question", "expected": "ok"}\n', encoding="utf-8")
    monkeypatch.setattr(
        quant_check, "make_model_generator", lambda *a, **kw: pytest.fail("Must not load a model"),
    )
    result = make_cli_runner().invoke(app, [
        "eval", "quant-check", "--before", "before", "--after", "after",
        "--tasks", str(tasks), *options,
    ])
    assert result.exit_code == 3, (result.output, result.exception)
    output = " ".join(strip_ansi(result.output).split())
    assert "--device and performance budgets require --performance." in output
