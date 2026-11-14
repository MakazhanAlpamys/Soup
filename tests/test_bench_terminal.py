"""Benchmark presentation preserves literal text and original measurement inputs."""

import builtins
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import bench as command
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

VALUES = [
    pytest.param("ordinary", "ordinary", id="plain-control"),
    pytest.param(":rocket:", ":rocket:", id="emoji-alias"),
    pytest.param("x[red]name", "x[red]name", id="markup"),
    pytest.param(r"x\[unfinished", r"x\[unfinished", id="backslash"),
    pytest.param("x\x1b\x07\x9b\u202e[end]", "x[end]", id="controls"),
]
FILE_VALUES = [
    pytest.param(*case.values, id=case.id,
                 marks=skip_colon_on_windows if ":" in case.values[0] else ())
    for case in VALUES
]


def _plain(text):
    text = strip_ansi(text)
    assert not any(c in text for c in ("\x1b", "\x07", "\x9b", "\u202e")), repr(text)
    return " ".join(text.split())


@pytest.fixture
def printed(tmp_path, monkeypatch):
    from soup_cli.commands import infer
    from soup_cli.utils import gpu

    monkeypatch.chdir(tmp_path)
    stream = io.StringIO()
    monkeypatch.setattr(command, "console", Console(
        file=stream, width=600, height=200, force_terminal=True,
        color_system="standard", no_color=False,
    ))
    monkeypatch.setattr(gpu, "detect_device", lambda **kwargs: ("cpu", 0))
    monkeypatch.setattr(infer, "_resolve_model_source", lambda model: ("local", model))
    monkeypatch.setattr(infer, "_load_model", lambda *a, **k: ("model", "tokenizer"))
    monkeypatch.setattr(infer, "_generate", lambda *a, **k: ("reply", 2))
    return stream


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("kind", ["resolve", "backend", "missing-prompts", "read", "load",
                                  "preflight", "postflight"])
def test_bench_infer_refusals_keep_literal_outside_text(
    printed, monkeypatch, value, shown, kind
):
    from soup_cli.commands import infer
    from soup_cli.utils import cuda_graphs

    args = ["bench", "infer", "model", "--backend", "transformers", "-n", "1"]
    expected = f"Failure {shown}"

    def fail(*args, **kwargs):
        error = FileNotFoundError if kind == "resolve" else RuntimeError
        raise error(f"Failure {value}")

    if kind == "resolve":
        monkeypatch.setattr(infer, "_resolve_model_source", fail)
    elif kind == "backend":
        args[args.index("--backend") + 1] = value
        expected = f"Unknown --backend: {shown}"
    elif kind == "missing-prompts":
        args += ["--prompts-file", f"missing-{value}.txt"]
        expected = f"Prompts file not found: missing-{shown}.txt"
    elif kind == "read":
        Path("prompts.txt").write_text("hello", encoding="utf-8")
        args += ["--prompts-file", "prompts.txt"]
        original_open = builtins.open

        def open_file(file, *args, **kwargs):
            if isinstance(file, (str, Path)) and Path(file).name == "prompts.txt":
                raise OSError(f"Failure {value}")
            return original_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", open_file)
        expected = f"Failed to read prompts file: Failure {shown}"
    elif kind == "load":
        monkeypatch.setattr(infer, "_load_model", fail)
        expected = f"Failed to load model: Failure {shown}"
    else:
        args += ["--cuda-graphs"]
        monkeypatch.setattr(cuda_graphs, "check_cuda_graphs_preflight",
                            fail if kind == "preflight" else lambda **kwargs: None)
        monkeypatch.setattr(cuda_graphs, "cuda_graph_generation_kwargs", fail)
        expected = " ".join(infer._with_omit_hint(f"Failure {shown}").split())
    result = CliRunner().invoke(app, args)
    assert result.exit_code == (2 if kind == "backend" else 1), result.exception
    assert expected in _plain(printed.getvalue())
    assert "\x1b[31m" in printed.getvalue()
    if kind == "resolve":
        assert "If you meant a HuggingFace repo" in _plain(printed.getvalue())


@pytest.mark.parametrize("value,shown", VALUES)
def test_bench_infer_plan_is_literal_and_preserves_hf_id_and_generation(
    printed, monkeypatch, value, shown
):
    from soup_cli.commands import infer

    loads, generations = [], []
    target = f"org/{value}"
    monkeypatch.setattr(infer, "_resolve_model_source", lambda model: ("hf", model))

    def load(model, base, device, **kwargs):
        loads.append((model, base, device, kwargs))
        return "model", "tokenizer"

    def generate(model, tokenizer, messages, **kwargs):
        generations.append((copy.deepcopy(messages), kwargs))
        return "reply", 2

    monkeypatch.setattr(infer, "_load_model", load)
    monkeypatch.setattr(infer, "_generate", generate)
    result = CliRunner().invoke(app, [
        "bench", "infer", target, "--base", value, "--backend", "transformers",
        "-n", "1", "--max-tokens", "8", "--p50", "--p95",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    text = _plain(printed.getvalue())
    # The existing advisory uses repr; controls already escaped by repr stay escaped.
    display_repr = repr(target).replace("\x9b", "").replace("\u202e", "")
    assert f"treating {display_repr} as a HF repo id." in text
    assert f"Model: org/{shown}" in text
    assert "Inference Benchmark Results" in text and "Per-prompt latency" in text
    assert loads == [(target, value, "cpu", {"is_local": False})]
    assert len(generations) == 2
    assert generations[0] == generations[1]
    assert generations[0][1] == {"max_tokens": 8, "temperature": 0.0}


def _report(value="ordinary", valid=True):
    return {"valid": valid, "steps_measured": 2,
            "timing": {"warmup_steps_discarded": 1, "median_seconds": .1,
                       "p95_seconds": .2}, "tokens": {"useful": 8, "total": 10},
            "throughput": {"useful_tokens_per_second": 40.0},
            "failures": [] if valid else [{"check": value, "message": f"Failure {value}"}]}


def _prepare_train(monkeypatch, report):
    from soup_cli.bench import train_run
    from soup_cli.config import loader
    from soup_cli.data import loader as data_loader

    Path("soup.yaml").write_bytes(b"# original config\n")
    cfg = SimpleNamespace(backend="transformers", data=object())
    calls = []
    monkeypatch.setattr(loader, "load_config", lambda path: cfg)
    monkeypatch.setattr(train_run, "bench_scope_error", lambda config: None)
    monkeypatch.setattr(data_loader, "load_dataset", lambda data: data)

    def run(config, **kwargs):
        assert config is cfg
        assert kwargs.pop("load_dataset")(config) is cfg.data
        calls.append(kwargs)
        return report

    monkeypatch.setattr(train_run, "run_bench_train", run)
    return calls


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("kind", ["output", "missing-config", "config", "scope", "run", "write"])
def test_bench_train_refusals_keep_literal_errors_and_do_not_write_report(
    printed, monkeypatch, value, shown, kind
):
    from soup_cli.bench import train_run
    from soup_cli.config import loader
    from soup_cli.utils import paths

    _prepare_train(monkeypatch, _report())

    def fail(*args, **kwargs):
        raise ValueError(f"Failure {value}")

    args = ["bench", "train", "--steps", "3", "--warmup", "1", "-o", "report.json"]
    if kind == "missing-config":
        args += ["--config", f"missing-{value}.yaml"]
        expected = f"Config not found: missing-{shown}.yaml"
    else:
        site = {"output": (paths, "check_report_output_path"), "config": (loader, "load_config"),
                "run": (train_run, "run_bench_train"), "write": (paths, "atomic_write_text")}
        if kind == "scope":
            monkeypatch.setattr(train_run, "bench_scope_error", lambda cfg: f"Failure {value}")
        else:
            module, name = site[kind]
            monkeypatch.setattr(module, name, fail)
        prefix = {"output": "Invalid output: ", "config": "Config invalid: ",
                  "scope": "Cannot bench this config: ", "run": "",
                  "write": "Cannot write the report: "}[kind]
        expected = prefix + f"Failure {shown}"
    result = CliRunner().invoke(app, args)
    assert result.exit_code == (2 if kind == "output" else 1), result.exception
    assert _plain(printed.getvalue()) == expected
    assert "\x1b[31m" in printed.getvalue()
    assert not Path("report.json").exists()
    assert Path("soup.yaml").read_bytes() == b"# original config\n"


@pytest.mark.parametrize("value,shown", VALUES)
def test_bench_train_failure_receipt_keeps_original_json_report(printed, monkeypatch, value, shown):
    report = _report(value, valid=False)
    calls = _prepare_train(monkeypatch, report)
    result = CliRunner().invoke(app, [
        "bench", "train", "--steps", "3", "--warmup", "1", "-o", "report.json",
    ])
    assert result.exit_code == 1, (result.output, result.exception)
    assert f"FAILED {shown}: Failure {shown}" in _plain(printed.getvalue())
    assert "NOT valid" in _plain(printed.getvalue())
    assert json.loads(Path("report.json").read_text()) == report
    assert calls == [{"steps": 3, "warmup": 1, "device": "cpu"}]


@pytest.mark.parametrize("value,shown", FILE_VALUES)
def test_bench_train_report_receipt_preserves_real_output_path(printed, monkeypatch, value, shown):
    report = _report()
    calls = _prepare_train(monkeypatch, report)
    name = "report-" + value.replace("\x1b", "").replace("\x07", "") + ".json"
    output = Path(name)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = CliRunner().invoke(app, [
        "bench", "train", "--steps", "3", "--warmup", "1", "-o", str(output),
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"Report: {Path('report-' + shown + '.json')}" in _plain(printed.getvalue())
    assert json.loads(output.read_text()) == report
    assert calls == [{"steps": 3, "warmup": 1, "device": "cpu"}]
