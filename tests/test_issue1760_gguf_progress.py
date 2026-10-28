"""GGUF progress is live, and managed dependencies never invoke pip (#1760)."""

import importlib.util
import io
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import export as export_module
from tests.conftest import strip_ansi


def test_verbose_conversion_delivers_output_before_child_exit(monkeypatch, tmp_path):
    ack = tmp_path / "ack"
    script = tmp_path / "convert.py"
    script.write_text(
        "import pathlib, sys, time\n"
        "print('child-converter-started', flush=True)\n"
        f"ack = pathlib.Path({str(ack)!r})\n"
        "deadline = time.monotonic() + 5\n"
        "while not ack.exists() and time.monotonic() < deadline: time.sleep(.01)\n"
        "if not ack.exists(): sys.exit(7)\n"
        "print('[red]literal[/]\\x1b[2J', file=sys.stderr, flush=True)\n"
        "pathlib.Path(sys.argv[sys.argv.index('--outfile') + 1]).write_bytes(b'fixture')\n",
        encoding="utf-8",
    )
    output = io.StringIO()

    class AcknowledgingConsole(Console):
        def print(self, *objects, **kwargs):
            if "child-converter-started" in " ".join(str(obj) for obj in objects):
                ack.write_text("seen before exit")
            return super().print(*objects, **kwargs)

    monkeypatch.setattr(export_module, "console", AcknowledgingConsole(file=output, width=120))
    destination = tmp_path / "out.gguf"
    export_module._run_convert(script, tmp_path, destination, "f16", verbose=True)
    assert ack.exists()
    assert destination.read_bytes() == b"fixture"
    rendered = strip_ansi(output.getvalue())
    assert "[red]literal[/]" in rendered
    assert "\x1b" not in rendered


def test_verbose_quantization_streams_stdout_and_stderr(monkeypatch, tmp_path):
    # Python stands in for the binary; its first argument is an actual child script.
    source = tmp_path / "input.gguf"
    source.write_text(
        "import pathlib, sys\n"
        "print('quant-out', flush=True)\n"
        "print('quant-error-stream', file=sys.stderr, flush=True)\n"
        "assert sys.argv[2] == 'Q4_0'\n"
        "pathlib.Path(sys.argv[1]).write_bytes(b'quantized fixture')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(export_module, "_find_quantize_binary", lambda path: Path(sys.executable))
    output = io.StringIO()
    monkeypatch.setattr(export_module, "console", Console(file=output))
    destination = tmp_path / "out.gguf"
    export_module._run_quantize(tmp_path, source, destination, "q4_0", verbose=True)
    assert destination.read_bytes() == b"quantized fixture"
    rendered = strip_ansi(output.getvalue())
    assert "quant-out" in rendered
    assert "quant-error-stream" in rendered


@pytest.mark.parametrize("verbose", [False, True])
def test_conversion_failure_keeps_diagnostic_summary(monkeypatch, tmp_path, verbose):
    import typer

    script = tmp_path / "fail.py"
    payload = "distinctive failure [/x] [bold]b[/]\x1b[2J\x9b31m"
    script.write_text(f"import sys\nprint({payload!r}, file=sys.stderr)\nsys.exit(4)\n")
    output = io.StringIO()
    monkeypatch.setattr(export_module, "console", Console(file=output))
    with pytest.raises(typer.Exit) as exc:
        export_module._run_convert(script, tmp_path, tmp_path / "out.gguf", "f16", verbose=verbose)
    assert exc.value.exit_code == 1
    rendered = strip_ansi(output.getvalue())
    assert "Conversion failed" in rendered
    assert rendered.count("distinctive failure [/x] [bold]b[/][2J31m") == (2 if verbose else 1)
    assert "\x1b" not in rendered
    assert "\x9b" not in rendered


def test_default_conversion_success_stays_quiet(monkeypatch, tmp_path):
    script = tmp_path / "quiet.py"
    script.write_text("print('child-success-log')\n")
    output = io.StringIO()
    monkeypatch.setattr(export_module, "console", Console(file=output))
    export_module._run_convert(script, tmp_path, tmp_path / "out.gguf", "f16")
    assert "child-success-log" not in strip_ansi(output.getvalue())


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("LLAMA_CPP_PATH", raising=False)
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"llama","architectures":["LlamaForCausalLM"]}', encoding="utf-8",
    )
    llama = tmp_path / "llama.cpp"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("# fixture")
    return model, llama, tmp_path / "out.gguf"


def test_no_install_deps_missing_packages_refuses_before_merge_or_lookup(
    monkeypatch, cli_inputs,
):
    model, llama, destination = cli_inputs
    destination = destination.parent / "must-not-create" / "out.gguf"
    (model / "adapter_config.json").write_text('{"base_model_name_or_path":"unused"}')
    monkeypatch.setattr(export_module, "_missing_convert_deps", lambda: ["gguf", "protobuf"])

    def unexpected(*args, **kwargs):
        pytest.fail("managed missing deps must fail before expensive work or pip")

    monkeypatch.setattr(export_module, "_merge_adapter", unexpected)
    monkeypatch.setattr(export_module, "_find_llama_cpp", unexpected)
    monkeypatch.setattr(export_module.subprocess, "run", unexpected)
    result = CliRunner().invoke(app, ["export", "--model", str(model), "--output",
                                    str(destination), "--no-install-deps"])
    assert result.exit_code == 1, result.output
    output = strip_ansi(result.output)
    assert "gguf" in output and "protobuf" in output
    assert "--no-install-deps" in output
    assert not (model.parent / ".soup_merge_tmp_model").exists()
    assert not destination.parent.exists()


def test_no_install_deps_present_never_runs_pip_and_reports_phases(monkeypatch, cli_inputs):
    model, llama, destination = cli_inputs
    calls = []
    monkeypatch.setattr(export_module, "_missing_convert_deps", lambda: [])
    monkeypatch.setattr(export_module, "_find_llama_cpp", lambda path: llama)

    def convert(script, source, output, outtype, **kwargs):
        calls.append(("convert", kwargs))
        output.write_bytes(b"f16 fixture")

    def quantize(toolchain, source, output, quant, **kwargs):
        calls.append(("quantize", kwargs))
        output.write_bytes(b"q4 fixture")

    def unexpected(*args, **kwargs):
        pytest.fail("no pip call is allowed")

    monkeypatch.setattr(export_module.subprocess, "run", unexpected)
    monkeypatch.setattr(export_module, "_run_convert", convert)
    monkeypatch.setattr(export_module, "_run_quantize", quantize)
    result = CliRunner().invoke(app, ["export", "--model", str(model), "--output",
                                    str(destination), "--no-install-deps", "--verbose"])
    assert result.exit_code == 0, result.output
    assert destination.read_bytes() == b"q4 fixture"
    assert calls == [("convert", {"verbose": True}), ("quantize", {"verbose": True})]
    output = strip_ansi(result.output)
    assert "GGUF: toolchain" in output
    assert "GGUF: convert" in output
    assert "GGUF: quantize" in output
    assert not list(destination.parent.glob(".soup_gguf_*"))


@pytest.mark.parametrize("flag", ["--verbose", "--no-install-deps"])
def test_gguf_only_flags_are_not_silently_ignored(monkeypatch, cli_inputs, flag):
    model, _, destination = cli_inputs
    result = CliRunner().invoke(app, ["export", "--model", str(model), "--format", "onnx",
                                    "--output", str(destination), flag])
    assert result.exit_code == 2, result.output
    output = strip_ansi(result.output)
    assert "--format gguf" in output
    assert flag in output
    other = "--no-install-deps" if flag == "--verbose" else "--verbose"
    assert other not in output


def test_dependency_probe_names_all_missing_packages(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert export_module._missing_convert_deps() == ["gguf", "sentencepiece", "protobuf"]


def test_only_protobuf_missing_is_probed_and_triggers_the_pip_attempt(monkeypatch):
    real_find_spec = importlib.util.find_spec
    probed = []

    def find_spec(name, *args, **kwargs):
        probed.append(name)
        if name == "google.protobuf":
            return None
        return object()  # gguf and sentencepiece look installed

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert export_module._missing_convert_deps() == ["protobuf"]
    assert "google.protobuf" in probed
    assert export_module._convert_deps_present() is False

    runs = []
    monkeypatch.setattr(
        export_module.subprocess, "run",
        lambda cmd, *args, **kwargs: runs.append(cmd) or type("R", (), {"returncode": 0})(),
    )
    export_module._install_convert_deps()
    assert runs and "pip" in " ".join(map(str, runs[0]))
    assert real_find_spec is not find_spec


def test_stream_failure_keeps_only_a_bounded_tail(monkeypatch):
    import typer

    printed = []

    class OutputConsole:
        def print(self, text, **kwargs):
            printed.append(text)

    class Process:
        stdout = io.StringIO("".join(f"row-{index} " + "x" * 3000 + "\n" for index in range(110)))

        def wait(self):
            return 5

    monkeypatch.setattr(export_module, "console", OutputConsole())
    monkeypatch.setattr(export_module.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(typer.Exit):
        export_module._run_export_tool(["fixture"], "Conversion", verbose=True)
    assert len(printed[-1].splitlines()) == 100
    assert len(printed[-1]) <= 100 * 2000 + 99


def test_output_failure_terminates_the_child_and_closes_its_pipe(monkeypatch):
    actions = []

    class BrokenConsole:
        def print(self, text, **kwargs):
            raise OSError("output closed")

    class Process:
        stdout = io.StringIO("progress\n")

        def poll(self):
            return None

        def terminate(self):
            actions.append("terminate")

        def wait(self, **kwargs):
            actions.append("wait")

    process = Process()
    monkeypatch.setattr(export_module, "console", BrokenConsole())
    monkeypatch.setattr(export_module.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(OSError, match="output closed"):
        export_module._run_export_tool(["fixture"], "Conversion", verbose=True)
    assert actions == ["terminate", "wait"]
    assert process.stdout.closed


@pytest.mark.parametrize("quant", ["f16", "f32", "q4_0"])
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("adapter", [False, True])
def test_cli_reports_each_reached_phase_once(monkeypatch, cli_inputs, quant, managed, adapter):
    model, llama, destination = cli_inputs
    if adapter:
        (model / "adapter_config.json").write_text('{"base_model_name_or_path":"base"}')
    calls = []
    convert_options = []

    def deps(*, install=True):
        calls.append(("dependencies", install))

    def merge(source, base, output, trust_remote_code):
        calls.append(("merge", base))
        Path(output).mkdir()
        (Path(output) / "config.json").write_text(
            '{"architectures":["LlamaForCausalLM"]}', encoding="utf-8",
        )

    def convert(script, source, output, outtype, **kwargs):
        calls.append(("convert", outtype))
        convert_options.append(kwargs)
        output.write_bytes(b"converted fixture")

    def quantize(toolchain, source, output, quant, **kwargs):
        calls.append(("quantize", quant))
        output.write_bytes(b"quantized fixture")

    monkeypatch.setattr(export_module, "_install_convert_deps", deps)
    monkeypatch.setattr(export_module, "_merge_adapter", merge)
    monkeypatch.setattr(export_module, "_find_llama_cpp", lambda path: llama)
    monkeypatch.setattr(export_module, "_run_convert", convert)
    monkeypatch.setattr(export_module, "_run_quantize", quantize)
    flags = ["--no-install-deps"] if managed else []
    if quant in ("f16", "f32"):
        flags.append("--verbose")
    result = CliRunner().invoke(app, [
        "export", "--model", str(model), "--output", str(destination), "--quant", quant, *flags,
    ])
    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert output.count("GGUF: merge adapter") == int(adapter)
    assert output.count("GGUF: conversion dependencies") == int(not managed)
    assert output.count("GGUF: dependencies (managed environment)") == int(managed)
    assert output.count("GGUF: toolchain lookup") == 1
    outtype = "f16" if quant == "q4_0" else quant
    assert output.count(f"GGUF: convert — Converting to GGUF ({outtype})") == 1
    assert output.count("GGUF: quantize") == int(quant == "q4_0")
    assert ("dependencies", not managed) in calls
    assert ("convert", "f16" if quant == "q4_0" else quant) in calls
    assert convert_options == [{"verbose": True} if quant in ("f16", "f32") else {}]
    if quant == "q4_0":
        assert ("quantize", "q4_0") in calls
    else:
        assert all(name != "quantize" for name, _ in calls)
    assert not (model.parent / ".soup_merge_tmp_model").exists()
    assert not list(destination.parent.glob(".soup_gguf_*"))
