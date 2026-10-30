"""#1911: diagnose converter capability before expensive GGUF work."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import export as export_mod
from tests.conftest import strip_ansi


def _converter(path: Path, text: tuple[str, ...], *, vision=(), failed=False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    output = "Supported models:\nERROR:hf-to-gguf:TEXT models:\n"
    output += "".join(f"ERROR:hf-to-gguf:  - {name}\n" for name in text)
    output += "ERROR:hf-to-gguf:VISION models:\n"
    output += "".join(f"ERROR:hf-to-gguf:  - {name}\n" for name in vision)
    (path / "convert_hf_to_gguf.py").write_text(
        "import sys\n"
        f"sys.stderr.write({output!r})\n"
        f"sys.exit({1 if failed else 0})\n", encoding="utf-8",
    )


@pytest.fixture
def export_flow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLAMA_CPP_PATH", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    calls = []
    unexpected = []
    original_run = subprocess.run
    # Every program lookup answers with a stand-in path, so the only commands
    # that reach ``run`` are the ones this fixture knows.
    monkeypatch.setattr("soup_cli.utils.tool_paths.find_tool",
                        lambda name, path=None: "/fake/" + name)

    def run(command, **kwargs):
        if list(command[:2]) == ["/fake/git", "clone"]:
            calls.append("clone")
            _converter(Path(command[-1]), ("LlamaForCausalLM",))
            return subprocess.CompletedProcess(command, 0, "", "")
        if (len(command) == 3 and command[0] == sys.executable
                and command[2] == "--print-supported-models"):
            # The synthetic converter this fixture wrote: a local script.
            calls.append("probe")
            return original_run(command, **kwargs)
        unexpected.append(command)
        raise AssertionError(f"export_flow: unexpected command {command!r}")

    def convert(script, model, output, outtype, **kwargs):
        calls.append("convert")
        output.write_bytes(b"synthetic conversion")

    def quantize(llama, source, output, quant, **kwargs):
        calls.append("quantize")
        output.write_bytes(b"synthetic quantization")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(export_mod, "_install_convert_deps", lambda **kwargs: calls.append("deps"))
    monkeypatch.setattr(export_mod, "_run_convert", convert)
    monkeypatch.setattr(export_mod, "_run_quantize", quantize)
    monkeypatch.setattr(export_mod, "_maybe_attach_export", lambda **kwargs: None)

    def invoke(config, *, selection=None, converter_text=("Qwen3_5ForCausalLM",),
               converter_vision=(), failed=False, managed=False, adapter=False):
        model = tmp_path / "model"
        model.mkdir(exist_ok=True)
        if config is not None and not adapter:
            (model / "config.json").write_text(
                config if isinstance(config, str) else json.dumps(config), encoding="utf-8",
            )
        if adapter:
            (model / "adapter_config.json").write_text(
                '{"base_model_name_or_path":"synthetic-base"}', encoding="utf-8",
            )

            def merge(source, base, output, trust_remote_code=False):
                calls.append("merge")
                Path(output).mkdir()
                (Path(output) / "config.json").write_text(json.dumps(config), encoding="utf-8")

            monkeypatch.setattr(export_mod, "_merge_adapter", merge)
        arguments = ["export", "--model", str(model), "--output", str(tmp_path / "out.gguf")]
        if managed:
            arguments.append("--no-install-deps")
        if selection:
            converter_dir = (home / ".soup" / "llama.cpp" if selection == "cached" else
                             tmp_path / "new-llama.cpp")
            _converter(converter_dir, converter_text, vision=converter_vision, failed=failed)
            if selection == "explicit":
                arguments += ["--llama-cpp", str(converter_dir)]
            elif selection == "env":
                monkeypatch.setenv("LLAMA_CPP_PATH", str(converter_dir))
        result = CliRunner().invoke(app, arguments)
        # Rich may emit per-character style sequences under FORCE_COLOR=1.
        return SimpleNamespace(exit_code=result.exit_code, output=strip_ansi(result.output),
                               exception=result.exception), calls

    yield invoke
    # A test that expects a refusal must not pass because a command was refused here.
    assert unexpected == [], unexpected


@pytest.mark.parametrize("managed", [False, True])
def test_unsupported_fresh_default_refuses_before_any_work(export_flow, managed):
    result, calls = export_flow({"architectures": ["Qwen3_5ForCausalLM"]}, managed=managed)
    assert result.exit_code == 1, result.output
    assert "Qwen3_5ForCausalLM" in result.output
    assert "b5270" in result.output
    assert "--llama-cpp" in result.output
    assert calls == []


def test_supported_fresh_default_still_exports(export_flow):
    result, calls = export_flow({"architectures": ["LlamaForCausalLM"]})
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["clone", "deps", "convert", "quantize"]


@pytest.mark.parametrize("architecture,allowed", [
    ("LlamaForCausalLM", True), ("Qwen3_5ForCausalLM", False),
])
def test_adapter_checks_merged_metadata_and_cleans_up(export_flow, tmp_path, architecture, allowed):
    result, calls = export_flow({"architectures": [architecture]}, adapter=True)
    assert result.exit_code == (0 if allowed else 1), (result.output, result.exception)
    assert calls == (["merge", "clone", "deps", "convert", "quantize"] if allowed else ["merge"])
    assert not (tmp_path / ".soup_merge_tmp_model").exists()


def test_managed_missing_dependencies_refuse_before_running_selected_converter(
    export_flow, monkeypatch,
):
    calls = []

    def missing(*, install=True):
        import typer

        assert install is False
        calls.append("managed refusal")
        raise typer.Exit(1)

    monkeypatch.setattr(export_mod, "_install_convert_deps", missing)
    result, work = export_flow({"architectures": ["LlamaForCausalLM"]}, selection="explicit",
                               converter_text=("LlamaForCausalLM",), managed=True)
    assert result.exit_code == 1
    assert work == []
    assert calls == ["managed refusal"]


@pytest.mark.parametrize("selection", ["explicit", "env", "cached"])
def test_selected_newer_converter_uses_its_own_capabilities(export_flow, selection):
    result, calls = export_flow({"architectures": ["Qwen3_5ForCausalLM"]}, selection=selection)
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["probe", "deps", "convert", "quantize"]


@pytest.mark.parametrize("failed", [False, True])
def test_unavailable_capability_probe_is_inconclusive(export_flow, failed):
    result, calls = export_flow({"architectures": ["Qwen3_5ForCausalLM"]}, selection="explicit",
                               converter_text=(), failed=failed)
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["probe", "deps", "convert", "quantize"]


def test_vision_support_does_not_imply_text_support(export_flow):
    result, calls = export_flow({"architectures": ["Qwen3_5ForCausalLM"]}, selection="explicit",
                               converter_text=("LlamaForCausalLM",),
                               converter_vision=("Qwen3_5ForCausalLM",))
    assert result.exit_code == 1, result.output
    assert "Qwen3_5ForCausalLM" in result.output
    assert "--llama-cpp" in result.output
    assert calls == ["probe"]


@pytest.mark.parametrize("root,nested,allowed", [
    ("Qwen3_5ForCausalLM", "LlamaForCausalLM", True),
    ("LlamaForCausalLM", "Qwen3_5ForCausalLM", False),
])
def test_nested_text_architecture_takes_precedence(export_flow, root, nested, allowed):
    result, calls = export_flow({"architectures": [root],
                                "text_config": {"architectures": [nested]}})
    assert result.exit_code == (0 if allowed else 1), result.output
    assert calls == (["clone", "deps", "convert", "quantize"] if allowed else [])


def test_null_nested_architectures_keeps_root_architecture(export_flow):
    result, _ = export_flow({"architectures": ["LlamaForCausalLM"],
                             "text_config": {"architectures": None}})
    assert result.exit_code == 0, (result.output, result.exception)


@pytest.mark.parametrize("config", [
    None, "{bad", [], {}, {"architectures": []}, {"architectures": "LlamaForCausalLM"},
    {"architectures": [None]}, {"architectures": ["LlamaForCausalLM"], "text_config": []},
    {"architectures": ["LlamaForCausalLM"], "text_config": {"architectures": []}},
])
@pytest.mark.parametrize("console_width", [40, 80, 120])
def test_malformed_metadata_has_its_own_diagnostic_without_work(
    export_flow, config, console_width, monkeypatch,
):
    from rich.console import Console

    monkeypatch.setattr(export_mod, "console", Console(width=console_width))
    result, calls = export_flow(config)
    assert result.exit_code == 1, result.output
    # Rich may fold a long POSIX path inside the filename on a narrow terminal.
    plain = strip_ansi(result.output).replace("\n", "")
    assert "metadata" in plain.lower()
    assert "config.json" in plain
    assert calls == []


def test_generated_capabilities_match_pin_and_record_immutable_source():
    from soup_cli.utils.gguf_capabilities import pinned_text_architectures

    path = Path(export_mod.__file__).parents[1] / "utils/llama_cpp_capabilities.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["tag"] == export_mod.LLAMA_CPP_TAG
    assert data["commit"] == "3bf785f3efa89ed28294fbf73054558a2b034bfb"
    assert data["source_git_blob"] == "838999531e58041976c8e612bc97b382874bf227"
    assert data["source_url"].endswith(f'{data["commit"]}/convert_hf_to_gguf.py')
    supported = pinned_text_architectures(export_mod.LLAMA_CPP_TAG)
    assert {"LlamaForCausalLM", "GPT2LMHeadModel", "Qwen2ForCausalLM"} <= supported
    assert "Qwen3_5ForCausalLM" not in supported
    with pytest.raises(ValueError, match="regenerate"):
        pinned_text_architectures("changed-pin")


def test_generator_resolves_inheritance_and_keeps_model_kinds_separate():
    from scripts.gen_llama_cpp_capabilities import extract_capabilities

    source = b'''
raise AssertionError("source must never execute")
class ModelBase: pass
class TextModel(ModelBase): model_arch = gguf.MODEL_ARCH.LLAMA
class VisionModel(ModelBase): model_arch = gguf.MODEL_ARCH.CLIP_VISION
@ModelBase.register("TextAlias", "SharedName")
class Decoder(TextModel): pass
@ModelBase.register("InheritedText")
class Child(Decoder): pass
@ModelBase.register("VisionAlias", "SharedName")
class Encoder(VisionModel): pass
'''
    assert extract_capabilities(source) == {
        "TEXT": ["InheritedText", "SharedName", "TextAlias"],
        "VISION": ["SharedName", "VisionAlias"],
    }


def test_timed_out_probe_is_inconclusive(monkeypatch, tmp_path):
    from soup_cli.utils.gguf_capabilities import probe_text_architectures

    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 15
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timed_out)
    assert probe_text_architectures(tmp_path / "convert_hf_to_gguf.py") is None


@pytest.mark.parametrize("tag,commit,diagnostic", [
    ("different-pin", "a" * 40, "LLAMA_CPP_TAG"),
    ("b5270", "short-sha", "immutable Git commit"),
])
def test_regeneration_refuses_wrong_pin_or_mutable_source(tmp_path, tag, commit, diagnostic):
    script = Path(__file__).resolve().parents[1] / "scripts/gen_llama_cpp_capabilities.py"
    output = tmp_path / "capabilities.json"
    result = subprocess.run([
        sys.executable, str(script), "--converter", str(tmp_path / "unused.py"),
        "--tag", tag, "--commit", commit, "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert diagnostic in result.stderr
    assert not output.exists()


def test_oversized_metadata_is_refused_without_work(export_flow):
    result, calls = export_flow(" " * (1024 * 1024 + 1))
    assert result.exit_code == 1
    plain = " ".join(strip_ansi(result.output).split())
    assert "metadata" in plain.lower()
    assert "1 MiB" in plain
    assert calls == []
