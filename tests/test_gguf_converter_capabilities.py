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
                Path(output).mkdir(exist_ok=True)
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
    result, calls = export_flow(
        {"architectures": ["BartForConditionalGeneration"]}, managed=managed)
    assert result.exit_code == 1, result.output
    assert "BartForConditionalGeneration" in result.output
    assert "b12423" in result.output
    assert "--llama-cpp" in result.output
    assert calls == []


def test_supported_fresh_default_still_exports(export_flow):
    result, calls = export_flow({"architectures": ["LlamaForCausalLM"]})
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["clone", "deps", "convert", "quantize"]


def test_missing_transformers_refuses_before_clone_and_install(export_flow, monkeypatch):
    # #1911 — the pinned b12423 converter imports transformers at load time, but
    # Soup never auto-installs torch/transformers (a llama.cpp requirements
    # install once downgraded a user's CUDA torch). Refuse by name BEFORE any
    # clone or install, pointing at the supported way to get it.
    monkeypatch.setattr(export_mod, "_transformers_missing", lambda: True)
    result, calls = export_flow({"architectures": ["LlamaForCausalLM"]})
    assert result.exit_code == 1, result.output
    plain = " ".join(strip_ansi(result.output).split())
    assert "transformers" in plain
    assert 'pip install "soup-cli[train]"' in plain
    assert calls == []


def test_missing_transformers_refuses_before_adapter_merge(export_flow, monkeypatch):
    monkeypatch.setattr(export_mod, "_transformers_missing", lambda: True)
    result, calls = export_flow({"architectures": ["LlamaForCausalLM"]}, adapter=True)
    assert result.exit_code == 1, result.output
    plain = " ".join(strip_ansi(result.output).split())
    assert "transformers" in plain
    assert 'pip install "soup-cli[train]"' in plain
    assert calls == []


def test_present_transformers_lets_fresh_default_export(export_flow, monkeypatch):
    monkeypatch.setattr(export_mod, "_transformers_missing", lambda: False)
    result, calls = export_flow({"architectures": ["LlamaForCausalLM"]})
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls == ["clone", "deps", "convert", "quantize"]


@pytest.mark.parametrize("spec,raises,expected", [
    (SimpleNamespace(), False, False),
    (None, False, True),
    (None, True, True),
])
def test_transformers_missing_checks_import_spec(monkeypatch, spec, raises, expected):
    def find_spec(name):
        assert name == "transformers"
        if raises:
            raise ValueError("transformers.__spec__ is None")
        return spec

    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    assert export_mod._transformers_missing() is expected


@pytest.mark.parametrize("architecture,allowed", [
    ("LlamaForCausalLM", True), ("BartForConditionalGeneration", False),
])
def test_adapter_checks_merged_metadata_and_cleans_up(export_flow, tmp_path, architecture, allowed):
    result, calls = export_flow({"architectures": [architecture]}, adapter=True)
    assert result.exit_code == (0 if allowed else 1), (result.output, result.exception)
    assert calls == (["merge", "clone", "deps", "convert", "quantize"] if allowed else ["merge"])
    assert not list(tmp_path.glob(".soup_merge_tmp_model*"))


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
    ("BartForConditionalGeneration", "LlamaForCausalLM", True),
    ("LlamaForCausalLM", "BartForConditionalGeneration", False),
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
    assert data["commit"] == "f4d0673917c75493c73fc9b64ab42841600ba944"
    assert data["source_git_blob"] == "8cee1f8646e113accba4de2c5e68b7b0804252eb"
    assert data["sources"]["conversion/__init__.py"] == {
        "sha256": "c922e7710c53a8352789c999f00b7c8cc49c1733c21453c566af73334194300b",
        "git_blob": "fb6b8464c4a1e99fbefd58aa856d89b1f8b6d5cd",
    }
    assert data["source_url"].endswith(f'{data["commit"]}/convert_hf_to_gguf.py')
    supported = pinned_text_architectures(export_mod.LLAMA_CPP_TAG)
    assert {"LlamaForCausalLM", "GPT2LMHeadModel", "Qwen2ForCausalLM"} <= supported
    # b12423 registers Qwen3.5 dense and MoE text models.
    assert {"Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"} <= supported
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


def test_generator_prefers_init_model_maps_for_multi_file_layout():
    from scripts.gen_llama_cpp_capabilities import (
        ENTRY_NAME,
        extract_capabilities_from_sources,
    )

    # b12423 splits the converter into convert_hf_to_gguf.py plus conversion/*.py,
    # and its printed listing is the static __init__.py maps' keys — not the
    # decorator buckets. A name registered only by an mmproj class still counts
    # as TEXT, and dotted remote-code aliases are never listed.
    init = b'''
raise AssertionError("source must never execute")
TEXT_MODEL_MAP: dict[str, str] = {
    "GroveMoeForCausalLM": "grovemoe",
    "modeling_grove_moe.GroveMoeForCausalLM": "grovemoe",
    "GraniteSpeechForConditionalGeneration": "granite",
}
MMPROJ_MODEL_MAP: dict[str, str] = {"MmprojOnly": "somemodule"}
'''
    models = b'''
raise AssertionError("source must never execute")
class ModelBase: pass
class TextModel(ModelBase): model_arch = gguf.MODEL_ARCH.LLAMA
@ModelBase.register("GroveMoeForCausalLM", "modeling_grove_moe.GroveMoeForCausalLM")
class GroveMoeModel(TextModel): pass
@ModelBase.register("GraniteSpeechForConditionalGeneration")
class GraniteSpeechMmprojModel(ModelBase): model_arch = gguf.MODEL_ARCH.MMPROJ
'''
    entry = b"import sys\nsys.exit(1)\n"
    capabilities = extract_capabilities_from_sources({
        "conversion/__init__.py": init, "conversion/models.py": models, ENTRY_NAME: entry,
    })
    assert capabilities == {
        "TEXT": ["GraniteSpeechForConditionalGeneration", "GroveMoeForCausalLM"],
        "VISION": ["MmprojOnly"],
    }


def test_regeneration_records_per_file_sources(tmp_path):
    """The manifest must hash EVERY file the data was read from, not only the
    entry script (each conversion/*.py module feeds the AST walk too)."""
    import hashlib

    script = Path(__file__).resolve().parents[1] / "scripts/gen_llama_cpp_capabilities.py"
    conv = tmp_path / "conversion"
    conv.mkdir()
    init_bytes = (
        b'TEXT_MODEL_MAP = {"LlamaForCausalLM": "llama"}\n'
        b'MMPROJ_MODEL_MAP = {"MmprojOnly": "somemodule"}\n'
    )
    (conv / "__init__.py").write_bytes(init_bytes)
    (conv / "models.py").write_bytes(b"")
    entry = tmp_path / "convert_hf_to_gguf.py"
    entry.write_bytes(b"")
    output = tmp_path / "capabilities.json"
    result = subprocess.run([
        sys.executable, str(script), "--converter", str(entry),
        "--package-dir", str(conv), "--tag", export_mod.LLAMA_CPP_TAG,
        "--commit", "f4d0673917c75493c73fc9b64ab42841600ba944",
        "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    assert set(data["sources"]) == {
        "convert_hf_to_gguf.py", "conversion/__init__.py", "conversion/models.py",
    }
    recorded = data["sources"]["conversion/__init__.py"]
    assert recorded["sha256"] == hashlib.sha256(init_bytes).hexdigest()
    blob = f"blob {len(init_bytes)}\0".encode() + init_bytes
    assert recorded["git_blob"] == hashlib.sha1(blob).hexdigest()
    # Backward-compat: the entry-script top-level fields survive.
    assert data["source_sha256"] == data["sources"]["convert_hf_to_gguf.py"]["sha256"]
    assert data["source_git_blob"] == data["sources"]["convert_hf_to_gguf.py"]["git_blob"]


def test_generator_multifile_without_maps_resolves_decorators():
    """Without conversion/__init__.py model maps, the decorator walk rules —
    including the b12423 MMPROJ model_arch, which lands in VISION."""
    from scripts.gen_llama_cpp_capabilities import (
        ENTRY_NAME,
        extract_capabilities_from_sources,
    )

    entry = b"import sys\nsys.exit(1)\n"
    models = b'''
raise AssertionError("source must never execute")
class ModelBase: pass
class TextModel(ModelBase): model_arch = gguf.MODEL_ARCH.LLAMA
@ModelBase.register("TextAlias")
class Decoder(TextModel): pass
@ModelBase.register("MmprojAlias")
class Encoder(ModelBase): model_arch = gguf.MODEL_ARCH.MMPROJ
'''
    capabilities = extract_capabilities_from_sources(
        {ENTRY_NAME: entry, "conversion/models.py": models})
    assert capabilities == {"TEXT": ["TextAlias"], "VISION": ["MmprojAlias"]}


@pytest.mark.parametrize("sources,diagnostic", [
    pytest.param(
        {"conversion/__init__.py": b'TEXT_MODEL_MAP = {"LlamaForCausalLM": "llama"}\n'},
        "TEXT_MODEL_MAP and MMPROJ_MODEL_MAP", id="partial-maps",
    ),
    pytest.param(
        {
            "conversion/__init__.py": (
                b'TEXT_MODEL_MAP = {"LlamaForCausalLM": "llama"}\n'
                b'MMPROJ_MODEL_MAP = {"MmprojOnly": "mm"}\n'
            ),
            "conversion/other.py": (
                b'TEXT_MODEL_MAP = {"GPT2LMHeadModel": "gpt2"}\n'
                b'MMPROJ_MODEL_MAP = {"Other": "o"}\n'
            ),
        },
        "multiple sources define converter model maps", id="multiple-map-sources",
    ),
    pytest.param(
        {
            "conversion/__init__.py": (
                b"TEXT_MODEL_MAP: dict[str, str] = {}\n"
                b'MMPROJ_MODEL_MAP: dict[str, str] = {"MmprojOnly": "mm"}\n'
            ),
        },
        "TEXT_MODEL_MAP lists no architectures", id="empty-text-map",
    ),
    pytest.param(
        {"convert_hf_to_gguf.py": (
            b"class ModelBase: pass\n"
            b'@ModelBase.register("Alias")\n'
            b"class Orphan(ModelBase): pass\n"
        )},
        "unrecognized registration on Orphan", id="unrecognized-registration",
    ),
])
def test_generator_refuses_broken_converter_layouts(sources, diagnostic):
    from scripts.gen_llama_cpp_capabilities import extract_capabilities_from_sources

    with pytest.raises(ValueError, match=diagnostic):
        extract_capabilities_from_sources(sources)


def test_timed_out_probe_is_inconclusive(monkeypatch, tmp_path):
    from soup_cli.utils.gguf_capabilities import probe_text_architectures

    def timed_out(command, **kwargs):
        assert kwargs["timeout"] == 15
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timed_out)
    assert probe_text_architectures(tmp_path / "convert_hf_to_gguf.py") is None


def test_probe_mmproj_section_does_not_leak_into_text(tmp_path):
    """The conversion/ package listing prints TEXT then MMPROJ sections
    (conversion/__init__.py print_registered_models); MMPROJ names must not
    land in the TEXT set (b12423 layout, #1911)."""
    from soup_cli.utils.gguf_capabilities import probe_text_architectures

    stub = tmp_path / "convert_hf_to_gguf.py"
    stub.write_text(
        "import sys\n"
        "print('ERROR:root:TEXT models:', file=sys.stderr)\n"
        "print('  - Qwen3_5ForCausalLM', file=sys.stderr)\n"
        "print('ERROR:root:MMPROJ models:', file=sys.stderr)\n"
        "print('  - Qwen3_5VLForConditionalGeneration', file=sys.stderr)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [__import__("sys").executable, str(stub), "--print-supported-models"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0
    names = probe_text_architectures(stub)
    assert names == frozenset({"Qwen3_5ForCausalLM"})


@pytest.mark.parametrize("tag,commit,diagnostic", [
    ("different-pin", "a" * 40, "LLAMA_CPP_TAG"),
    ("b12423", "short-sha", "immutable Git commit"),
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


def test_regeneration_refuses_empty_package_dir(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/gen_llama_cpp_capabilities.py"
    entry = tmp_path / "convert_hf_to_gguf.py"
    entry.write_text("", encoding="utf-8")
    package = tmp_path / "conversion"
    package.mkdir()
    output = tmp_path / "capabilities.json"
    result = subprocess.run([
        sys.executable, str(script), "--converter", str(entry),
        "--package-dir", str(package), "--tag", export_mod.LLAMA_CPP_TAG,
        "--commit", "f4d0673917c75493c73fc9b64ab42841600ba944",
        "--output", str(output),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--package-dir" in result.stderr
    assert "contains no *.py modules" in result.stderr
    assert not output.exists()


def test_oversized_metadata_is_refused_without_work(export_flow):
    result, calls = export_flow(" " * (1024 * 1024 + 1))
    assert result.exit_code == 1
    plain = " ".join(strip_ansi(result.output).split())
    assert "metadata" in plain.lower()
    assert "1 MiB" in plain
    assert calls == []
