"""Tests for Issue #1464: soup export leaves .soup_merge_tmp_<adapter>/ behind
when a later step fails.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import typer
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import export as export_mod

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Help/console text with ANSI color removed and whitespace collapsed."""
    return " ".join(_ANSI_RE.sub("", text).split())


def _adapter_dir(tmp_path: Path, base_model: str = "some/base") -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model}),
        encoding="utf-8",
    )
    return adapter


def _leftovers(tmp_path: Path) -> list[Path]:
    return list(tmp_path.glob(".soup_merge_tmp_*"))


def _make_fake_merge(merges: list[str]):
    def fake_merge(adapter_path, base_model, output_dir, trust_remote_code=False):
        merges.append(output_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(
            '{"architectures":["LlamaForCausalLM"]}', encoding="utf-8"
        )
        (out / "model.safetensors").write_bytes(b"\0" * 16)

    return fake_merge


def _make_partial_merge(merges: list[str]):
    def partial_merge(adapter_path, base_model, output_dir, trust_remote_code=False):
        merges.append(output_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(
            '{"architectures":["LlamaForCausalLM"]}', encoding="utf-8"
        )
        (out / "model.safetensors").write_bytes(b"\0" * 16)
        raise RuntimeError("tokenizer failed")

    return partial_merge


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Hermetic test environment: redirect home and clear toolchain env vars."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("SOUP_HOME", str(tmp_path / ".soup"))
    monkeypatch.delenv("LLAMA_CPP_PATH", raising=False)


# ---------------------------------------------------------------------------
# Hermeticity verification (Section 4a)
# ---------------------------------------------------------------------------


def test_environment_isolation_points_home_to_tmp(tmp_path):
    """4a hermeticity: Path.home() resolves inside tmp_path, never real user home."""
    assert Path.home().resolve() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Failing tests (each RED on main before #1464 fix, GREEN on branch HEAD)
# ---------------------------------------------------------------------------


def test_gguf_auto_clone_fails_cleans_up(tmp_path, monkeypatch):
    """Test 2: gguf, auto-clone fails."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))
    monkeypatch.setattr(export_mod, "_existing_llama_cpp", lambda p: None)

    def fake_subprocess_run(cmd, *args, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
        ],
    )

    assert merges != []
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "Failed to clone llama.cpp" in _plain(result.output)
    assert _leftovers(tmp_path) == []


def test_gguf_dep_install_fails_cleans_up(tmp_path, monkeypatch):
    """Test 3: gguf, dependency install fails."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(export_mod, "_find_llama_cpp", lambda p=None: llama)
    monkeypatch.setattr(export_mod, "_existing_llama_cpp", lambda p=None: llama)

    def failing_install():
        raise typer.Exit(1)

    monkeypatch.setattr(export_mod, "_install_convert_deps", failing_install)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
        ],
    )

    assert merges != []
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    ("fmt", "extra_args"),
    [
        ("bitnet", []),
        ("gguf-ud", ["--gguf-flavour", "Q4_0_4_4"]),
    ],
)
def test_bitnet_and_gguf_ud_missing_llama_cpp_cleans_up(tmp_path, monkeypatch, fmt, extra_args):
    """Test 4: bitnet and gguf-ud, llama.cpp path missing, real code."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            fmt,
            "--llama-cpp",
            str(tmp_path / "missing"),
            *extra_args,
        ],
    )

    assert merges != []
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "llama.cpp not found at" in _plain(result.output)
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    ("fmt", "extra_args"),
    [
        ("gguf", ["--quant", "q4_0"]),
        ("bitnet", []),
        ("gguf-ud", ["--gguf-flavour", "Q4_0_4_4"]),
        ("onnx", []),
        ("tensorrt", []),
    ],
)
def test_merge_dies_halfway_matrix_cleans_up(tmp_path, monkeypatch, fmt, extra_args):
    """Test 5: Merge dies halfway across all 5 export formats."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_partial_merge(merges))

    # Provide modules for onnx/tensorrt imports if needed
    monkeypatch.setitem(sys.modules, "optimum", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "optimum.exporters", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "optimum.exporters.onnx", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "tensorrt_llm", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "tensorrt_llm.commands", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "tensorrt_llm.commands.convert_checkpoint", mock.MagicMock())

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            fmt,
            *extra_args,
        ],
    )

    assert merges != []
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, RuntimeError)
    assert "tokenizer failed" in str(result.exception)
    assert _leftovers(tmp_path) == []


def test_real_tokenizerless_adapter_e2e_cleans_up(tmp_path, monkeypatch):
    """Test 6: Real tokenizer-less adapter (issue repro a), end to end."""
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from tests._tiny_hf_models import tiny_model_dir

    tiny_base = tiny_model_dir("llama")
    base = AutoModelForCausalLM.from_pretrained(tiny_base)
    adapter = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    adapter.peft_config["default"].base_model_name_or_path = str(tiny_base)
    adapter_dir = tmp_path / "adapter"
    adapter.save_pretrained(adapter_dir)
    # Deliberately do not save tokenizer into adapter_dir

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter_dir),
            "--base",
            str(tiny_base),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, ValueError)
    assert "Couldn't instantiate the backend tokenizer" in str(result.exception)
    assert _leftovers(tmp_path) == []


def test_merge_adapter_direct_cleans_own_created_directory(tmp_path):
    """Test 7: _merge_adapter direct, same adapter: created directory is removed on failure."""
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from tests._tiny_hf_models import tiny_model_dir

    tiny_base = tiny_model_dir("llama")
    base = AutoModelForCausalLM.from_pretrained(tiny_base)
    adapter = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    adapter.peft_config["default"].base_model_name_or_path = str(tiny_base)
    adapter_dir = tmp_path / "adapter"
    adapter.save_pretrained(adapter_dir)

    out = tmp_path / "merge_output"
    assert not out.exists()

    with pytest.raises(ValueError, match=r"Couldn't instantiate the backend tokenizer"):
        export_mod._merge_adapter(str(adapter_dir), str(tiny_base), str(out))

    assert not out.exists()


def test_ctrl_c_during_gguf_export_cleans_up(tmp_path, monkeypatch):
    """Test 3a: Ctrl-C in CLI step after merge removes temporary merge directory."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(export_mod, "_find_llama_cpp", lambda p=None: llama)
    monkeypatch.setattr(export_mod, "_existing_llama_cpp", lambda p=None: llama)
    monkeypatch.setattr(
        export_mod, "_install_convert_deps", mock.Mock(side_effect=KeyboardInterrupt)
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
        ],
    )

    assert merges != []
    assert result.exit_code == 130, (result.output, repr(result.exception))
    assert "Cleaning up temporary merge files..." in _plain(result.output)
    assert _leftovers(tmp_path) == []


def test_ctrl_c_during_merge_adapter_direct_cleans_up(tmp_path):
    """Test 3b: Ctrl-C during tokenizer save inside _merge_adapter cleans up created output dir."""
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from tests._tiny_hf_models import tiny_model_dir

    tiny_base = tiny_model_dir("llama")
    base = AutoModelForCausalLM.from_pretrained(tiny_base)
    adapter = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    adapter.peft_config["default"].base_model_name_or_path = str(tiny_base)
    adapter_dir = tmp_path / "adapter"
    adapter.save_pretrained(adapter_dir)

    out = tmp_path / "merge_output_ctrl_c"
    assert not out.exists()

    fake_tok = mock.MagicMock()
    fake_tok.save_pretrained.side_effect = KeyboardInterrupt("Ctrl-C during tokenizer save")

    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok):
        with pytest.raises(KeyboardInterrupt, match=r"Ctrl-C during tokenizer save"):
            export_mod._merge_adapter(str(adapter_dir), str(tiny_base), str(out))

    assert not out.exists()


# ---------------------------------------------------------------------------
# Control tests (GREEN on main AND after fix)
# ---------------------------------------------------------------------------


def test_control_gguf_llama_cpp_path_missing_cleans_up(tmp_path, monkeypatch):
    """Control (4b): gguf, llama.cpp path missing: handled cleanly by inner guard on main too."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
            "--llama-cpp",
            str(tmp_path / "missing"),
        ],
    )

    assert merges != []
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "llama.cpp not found at" in _plain(result.output)
    assert _leftovers(tmp_path) == []


def test_control_merge_adapter_direct_preserves_caller_owned_directory(tmp_path):
    """Test 8: _merge_adapter direct, caller-owned directory: directory and marker are kept."""
    pytest.importorskip("peft")
    pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from tests._tiny_hf_models import tiny_model_dir

    tiny_base = tiny_model_dir("llama")
    base = AutoModelForCausalLM.from_pretrained(tiny_base)
    adapter = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    adapter.peft_config["default"].base_model_name_or_path = str(tiny_base)
    adapter_dir = tmp_path / "adapter"
    adapter.save_pretrained(adapter_dir)

    out = tmp_path / "existing_merge_dir"
    out.mkdir()
    marker = out / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Couldn't instantiate the backend tokenizer"):
        export_mod._merge_adapter(str(adapter_dir), str(tiny_base), str(out))

    assert out.exists()
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "keep me"


def test_control_gguf_export_success_cleans_up(tmp_path, monkeypatch):
    """Test 9: Control: full gguf export succeeds, output exists, leftovers() == []."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(export_mod, "_find_llama_cpp", lambda p=None: llama)
    monkeypatch.setattr(export_mod, "_install_convert_deps", lambda: None)

    def fake_convert(script, model_path, output_path, outtype, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"F16_GGUF")

    def fake_quantize(llama_dir, src, dst, quant, **kwargs):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"Q4_0_GGUF")

    monkeypatch.setattr(export_mod, "_run_convert", fake_convert)
    monkeypatch.setattr(export_mod, "_run_quantize", fake_quantize)

    out_file = tmp_path / "output.gguf"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
            "--output",
            str(out_file),
        ],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert out_file.exists()
    assert out_file.read_bytes() == b"Q4_0_GGUF"
    assert _leftovers(tmp_path) == []


def test_control_failure_in_already_guarded_region_cleans_up(tmp_path, monkeypatch):
    """Test 10: Control: Failure in already-guarded region (_run_convert raises) -> cleaned up."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(export_mod, "_find_llama_cpp", lambda p=None: llama)
    monkeypatch.setattr(export_mod, "_install_convert_deps", lambda: None)

    def failing_convert(*args, **kwargs):
        raise RuntimeError("convert step crashed")

    monkeypatch.setattr(export_mod, "_run_convert", failing_convert)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert isinstance(result.exception, RuntimeError)
    assert "convert step crashed" in str(result.exception)
    assert _leftovers(tmp_path) == []


def test_control_unrelated_sibling_survives(tmp_path, monkeypatch):
    """Test 11: Control: Unrelated sibling .soup_merge_tmp_other survives failing export."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    sibling = tmp_path / ".soup_merge_tmp_other"
    sibling.mkdir()
    marker = sibling / "marker.txt"
    marker.write_text("precious data", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
            "--llama-cpp",
            str(tmp_path / "missing"),
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert sibling.exists()
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "precious data"


def test_control_non_adapter_model_creates_no_merge_dir(tmp_path, monkeypatch):
    """Test 12: Control: Non-adapter model failing step creates no merge dir and deletes nothing."""
    model_dir = tmp_path / "full_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"architectures":["LlamaForCausalLM"]}', encoding="utf-8"
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(model_dir),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
            "--llama-cpp",
            str(tmp_path / "missing"),
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _leftovers(tmp_path) == []
    assert (model_dir / "config.json").exists()


def test_control_error_preserved_on_llama_cpp_missing(tmp_path, monkeypatch):
    """Test 13: Control: Error is preserved with exact exit code 1 and toolchain message."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            "gguf",
            "--quant",
            "q4_0",
            "--llama-cpp",
            str(tmp_path / "missing"),
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "llama.cpp not found at" in _plain(result.output)


@pytest.mark.parametrize("fmt", ["awq", "gptq"])
def test_control_awq_gptq_refused_no_merge(tmp_path, monkeypatch, fmt):
    """Test 14: Control: awq and gptq refused with non-zero exit and no merge started (#1710)."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export",
            "--model",
            str(adapter),
            "--format",
            fmt,
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"--format {fmt} was removed in v0.77" in _plain(result.output)
    assert merges == []
    assert _leftovers(tmp_path) == []


def test_control_onnx_main_export_failure_cleans_up(tmp_path, monkeypatch):
    """Control (4d): onnx with main_export raising after successful merge cleans up."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    mock_main_export = mock.MagicMock(side_effect=RuntimeError("onnx export failed"))
    monkeypatch.setitem(sys.modules, "optimum", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "optimum.exporters", mock.MagicMock())
    monkeypatch.setitem(
        sys.modules, "optimum.exporters.onnx", mock.MagicMock(main_export=mock_main_export)
    )

    runner = CliRunner()
    result = runner.invoke(app, ["export", "--model", str(adapter), "--format", "onnx"])
    assert merges != []
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "ONNX export failed" in _plain(result.output)
    assert _leftovers(tmp_path) == []


def test_control_tensorrt_build_failure_cleans_up(tmp_path, monkeypatch):
    """Control (4d): tensorrt with trtllm-build failure after merge cleans up."""
    adapter = _adapter_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    monkeypatch.setitem(sys.modules, "optimum", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "optimum.exporters", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "optimum.exporters.onnx", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "tensorrt_llm", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "tensorrt_llm.commands", mock.MagicMock())
    monkeypatch.setitem(sys.modules, "tensorrt_llm.commands.convert_checkpoint", mock.MagicMock())

    # Step 1 succeeds (checkpoint conversion), Step 2 fails (trtllm-build)
    mock_ckpt_res = mock.MagicMock(returncode=0)
    mock_build_res = mock.MagicMock(returncode=1, stderr="build failed")
    fake_run = mock.MagicMock(side_effect=[mock_ckpt_res, mock_build_res])
    monkeypatch.setattr(subprocess, "run", fake_run)
    # trtllm-build is looked up on PATH before it runs: answer with a stand-in path.
    monkeypatch.setattr(
        "soup_cli.utils.tool_paths.find_tool", lambda name, path=None: "/fake/" + name
    )

    runner = CliRunner()
    result = runner.invoke(app, ["export", "--model", str(adapter), "--format", "tensorrt"])
    assert merges != []
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "TensorRT engine build failed" in _plain(result.output)
    assert [call.args[0][0] for call in fake_run.call_args_list] == [
        sys.executable, "/fake/trtllm-build",
    ]
    assert _leftovers(tmp_path) == []


# ---------------------------------------------------------------------------
# The export removes only the merge directory it created
# ---------------------------------------------------------------------------

_FORMATS = [
    ("gguf", ["--quant", "q4_0"]),
    ("bitnet", []),
    ("gguf-ud", ["--gguf-flavour", "Q4_0_4_4"]),
    ("onnx", []),
    ("tensorrt", []),
]


def _preexisting_merge_dir(tmp_path: Path) -> tuple[Path, Path]:
    existing = tmp_path / ".soup_merge_tmp_adapter"
    existing.mkdir()
    marker = existing / "keep-me.txt"
    marker.write_text("not created by this export", encoding="utf-8")
    return existing, marker


def _stub_optional_modules(monkeypatch) -> None:
    for name in (
        "optimum", "optimum.exporters", "optimum.exporters.onnx", "tensorrt_llm",
        "tensorrt_llm.commands", "tensorrt_llm.commands.convert_checkpoint",
    ):
        monkeypatch.setitem(sys.modules, name, mock.MagicMock())


@pytest.mark.parametrize(("fmt", "extra_args"), _FORMATS)
def test_a_merge_that_fails_early_leaves_a_directory_it_did_not_create(
    tmp_path, monkeypatch, fmt, extra_args
):
    adapter = _adapter_dir(tmp_path)
    existing, marker = _preexisting_merge_dir(tmp_path)
    _stub_optional_modules(monkeypatch)

    def failing_merge(adapter_path, base_model, output_dir, trust_remote_code=False):
        raise RuntimeError("merge failed before writing")

    monkeypatch.setattr(export_mod, "_merge_adapter", failing_merge)

    result = CliRunner().invoke(
        app, ["export", "--model", str(adapter), "--format", fmt, *extra_args]
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "merge failed before writing" in str(result.exception)
    assert marker.read_text(encoding="utf-8") == "not created by this export"
    assert existing.is_dir()


@pytest.mark.parametrize(("fmt", "extra_args"), _FORMATS)
def test_a_later_failure_also_leaves_a_directory_it_did_not_create(
    tmp_path, monkeypatch, fmt, extra_args
):
    adapter = _adapter_dir(tmp_path)
    existing, marker = _preexisting_merge_dir(tmp_path)
    _stub_optional_modules(monkeypatch)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_partial_merge(merges))

    result = CliRunner().invoke(
        app, ["export", "--model", str(adapter), "--format", fmt, *extra_args]
    )

    assert merges == [str(existing)]
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert marker.read_text(encoding="utf-8") == "not created by this export"
    assert existing.is_dir()


def test_the_run_says_it_left_the_directory(tmp_path, monkeypatch):
    adapter = _adapter_dir(tmp_path)
    existing, _ = _preexisting_merge_dir(tmp_path)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_fake_merge(merges))

    result = CliRunner().invoke(
        app,
        [
            "export", "--model", str(adapter), "--format", "gguf", "--quant", "q4_0",
            "--llama-cpp", str(tmp_path / "missing"),
        ],
    )

    assert result.exit_code == 1, (result.output, repr(result.exception))
    text = _plain(result.output)
    assert "existed before this export" in text
    assert "Cleaning up temporary merge files" not in text
    assert existing.is_dir()


@pytest.mark.parametrize(("fmt", "extra_args"), _FORMATS)
def test_a_directory_the_export_created_is_still_removed(tmp_path, monkeypatch, fmt, extra_args):
    adapter = _adapter_dir(tmp_path)
    _stub_optional_modules(monkeypatch)
    merges: list[str] = []
    monkeypatch.setattr(export_mod, "_merge_adapter", _make_partial_merge(merges))

    result = CliRunner().invoke(
        app, ["export", "--model", str(adapter), "--format", fmt, *extra_args]
    )

    assert merges != []
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _leftovers(tmp_path) == []
