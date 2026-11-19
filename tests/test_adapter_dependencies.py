"""Acceptance tests for adapter dependencies (#2056).

Read-only commands (`check`, `diff`) require only `numpy` (header parser introduced in #2063).
Writing commands (`merge`, `arithmetic`) require both `numpy` and `safetensors`.

When required dependencies are absent:
1. Commands fail with exit code 1.
2. Error message clearly names the missing package(s) and provides the install recipe:
   `pip install "soup-cli[train]"` (or pip install ...)
3. Missing dependencies are never misdiagnosed as "Corrupt or unreadable safetensors weights"
   or scan errors ("Could not scan", "Pass --allow-unscanned to proceed anyway").
4. Genuine corrupt/unreadable files continue to report corruption when dependencies are present.
"""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.utils.adapter_check import _load_safetensors_numpy, check_adapter
from soup_cli.utils.adapter_diff import (
    READ_ADAPTER_DEPS,
    READ_ADAPTER_DEPS_MSG,
    WRITE_ADAPTER_DEPS,
    WRITE_ADAPTER_DEPS_MSG,
    _load_safetensors,
    compute_adapter_diff,
    require_adapter_dependencies,
)
from soup_cli.utils.adapter_merge import merge_adapters, write_merged_adapter
from tests.conftest import strip_ansi

runner = CliRunner()


def _clean(output: str) -> str:
    """Strip ANSI color codes and collapse whitespace for resilient assertions."""
    return " ".join(strip_ansi(output).split())


def _make_dummy_adapter(dir_path: Path) -> Path:
    """Create a minimal valid adapter with safetensors and config."""
    import safetensors.numpy

    dir_path.mkdir(parents=True, exist_ok=True)
    weights = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones(
            (4, 16), dtype=np.float32
        ),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.ones(
            (16, 4), dtype=np.float32
        ),
    }
    safetensors.numpy.save_file(
        weights, str(dir_path / "adapter_model.safetensors")
    )
    cfg = {
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 8.0,
        "target_modules": ["q_proj"],
    }
    (dir_path / "adapter_config.json").write_text(
        json.dumps(cfg), encoding="utf-8"
    )
    return dir_path


def _hide_dependencies(monkeypatch: pytest.MonkeyPatch, hidden_module: str) -> None:
    """Simulate absence of dependencies on a plain/lean installation."""
    if hidden_module in ("numpy", "both"):
        monkeypatch.setitem(sys.modules, "numpy", None)
    if hidden_module in ("safetensors", "both"):
        monkeypatch.setitem(sys.modules, "safetensors", None)
        monkeypatch.setitem(sys.modules, "safetensors.numpy", None)


def test_plain_install_reports_missing_numpy_before_cli_import() -> None:
    """A fresh interpreter must reach the friendly refusal without numpy installed."""
    probe = (
        "import sys\n"
        "sys.modules['numpy'] = None\n"
        "from soup_cli.cli import app\n"
        "from typer.testing import CliRunner\n"
        "from rich.text import Text\n"
        "result = CliRunner().invoke(app, ['adapters', 'check', 'missing_adapter'])\n"
        "output = ' '.join(Text.from_ansi(result.output).plain.split())\n"
        "assert result.exit_code == 1, (result.output, repr(result.exception))\n"
        "assert 'numpy package required' in output, output\n"
        "assert 'pip install numpy' in output, output\n"
        "assert 'Corrupt or unreadable' not in output, output\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)


@pytest.mark.parametrize("hidden_module", ["numpy", "both"])
def test_adapters_check_missing_numpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden_module: str,
):
    """adapters check exits 1 and names numpy when numpy is missing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")
    adapter_dir = _make_dummy_adapter(tmp_path / "test_adapter")

    _hide_dependencies(monkeypatch, hidden_module)

    result = runner.invoke(soup_app, ["adapters", "check", str(adapter_dir)])
    clean_out = _clean(result.output)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "numpy package required" in clean_out
    assert 'pip install "soup-cli[train]" (or pip install numpy)' in clean_out
    assert "Corrupt or unreadable" not in clean_out


def test_adapters_check_safetensors_missing_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """adapters check succeeds when safetensors is missing (pins #2063)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")
    adapter_dir = _make_dummy_adapter(tmp_path / "test_adapter")

    _hide_dependencies(monkeypatch, "safetensors")

    result = runner.invoke(soup_app, ["adapters", "check", str(adapter_dir)])
    clean_out = _clean(result.output)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert clean_out.endswith("alive")


@pytest.mark.parametrize("hidden_module", ["numpy", "both"])
def test_adapters_diff_missing_numpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden_module: str,
):
    """adapters diff exits 1 and names numpy when numpy is missing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")
    adapter1 = _make_dummy_adapter(tmp_path / "adapter1")
    adapter2 = _make_dummy_adapter(tmp_path / "adapter2")

    _hide_dependencies(monkeypatch, hidden_module)

    result = runner.invoke(
        soup_app, ["adapters", "diff", str(adapter1), str(adapter2)]
    )
    clean_out = _clean(result.output)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "numpy package required" in clean_out
    assert 'pip install "soup-cli[train]" (or pip install numpy)' in clean_out
    assert "Corrupt or unreadable" not in clean_out


def test_adapters_diff_safetensors_missing_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """adapters diff succeeds when safetensors is missing (pins #2063)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")
    adapter1 = _make_dummy_adapter(tmp_path / "adapter1")
    adapter2 = _make_dummy_adapter(tmp_path / "adapter2")

    _hide_dependencies(monkeypatch, "safetensors")

    result = runner.invoke(
        soup_app, ["adapters", "diff", str(adapter1), str(adapter2)]
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))


@pytest.mark.parametrize("hidden_module", ["numpy", "safetensors", "both"])
def test_adapters_merge_missing_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden_module: str
):
    """adapters merge exits 1 naming dependencies when any write dependency is missing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")
    adapter1 = _make_dummy_adapter(tmp_path / "adapter1")
    adapter2 = _make_dummy_adapter(tmp_path / "adapter2")

    _hide_dependencies(monkeypatch, hidden_module)

    result = runner.invoke(
        soup_app,
        ["adapters", "merge", str(adapter1), str(adapter2), "-o", "merged_out"],
    )
    clean_out = _clean(result.output)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "numpy and safetensors packages required" in clean_out
    assert 'pip install "soup-cli[train]" (or pip install safetensors numpy)' in clean_out
    assert "Could not scan" not in clean_out
    assert "Corrupt or unreadable" not in clean_out


@pytest.mark.parametrize("hidden_module", ["numpy", "safetensors", "both"])
def test_adapters_arithmetic_missing_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden_module: str
):
    """adapters arithmetic exits 1 naming dependencies without scan bypass hint."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")
    adapter1 = _make_dummy_adapter(tmp_path / "adapter1")
    adapter2 = _make_dummy_adapter(tmp_path / "adapter2")

    _hide_dependencies(monkeypatch, hidden_module)

    result = runner.invoke(
        soup_app,
        [
            "adapters",
            "arithmetic",
            "a + b",
            "--adapter",
            f"a={adapter1}",
            "--adapter",
            f"b={adapter2}",
            "-o",
            "arithmetic_out",
        ],
    )
    clean_out = _clean(result.output)

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "numpy and safetensors packages required" in clean_out
    assert 'pip install "soup-cli[train]" (or pip install safetensors numpy)' in clean_out
    assert "Pass --allow-unscanned to proceed anyway" not in clean_out
    assert "Could not scan" not in clean_out


@pytest.mark.parametrize("hidden_module", ["numpy", "both"])
def test_direct_read_functions_missing_numpy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden_module: str,
):
    """Read-only functions raise RuntimeError when numpy is missing."""
    monkeypatch.chdir(tmp_path)
    adapter1 = _make_dummy_adapter(tmp_path / "adapter1")
    adapter2 = _make_dummy_adapter(tmp_path / "adapter2")
    weights_file = adapter1 / "adapter_model.safetensors"

    _hide_dependencies(monkeypatch, hidden_module)

    with pytest.raises(RuntimeError) as exc_info:
        require_adapter_dependencies(READ_ADAPTER_DEPS)
    assert READ_ADAPTER_DEPS_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        _load_safetensors_numpy(weights_file)
    assert READ_ADAPTER_DEPS_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        _load_safetensors(weights_file)
    assert READ_ADAPTER_DEPS_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        check_adapter(str(adapter1))
    assert READ_ADAPTER_DEPS_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        compute_adapter_diff(str(adapter1), str(adapter2))
    assert READ_ADAPTER_DEPS_MSG in str(exc_info.value)

    # Dependency diagnostics take precedence over path/stat failures.
    with pytest.raises(RuntimeError) as exc_info:
        _load_safetensors_numpy(tmp_path / "missing.safetensors")
    assert READ_ADAPTER_DEPS_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        check_adapter(str(tmp_path / "missing"))
    assert READ_ADAPTER_DEPS_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        compute_adapter_diff("missing_a", "missing_b")
    assert READ_ADAPTER_DEPS_MSG in str(exc_info.value)


def test_direct_read_functions_safetensors_missing_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Read-only functions succeed when safetensors is missing (pins #2063)."""
    monkeypatch.chdir(tmp_path)
    adapter1 = _make_dummy_adapter(tmp_path / "adapter1")
    adapter2 = _make_dummy_adapter(tmp_path / "adapter2")
    weights_file = adapter1 / "adapter_model.safetensors"

    _hide_dependencies(monkeypatch, "safetensors")

    # None of these should raise
    require_adapter_dependencies(READ_ADAPTER_DEPS)
    tensors = _load_safetensors_numpy(weights_file)
    assert len(tensors) > 0

    tensors_diff = _load_safetensors(weights_file)
    assert len(tensors_diff) > 0

    report = check_adapter(str(adapter1))
    assert report.verdict == "alive"

    diff_report = compute_adapter_diff(str(adapter1), str(adapter2))
    assert diff_report.shared_layers > 0


@pytest.mark.parametrize("hidden_module", ["numpy", "safetensors", "both"])
def test_direct_write_functions_raise_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hidden_module: str
):
    """Write functions raise RuntimeError when any write dependency is missing."""
    monkeypatch.chdir(tmp_path)
    adapter1 = _make_dummy_adapter(tmp_path / "adapter1")
    adapter2 = _make_dummy_adapter(tmp_path / "adapter2")

    _hide_dependencies(monkeypatch, hidden_module)

    with pytest.raises(RuntimeError) as exc_info:
        require_adapter_dependencies(WRITE_ADAPTER_DEPS)
    assert WRITE_ADAPTER_DEPS_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        merge_adapters([str(adapter1), str(adapter2)], "out", strategy="linear")
    assert WRITE_ADAPTER_DEPS_MSG in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info:
        write_merged_adapter("out", str(adapter1), {})
    assert WRITE_ADAPTER_DEPS_MSG in str(exc_info.value)


def test_positive_control_with_dependencies_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When dependencies are present, check, diff, merge, and arithmetic work normally."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")
    adapter1 = _make_dummy_adapter(tmp_path / "adapter1")
    adapter2 = _make_dummy_adapter(tmp_path / "adapter2")

    # Positive control: check
    res_check = runner.invoke(soup_app, ["adapters", "check", str(adapter1)])
    assert res_check.exit_code == 0, (res_check.output, repr(res_check.exception))
    assert _clean(res_check.output).endswith("alive")

    # Positive control: diff
    res_diff = runner.invoke(
        soup_app, ["adapters", "diff", str(adapter1), str(adapter2)]
    )
    assert res_diff.exit_code == 0, (res_diff.output, repr(res_diff.exception))

    # Positive control: merge
    res_merge = runner.invoke(
        soup_app,
        [
            "adapters",
            "merge",
            str(adapter1),
            str(adapter2),
            "-o",
            "merged_out",
            "--allow-unscanned",
        ],
    )
    assert res_merge.exit_code == 0, (res_merge.output, repr(res_merge.exception))
    assert (tmp_path / "merged_out" / "adapter_model.safetensors").is_file()

    # Positive control: arithmetic
    res_arith = runner.invoke(
        soup_app,
        [
            "adapters",
            "arithmetic",
            "a + b",
            "--adapter",
            f"a={adapter1}",
            "--adapter",
            f"b={adapter2}",
            "-o",
            "arithmetic_out",
            "--allow-unscanned",
        ],
    )
    assert res_arith.exit_code == 0, (res_arith.output, repr(res_arith.exception))
    assert (tmp_path / "arithmetic_out" / "adapter_model.safetensors").is_file()


@pytest.mark.parametrize(
    "command, expected_exit", [("check", 1), ("diff", 2), ("merge", 2), ("arithmetic", 1),
                               ("scan", 2)],
)
def test_corrupt_weights_reports_corrupt_when_dependencies_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, expected_exit: int,
):
    """A genuinely corrupt safetensors file reports 'Corrupt or unreadable'."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")
    adapter_dir = tmp_path / "corrupt_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(
        b"this is not valid safetensors data"
    )
    cfg = {"peft_type": "LORA", "r": 4}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")

    good_adapter = _make_dummy_adapter(tmp_path / "valid_adapter")
    args = ["adapters", command, str(adapter_dir)]
    if command == "diff":
        args.append(str(good_adapter))
    elif command == "merge":
        args.extend([str(good_adapter), "-o", "merged", "--allow-unscanned"])
    elif command == "arithmetic":
        args = ["adapters", command, "a + b", "--adapter", f"a={adapter_dir}",
                "--adapter", f"b={good_adapter}", "-o", "merged", "--allow-unscanned"]

    result = runner.invoke(soup_app, args)
    clean_out = _clean(result.output)

    assert result.exit_code == expected_exit, (result.output, repr(result.exception))
    assert "Corrupt or unreadable safetensors weights" in clean_out
    assert "packages required" not in clean_out


def test_merge_reports_writer_import_failure_after_successful_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The root package can import while its numpy writer is unavailable."""
    monkeypatch.chdir(tmp_path)
    adapter1 = _make_dummy_adapter(tmp_path / "adapter1")
    adapter2 = _make_dummy_adapter(tmp_path / "adapter2")
    original_import = builtins.__import__

    def unavailable_writer(name, *args, **kwargs):
        if name == "safetensors.numpy":
            raise ModuleNotFoundError("synthetic unavailable safetensors numpy writer")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_writer)
    result = runner.invoke(soup_app, [
        "adapters", "merge", str(adapter1), str(adapter2), "-o", "merged", "--allow-unscanned",
    ])

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert WRITE_ADAPTER_DEPS_MSG in _clean(result.output)
    assert "Corrupt or unreadable" not in _clean(result.output)
    assert not (tmp_path / "merged").exists()
