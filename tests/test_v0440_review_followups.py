"""v0.44.0 — review-fix coverage gaps surfaced by code-/tdd-/security-reviews.

Bundles the additional negative-path + boundary tests required by the
reviewer findings.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import threading

import pytest

from soup_cli.commands.llama import _LLAMA_ENV_ALLOWLIST, _filtered_env
from soup_cli.ui.plugins import (
    clear_tabs,
    list_tabs,
    load_plugins,
    register_tab,
)
from soup_cli.utils.checkpoint_trigger import write_trigger
from soup_cli.utils.delinearize_llama4 import discover_weight_files
from soup_cli.utils.fetch_examples import fetch_examples_dir
from soup_cli.utils.fsdp_consolidate import discover_shards, plan_consolidation
from soup_cli.utils.gpu_monitor import (
    detect_apple_silicon,
    query_nvidia_smi,
)
from soup_cli.utils.llama_proxy import resolve
from soup_cli.utils.llama_server_timings import format_kv_bar, parse_timings
from soup_cli.utils.onboarding import render_onboarding_yaml
from soup_cli.utils.qr_url import build_phone_url, render_qr_ascii, validate_token
from soup_cli.utils.shortcuts import (
    build_macos_command_file,
    build_windows_cmd,
)
from soup_cli.utils.sweep_config import parse_sweep_yaml
from soup_cli.utils.tail_latency import percentile, summarise_latency, update_ema
from soup_cli.utils.tool_outputs import ToolCallTimer, ToolOutputsBuffer
from soup_cli.utils.ui_env import resolve_ui_env

# --- gpu_monitor coverage ---------------------------------------------------

def test_query_nvidia_smi_no_smi_returns_false_empty(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda _name: None)
    ok, samples = query_nvidia_smi()
    assert ok is False
    assert samples == []


def test_detect_apple_silicon_matches_platform_pair():
    expected = (
        platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"}
    )
    assert detect_apple_silicon() is expected


# --- tail_latency boundary --------------------------------------------------

def test_update_ema_rejects_non_finite_prev():
    with pytest.raises(ValueError):
        update_ema(float("inf"), 1.0, 0.1)


# --- tool_outputs -----------------------------------------------------------

def test_tool_call_timer_set_error_is_recorded():
    buffer = ToolOutputsBuffer()
    with ToolCallTimer(buffer, name="x") as timer:
        timer.set_error("bad input")
    snap = buffer.snapshot()
    assert snap[0].success is False
    assert snap[0].error == "bad input"


def test_tool_call_timer_set_output_set_error_type_check():
    buffer = ToolOutputsBuffer()
    timer = ToolCallTimer(buffer, name="x")
    with pytest.raises(TypeError):
        timer.set_output(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        timer.set_error(123)  # type: ignore[arg-type]


def test_tool_outputs_snapshot_limit_zero_returns_empty():
    buffer = ToolOutputsBuffer()
    buffer.record_call(
        name="x",
        started_ts=1.0,
        duration_ms=1.0,
        success=True,
        output_preview="",
    )
    assert buffer.snapshot(limit=0) == []


def test_tool_outputs_concurrent_writes():
    buffer = ToolOutputsBuffer()

    def worker(prefix: str) -> None:
        for idx in range(50):
            buffer.record_call(
                name=f"{prefix}-{idx}",
                started_ts=float(idx),
                duration_ms=1.0,
                success=True,
                output_preview="",
            )

    threads = [threading.Thread(target=worker, args=(f"t{n}",)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    snap = buffer.snapshot()
    # No torn writes; total count is exactly 8 * 50 = 400 (under cap).
    assert len(snap) == 400


def test_tool_outputs_ring_drops_oldest_after_overflow():
    buffer = ToolOutputsBuffer()
    # Write more than the ring's max — deque auto-evicts oldest.
    from soup_cli.utils.tool_outputs import _MAX_RECORDS

    for idx in range(_MAX_RECORDS + 5):
        buffer.record_call(
            name=f"r{idx}",
            started_ts=float(idx),
            duration_ms=1.0,
            success=True,
            output_preview="",
        )
    snap = buffer.snapshot()
    assert len(snap) == _MAX_RECORDS
    # First record dropped; tail must be the newest.
    assert snap[-1].name == f"r{_MAX_RECORDS + 4}"


# --- llama_server_timings --------------------------------------------------

def test_format_kv_bar_upper_bound_rejected():
    with pytest.raises(ValueError):
        format_kv_bar(50.0, width=201)


def test_parse_timings_rejects_negative_kv():
    timings = parse_timings({"kv_cache_used": -1, "kv_cache_size": 100})
    # Negative coerces to None; pct can't be computed.
    assert timings.kv_cache_used is None
    assert timings.kv_cache_pct is None


# --- qr_url -----------------------------------------------------------------

def test_validate_token_rejects_non_string():
    with pytest.raises(TypeError):
        validate_token(123)  # type: ignore[arg-type]


def test_build_phone_url_empty_host_rejected():
    with pytest.raises(ValueError):
        build_phone_url(scheme="https", host="", port=80, token="x" * 32)


def test_build_phone_url_null_byte_host_rejected():
    with pytest.raises(ValueError):
        build_phone_url(
            scheme="https", host="x\x00y", port=80, token="x" * 32
        )


def test_build_phone_url_token_in_query_string():
    url = build_phone_url(
        scheme="https", host="x", port=443, token="x" * 32
    )
    # Token MUST be in the query string so the server can read it; not in
    # the fragment (which never reaches the server).
    assert "?token=" in url
    assert "#token=" not in url


def test_render_qr_ascii_rejects_non_string():
    with pytest.raises(ValueError):
        render_qr_ascii(123)  # type: ignore[arg-type]


# --- ui plugins -------------------------------------------------------------

def test_load_plugins_returns_int(monkeypatch):
    clear_tabs()
    count = load_plugins()
    assert isinstance(count, int)
    # No bundled plugins ship in v0.44.0 — count is 0.
    assert count == 0


def test_register_tab_clear_resets_limit():
    clear_tabs()
    for idx in range(32):
        register_tab(name=f"t{idx}", title="T", render=lambda: "x")
    clear_tabs()
    register_tab(name="fresh", title="T", render=lambda: "x")
    assert "fresh" in list_tabs()


# --- ui_env -----------------------------------------------------------------

def test_resolve_ui_env_default_reads_environ(monkeypatch):
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("GRADIO_HOST", raising=False)
    monkeypatch.delenv("GRADIO_PORT", raising=False)
    env = resolve_ui_env(None)
    assert env.api_host is None
    assert env.api_port is None


# --- shortcuts --------------------------------------------------------------

def test_macos_command_oversize_command_rejected():
    with pytest.raises(ValueError):
        build_macos_command_file(name="x", command="x" * 2000)


def test_windows_cmd_oversize_command_rejected():
    with pytest.raises(ValueError):
        build_windows_cmd(name="x", command="x" * 2000)


# --- onboarding -------------------------------------------------------------

def test_onboarding_output_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = str((tmp_path.parent / "elsewhere").resolve())
    with pytest.raises(ValueError, match="under cwd"):
        render_onboarding_yaml(
            {
                "base": "x/y",
                "dataset": "d",
                "task": "sft",
                "epochs": 1,
                "output": other,
            }
        )


def test_onboarding_empty_dataset_rejected():
    with pytest.raises(ValueError):
        render_onboarding_yaml(
            {"base": "x/y", "dataset": "", "task": "sft", "epochs": 1}
        )


# --- sweep_config -----------------------------------------------------------

def test_sweep_spec_frozen():
    spec = parse_sweep_yaml("strategy: grid\n")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.strategy = "random"  # type: ignore[misc]


def test_sweep_spec_params_immutable():
    spec = parse_sweep_yaml(
        "strategy: grid\nparams:\n  lr: [0.001, 0.002]\n"
    )
    # The mapping itself is a MappingProxyType — readonly.
    with pytest.raises(TypeError):
        spec.params["lr"] = (0.999,)  # type: ignore[index]
    # Each value is a tuple — also immutable.
    with pytest.raises(AttributeError):
        spec.params["lr"].append(0.999)  # type: ignore[attr-defined]


def test_sweep_yaml_n_runs_zero_accepted():
    spec = parse_sweep_yaml("n_runs: 0\n")
    assert spec.n_runs == 0


def test_sweep_yaml_n_runs_upper_bound_accepted():
    spec = parse_sweep_yaml("n_runs: 10000\n")
    assert spec.n_runs == 10000


def test_sweep_yaml_param_key_oversize_rejected():
    long_key = "k" * 200
    with pytest.raises(ValueError, match="exceeds"):
        parse_sweep_yaml(f"params:\n  {long_key}: [1]\n")


def test_sweep_yaml_param_value_non_scalar_rejected():
    with pytest.raises(ValueError, match="non-scalar"):
        parse_sweep_yaml("params:\n  lr: [{nested: 1}]\n")


# --- fsdp_consolidate -------------------------------------------------------

def test_plan_consolidation_output_outside_cwd_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "shards"
    out.mkdir()
    (out / "pytorch_model_fsdp_0.bin").write_bytes(b"")
    other = str((tmp_path.parent / "evil.safetensors").resolve())
    with pytest.raises(ValueError, match="outside cwd"):
        plan_consolidation(str(out), other)


def test_plan_consolidation_null_byte_output_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="NUL byte"):
        plan_consolidation(str(tmp_path), "bad\x00.safetensors")


def test_discover_shards_non_string_rejected():
    with pytest.raises(TypeError):
        discover_shards(123)  # type: ignore[arg-type]


# --- delinearize_llama4 -----------------------------------------------------

def test_discover_weight_files_non_string_rejected():
    with pytest.raises(TypeError):
        discover_weight_files(123)  # type: ignore[arg-type]


# --- llama_proxy ------------------------------------------------------------

def test_llama_resolve_rejects_null_byte_arg(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda *_a, **_k: "/fake/llama-cli")
    with pytest.raises(ValueError, match="control"):
        resolve("cli", ["bad\x00arg"])


# --- fetch (security review fixes) ------------------------------------------

def test_fetch_examples_dir_under_realpath():
    # The bundled dir must exist and be a directory.
    path = fetch_examples_dir()
    assert os.path.isdir(path)


def test_cli_fetch_force_overwrites(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    runner.invoke(app, ["fetch", "examples", "llama-3.1-8b-lora"])
    target = tmp_path / "llama-3.1-8b-lora.yaml"
    target.write_text("# stomp")
    result = runner.invoke(
        app, ["fetch", "examples", "llama-3.1-8b-lora", "--force"]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "stomp" not in target.read_text()


# --- llama env filter -------------------------------------------------------

def test_filtered_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = _filtered_env()
    assert "HF_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env.get("PATH") == "/usr/bin"


def test_llama_env_allowlist_immutable():
    with pytest.raises(AttributeError):
        _LLAMA_ENV_ALLOWLIST.add("EVIL")  # type: ignore[attr-defined]


# --- write_trigger symlink rejection (security review M2) -------------------

def test_write_trigger_rejects_pre_existing_symlink(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("Symlink test requires POSIX permissions.")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "elsewhere"
    target.write_text("victim")
    trigger = out / ".checkpoint_now"
    os.symlink(str(target), str(trigger))
    with pytest.raises(OSError, match="symlink"):
        write_trigger(str(out))


# --- tail_latency MAX_SAMPLES cap (TDD review C2) ---------------------------

def test_percentile_max_samples_cap():
    """A well-formed but too-large iterable must raise ValueError."""

    def too_many():
        # Use a generator to avoid actually allocating 1M+ floats in memory.
        for idx in range(1_000_005):
            yield float(idx)

    with pytest.raises(ValueError, match="too many"):
        percentile(too_many(), 50)


def test_summarise_latency_max_samples_cap():
    def too_many():
        for idx in range(1_000_005):
            yield float(idx)

    with pytest.raises(ValueError, match="too many"):
        summarise_latency(too_many())


# --- graceful_save additional coverage (TDD review H3, H4) -----------------

def test_graceful_save_restore_idempotent(monkeypatch):
    import signal as _signal

    from soup_cli.utils.graceful_save import GracefulSaveHandler

    calls: list = []
    monkeypatch.setattr(_signal, "signal", lambda *_a, **_k: calls.append(_a) or _signal.SIG_DFL)
    handler = GracefulSaveHandler()
    handler.install()
    handler.restore()
    handler.restore()  # double-restore must not raise.
    # install() recorded one call; restore() recorded one call. No more.
    assert len(calls) == 2


def test_graceful_save_install_signal_failure_swallowed(monkeypatch):
    import signal as _signal

    from soup_cli.utils.graceful_save import GracefulSaveHandler

    def failing(*_a, **_k):
        raise ValueError("not main thread")

    monkeypatch.setattr(_signal, "signal", failing)
    handler = GracefulSaveHandler()
    handler.install()  # must not raise
    assert handler._installed is False
