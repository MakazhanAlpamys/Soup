"""Issue #99 — native Apple Silicon metrics for ``soup monitor``."""

from __future__ import annotations

import io
import math
import plistlib
import subprocess

import pytest
import typer
from rich.console import Console

from soup_cli.commands import monitor as monitor_command
from soup_cli.utils import gpu_monitor
from soup_cli.utils.gpu_monitor import (
    GpuSample,
    PowermetricsResult,
    PowermetricsStatus,
    parse_powermetrics_plist,
    query_powermetrics,
)


def _plist_sample(
    *,
    idle_ratio: object = 0.25,
    elapsed_ns: object = 2_000_000_000,
    gpu_power_mw: object | None = 12_500.0,
    gpu_energy_mj: object | None = 25_000,
    hw_model: object = "Mac16,5",
    gpu_sampler_energy_location: bool = False,
    padding: str | None = None,
) -> bytes:
    gpu: dict[str, object] = {"idle_ratio": idle_ratio, "freq_hz": 1_500}
    processor: dict[str, object] = {}
    if gpu_power_mw is not None:
        processor["gpu_power"] = gpu_power_mw
    if gpu_energy_mj is not None:
        destination = gpu if gpu_sampler_energy_location else processor
        destination["gpu_energy"] = gpu_energy_mj
    payload: dict[str, object] = {
        "is_delta": True,
        "elapsed_ns": elapsed_ns,
        "hw_model": hw_model,
        "gpu": gpu,
        "processor": processor,
    }
    if padding is not None:
        payload["padding"] = padding
    return plistlib.dumps(payload)


def test_parse_powermetrics_uses_direct_power_and_active_ratio():
    samples = parse_powermetrics_plist(_plist_sample())
    assert samples == [
        GpuSample(
            index=0,
            name="Apple Silicon (Mac16,5)",
            util_gpu_pct=75.0,
            util_mem_pct=None,
            mem_used_mb=None,
            mem_total_mb=None,
            temp_c=None,
            power_w=12.5,
        )
    ]


def test_parse_powermetrics_derives_power_from_energy_and_elapsed_time():
    payload = _plist_sample(gpu_power_mw=None, gpu_energy_mj=12_000)
    sample = parse_powermetrics_plist(payload)[0]
    assert sample.power_w == pytest.approx(6.0)


def test_parse_powermetrics_accepts_gpu_sampler_energy_location():
    payload = _plist_sample(
        gpu_power_mw=None,
        gpu_energy_mj=9_000,
        gpu_sampler_energy_location=True,
    )
    sample = parse_powermetrics_plist(payload)[0]
    assert sample.power_w == pytest.approx(4.5)


def test_parse_powermetrics_handles_nul_separated_samples():
    first = _plist_sample(idle_ratio=0.9, gpu_power_mw=1_000)
    second = _plist_sample(idle_ratio=0.1, gpu_power_mw=2_000)
    samples = parse_powermetrics_plist(first + b"\x00" + second)
    assert [sample.util_gpu_pct for sample in samples] == pytest.approx([10.0, 90.0])
    assert [sample.power_w for sample in samples] == pytest.approx([1.0, 2.0])


@pytest.mark.parametrize("bad_ratio", [-0.1, 1.1, True, math.nan, math.inf, "busy"])
def test_parse_powermetrics_rejects_invalid_idle_ratio(bad_ratio):
    sample = parse_powermetrics_plist(_plist_sample(idle_ratio=bad_ratio))[0]
    assert sample.util_gpu_pct is None


@pytest.mark.parametrize("bad_power", [-1, True, math.nan, math.inf, "watts"])
def test_parse_powermetrics_rejects_invalid_power(bad_power):
    sample = parse_powermetrics_plist(
        _plist_sample(gpu_power_mw=bad_power, gpu_energy_mj=None)
    )[0]
    assert sample.power_w is None


def test_parse_powermetrics_rejects_invalid_model_name():
    sample = parse_powermetrics_plist(_plist_sample(hw_model="x" * 129))[0]
    assert sample.name == "Apple Silicon"


def test_parse_powermetrics_skips_malformed_and_oversize_payloads():
    assert parse_powermetrics_plist(b"not a plist") == []
    oversized_plist = _plist_sample(
        padding="x" * gpu_monitor._MAX_POWERMETRICS_OUTPUT_BYTES,
    )
    assert len(oversized_plist) > gpu_monitor._MAX_POWERMETRICS_OUTPUT_BYTES
    assert plistlib.loads(oversized_plist)["gpu"]["idle_ratio"] == 0.25
    assert parse_powermetrics_plist(oversized_plist) == []


def test_parse_powermetrics_requires_bytes():
    with pytest.raises(TypeError):
        parse_powermetrics_plist("not bytes")  # type: ignore[arg-type]


def _completed(*, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def test_query_powermetrics_root_uses_fixed_binary_without_shell(monkeypatch):
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(gpu_monitor.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(gpu_monitor, "_effective_uid", lambda: 0)

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _completed(stdout=_plist_sample())

    monkeypatch.setattr(gpu_monitor.subprocess, "run", _run)
    result = query_powermetrics(0.25)

    assert result.status is PowermetricsStatus.OK
    assert len(result.samples) == 1
    argv, kwargs = calls[0]
    assert argv[0] == "/usr/bin/powermetrics"
    assert argv[argv.index("--samplers") + 1] == "gpu_power"
    assert argv[argv.index("--sample-rate") + 1] == "250"
    assert argv[argv.index("--sample-count") + 1] == "1"
    assert argv[argv.index("--format") + 1] == "plist"
    assert "--handle-invalid-values" in argv
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["timeout"] == pytest.approx(5.25)
    assert kwargs["shell"] is False
    assert kwargs["text"] is False
    assert kwargs["check"] is False


def test_query_powermetrics_non_root_uses_non_interactive_sudo(monkeypatch):
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(gpu_monitor.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(gpu_monitor, "_effective_uid", lambda: 501)

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _completed(stdout=_plist_sample())

    monkeypatch.setattr(gpu_monitor.subprocess, "run", _run)
    result = query_powermetrics(2.0)

    assert result.status is PowermetricsStatus.OK
    argv, kwargs = calls[0]
    assert argv[:3] == ["/usr/bin/sudo", "-n", "/usr/bin/powermetrics"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["timeout"] == pytest.approx(7.0)


def test_query_powermetrics_names_missing_sudo_ticket(monkeypatch):
    monkeypatch.setattr(gpu_monitor.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(gpu_monitor, "_effective_uid", lambda: 501)
    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed(
            returncode=1,
            stderr=b"sudo: a password is required\n",
        ),
    )
    result = query_powermetrics(1.0)
    assert result == PowermetricsResult(PowermetricsStatus.PERMISSION_DENIED)


def test_query_powermetrics_reports_missing_binary(monkeypatch):
    monkeypatch.setattr(gpu_monitor.os.path, "isfile", lambda _path: False)
    assert query_powermetrics(1.0) == PowermetricsResult(PowermetricsStatus.UNAVAILABLE)


def test_query_powermetrics_reports_timeout_and_invalid_output(monkeypatch):
    monkeypatch.setattr(gpu_monitor.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(gpu_monitor, "_effective_uid", lambda: 0)

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["powermetrics"], 1)

    monkeypatch.setattr(gpu_monitor.subprocess, "run", _timeout)
    assert query_powermetrics(1.0).status is PowermetricsStatus.FAILED

    monkeypatch.setattr(
        gpu_monitor.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed(stdout=b"not plist"),
    )
    assert query_powermetrics(1.0).status is PowermetricsStatus.INVALID_OUTPUT


@pytest.mark.parametrize("bad_interval", [True, 0.0, 0.24, 30.01, math.nan, math.inf])
def test_query_powermetrics_validates_interval(bad_interval):
    with pytest.raises((TypeError, ValueError)):
        query_powermetrics(bad_interval)  # type: ignore[arg-type]


def test_monitor_routes_apple_silicon_before_nvidia(monkeypatch):
    output = io.StringIO()
    sample = GpuSample(0, "Apple M4 Max", 87.0, None, None, None, None, 31.5)
    monkeypatch.setattr(
        monitor_command,
        "console",
        Console(file=output, color_system=None, width=160),
    )
    monkeypatch.setattr(monitor_command, "detect_apple_silicon", lambda: True)
    monkeypatch.setattr(
        monitor_command,
        "query_powermetrics",
        lambda _refresh: PowermetricsResult(PowermetricsStatus.OK, (sample,)),
    )
    monkeypatch.setattr(
        monitor_command,
        "query_nvidia_smi",
        lambda: pytest.fail("Apple path must not query nvidia-smi"),
    )

    monitor_command.monitor(refresh=0.25, once=True)
    rendered = output.getvalue()
    assert "Apple M4 Max" in rendered
    assert "87.0%" in rendered
    assert "31.5 W" in rendered


def test_monitor_non_apple_routes_to_nvidia(monkeypatch):
    output = io.StringIO()
    sample = GpuSample(0, "NVIDIA Test GPU", 42.0, 7.0, 512.0, 8192.0, 55.0, 75.0)
    monkeypatch.setattr(
        monitor_command,
        "console",
        Console(file=output, color_system=None, width=160),
    )
    monkeypatch.setattr(monitor_command, "detect_apple_silicon", lambda: False)
    monkeypatch.setattr(
        monitor_command,
        "query_powermetrics",
        lambda _refresh: pytest.fail("Non-Apple path must not query powermetrics"),
    )
    monkeypatch.setattr(monitor_command, "query_nvidia_smi", lambda: (True, [sample]))

    monitor_command.monitor(refresh=0.25, once=True)
    rendered = output.getvalue()
    assert "NVIDIA Test GPU" in rendered
    assert "42.0%" in rendered


def test_monitor_apple_permission_error_is_actionable(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(monitor_command, "console", Console(file=output, color_system=None))
    monkeypatch.setattr(monitor_command, "detect_apple_silicon", lambda: True)
    monkeypatch.setattr(
        monitor_command,
        "query_powermetrics",
        lambda _refresh: PowermetricsResult(PowermetricsStatus.PERMISSION_DENIED),
    )

    with pytest.raises(typer.Exit) as exc:
        monitor_command.monitor(refresh=0.25, once=True)
    assert exc.value.exit_code == 1
    assert "sudo -v" in output.getvalue()
    assert "never prompts" in output.getvalue()
