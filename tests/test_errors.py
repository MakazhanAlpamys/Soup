"""Tests for friendly error handling and --verbose flag."""

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.errors import format_friendly_error

runner = CliRunner()


# --- format_friendly_error tests ---


def test_cuda_oom_error():
    """CUDA OOM gets a friendly message."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "GPU ran out of memory" in output
    # v0.53.4 #11 — message now suggests --batch-size / --grad-accum CLI flags.
    assert "--batch-size" in output


def test_missing_fastapi_error():
    """Missing fastapi gives install hint."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ModuleNotFoundError("No module named 'fastapi'")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "soup-cli[serve]" in output


def test_missing_datasketch_error():
    """Missing datasketch gives install hint."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ModuleNotFoundError("No module named 'datasketch'")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "soup-cli[data]" in output


def test_missing_wandb_error():
    """Missing wandb gives install hint."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ModuleNotFoundError("No module named 'wandb'")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "pip install wandb" in output


def test_missing_deepspeed_error():
    """Missing deepspeed gives install hint."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ModuleNotFoundError("No module named 'deepspeed'")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "soup-cli[deepspeed]" in output


@pytest.mark.parametrize(
    "module",
    ["torch", "transformers", "peft", "trl", "datasets", "bitsandbytes", "accelerate"],
)
def test_missing_train_dep_points_at_train_extra(module):
    """v0.71.0 — any missing heavy training dep points at the [train] extra."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ModuleNotFoundError(f"No module named '{module}'")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    # The `\[train]` escape renders to a literal `[train]` in the output.
    assert "soup-cli[train]" in output
    assert "[train] extra" in output


def test_connection_error():
    """Connection error gets friendly message."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ConnectionError("Failed to connect")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "Network connection failed" in output


def test_unknown_error_shows_type():
    """Unknown errors show the exception type and message."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = ZeroDivisionError("division by zero")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "ZeroDivisionError" in output
    assert "division by zero" in output
    assert "--verbose" in output


def test_verbose_shows_traceback():
    """Verbose mode shows full traceback."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        except RuntimeError as exc:
            format_friendly_error(exc, verbose=True)
    output = buf.getvalue()
    assert "GPU ran out of memory" in output
    assert "Traceback" in output or "RuntimeError" in output


def test_verbose_unknown_error():
    """Verbose mode for unknown errors shows traceback."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        try:
            raise ValueError("something weird")
        except ValueError as exc:
            format_friendly_error(exc, verbose=True)
    output = buf.getvalue()
    assert "ValueError" in output
    assert "something weird" in output


def test_yaml_error():
    """YAML syntax error gets friendly message."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = Exception("yaml.scanner.ScannerError: mapping values are not allowed here")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "YAML syntax" in output


def test_validation_error():
    """Pydantic validation error gets friendly message."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = Exception("2 validation error for SoupConfig")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "Config validation failed" in output


def test_auth_401_error():
    """401 error gets auth hint."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = Exception("401 Client Error: Unauthorized")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "Authentication failed" in output


def test_file_not_found_error():
    """File not found gets friendly message."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        exc = FileNotFoundError("No such file or directory: 'model.bin'")
        format_friendly_error(exc, verbose=False)
    output = buf.getvalue()
    assert "No such file or directory" in output
    assert "soup init" in output


# --- CLI --verbose flag tests ---


def test_verbose_flag_in_help():
    """--verbose flag is shown in help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "verbose" in result.output


def test_help_shows_doctor_and_quickstart():
    """New commands are visible in help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output
    assert "quickstart" in result.output
