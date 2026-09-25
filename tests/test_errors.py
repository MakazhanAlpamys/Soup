"""Tests for friendly error handling and --verbose flag."""

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.errors import format_friendly_error
from tests.conftest import strip_ansi

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
    assert "--batch-size" in output
    assert "GPU ran out of memory" in output
    # v0.53.4 #11 — message now suggests --batch-size / --grad-accum CLI flags.
    assert "gradient_checkpointing" in output
    assert "4bit" in output


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

def test_cuda_oom_mentions_gradient_checkpointing():
    """CUDA OOM hint should mention gradient_checkpointing and 4bit."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)

    with patch("soup_cli.utils.errors.console", test_console):
        exc = RuntimeError("CUDA out of memory")
        format_friendly_error(exc, verbose=False)

    output = buf.getvalue()
    assert "gradient_checkpointing" in output
    assert "4bit" in output


def test_gated_repo_error():
    """HF gated repos should suggest login and HF_TOKEN."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)

    with patch("soup_cli.utils.errors.console", test_console):
        exc = Exception("GatedRepoError: Cannot access gated repo")
        format_friendly_error(exc, verbose=False)

    output = buf.getvalue()
    assert "huggingface-cli login" in output
    assert "HF_TOKEN" in output


def test_gated_repo_hyphenated_error():
    """Hyphenated gated-repo errors should match."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)

    with patch("soup_cli.utils.errors.console", test_console):
        exc = Exception("Access denied to gated-repo")
        format_friendly_error(exc, verbose=False)

    output = buf.getvalue()
    assert "huggingface-cli login" in output


def test_trust_remote_code_error():
    """trust_remote_code errors should provide actionable hint."""
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)

    with patch("soup_cli.utils.errors.console", test_console):
        exc = Exception(
            "This repository requires you to execute the configuration file"
        )
        format_friendly_error(exc, verbose=False)

    output = buf.getvalue()
    assert "trust_remote_code=True" in output


_HINT = 'pip install "soup-cli[audio]"'


@pytest.mark.parametrize("verbose", [False, True], ids=["plain", "verbose"])
@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError(f"No such file or directory: \x1b]0;pwned\x07 {_HINT}"),
        ImportError(f"XCodec2 needs torchaudio: \x1b]0;pwned\x07 {_HINT}"),
        # Message contains Rich markup close tag: must not be parsed as markup
        # (on main this raises rich.errors.MarkupError, and under --verbose the
        # last traceback line reintroduces it one panel lower).
        RuntimeError(f"broken [/] pipe {_HINT}"),
    ],
    ids=["known-pattern", "unknown-error", "markup-close-tag"],
)
def test_exception_text_is_not_parsed_as_markup(exc, verbose):
    """#1200: `[extra]` in exception text survives, control bytes are stripped,
    and a `[/]` in the message never raises MarkupError — in both the plain and
    the --verbose panel, whose traceback ends with the same exception text."""
    buf = StringIO()
    test_console = Console(file=buf, force_terminal=True, color_system="truecolor", width=300)
    # Raise-and-catch so traceback.format_exc() carries the exception text into
    # the verbose panel, matching how cli.py calls the handler from an except block.
    with patch("soup_cli.utils.errors.console", test_console):
        try:
            raise exc
        except Exception as caught:
            format_friendly_error(caught, verbose=verbose)
    output = strip_ansi(buf.getvalue())
    assert "soup-cli[audio]" in output
    assert "\x1b" not in output
    assert "\x07" not in output


# --- #1267: 401/403 must be decided from the exception object or HTTP
# phrasing, never from bare digits in the message ---

def _run_format(exc):
    buf = StringIO()
    test_console = Console(file=buf, stderr=False)
    with patch("soup_cli.utils.errors.console", test_console):
        format_friendly_error(exc, verbose=False)
    return buf.getvalue()


def test_403_status_code_from_response_object():
    """A wrapped HTTP error carrying a 403 response gets the permission hint."""

    class FakeResponse:
        status_code = 403

    class FakeHTTPError(Exception):
        response = FakeResponse()

    output = _run_format(FakeHTTPError("404 Client Error for url: x"))
    assert "Access denied" in output


def test_401_via_cause_chain():
    """peft/transformers wrap hub errors; the cause's response counts."""

    class FakeResponse:
        status_code = 401

    class FakeHTTPError(Exception):
        response = FakeResponse()

    try:
        try:
            raise FakeHTTPError("boom")
        except FakeHTTPError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as wrapped:
        output = _run_format(wrapped)
    assert "Authentication failed" in output


def test_http_401_text_phrasings_still_match():
    """Genuine HTTP phrasings keep their hints."""
    for msg in (
        "Client error '401 Unauthorized' for url 'https://x'",
        "401 Client Error: Unauthorized for url: https://x",
        "HTTP Error 401: Unauthorized",
        "Error code: 401",
    ):
        assert "Authentication failed" in _run_format(Exception(msg)), msg
    for msg in (
        "Client error '403 Forbidden' for url 'https://x'",
        "403 Client Error: Forbidden",
        "HTTP Error 403: Forbidden",
    ):
        assert "Access denied" in _run_format(Exception(msg)), msg


@pytest.mark.parametrize(
    "msg",
    [
        "The size of tensor a (4013) must match the size of tensor b (4096)",
        "mat1 and mat2 shapes cannot be multiplied (2x4030 and 4096x8)",
        "index 40132 is out of bounds for dimension 0 with size 32000",
        "train row 401: no causal-loss target remains after tokenization",
        "train row 403: no causal-loss target remains after tokenization",
        "no causal-loss target remains ... at data.max_length=4030",
        "Can't find 'adapter_config.json' at 'adapters/code-20260403'",
        "Can't find 'adapter_config.json' at 'output/checkpoint-4010'",
        "[Errno 13] Permission denied: '...\\runs\\20260401'",
        "404 Client Error. Revision Not Found (revision 9c1f4012 in the URL)",
        "404 Client Error. Entry Not Found (Request ID: Root=1-68d4b2c1-4a4039d2)",
    ],
)
def test_bare_digits_no_longer_reported_as_auth(msg):
    """Digits 401/403 inside numbers, rows, paths and hashes are not auth."""
    output = _run_format(Exception(msg))
    assert "Authentication failed" not in output, msg
    assert "Access denied" not in output, msg
