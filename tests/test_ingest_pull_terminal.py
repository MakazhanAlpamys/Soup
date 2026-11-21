"""``soup ingest --pull`` shows outside text as typed, on the display only (#2489).

The ``--source`` and ``--output`` the operator typed, a bad ``--since``, the
Langfuse host, a credential refusal and an error from the server keep
``:name:`` aliases, markup and a backslash before a bracket as written, and lose
their control and bidi characters. No test here reaches the network.
"""

import json
import sys
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import ingest
from soup_cli.utils import ingest_pull
from soup_cli.utils.ingest_sources import SUPPORTED_INGEST_SOURCES
from soup_cli.utils.paths import quote_path
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

VALUES = ["plain", "x[red]n", "[/]n", "n:rocket:", "x\\[n", "x\x9b\u202en", "x\x1b\x07n"]
#: What the display drops: C0 controls, a C1 CSI and a bidi override.
DROPPED = ("\x1b", "\x07", "\x9b", "\u202e")

_NOT_A_WINDOWS_NAME = "{} cannot be part of a file name on Windows"
#: ``VALUES`` that the receipt test writes as part of a real file name.
FILE_VALUES = [
    "plain",
    "x[red]n",
    "[/]n",
    pytest.param("n:rocket:", marks=skip_colon_on_windows),
    pytest.param(
        "x\\[n",
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason=_NOT_A_WINDOWS_NAME.format("'\\'")
        ),
    ),
    "x\x9b\u202en",
    pytest.param(
        "x\x1b\x07n",
        marks=pytest.mark.skipif(
            sys.platform == "win32", reason=_NOT_A_WINDOWS_NAME.format("A C0 control")
        ),
    ),
]

GENERATION = {"id": "gen-1", "input": "hello", "output": "world", "model": "test-model"}
DEFAULT_HOST = "https://cloud.langfuse.com"


def _shown(value: str) -> str:
    for char in DROPPED:
        value = value.replace(char, "")
    return value


def _assert_shown(result, exit_code: int, text: str) -> None:
    """``text`` is in the output, and no dropped character is anywhere in it."""
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert " ".join(text.split()) in " ".join(output.split()), output
    assert [char for char in DROPPED if char in output] == []


def _ingest(*args: str):
    return make_cli_runner().invoke(app, ["ingest", *args])


def _pull(*args: str):
    return _ingest("--source", "langfuse", "--pull", *args)


def _raising(error: Exception):
    def fail(*_args, **_kwargs):
        raise error

    return fail


def _no_network(*_args, **_kwargs):
    raise AssertionError("a --pull display test reached the Langfuse transport")


def _connect(monkeypatch, host: str = DEFAULT_HOST) -> None:
    """Credentials load as the real function builds them, with ``host`` as the base URL."""
    credentials = ingest_pull.LangfuseCredentials(
        public_key="pk-test", secret_key="sk-test", host=host
    )
    monkeypatch.setattr(
        ingest_pull,
        "load_langfuse_credentials",
        lambda environ, *, allow_private_host=False: credentials,
    )


def _serve(monkeypatch, rows=(), error=None) -> None:
    """The pull yields ``rows`` as GENERATION rows, counting them, then raises ``error``."""

    def pull(credentials, *, stats, **_kwargs):
        stats.pages += 1
        for row in rows:
            stats.observations += 1
            stats.generations += 1
            yield row
        if error is not None:
            raise error

    monkeypatch.setattr(ingest_pull, "pull_langfuse_generations", pull)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ingest, "console", Console(width=1000, height=100))
    for name in ("PUBLIC_KEY", "SECRET_KEY", "HOST", "BASE_URL"):
        monkeypatch.delenv(f"LANGFUSE_{name}", raising=False)
    monkeypatch.setattr(ingest_pull, "_urllib_transport", _no_network)
    return tmp_path


# ---------------------------------------------------------------------------
# Refusals of what the operator typed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", VALUES)
def test_an_unknown_source_is_named_as_typed(workdir, value):
    result = _ingest("--source", value, "--pull")

    _assert_shown(
        result,
        2,
        f"unknown ingest source {value!r}; supported: {sorted(SUPPORTED_INGEST_SOURCES)}",
    )


@pytest.mark.parametrize("value", VALUES)
def test_an_output_outside_cwd_is_named_as_typed(workdir, value):
    path = "../" + value

    result = _pull("--output", path)

    # The path is checked, and named, as ``str(Path(...))``: separators are native.
    named = quote_path(str(Path(path)))
    _assert_shown(result, 1, _shown(f"--output {named} must stay under cwd"))


@pytest.mark.parametrize("value", VALUES)
def test_a_bad_since_is_named_as_typed(workdir, value):
    result = _pull("--since", value)

    _assert_shown(result, 2, f"--since {value!r} must look like 30m, 24h or 7d")
    assert not Path("traces.jsonl").exists()


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize(
    ("error", "exit_code"), [(ingest_pull.PullError, 1), (ValueError, 2)], ids=["pull", "host"]
)
def test_a_credential_refusal_is_shown_as_raised(workdir, monkeypatch, error, exit_code, value):
    message = "LANGFUSE_HOST {} is not an allowed Langfuse host"
    monkeypatch.setattr(
        ingest_pull, "load_langfuse_credentials", _raising(error(message.format(value)))
    )

    result = _pull()

    _assert_shown(result, exit_code, message.format(_shown(value)))
    assert not Path("traces.jsonl").exists()


# ---------------------------------------------------------------------------
# The pull itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", VALUES)
def test_the_progress_line_names_the_host_as_loaded(workdir, monkeypatch, value):
    _connect(monkeypatch, host=value)
    _serve(monkeypatch)

    result = _pull("--since", "24h")

    _assert_shown(
        result, 0, f"Pulling GENERATION observations from {_shown(value)} for the last 24h..."
    )


@pytest.mark.parametrize("value", VALUES)
def test_a_failed_pull_is_shown_as_raised(workdir, monkeypatch, value):
    _connect(monkeypatch)
    _serve(monkeypatch, rows=(GENERATION,), error=ingest_pull.PullError(f"HTTP 500: {value}"))

    result = _pull("--output", "out.jsonl")

    _assert_shown(result, 1, f"HTTP 500: {_shown(value)}")
    _assert_shown(result, 1, "Nothing was written to out.jsonl.")
    assert not Path("out.jsonl").exists()


@pytest.mark.parametrize("value", VALUES)
def test_nothing_written_names_the_output_as_typed(workdir, monkeypatch, value):
    # No file is created on this path, so every value is a usable --output here.
    output = f"out-{value}.jsonl"
    _connect(monkeypatch)
    _serve(monkeypatch, error=ingest_pull.PullError("HTTP 500: (empty body)"))

    result = _pull("--output", output)

    _assert_shown(result, 1, f"Nothing was written to {_shown(Path(output).name)}.")
    assert not Path(output).exists()


@pytest.mark.parametrize("value", FILE_VALUES)
def test_the_receipt_names_the_output_as_typed(workdir, monkeypatch, value):
    output = f"out-{value}.jsonl"
    _connect(monkeypatch)
    _serve(monkeypatch, rows=(GENERATION,))

    result = _pull("--output", output)

    _assert_shown(
        result,
        0,
        f"Wrote 1 traces from langfuse (1 generations pulled) -> {_shown(Path(output).name)}",
    )
    row = json.loads(Path(output).read_text(encoding="utf-8"))
    assert (row["trace_id"], row["prompt"], row["output"]) == ("gen-1", "hello", "world")


# ---------------------------------------------------------------------------
# The lines keep their colours
# ---------------------------------------------------------------------------


@pytest.fixture
def coloured(workdir, monkeypatch):
    monkeypatch.setattr(
        ingest,
        "console",
        Console(
            width=1000, height=100, force_terminal=True, color_system="standard", no_color=False
        ),
    )


def test_a_refusal_stays_red(coloured):
    result = _pull("--output", "../out.jsonl")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert f"\x1b[31m--output '{Path('../out.jsonl')}' must stay under cwd\x1b[0m" in result.output


def test_a_failed_pull_stays_red(coloured, monkeypatch):
    _connect(monkeypatch)
    _serve(monkeypatch, error=ingest_pull.PullError("Langfuse answered HTTP 500: (empty body)"))

    result = _pull("--output", "out.jsonl")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "\x1b[31mLangfuse answered HTTP 500: (empty body)\x1b[0m" in result.output
    assert "\x1b[31mNothing was written to out.jsonl.\x1b[0m" in result.output


def test_the_progress_line_stays_dim(coloured, monkeypatch):
    _connect(monkeypatch)
    _serve(monkeypatch)

    result = _pull()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (
        f"\x1b[2mPulling GENERATION observations from {DEFAULT_HOST} for the last 7d...\x1b[0m"
        in result.output
    )


def test_the_receipt_stays_green(coloured, monkeypatch):
    _connect(monkeypatch)
    _serve(monkeypatch, rows=(GENERATION,))

    result = _pull()

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (
        "\x1b[32mWrote 1 traces from langfuse (1 generations pulled) -> traces.jsonl\x1b[0m"
        in result.output
    )
