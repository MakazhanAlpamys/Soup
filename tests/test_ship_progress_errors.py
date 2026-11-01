"""Real entry-point error diagnostics stay separate from JSONL progress."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import strip_ansi

_ENTRY_POINT = """
import json
import sys

from soup_cli.cli import run
from soup_cli.commands import ship

kind, message, arguments = sys.argv[1:]
message = json.loads(message)
if kind != "none":
    error_type = {"key": KeyError, "runtime": RuntimeError,
                  "interrupt": KeyboardInterrupt}[kind]
    def fail(*args, **kwargs):
        raise error_type(message)
    ship._parse_chat_template_kwargs = fail
sys.argv = ["soup", "--no-audit-log", "--no-telemetry", "ship", *json.loads(arguments)]
run()
"""


def _invoke(
    tmp_path: Path, *, kind: str = "none", message: str = "probe_failure",
    progress: bool = True, color: bool = False, arguments: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    args = arguments if arguments is not None else ["--evidence", "missing.json"]
    if progress:
        args = [*args, "--progress", "jsonl"]
    env = os.environ.copy()
    env.pop("NO_COLOR", None)
    if color:
        env["FORCE_COLOR"] = "1"
    else:
        env.pop("FORCE_COLOR", None)
        env["NO_COLOR"] = "1"
    env["COLUMNS"] = "80"
    return subprocess.run(
        [sys.executable, "-c", _ENTRY_POINT, kind, json.dumps(message), json.dumps(args)],
        cwd=tmp_path, env=env, capture_output=True, encoding="utf-8", timeout=60,
    )


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _assert_aborted_jsonl(result: subprocess.CompletedProcess[str]) -> None:
    events = [json.loads(line) for line in result.stderr.splitlines()]
    assert [(event["phase"], event["event"]) for event in events] == [
        ("verdict", "start"), ("verdict", "error"),
    ]
    assert events[-1]["error"] == "execution_aborted"
    assert "probe_failure" not in result.stderr


@pytest.mark.parametrize("kind, error_type", [("key", "KeyError"), ("runtime", "RuntimeError")])
def test_unexpected_error_names_its_type_and_cause_on_stdout(
    kind: str, error_type: str, tmp_path: Path,
) -> None:
    result = _invoke(tmp_path, kind=kind)

    assert result.returncode == 1, (result.stdout, result.stderr)
    assert f"Error: {error_type}:" in _plain(result.stdout)
    assert "probe_failure" in _plain(result.stdout)
    _assert_aborted_jsonl(result)
    assert error_type not in result.stderr


@pytest.mark.parametrize("progress", [False, True], ids=["ordinary", "jsonl"])
def test_interrupt_exits_130_and_jsonl_still_ends_in_an_error_event(
    progress: bool, tmp_path: Path,
) -> None:
    result = _invoke(tmp_path, kind="interrupt", progress=progress)

    assert result.returncode == 130, (result.stdout, result.stderr)
    assert "probe_failure" not in result.stdout + result.stderr
    if progress:
        _assert_aborted_jsonl(result)
        assert result.stdout == ""
    else:
        assert "schema_version" not in result.stderr


@pytest.mark.parametrize("color", [False, True], ids=["plain", "color"])
def test_error_text_strips_controls_and_displays_markup_literally(
    color: bool, tmp_path: Path,
) -> None:
    message = (
        "[bold]literal[/bold] [/] ctrl:\x00A\x07B\x1bC\x7fD\x85E"
        "\u202eF\u2067G\u200bH\ufeffI end"
    )
    result = _invoke(tmp_path, kind="runtime", message=message, color=color)

    assert result.returncode == 1, (result.stdout, result.stderr)
    rendered = strip_ansi(result.stdout)
    assert "[bold]literal[/bold] [/] ctrl:ABCDEFGHI end" in _plain(rendered)
    for control in ("\x00", "\x07", "\x1b", "\x7f", "\x85", "\u202e", "\u2067", "\u200b", "\ufeff"):
        assert control not in rendered
    _assert_aborted_jsonl(result)
    assert "literal" not in result.stderr and "ctrl:" not in result.stderr


@pytest.mark.parametrize("color", [False, True], ids=["plain", "color"])
@pytest.mark.parametrize("kind", ["key", "runtime"])
def test_error_markup_is_literal_for_each_exception_type(
    color: bool, kind: str, tmp_path: Path,
) -> None:
    message = "bad[/] [bold]literal[/bold]"
    result = _invoke(tmp_path, kind=kind, message=message, color=color)

    assert result.returncode == 1, (result.stdout, result.stderr)
    assert message in _plain(result.stdout)
    _assert_aborted_jsonl(result)
    assert "literal" not in result.stderr


@pytest.mark.parametrize("kind, error_type", [("key", "KeyError"), ("runtime", "RuntimeError")])
def test_without_progress_the_friendly_error_stays_on_stderr(
    kind: str, error_type: str, tmp_path: Path,
) -> None:
    result = _invoke(tmp_path, kind=kind, progress=False)

    assert result.returncode == 1
    assert result.stdout == ""
    assert f"Error: {error_type}:" in _plain(result.stderr)
    assert "probe_failure" in result.stderr
    assert "Run with --verbose for the full traceback." in _plain(result.stderr)
    assert "schema_version" not in result.stderr


@pytest.mark.parametrize("malformed", [False, True], ids=["missing", "malformed"])
def test_handled_evidence_failure_keeps_its_output_and_exit_code(
    malformed: bool, tmp_path: Path,
) -> None:
    if malformed:
        (tmp_path / "missing.json").write_text("{broken json", encoding="utf-8")
    ordinary = _invoke(tmp_path, progress=False)
    progress = _invoke(tmp_path)

    assert ordinary.returncode == progress.returncode == 3
    assert progress.stdout == ordinary.stdout
    assert "cannot read --evidence" in _plain(progress.stdout)
    assert progress.stdout.count("Error:") == 1
    _assert_aborted_jsonl(progress)


@pytest.mark.parametrize("tuned, exit_code", [(0.8, 0), (0.5, 2)], ids=["ship", "dont-ship"])
def test_completed_offline_verdict_keeps_its_output_and_end_event(
    tuned: float, exit_code: int, tmp_path: Path,
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({
        "task": {"mode": "metric", "base": 0.5, "tuned": tuned},
        "benchmarks": {"a": {"base": 0.8, "tuned": 0.8}},
    }), encoding="utf-8")
    args = ["--evidence", str(path)]
    ordinary = _invoke(tmp_path, progress=False, arguments=args)
    progress = _invoke(tmp_path, arguments=args)

    assert ordinary.returncode == progress.returncode == exit_code
    assert progress.stdout == ordinary.stdout
    events = [json.loads(line) for line in progress.stderr.splitlines()]
    assert [event["event"] for event in events] == ["start", "end"]
    assert all("error" not in event for event in events)
