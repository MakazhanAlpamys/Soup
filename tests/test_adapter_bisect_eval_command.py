"""A command that never starts is not evidence of a broken checkpoint."""

from __future__ import annotations

import errno
import json
import ntpath
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import adapters
from tests.conftest import strip_ansi


def _invoke(command, *history, output="bisect.json"):
    return CliRunner().invoke(
        app, ["adapters", "bisect", *(history or ("ckpt-1", "ckpt-2")),
              "--eval-command", command, "--output", output]
    )


def _plain(result):
    return " ".join(strip_ansi(result.output).split())


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (r".\tool.bat {ckpt}", [r".\tool.bat", "{ckpt}"]),
        (r"C:\dir\tool.bat {ckpt}", [r"C:\dir\tool.bat", "{ckpt}"]),
        (r'"C:\dir with space\tool.bat" {ckpt}', [r"C:\dir with space\tool.bat", "{ckpt}"]),
        (r'".\tool.bat" "" "prefix {ckpt}"', [r".\tool.bat", "", "prefix {ckpt}"]),
    ],
)
def test_windows_split_preserves_paths_and_argument_boundaries(command, expected):
    assert adapters._split_bisect_eval_command(command, windows=True) == expected


@pytest.mark.parametrize(
    "program", [r".\tool.bat", r"C:\dir\tool.bat", r"C:\dir with space\tool.bat"]
)
def test_windows_cli_passes_the_written_path_and_checkpoint_as_argv(
    workdir, monkeypatch, program
):
    simulated = SimpleNamespace(**vars(os))
    simulated.name = "nt"
    simulated.path = SimpleNamespace(**vars(ntpath))
    simulated.path.isfile = lambda path: path == program
    simulated.access = lambda path, mode: True
    monkeypatch.setattr(adapters, "os", simulated)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    ids = ("checkpoint with spaces", "checkpoint 'single' \"double\" ; $literal")
    written_program = f'"{program}"' if " " in program else program
    command = f'{written_program} --model "{{ckpt}}"'
    result = _invoke(command, *ids)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert [argv for argv, _ in calls] == [[program, "--model", ckpt] for ckpt in ids]
    assert all(not kwargs.get("shell", False) for _, kwargs in calls)
    assert json.loads(Path("bisect.json").read_text())["verdict"] == "ALL_OK"


@pytest.mark.parametrize("existing_output", [False, True])
def test_missing_explicit_program_refuses_before_any_probe(
    workdir, monkeypatch, existing_output
):
    if existing_output:
        Path("bisect.json").write_text("keep existing result", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("probe started"))

    result = _invoke("./missing.sh {ckpt}")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "./missing.sh" in _plain(result)
    assert "BROKEN_AT" not in _plain(result) and "Verdict:" not in _plain(result)
    if existing_output:
        assert Path("bisect.json").read_text() == "keep existing result"
    else:
        assert not Path("bisect.json").exists()


@pytest.mark.parametrize("kind", ["directory", "not-executable"])
def test_unusable_explicit_program_refuses_before_any_probe(workdir, monkeypatch, kind):
    program = workdir / "unusable"
    if kind == "directory":
        program.mkdir()
    else:
        program.write_text("not executable", encoding="utf-8")
        local_os = SimpleNamespace(**vars(os))
        local_os.access = lambda path, mode: False
        monkeypatch.setattr(adapters, "os", local_os)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("probe started"))

    result = _invoke(f'"{program}" {{ckpt}}')

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "could not start" in _plain(result)
    expected_error = "not a regular file" if kind == "directory" else "not executable"
    assert expected_error in _plain(result)
    assert "Verdict:" not in _plain(result)
    assert not Path("bisect.json").exists()


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError(errno.ENOENT, "gone"), PermissionError(errno.EACCES, "denied"),
     OSError(errno.ENOEXEC, "invalid executable")],
)
def test_program_start_errors_stop_without_a_verdict_or_result(workdir, monkeypatch, error):
    calls = []

    def cannot_start(argv, **kwargs):
        calls.append(argv)
        raise error

    monkeypatch.setattr(subprocess, "run", cannot_start)
    result = _invoke(f'"{sys.executable}" {{ckpt}}')

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert len(calls) == 1
    assert repr(sys.executable) in _plain(result)
    assert error.strerror in _plain(result)
    assert "BROKEN_AT" not in _plain(result) and "Verdict:" not in _plain(result)
    assert not Path("bisect.json").exists()


def test_checkpoint_text_is_one_argument_after_template_split(workdir, monkeypatch):
    calls = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    ids = ("first 'quoted' checkpoint", "last \"quoted\" checkpoint; $not_a_shell")
    result = _invoke(f'"{sys.executable}" "prefix {{ckpt}}"', *ids)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert calls == [[sys.executable, "prefix " + ckpt] for ckpt in ids]


@pytest.mark.parametrize("returncode", [0, 1])
def test_started_program_exit_code_still_means_pass_or_fail(workdir, monkeypatch, returncode):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=returncode)
    )
    result = _invoke(f'"{sys.executable}" {{ckpt}}')

    assert result.exit_code == (0 if returncode == 0 else 2)
    document = json.loads(Path("bisect.json").read_text())
    assert document["verdict"] == ("ALL_OK" if returncode == 0 else "BROKEN_AT")


def test_timeout_remains_a_failed_probe(workdir, monkeypatch):
    def timed_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, timeout=3600)

    monkeypatch.setattr(subprocess, "run", timed_out)
    result = _invoke(f'"{sys.executable}" {{ckpt}}')

    assert result.exit_code == 2
    assert "Eval failed for ckpt-1" in _plain(result)
    assert json.loads(Path("bisect.json").read_text())["verdict"] == "BROKEN_AT"


def test_start_error_is_literal_and_terminal_safe(workdir, monkeypatch):
    def failed(argv, **kwargs):
        raise OSError(errno.EIO, "[bold]bad[/]\x1b\x9b\u202epath")

    monkeypatch.setattr(subprocess, "run", failed)
    result = _invoke(f'"{sys.executable}" {{ckpt}}')

    assert result.exit_code == 2
    text = strip_ansi(result.output)
    assert "[bold]bad[/]path" in text
    assert not any(char in text for char in ("\x1b", "\x9b", "\u202e"))
    assert not Path("bisect.json").exists()


def test_a_real_child_receives_checkpoint_text_without_shell_interpretation(workdir):
    probe = workdir / "argument_probe.py"
    probe.write_text(
        "import json, sys\n"
        "with open('received.jsonl', 'a', encoding='utf-8') as target:\n"
        "    target.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    ids = ("first 'quoted' checkpoint", 'second "quoted"; $(literal)')
    result = _invoke(f'"{sys.executable}" "{probe}" "{{ckpt}}"', *ids)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    received = [
        json.loads(line) for line in Path("received.jsonl").read_text().splitlines()
    ]
    assert received == [[ckpt] for ckpt in ids]
    assert json.loads(Path("bisect.json").read_text())["verdict"] == "ALL_OK"
