"""Env lock contents and paths are data, never terminal control sequences."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import env
from soup_cli.utils.env_lock import EnvEntry, EnvLock, read_lock, write_lock
from tests.conftest import strip_ansi

DIRTY = "[bold]value\x1b\x07\x9b\u202e[/]"
CLEAN = "[bold]value[/]"


def _lock():
    return EnvLock(
        soup_version="0.77.1", python_version="3.11.12", platform="linux-x86_64",
        cuda_version="12.4", entries=(EnvEntry("torch", "2.5.1", "pip"),),
        created_at="2026-10-28T00:00:00Z",
    )


def _plain(result):
    assert result.exit_code == 0, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert not any(c in output for c in ("\x1b", "\x07", "\x9b", "\u202e")), repr(output)
    return " ".join(output.split())


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env, "console", Console(width=180))
    monkeypatch.setattr(
        env, "current_declared_bounds_check", lambda: SimpleNamespace(ok=True)
    )
    return tmp_path


@pytest.mark.parametrize(
    "field", ["soup_version", "python_version", "platform", "cuda_version", "created_at",
              "package_name", "package_version"]
)
def test_status_sanitizes_each_metadata_field_without_changing_the_lock(workdir, field):
    lock = _lock()
    if field.startswith("package_"):
        entry = replace(lock.entries[0], **{field.removeprefix("package_"): DIRTY})
        lock = replace(lock, entries=(entry,))
    else:
        lock = replace(lock, **{field: DIRTY})
    write_lock(lock, "soup-env.lock")
    before = Path("soup-env.lock").read_bytes()

    output = _plain(CliRunner().invoke(app, ["env", "status"]))

    assert CLEAN in output
    assert Path("soup-env.lock").read_bytes() == before
    assert read_lock("soup-env.lock") == lock


def test_lock_sanitizes_snapshot_and_success_path_but_keeps_file_contents(workdir, monkeypatch):
    lock = replace(_lock(), platform=DIRTY, cuda_version=DIRTY)
    monkeypatch.setattr(env, "snapshot_env", lambda: lock)
    name = "[bold]lock\x9b\u202e[end].json"
    output = _plain(CliRunner().invoke(app, ["env", "lock", "--output", name]))

    assert CLEAN in output
    assert "[bold]lock[end].json" in output
    assert read_lock(name) == lock


@pytest.mark.parametrize("command", ["status", "check", "fix"])
def test_missing_lock_path_is_literal_and_terminal_safe(workdir, command):
    name = "[bold]missing\x9b\u202e[end].lock"
    result = CliRunner().invoke(app, ["env", command, "--lock", name])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert "[bold]missing[end].lock" in output
    assert "\x9b" not in output and "\u202e" not in output
    assert not Path(name).exists()


@pytest.mark.parametrize("fmt", ["requirements", "uv-pip"])
def test_fix_sanitizes_only_displayed_plan_and_output_path(workdir, fmt):
    lock = replace(_lock(), entries=(EnvEntry("torch", DIRTY, "pip"),))
    write_lock(lock, "soup-env.lock")
    output_path = "requirements\x9b\u202e.txt"
    result = CliRunner().invoke(
        app, ["env", "fix", "--format", fmt, "--output", output_path]
    )
    output = _plain(result)
    assert CLEAN in output
    assert "Wrote requirements to requirements.txt" in output
    assert f"torch=={DIRTY}" in Path(output_path).read_text()
    assert read_lock("soup-env.lock") == lock


def test_check_sanitizes_real_abi_drift_description(workdir, monkeypatch):
    write_lock(replace(_lock(), entries=(EnvEntry("torch", DIRTY, "pip"),)), "soup-env.lock")
    monkeypatch.setattr(env, "snapshot_env", _lock)
    result = CliRunner().invoke(app, ["env", "check"])
    assert result.exit_code == 3, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert CLEAN in output
    assert not any(c in output for c in ("\x1b", "\x07", "\x9b", "\u202e"))


def test_check_sanitizes_declared_bound_metadata(workdir, monkeypatch):
    write_lock(_lock(), "soup-env.lock")
    monkeypatch.setattr(env, "snapshot_env", _lock)
    monkeypatch.setattr(
        env, "current_declared_bounds_check",
        lambda: SimpleNamespace(
            ok=False, violation_count=1,
            violations=[SimpleNamespace(name=DIRTY, installed=DIRTY, specifier=DIRTY)],
        ),
    )
    result = CliRunner().invoke(app, ["env", "check"])
    assert result.exit_code == 3, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert output.count(CLEAN) == 3
    assert not any(c in output for c in ("\x1b", "\x07", "\x9b", "\u202e"))
