"""Env lock contents and paths are data, never terminal control sequences."""

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import env
from soup_cli.utils.env_lock import (
    BoundsCheck,
    BoundViolation,
    EnvEntry,
    EnvLock,
    read_lock,
    write_lock,
)
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

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


@skip_colon_on_windows
def test_lock_receipt_keeps_alias_in_the_written_path(workdir, monkeypatch):
    lock = _lock()
    monkeypatch.setattr(env, "snapshot_env", lambda: lock)
    name = "[bold]snapshot-:warning:\x9b\u202e.lock"

    result = make_cli_runner().invoke(app, ["env", "lock", "--output", name])

    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    assert "Locked 1 packages to [bold]snapshot-:warning:.lock" in _plain(result)
    assert read_lock(name) == lock
    assert not Path("[bold]snapshot-:warning:.lock").exists()


def test_lock_receipt_keeps_alias_in_snapshot_metadata(workdir, monkeypatch):
    lock = replace(_lock(), platform="[bold]platform-:warning:\x9b\u202e[/]")
    monkeypatch.setattr(env, "snapshot_env", lambda: lock)

    result = make_cli_runner().invoke(app, ["env", "lock"])

    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    assert "Platform: [bold]platform-:warning:[/] | CUDA: 12.4" in _plain(result)
    assert read_lock("soup-env.lock") == lock


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


def test_lock_receipt_preserves_backslash_in_written_path(workdir, monkeypatch):
    lock = _lock()
    monkeypatch.setattr(env, "snapshot_env", lambda: lock)
    name = "nested\\[unclosed\x9b\u202e.lock"
    Path(name).parent.mkdir(exist_ok=True)

    result = make_cli_runner().invoke(app, ["env", "lock", "--output", name])

    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    assert r"Locked 1 packages to nested\[unclosed.lock" in _plain(result)
    assert read_lock(name) == lock


def test_lock_receipt_preserves_backslash_in_metadata(workdir, monkeypatch):
    lock = replace(_lock(), platform="host\\[unclosed\x9b\u202e")
    monkeypatch.setattr(env, "snapshot_env", lambda: lock)

    result = make_cli_runner().invoke(app, ["env", "lock"])

    assert "\x9b" not in result.stdout and "\u202e" not in result.stdout
    assert r"Platform: host\[unclosed | CUDA: 12.4" in _plain(result)
    assert read_lock("soup-env.lock") == lock


# ---------------------------------------------------------------------------
# #2489: every line that carries outside text shows it as typed
# ---------------------------------------------------------------------------
#
# A lock-file field, a typed path or format, a declared bound and an error
# message keep ``:name:`` aliases, markup and a backslash before a bracket as
# written, and lose their control and bidi characters, on the display only.

VALUES = ["plain", "x[red]n", "[/]n", "n:rocket:", "x\\[n", "x\x9b\u202en", "x\x1b\x07n"]
#: What the display drops: C0 controls, a C1 CSI and a bidi override.
DROPPED = ("\x1b", "\x07", "\x9b", "\u202e")
BOUND = {"name": "transformers", "installed": "4.40.0", "specifier": ">=4.45"}


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


def _cells(result, first: str) -> list:
    """The cells of the table row whose first cell is ``first``."""
    assert result.exit_code == 0, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert [char for char in DROPPED if char in output] == []
    for line in output.splitlines():
        cells = [cell.strip() for cell in re.split("[│┃|]", line) if cell.strip()]
        if cells and cells[0] == first:
            return cells
    raise AssertionError(f"no row starting {first!r} in:\n{output}")


def _fail_with(message: str):
    def fail(*_args, **_kwargs):
        raise ValueError(message)

    return fail


def _env(*args: str):
    return make_cli_runner().invoke(app, ["env", *args])


def _lock_with(field: str, value: str) -> EnvLock:
    lock = _lock()
    if field in ("name", "version"):
        return replace(lock, entries=(replace(lock.entries[0], **{field: value}),))
    return replace(lock, **{field: value})


def _violating(monkeypatch, **fields: str) -> None:
    violation = BoundViolation(**{**BOUND, **fields}, extra=None)
    monkeypatch.setattr(
        env,
        "current_declared_bounds_check",
        lambda: BoundsCheck(ok=False, violation_count=1, violations=(violation,)),
    )


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("command", ["lock", "fix"])
def test_an_output_outside_cwd_is_named_as_typed(workdir, command, value):
    write_lock(_lock(), "soup-env.lock")
    path = "../" + value

    result = _env(command, "--output", path)

    _assert_shown(result, 2, f"output {path!r} is outside cwd")


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("command", ["status", "check", "fix"])
def test_a_lock_outside_cwd_is_named_as_typed(workdir, command, value):
    path = "../" + value

    result = _env(command, "--lock", path)

    _assert_shown(result, 2, f"lock {path!r} is outside cwd")


@pytest.mark.parametrize("value", VALUES)
def test_lock_shows_a_snapshot_failure_as_raised(workdir, monkeypatch, value):
    monkeypatch.setattr(env, "snapshot_env", _fail_with(f"cannot snapshot {value}"))

    result = _env("lock")

    _assert_shown(result, 2, f"cannot snapshot {_shown(value)}")
    assert not Path("soup-env.lock").exists()


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize(
    "field", ["soup_version", "python_version", "platform", "cuda_version", "created_at"]
)
def test_status_shows_each_lock_field_as_written(workdir, field, value):
    write_lock(_lock_with(field, value), "soup-env.lock")

    assert _cells(_env("status"), field) == [field, _shown(value)]


@pytest.mark.parametrize("value", VALUES)
def test_status_shows_package_names_and_versions_as_written(workdir, value):
    entries = (EnvEntry(value, "2.5.1", "pip"), EnvEntry("torch", value, "conda"))
    write_lock(replace(_lock(), entries=entries), "soup-env.lock")

    result = _env("status")

    assert _cells(result, _shown(value)) == [_shown(value), "2.5.1", "pip"]
    assert _cells(result, "torch") == ["torch", _shown(value), "conda"]


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize(
    "field,line",
    [
        ("platform", "- platform: {} -> linux-x86_64"),
        ("cuda_version", "- cuda: {} -> 12.4"),
        ("version", "- torch: {} -> 2.5.1"),
    ],
)
def test_check_shows_each_drift_line_as_written(workdir, monkeypatch, field, line, value):
    write_lock(_lock_with(field, value), "soup-env.lock")
    monkeypatch.setattr(env, "snapshot_env", _lock)

    result = _env("check")

    _assert_shown(result, 3, "1 ABI-sensitive drift(s):")
    _assert_shown(result, 3, line.format(_shown(value)))


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("field", sorted(BOUND))
def test_check_shows_each_bound_violation_as_written(workdir, monkeypatch, field, value):
    write_lock(_lock(), "soup-env.lock")
    monkeypatch.setattr(env, "snapshot_env", _lock)
    _violating(monkeypatch, **{field: value})
    shown = {**BOUND, field: _shown(value)}

    result = _env("check")

    _assert_shown(
        result,
        3,
        f"- {shown['name']} {shown['installed']} violates declared bound {shown['specifier']}",
    )


@pytest.mark.parametrize("value", VALUES)
def test_fix_names_an_unknown_format_as_typed(workdir, value):
    write_lock(_lock(), "soup-env.lock")

    result = _env("fix", "--format", value)

    _assert_shown(result, 2, f"format must be one of {{requirements, uv-pip}}, got {value!r}")


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize(
    "fmt,field,line",
    [
        ("uv-pip", "platform", "# python 3.11.12 | platform {}"),
        ("uv-pip", "cuda_version", "# CUDA: {}"),
        ("uv-pip", "name", "uv pip install --python 3.11 '{}==2.5.1'"),
        ("uv-pip", "version", "uv pip install --python 3.11 'torch=={}'"),
        ("requirements", "name", "{}==2.5.1"),
        ("requirements", "version", "torch=={}"),
    ],
)
def test_fix_shows_the_install_plan_as_rendered(workdir, fmt, field, line, value):
    write_lock(_lock_with(field, value), "soup-env.lock")

    result = _env("fix", "--format", fmt)

    _assert_shown(result, 0, f"env fix — install plan ({fmt})")
    _assert_shown(result, 0, line.format(_shown(value)))


@pytest.mark.parametrize("value", VALUES)
def test_fix_shows_a_write_failure_as_raised(workdir, monkeypatch, value):
    write_lock(_lock(), "soup-env.lock")
    monkeypatch.setattr(env, "write_requirements_txt", _fail_with(f"cannot write {value}"))

    result = _env("fix", "--output", "requirements.txt")

    _assert_shown(result, 2, f"cannot write {_shown(value)}")


# ---------------------------------------------------------------------------
# The lines keep their colours
# ---------------------------------------------------------------------------


@pytest.fixture
def coloured(workdir, monkeypatch):
    monkeypatch.setattr(
        env,
        "console",
        Console(
            width=1000, height=100, force_terminal=True, color_system="standard", no_color=False
        ),
    )


def test_a_refusal_stays_red(coloured):
    result = _env("lock", "--output", "../out.lock")

    assert result.exit_code == 2
    assert "\x1b[31moutput '../out.lock' is outside cwd\x1b[0m" in result.output


def test_the_drift_panel_keeps_its_red_header_and_plain_lines(coloured, monkeypatch):
    write_lock(_lock_with("platform", "other"), "soup-env.lock")
    monkeypatch.setattr(env, "snapshot_env", _lock)

    result = _env("check")

    assert result.exit_code == 3
    assert "\x1b[31m1 ABI-sensitive drift(s):\x1b[0m" in result.output
    assert "\x1b[31m│\x1b[0m - platform: other -> linux-x86_64 " in result.output


def test_the_bound_panel_keeps_its_red_header_and_yellow_advice(coloured, monkeypatch):
    write_lock(_lock(), "soup-env.lock")
    monkeypatch.setattr(env, "snapshot_env", _lock)
    _violating(monkeypatch)

    result = _env("check")

    assert result.exit_code == 3
    assert (
        "\x1b[31m1 package(s) violate this distribution's own declared bounds:\x1b[0m"
        in result.output
    )
    assert (
        "\x1b[31m│\x1b[0m - transformers 4.40.0 violates declared bound >=4.45 "
        in result.output
    )
    assert (
        "\x1b[33mA later install (e.g. `pip install vllm`) likely moved these packages "
        "outside the range soup-cli declares. Reinstall soup-cli, or pin the listed "
        "package(s) back into range.\x1b[0m" in result.output
    )


def test_the_install_plan_keeps_its_green_border(coloured):
    write_lock(_lock(), "soup-env.lock")

    result = _env("fix", "--format", "requirements")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "\x1b[32m│\x1b[0m torch==2.5.1 " in result.output
