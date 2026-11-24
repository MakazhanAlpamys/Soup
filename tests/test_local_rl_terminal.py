"""``soup local-rl`` shows outside text as typed, on the display only (#2489).

The database, prompt, output and scheduler paths the operator typed, and the
errors raised about them, keep ``:name:`` aliases, markup and a backslash
before a bracket as written, and lose their control and bidi characters. A
closing tag in ``--db`` no longer crashes ``status``. No test here trains a
model or writes a real scheduler unit.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.commands import local_rl
from soup_cli.utils import local_rl_scheduler
from soup_cli.utils.local_rl import NightlyTrainResult, init_local_rl_db, record_thumb
from soup_cli.utils.paths import quote_path
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

VALUES = ["plain", "x[red]n", "[/]n", "n:rocket:", "x\\[n", "x\x9b\u202en", "x\x1b\x07n"]
#: What the display drops: C0 controls, a C1 CSI and a bidi override.
DROPPED = ("\x1b", "\x07", "\x9b", "\u202e")
#: A closing tag with nothing to close: ``status`` raised MarkupError on it.
CLOSING_TAG = "a[/x]b"
DASH = "—"
ELLIPSIS = "…"

_NOT_A_WINDOWS_NAME = "{} cannot be part of a file name on Windows"
#: ``VALUES`` that a test writes as part of a real file name.
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

SCAFFOLD_ADVICE = (
    "Linux: cp soup-local-rl.{service,timer} ~/.config/systemd/user/ "
    "&& systemctl --user enable --now soup-local-rl.timer",
    "macOS: cp com.soup.local-rl.plist ~/Library/LaunchAgents/ "
    "&& launchctl load ~/Library/LaunchAgents/com.soup.local-rl.plist",
    "Now: re-run with --once for an ad-hoc train.",
)


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


def _local_rl(*args: str):
    return make_cli_runner().invoke(app, ["local-rl", *args])


def _raising(error: Exception):
    def fail(*_args, **_kwargs):
        raise error

    return fail


def _with_pair(db: str) -> None:
    """``db`` holds one prompt with an up and a down thumb: one DPO pair."""
    init_local_rl_db(db)
    record_thumb(db_path=db, prompt="hello", response="good", thumb="up")
    record_thumb(db_path=db, prompt="hello", response="bad", thumb="down")


def _scaffold(monkeypatch, names=("soup-local-rl.timer",)) -> list:
    """``write_scheduler_files`` returns ``names`` and writes nothing; returns its calls."""
    calls = []

    def write(target_dir, **kwargs):
        calls.append((target_dir, kwargs))
        return {name: f"/units/{name}" for name in names}

    monkeypatch.setattr(local_rl_scheduler, "write_scheduler_files", write)
    return calls


def _trained(monkeypatch, num_pairs: int = 3) -> list:
    """``run_nightly_train`` reports a train into the ``output_dir`` it was given."""
    calls = []

    def run(cfg, *, once, min_pairs, output_dir):
        calls.append((cfg, once, min_pairs, output_dir))
        return NightlyTrainResult("trained", num_pairs, output_dir, "trained")

    monkeypatch.setattr(local_rl, "run_nightly_train", run)
    return calls


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(local_rl, "console", Console(width=1000, height=100))
    return tmp_path


# ---------------------------------------------------------------------------
# Refusals of what the operator typed
# ---------------------------------------------------------------------------

_DB_COMMANDS = {
    "init": ("init",),
    "status": ("status",),
    "record": ("record", "--prompt", "p", "--response", "r", "--thumb", "up"),
    "harvest": ("harvest",),
    "train": ("train", "--model", "m"),
}


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("command", sorted(_DB_COMMANDS))
def test_a_database_outside_cwd_is_named_as_typed(workdir, command, value):
    path = "../" + value

    result = _local_rl(*_DB_COMMANDS[command], "--db", path)

    _assert_shown(result, 2, f"db_path {os.path.basename(path)!r} must stay under cwd")


@pytest.mark.parametrize("value", VALUES)
def test_record_names_an_unknown_thumb_as_typed(workdir, value):
    init_local_rl_db("local_rl.db")

    result = _local_rl("record", "--prompt", "p", "--response", "r", "--thumb", value)

    _assert_shown(result, 2, f"thumb must be one of ['down', 'up'], got {value!r}")


@pytest.mark.parametrize("value", VALUES)
def test_harvest_names_a_missing_database_as_typed(workdir, value):
    db = f"db-{value}.db"

    result = _local_rl("harvest", "--db", db)

    _assert_shown(result, 2, f"db_path not found: {db!r}")
    assert not Path("dpo_pairs.jsonl").exists()


@pytest.mark.parametrize("value", VALUES)
def test_harvest_names_an_output_outside_cwd_as_typed(workdir, value):
    _with_pair("local_rl.db")
    path = "../" + value

    result = _local_rl("harvest", "--output", path)

    _assert_shown(result, 2, _shown(f"output {quote_path(path)} must stay under cwd"))


@pytest.mark.parametrize("value", VALUES)
def test_train_names_an_unknown_train_method_as_typed(workdir, value):
    result = _local_rl("train", "--model", "m", "--train-method", value)

    _assert_shown(result, 2, f"unknown train_method {value!r}; supported: dpo, kto, orpo")


@pytest.mark.parametrize("value", VALUES)
def test_a_scaffold_refusal_is_shown_as_raised(workdir, monkeypatch, value):
    monkeypatch.setattr(
        local_rl_scheduler,
        "write_scheduler_files",
        _raising(ValueError(f"scheduler_file {value} must stay under cwd")),
    )

    result = _local_rl("train", "--model", "m")

    _assert_shown(result, 2, f"scheduler_file {_shown(value)} must stay under cwd")


@pytest.mark.parametrize("value", VALUES)
def test_a_train_refusal_is_shown_as_raised(workdir, monkeypatch, value):
    monkeypatch.setattr(
        local_rl, "run_nightly_train", _raising(ValueError(f"output_dir {value} refused"))
    )

    result = _local_rl("train", "--model", "m", "--once")

    _assert_shown(result, 2, f"output_dir {_shown(value)} refused")


# ---------------------------------------------------------------------------
# What each command reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", FILE_VALUES)
def test_init_names_the_database_as_typed(workdir, value):
    db = f"{value}.db"

    result = _local_rl("init", "--db", db)

    _assert_shown(result, 0, f"DB: {_shown(db)}")
    _assert_shown(result, 0, "Tables: interactions, thumbs, state")
    assert Path(db).is_file()


@pytest.mark.parametrize("value", [*FILE_VALUES, CLOSING_TAG])
def test_status_titles_the_table_with_the_database_as_typed(workdir, value):
    db = f"{value}.db"
    _with_pair(db)

    result = _local_rl("status", "--db", db)

    _assert_shown(result, 0, f"soup local-rl status {DASH} {_shown(db)}")
    _assert_shown(result, 0, "Thumbs up │ 1 │")


def test_status_prints_the_table_for_a_database_named_with_a_closing_tag(workdir):
    # On main this raised MarkupError from the table title instead of printing it.
    db = f"{CLOSING_TAG}.db"
    _with_pair(db)

    result = _local_rl("status", "--db", db)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert result.exception is None, repr(result.exception)
    _assert_shown(result, 0, f"soup local-rl status {DASH} {db}")
    _assert_shown(result, 0, "Interactions │")


@pytest.mark.parametrize("value", VALUES)
def test_status_names_a_missing_database_as_typed(workdir, value):
    db = f"db-{value}.db"

    result = _local_rl("status", "--db", db)

    _assert_shown(result, 2, f"db not found: {_shown(db)}")


@pytest.mark.parametrize("value", VALUES)
def test_the_receipt_names_the_prompt_as_typed(workdir, value):
    init_local_rl_db("local_rl.db")

    result = _local_rl("record", "--prompt", value, "--response", "r", "--thumb", "up")

    _assert_shown(result, 0, f"Recorded up for prompt {_shown(value)}{ELLIPSIS}")
    with sqlite3.connect("local_rl.db") as conn:
        assert conn.execute("SELECT prompt, thumb FROM thumbs").fetchall() == [(value, "up")]


def test_the_receipt_cuts_the_prompt_before_it_strips_it(workdir):
    # The receipt shows the first 32 characters of the typed prompt, then strips
    # the controls among them: the cut counts the controls, as it did before.
    head = "ab\x1b\x07cd\x9b\u202eef:rocket:x\\[n[red]g"
    prompt = head + "-" * (32 - len(head)) + "TAIL-NOT-SHOWN"
    assert len(prompt[:32]) == 32 and "TAIL" not in prompt[:32]
    init_local_rl_db("local_rl.db")

    result = _local_rl("record", "--prompt", prompt, "--response", "r", "--thumb", "down")

    _assert_shown(result, 0, f"Recorded down for prompt {_shown(prompt[:32])}{ELLIPSIS}")
    assert "TAIL" not in strip_ansi(result.output), result.output
    with sqlite3.connect("local_rl.db") as conn:
        assert conn.execute("SELECT prompt FROM thumbs").fetchall() == [(prompt,)]


@pytest.mark.parametrize("value", FILE_VALUES)
def test_harvest_names_the_database_and_output_as_typed(workdir, value):
    db, output = f"{value}.db", f"pairs-{value}.jsonl"
    _with_pair(db)

    result = _local_rl("harvest", "--db", db, "--output", output)

    _assert_shown(result, 0, f"DB: {_shown(db)}")
    _assert_shown(result, 0, f"Output: {_shown(output)}")
    _assert_shown(result, 0, "Pairs: 1")
    row = json.loads(Path(output).read_text(encoding="utf-8"))
    assert row == {"prompt": "hello", "chosen": "good", "rejected": "bad"}


@pytest.mark.parametrize("value", VALUES)
def test_the_scaffold_names_the_directory_and_files_as_written(workdir, monkeypatch, value):
    calls = _scaffold(monkeypatch, names=("soup-local-rl.timer", f"unit-{value}"))

    result = _local_rl("train", "--model", "m", "--scheduler-dir", value)

    _assert_shown(result, 0, f"Rendered the nightly-train scaffold into {_shown(value)}:")
    _assert_shown(result, 0, "soup-local-rl.timer")
    _assert_shown(result, 0, f"unit-{_shown(value)}")
    for line in SCAFFOLD_ADVICE:
        _assert_shown(result, 0, line)
    assert [target for target, _kwargs in calls] == [value]


@pytest.mark.parametrize("value", VALUES)
def test_a_failed_training_subprocess_is_shown_as_raised(workdir, monkeypatch, value):
    monkeypatch.setattr(
        local_rl,
        "run_nightly_train",
        _raising(subprocess.SubprocessError(f"soup train exited 1: {value}")),
    )

    result = _local_rl("train", "--model", "m", "--once")

    _assert_shown(result, 1, f"soup local-rl train {DASH} failed")
    _assert_shown(result, 1, f"Training subprocess failed: soup train exited 1: {_shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_a_skipped_train_is_shown_as_reported(workdir, monkeypatch, value):
    skipped = NightlyTrainResult("skipped_insufficient_pairs", 1, None, f"1 pairs < {value}")
    monkeypatch.setattr(local_rl, "run_nightly_train", lambda cfg, **_kwargs: skipped)

    result = _local_rl("train", "--model", "m", "--once")

    _assert_shown(result, 0, f"Skipped: 1 pairs < {_shown(value)}")


@pytest.mark.parametrize("value", VALUES)
def test_the_done_panel_names_the_output_as_typed(workdir, monkeypatch, value):
    calls = _trained(monkeypatch)

    result = _local_rl("train", "--model", "m", "--once", "--output", value)

    _assert_shown(result, 0, "Trained: 3 pairs via dpo")
    _assert_shown(result, 0, f"Output: {_shown(value)}")
    assert [output_dir for *_rest, output_dir in calls] == [value]


# ---------------------------------------------------------------------------
# The lines keep their colours
# ---------------------------------------------------------------------------


@pytest.fixture
def coloured(workdir, monkeypatch):
    monkeypatch.setattr(
        local_rl,
        "console",
        Console(
            width=1000, height=100, force_terminal=True, color_system="standard", no_color=False
        ),
    )


def test_a_refusal_stays_red(coloured):
    result = _local_rl("init", "--db", "../out.db")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "\x1b[31mdb_path 'out.db' must stay under cwd\x1b[0m" in result.output


def test_a_missing_database_stays_red(coloured):
    result = _local_rl("status", "--db", "missing.db")

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "\x1b[31mdb not found: missing.db\x1b[0m" in result.output


def test_the_receipt_stays_green(coloured):
    init_local_rl_db("local_rl.db")

    result = _local_rl("record", "--prompt", "hello", "--response", "r", "--thumb", "up")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"\x1b[32mRecorded up for prompt hello{ELLIPSIS}\x1b[0m" in result.output


def test_the_init_panel_keeps_its_bold_values(coloured):
    result = _local_rl("init")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "│ DB:     \x1b[1mlocal_rl.db\x1b[0m " in result.output
    assert (
        "│ Tables: \x1b[1minteractions\x1b[0m, \x1b[1mthumbs\x1b[0m, \x1b[1mstate\x1b[0m "
        in result.output
    )


def test_the_status_title_stays_italic(coloured):
    init_local_rl_db("local_rl.db")

    result = _local_rl("status")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert f"\x1b[3m soup local-rl status {DASH} \x1b[0m\n" in result.output
    assert "\x1b[3m      local_rl.db       \x1b[0m\n" in result.output


def test_the_harvest_panel_keeps_its_bold_values(coloured):
    _with_pair("local_rl.db")

    result = _local_rl("harvest", "--output", "pairs.jsonl")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "│ DB:     \x1b[1mlocal_rl.db\x1b[0m " in result.output
    assert "│ Output: \x1b[1mpairs.jsonl\x1b[0m " in result.output
    assert "│ Pairs:  \x1b[1m1\x1b[0m " in result.output


def test_the_scaffold_panel_keeps_its_bold_names_and_dim_labels(coloured, monkeypatch):
    _scaffold(monkeypatch, names=("soup-local-rl.timer", "soup-local-rl.service"))

    result = _local_rl("train", "--model", "m")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    output = result.output
    assert (
        "│ Rendered the nightly-train scaffold into \x1b[1mlocal-rl-scheduler\x1b[0m: " in output
    )
    assert "│   \x1b[1msoup-local-rl.service\x1b[0m " in output
    assert "│   \x1b[1msoup-local-rl.timer\x1b[0m " in output
    assert "│ \x1b[2mLinux:\x1b[0m cp soup-local-rl.{service,timer} " in output
    assert "│ \x1b[2mmacOS:\x1b[0m cp com.soup.local-rl.plist " in output
    assert (
        "│ \x1b[2mNow:\x1b[0m re-run with \x1b[1m--once\x1b[0m for an ad-hoc train. " in output
    )


def test_a_failed_training_subprocess_keeps_its_red_label(coloured, monkeypatch):
    monkeypatch.setattr(
        local_rl,
        "run_nightly_train",
        _raising(subprocess.SubprocessError("soup train exited 1")),
    )

    result = _local_rl("train", "--model", "m", "--once")

    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "│ \x1b[31mTraining subprocess failed:\x1b[0m soup train exited 1 " in result.output


def test_a_skipped_train_keeps_its_yellow_label(coloured, monkeypatch):
    skipped = NightlyTrainResult("skipped_insufficient_pairs", 1, None, "1 pairs < min_pairs 10")
    monkeypatch.setattr(local_rl, "run_nightly_train", lambda cfg, **_kwargs: skipped)

    result = _local_rl("train", "--model", "m", "--once")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "│ \x1b[33mSkipped:\x1b[0m 1 pairs < min_pairs 10 " in result.output


def test_the_done_panel_keeps_its_bold_values(coloured, monkeypatch):
    _trained(monkeypatch)

    result = _local_rl("train", "--model", "m", "--once", "--output", "adapter")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "│ Trained: \x1b[1m3\x1b[0m pairs via \x1b[1mdpo\x1b[0m " in result.output
    assert "│ Output:  \x1b[1madapter\x1b[0m " in result.output
