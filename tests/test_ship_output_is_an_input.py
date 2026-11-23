"""``soup ship`` refuses an output path that is one of the run's own input files (exit 3).

``soup ship --evidence ev.json --output ev.json`` used to exit 0 and REPLACE the evidence file it
had just read with the verdict; a live run (``--task-eval tasks.jsonl --output tasks.jsonl``)
evaluated both models and then replaced the task file, and ``--output soup.yaml`` with
``--config soup.yaml`` the same. Now every output (``--output``, ``--emit-evidence``) is compared
with every input file the run was given (``--evidence``, ``--task-eval`` as given or inherited
from ``--config``, ``--config``, ``--corrections``, ``--held-out``, a ``--baseline`` file, a
``--reward-fn`` file) and with the other output, by real path (``realpath`` and ``normcase``, so
a case difference on Windows, a link or a hard link is the same file), and the run is refused by
name before any model loads and before anything is written (the config, the corrections and the
held-out file are read first, to know what the run is). The task-rows file keeps its own check.
Windows also drops a trailing dot or space from a name, so two outputs that do not exist yet and
differ only by those are one file there.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

CONFIG = "base: org/model\ntask: sft\ndata:\n  train: ./train.jsonl\noutput: ./out\n"
TASKS = '{"prompt": "2+2", "expected": "4"}\n'
CORRECTION = (
    '{"edit_id": "e1", "expected": "four", "original": "five", "paraphrases": ["what is 2+2?"]}\n'
)
HELD_OUT = (
    '{"edit_id": "h1", "prompt": "2+3", "expected": "five", "original": "six"}\n'
)
LIVE = ["--base", "org/model", "--adapter", "adapter"]
GATE = ["--gate", "loop-v2"]

# input flag -> (the file's name, its content, the other arguments the run needs to reach the check)
INPUT_FILES = {
    "--evidence": ("ev.json", "{}", []),
    "--task-eval": ("tasks.jsonl", TASKS, LIVE),
    "--config": ("soup.yaml", CONFIG, ["--evidence", "other.json"]),
    "--corrections": ("corrections.jsonl", CORRECTION, [*LIVE, "--task-eval", "t.jsonl", *GATE]),
    "--held-out": ("held.jsonl", HELD_OUT, [*LIVE, "--task-eval", "t.jsonl", *GATE]),
    "--baseline": ("baseline.json", "{}", [*LIVE, "--task-eval", "t.jsonl"]),
    "--reward-fn": (
        "reward.py", "def reward(prompt, expected, answer):\n    return 1.0\n",
        [*LIVE, "--task-eval", "t.jsonl", "--task-mode", "reward"],
    ),
}
OUTPUT_FLAGS = ("--output", "--emit-evidence")


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).split())


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "t.jsonl").write_text(TASKS, encoding="utf-8")
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    return tmp_path


def _prepare(root: Path, input_flag: str) -> "tuple[str, str, list]":
    name, content, extra = INPUT_FILES[input_flag]
    (root / name).write_text(content, encoding="utf-8")
    return name, content, extra


def _reachable(input_flag: str, output_flag: str) -> bool:
    """``--gate loop-v2`` refuses ``--emit-evidence`` itself (live run only), before this check."""
    return not (input_flag in ("--corrections", "--held-out") and output_flag == "--emit-evidence")


@pytest.mark.parametrize(
    ("input_flag", "output_flag"),
    [
        (input_flag, output_flag)
        for input_flag in sorted(INPUT_FILES)
        for output_flag in OUTPUT_FLAGS
        if _reachable(input_flag, output_flag)
    ],
)
def test_an_output_that_is_an_input_file_is_refused_before_anything_runs(
    root, input_flag, output_flag
):
    name, content, extra = _prepare(root, input_flag)

    result = runner.invoke(app, ["ship", *extra, input_flag, name, output_flag, name])

    out = _plain(result.output)
    assert result.exit_code == 3, (out, repr(result.exception))
    assert f"{output_flag} '{name}' is the same file as {input_flag} '{name}'" in out, out
    assert f"choose a different {output_flag}" in out
    assert (root / name).read_text(encoding="utf-8") == content  # the input is untouched


def test_the_input_survives_when_the_output_names_it_in_another_spelling(root):
    name, content, extra = _prepare(root, "--evidence")

    result = runner.invoke(app, ["ship", "--evidence", name, "--output", "./" + name])

    assert result.exit_code == 3, (_plain(result.output), repr(result.exception))
    assert "is the same file as --evidence" in _plain(result.output)
    assert (root / name).read_text(encoding="utf-8") == content


def test_a_task_file_that_comes_from_the_config_is_an_input_too(root):
    (root / "tasks.jsonl").write_text(TASKS, encoding="utf-8")
    (root / "soup.yaml").write_text(
        CONFIG + "eval:\n  ship:\n    task_eval: tasks.jsonl\n", encoding="utf-8"
    )

    result = runner.invoke(
        app, ["ship", *LIVE, "--config", "soup.yaml", "--output", "tasks.jsonl"],
    )

    assert result.exit_code == 3, (_plain(result.output), repr(result.exception))
    assert "--output 'tasks.jsonl' is the same file as --task-eval 'tasks.jsonl'" in _plain(
        result.output
    )
    assert (root / "tasks.jsonl").read_text(encoding="utf-8") == TASKS


def test_the_two_outputs_may_not_be_the_same_file(root):
    (root / "ev.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app, ["ship", "--evidence", "ev.json", "--output", "out.json",
              "--emit-evidence", "out.json"],
    )

    out = _plain(result.output)
    assert result.exit_code == 3, (out, repr(result.exception))
    assert "--emit-evidence 'out.json' is the same file as --output 'out.json'" in out


def test_different_files_are_not_refused_by_this_check(root):
    (root / "ev.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app, ["ship", "--evidence", "ev.json", "--output", "verdict.json",
              "--emit-evidence", "emitted.json"],
    )

    assert "is the same file as" not in _plain(result.output)


def test_a_registry_baseline_is_not_a_file_and_is_not_compared(root):
    result = runner.invoke(
        app, ["ship", *LIVE, "--task-eval", "t.jsonl", "--baseline", "registry://abc12",
              "--output", "registry://abc12"],
    )

    assert "is the same file as" not in _plain(result.output)


@pytest.mark.skipif(sys.platform != "win32", reason="a case-insensitive file system is Windows")
def test_a_different_case_of_the_same_name_is_the_same_file_on_windows(root):
    name, content, _ = _prepare(root, "--evidence")

    result = runner.invoke(app, ["ship", "--evidence", name, "--output", name.upper()])

    assert result.exit_code == 3, (_plain(result.output), repr(result.exception))
    assert "is the same file as --evidence" in _plain(result.output)
    assert (root / name).read_text(encoding="utf-8") == content


def test_a_hard_link_to_an_input_is_the_same_file(root):
    name, content, _ = _prepare(root, "--evidence")
    try:
        os.link(root / name, root / "alias.json")
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("this file system has no hard links")

    result = runner.invoke(app, ["ship", "--evidence", name, "--output", "alias.json"])

    assert result.exit_code == 3, (_plain(result.output), repr(result.exception))
    assert "is the same file as --evidence" in _plain(result.output)
    assert (root / name).read_text(encoding="utf-8") == content


@pytest.mark.requires_symlink
def test_a_symlink_to_an_input_is_the_same_file(root):
    name, content, _ = _prepare(root, "--evidence")
    os.symlink(root / name, root / "alias.json")

    result = runner.invoke(app, ["ship", "--evidence", name, "--output", "alias.json"])

    assert result.exit_code == 3, (_plain(result.output), repr(result.exception))
    assert (root / name).read_text(encoding="utf-8") == content


# --- names that Windows makes equal, for files that do not exist yet ---------------------------
# ``realpath`` keeps a trailing dot or space when the file is not there, and ``samefile`` needs both
# files to exist, so ``--output x.json --emit-evidence x.json.`` passed the comparison and Win32
# then wrote both to ``x.json``. Inputs always exist (the OS resolves their spelling), so this is
# the outputs against each other.

WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32", reason="only Windows drops a trailing dot or space from a name"
)


@WINDOWS_ONLY
@pytest.mark.parametrize(
    "spelling",
    [
        "x.json.", "x.json..", "x.json...", "x.json ", "x.json. ", "X.JSON.", "./x.json.",
        "sub/../x.json.",
    ],
    ids=["dot", "two dots", "three dots", "space", "dot and space", "case and dot", "dot-slash",
         "parent"],
)
def test_two_outputs_that_differ_only_by_a_trailing_dot_or_space_are_one_file_on_windows(
    root, spelling
):
    (root / "ev.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app, ["ship", "--evidence", "ev.json", "--output", "x.json", "--emit-evidence", spelling],
    )

    out = _plain(result.output)
    assert result.exit_code == 3, (out, repr(result.exception))
    assert "--emit-evidence" in out and "is the same file as --output 'x.json'" in out, out
    assert "choose a different --emit-evidence" in out
    assert not (root / "x.json").exists()  # nothing was written


@WINDOWS_ONLY
def test_the_output_and_the_evidence_it_would_replace_are_one_file_whatever_the_spelling(root):
    name, content, _ = _prepare(root, "--evidence")

    for spelling in (name + ".", name + " ", name.upper() + "."):
        result = runner.invoke(app, ["ship", "--evidence", name, "--output", spelling])

        assert result.exit_code == 3, (spelling, _plain(result.output), repr(result.exception))
        assert "is the same file as --evidence" in _plain(result.output), spelling
    assert (root / name).read_text(encoding="utf-8") == content


@pytest.mark.skipif(sys.platform == "win32", reason="on POSIX a trailing dot is another name")
def test_a_trailing_dot_is_another_file_on_posix(root):
    (root / "ev.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(
        app, ["ship", "--evidence", "ev.json", "--output", "x.json", "--emit-evidence", "x.json."],
    )

    assert "is the same file as" not in _plain(result.output)


@pytest.mark.parametrize(
    ("first", "second", "same"),
    [
        ("x.json", "x.json.", True),
        ("x.json", "x.json ", True),
        ("x.json", "x.json. . ", True),
        ("a/x.json", "a/x.json.", True),
        ("dir./x.json", "dir/x.json", True),  # each part of the name, not only the last
        ("x.json", "y.json", False),
        ("x.json", "x.jsonl", False),
        ("x.json", "x.js on", False),  # a space inside a name is part of it
        ("x.json", ".x.json", False),  # a leading dot is part of it
    ],
)
def test_the_windows_name_drops_the_trailing_dots_and_spaces_of_each_part(
    monkeypatch, first, second, same
):
    from soup_cli.commands import ship

    monkeypatch.setattr(ship.os, "name", "nt")

    assert (ship._windows_name(first) == ship._windows_name(second)) is same


def test_the_name_is_left_alone_off_windows(monkeypatch):
    from soup_cli.commands import ship

    monkeypatch.setattr(ship.os, "name", "posix")
    # The host's own realpath must not decide this: on Windows, from Python 3.13, it already
    # drops a trailing dot or space of a file that does not exist, so the two names would be
    # one before the branch under test is reached.
    monkeypatch.setattr(ship.os.path, "realpath", lambda path: str(path))

    assert ship._windows_name("x.json.") != ship._windows_name("x.json")
    assert ship._windows_name("x.json ") != ship._windows_name("x.json")
