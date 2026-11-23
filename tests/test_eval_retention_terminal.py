"""``soup eval retention`` shows outside text as typed, on the display only (#2489).

Version labels (typed or read from the ledger), task names, the ledger, report
and numerics text, and the errors raised about them keep ``:name:`` aliases,
markup and a backslash before a bracket as written, and lose their control and
bidi characters. No model is loaded: ``live_eval.make_generator`` is replaced by
a fake whose correct answers per task the test sets.

Not every value reaches every line. A version label must be printable
(``validate_label``), so the labels with control or bidi characters are refused
with an ``Error:`` line before the title, the headers or ``Recorded``. A task
name must match ``[A-Za-z0-9][A-Za-z0-9_.-]*`` (``validate_task_name``, on
``--add-task`` and in the ledger), so of these values only ``plain`` reaches the
task cells, ``Control task``, the per-task score line and the regression list;
the rest are refused with an ``Error:`` line, and the renderer is checked with
the other names directly.
"""

import io
import json
import re
import sys
from pathlib import Path

import pytest
import typer
from rich.console import Console

from soup_cli.commands import _eval_retention as command
from soup_cli.utils import retention_ledger as rl
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import make_cli_runner, strip_ansi

VALUES = ["plain", "x[red]n", "[/]n", "n:rocket:", "x\\[n", "x\x9b\u202en", "x\x1b\x07n"]
#: What the display drops: C0 controls, a C1 CSI and a bidi override.
DROPPED = ("\x1b", "\x07", "\x9b", "\u202e")
#: The values ``validate_label`` accepts (printable text).
LABEL_VALUES = VALUES[:5]
#: The values it refuses (control or bidi characters).
REFUSED_LABEL_VALUES = VALUES[5:]
#: The values ``validate_task_name`` refuses: every one but ``plain``.
REFUSED_TASK_NAMES = VALUES[1:]

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

ROWS = 20
STAMP = "2026-11-05T00:00:00Z"
CPU_NUMERICS = rl.Numerics(
    dtype="float32", device="cpu", quantization=None, chat_template_kwargs=None
)
_BOX = re.compile("[│┃|]")


def _shown(value: str) -> str:
    for char in DROPPED:
        value = value.replace(char, "")
    return value


def _lines(result, exit_code: int) -> list:
    """The output lines, whitespace-collapsed; no dropped character is anywhere in it."""
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    output = strip_ansi(result.output)
    assert [char for char in DROPPED if char in output] == []
    return [" ".join(line.split()) for line in output.splitlines()]


def _assert_line(result, exit_code: int, line: str) -> None:
    lines = _lines(result, exit_code)
    assert line in lines, lines


def _cells(lines: list, first: str) -> list:
    """The cells of the table row whose first cell is ``first``."""
    for line in lines:
        cells = [cell.strip() for cell in _BOX.split(line) if cell.strip()]
        if cells and cells[0] == first:
            return cells
    raise AssertionError(f"no table row starts with {first!r}: {lines}")


def _console(**kwargs) -> Console:
    return Console(width=1000, height=100, **kwargs)


#: A console that emits ANSI colour whatever the environment says.
_COLOUR = {"force_terminal": True, "color_system": "standard", "no_color": False}


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def correct(monkeypatch):
    """No model: ``correct[task]`` of each task's rows are answered right."""
    from soup_cli.utils import gpu, live_eval

    profile: dict = {}

    def factory(model_id, **kwargs):
        def generate(prompt: str) -> str:
            task, _, index = prompt.rpartition("-")
            return "ok" if int(index) < profile.get(task, 0) else "bad"

        return generate

    monkeypatch.setattr(live_eval, "make_generator", factory)
    monkeypatch.setattr(gpu, "bf16_fp16_flags", lambda device: (False, False))
    return profile


def _task_file(name: str) -> str:
    rows = [json.dumps({"prompt": f"{name}-{index}", "expected": "ok"}) for index in range(ROWS)]
    Path(f"{name}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return f"{name}.jsonl"


def _ledger_with_earlier(label: str) -> None:
    """``retention.json`` with task ``alpha`` and one earlier version called ``label``."""
    path = _task_file("alpha")
    task = rl.RetentionTask("alpha", path, rl.file_sha256(path, "alpha"), STAMP, None)
    version = rl.RetentionVersion(label, "base-model", None, STAMP, {"alpha": 0.9}, CPU_NUMERICS)
    rl.save_ledger(rl.Ledger((task,), (version,)), "retention.json")


def _retention(label: str, *args: str, coloured: bool = False):
    """Invoke ``retention`` registered on a fresh app, its console writing to the runner.

    The console has no ``file``, so it prints to the ``sys.stdout`` the runner swaps in
    and the lines land in ``result.output``.
    """
    eval_app = typer.Typer()

    @eval_app.callback()
    def _root() -> None:
        """Keep ``retention`` a named subcommand, as under ``soup eval``."""

    command.register(eval_app, _console(**(_COLOUR if coloured else {})))
    return make_cli_runner().invoke(
        eval_app,
        ["retention", "--model", "base-model", "--device", "cpu", "--label", label, *args],
    )


# ---------------------------------------------------------------------------
# Error: label + plain message
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", VALUES)
def test_an_add_task_without_a_path_is_refused_as_typed(correct, value):
    result = _retention("v1", "--add-task", value)
    _assert_line(result, 3, f"Error: --add-task expects NAME=PATH, got {_shown(value)!r}")


@pytest.mark.parametrize("value", REFUSED_TASK_NAMES)
def test_a_task_name_other_than_plain_is_refused_before_any_task_line(correct, value):
    _task_file("alpha")
    result = _retention("v1", "--add-task", f"{value}=alpha.jsonl")
    _assert_line(
        result, 3,
        f"Error: task name {_shown(value)!r} is invalid: use 1-64 characters from "
        "A-Z a-z 0-9 _ . - and start with a letter or digit",
    )


@pytest.mark.parametrize("value", REFUSED_LABEL_VALUES)
def test_a_typed_label_with_control_characters_is_refused_as_typed(correct, value):
    result = _retention(value, "--add-task", "alpha=alpha.jsonl")
    _assert_line(
        result, 3,
        f"Error: version label {_shown(value)!r} must not contain control characters",
    )


@pytest.mark.parametrize("value", REFUSED_LABEL_VALUES)
def test_an_earlier_label_with_control_characters_in_the_ledger_is_refused_as_typed(
    correct, value
):
    _ledger_with_earlier(value)
    result = _retention("now")
    _assert_line(
        result, 3,
        f"Error: cannot read --ledger: version label {_shown(value)!r} must not contain "
        "control characters",
    )


@pytest.mark.parametrize("value", LABEL_VALUES)
def test_a_label_already_in_the_ledger_is_refused_as_typed(correct, value):
    _ledger_with_earlier(value)
    result = _retention(value)
    _assert_line(
        result, 3,
        f"Error: version label {value!r} is already recorded in the ledger; "
        "labels must be unique",
    )


# ---------------------------------------------------------------------------
# The table: title, headers, task cells
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", LABEL_VALUES)
def test_a_typed_label_is_shown_as_typed_in_the_title_and_the_now_header(correct, value):
    correct["alpha"] = 18
    _task_file("alpha")
    result = _retention(value, "--add-task", "alpha=alpha.jsonl")
    lines = _lines(result, 0)
    assert f"Retention: {value} vs 0 earlier version(s)" in lines, lines
    assert _cells(lines, "Task") == ["Task", f"{value} (now)", "Best", "Drop", "Status"]


@pytest.mark.parametrize("value", LABEL_VALUES)
def test_an_earlier_label_from_the_ledger_is_shown_as_typed_in_its_header(correct, value):
    correct["alpha"] = 18
    _ledger_with_earlier(value)
    result = _retention("now")
    lines = _lines(result, 0)
    assert "Retention: now vs 1 earlier version(s)" in lines, lines
    assert _cells(lines, "Task") == ["Task", value, "now (now)", "Best", "Drop", "Status"]
    assert _cells(lines, "alpha") == ["alpha", "0.900", "0.900", "0.900", "+0.000", "OK"]


def test_a_task_name_is_shown_in_its_lines_and_cells(correct):
    """``plain`` is the one value a task name may be; see the module docstring."""
    correct["plain"] = 18
    _task_file("plain")
    first = _retention("v1", "--add-task", "plain=plain.jsonl", "--record")
    lines = _lines(first, 0)
    assert "Control task plain: added." in lines, lines
    assert "plain: 0.900 (18/20)" in lines, lines
    assert _cells(lines, "plain") == ["plain", "0.900", "-", "-", "NEW"]

    correct["plain"] = 15
    second = _retention("v2")
    lines = _lines(second, 2)
    assert "plain: 0.750 (15/20)" in lines, lines
    assert _cells(lines, "plain") == ["plain", "0.900", "0.750", "0.900", "+0.150", "REGRESSED"]
    assert "RETENTION REGRESSION: plain (+0.150)" in lines, lines


@pytest.mark.parametrize("value", VALUES)
def test_the_report_shows_a_task_name_that_bypassed_validation_as_typed(value):
    """The renderer does not rely on the loader: the name cell and the regression list.

    This bypasses the CLI on purpose: ``--add-task`` and the ledger both validate a
    task name to ``[A-Za-z0-9][A-Za-z0-9_.-]*``, so no value here but ``plain`` can
    reach these lines through real input.
    """
    row = rl.TaskRetention(
        name=value, scores={"v1": 0.9}, current=0.75, first=0.9, best_previous=0.9,
        drop=0.15, status=rl.STATUS_REGRESSED, examples=20,
    )
    report = rl.RetentionReport(
        label="v2", threshold=0.05, versions=("v1", "v2"), tasks=(row,)
    )
    buffer = io.StringIO()
    command._print_report(
        _console(file=buffer), report, recorded=False, ledger_path="retention.json"
    )
    output = strip_ansi(buffer.getvalue())
    assert [char for char in DROPPED if char in output] == []
    lines = [" ".join(line.split()) for line in output.splitlines()]
    assert _cells(lines, _shown(value)) == [
        _shown(value), "0.900", "0.750", "0.900", "+0.150", "REGRESSED"
    ]
    assert f"RETENTION REGRESSION: {_shown(value)} (+0.150)" in lines, lines


# ---------------------------------------------------------------------------
# Dim lines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        "plain",
        "x[red]n",
        "[/]n",
        pytest.param("n:rocket:", marks=skip_colon_on_windows),
        "x\\[n",
        "x\x9b\u202en",
        "x\x1b\x07n",
    ],
)
def test_a_missing_ledger_path_is_shown_as_typed(correct, value):
    """The ledger is never written (no ``--record``), so no file has this name."""
    correct["alpha"] = 18
    _task_file("alpha")
    result = _retention("v1", "--ledger", f"{value}.json", "--add-task", "alpha=alpha.jsonl")
    _assert_line(result, 0, f"No ledger at {_shown(value)}.json; starting a new one.")


@pytest.mark.parametrize("value", VALUES)
def test_chat_template_kwargs_are_shown_as_recorded_in_the_numerics_line(correct, value):
    correct["alpha"] = 18
    _task_file("alpha")
    result = _retention(
        "v1", "--add-task", "alpha=alpha.jsonl",
        "--chat-template-kwargs", json.dumps({"k": value}),
    )
    recorded = json.dumps({"k": value}, separators=(",", ":"))
    _assert_line(
        result, 0,
        "Numerics: dtype float32, device cpu, quantization none, "
        f"chat_template_kwargs {recorded}",
    )


# ---------------------------------------------------------------------------
# Recorded / Wrote report
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", LABEL_VALUES)
def test_a_recorded_label_is_shown_as_typed(correct, value):
    correct["alpha"] = 18
    _task_file("alpha")
    result = _retention(value, "--add-task", "alpha=alpha.jsonl", "--record")
    _assert_line(result, 0, f"Recorded version {value} in retention.json")


@pytest.mark.parametrize("value", FILE_VALUES)
def test_a_recorded_ledger_path_is_shown_as_typed(correct, value):
    correct["alpha"] = 18
    _task_file("alpha")
    ledger = f"{value}.json"
    result = _retention("v1", "--ledger", ledger, "--add-task", "alpha=alpha.jsonl", "--record")
    _assert_line(result, 0, f"Recorded version v1 in {_shown(value)}.json")
    assert Path(ledger).is_file()


@pytest.mark.parametrize("value", FILE_VALUES)
def test_a_report_path_is_shown_as_typed(correct, value):
    correct["alpha"] = 18
    _task_file("alpha")
    output = f"{value}.json"
    result = _retention("v1", "--add-task", "alpha=alpha.jsonl", "--output", output)
    _assert_line(result, 0, f"Wrote report {_shown(value)}.json")
    assert json.loads(Path(output).read_text(encoding="utf-8"))["label"] == "v1"


# ---------------------------------------------------------------------------
# Styles on a colour console
# ---------------------------------------------------------------------------

RED, GREEN, DIM, ITALIC, RESET = "\x1b[31m", "\x1b[32m", "\x1b[2m", "\x1b[3m", "\x1b[0m"


def test_the_error_label_is_red_and_the_message_plain(correct):
    result = _retention("v1", "--add-task", "x[red]n", coloured=True)
    assert result.exit_code == 3, result.output
    # ansi-ok: asserts the colour codes themselves, on a console pinned to colour
    assert f"{RED}Error:{RESET} --add-task expects NAME=PATH, got 'x[red]n'\n" in result.output


def test_the_table_title_stays_italic(correct):
    correct["alpha"] = 18
    _task_file("alpha")
    result = _retention("x[red]n", "--add-task", "alpha=alpha.jsonl", coloured=True)
    assert result.exit_code == 0, result.output
    # The title is centred over the table; the padding is italic too.
    title = re.escape("Retention: x[red]n vs 0 earlier version(s)")
    assert re.search(f"{re.escape(ITALIC)} *{title} *{re.escape(RESET)}", result.output), (
        result.output
    )


def test_the_progress_lines_are_dim(correct):
    correct["alpha"] = 18
    _task_file("alpha")
    result = _retention(
        "v1", "--ledger", "x[red]n.json", "--add-task", "alpha=alpha.jsonl", coloured=True
    )
    assert result.exit_code == 0, result.output
    for line in (
        "No ledger at x[red]n.json; starting a new one.",
        "Numerics: dtype float32, device cpu, quantization none, chat_template_kwargs none",
        "Control task alpha: added.",
        "  alpha: 0.900 (18/20)",
    ):
        assert f"{DIM}{line}{RESET}\n" in result.output, (line, result.output)


def test_recorded_and_wrote_report_are_green_labels(correct):
    correct["alpha"] = 18
    _task_file("alpha")
    result = _retention(
        "x[red]n", "--add-task", "alpha=alpha.jsonl", "--record", "--output", "[b]rep.json",
        coloured=True,
    )
    assert result.exit_code == 0, result.output
    assert f"{GREEN}Recorded{RESET} version x[red]n in retention.json\n" in result.output
    assert f"{GREEN}Wrote report{RESET} [b]rep.json\n" in result.output


def test_the_regression_label_is_red_and_the_names_plain(correct):
    correct["alpha"] = 18
    _task_file("alpha")
    assert _retention("v1", "--add-task", "alpha=alpha.jsonl", "--record").exit_code == 0
    correct["alpha"] = 15
    result = _retention("v2", coloured=True)
    assert result.exit_code == 2, result.output
    # ansi-ok: asserts the colour codes themselves, on a console pinned to colour
    assert f"{RED}RETENTION REGRESSION:{RESET} alpha (+0.150)\n" in result.output
