"""``soup eval unlearning`` prints text it did not write as text.

The command interpolates the run id, the ``--output`` path, the registry entry
id, the evidence line of every metric and the text of the exceptions it
reports into Rich markup. Each went through ``rich.markup.escape`` alone,
which neutralises ``[...]`` tags and leaves ESC, DEL and the C1 range
U+0080-U+009F in place. Each now goes through
``soup_cli.utils.terminal.for_terminal``.

Every test drives the real command line: the command is registered on a fresh
Typer app with ``Console(file=StringIO(), color_system=None)``, so Rich emits
no styling of its own and any control byte in the capture came from the value.
Control characters are asserted on the UTF-8 bytes of the capture. Non-ASCII
characters are built with ``chr()`` so this file stays ASCII.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from io import StringIO
from pathlib import Path

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.commands._eval_v0610 as unlearning_cmd
import soup_cli.registry.attach as attach_mod
import soup_cli.utils.unlearning_eval as unlearning_eval

runner = CliRunner()

ESC = chr(27)
BEL = chr(7)
BACKSPACE = chr(8)
CARRIAGE_RETURN = chr(13)
DEL = chr(0x7F)
CSI_C1 = chr(0x9B)
ST_C1 = chr(0x9C)
OSC_C1 = chr(0x9D)
BACKSLASH = chr(92)

MARKUP = "[red]x[/red][link=http://h/]y[/link][/]"

#: name -> (text as given, text the terminal must show)
SEQUENCES = {
    "csi-colour": ("ZQ" + ESC + "[31mQZ", "ZQ[31mQZ"),
    "osc-title": ("ZQ" + ESC + "]0;T" + BEL + "QZ", "ZQ]0;TQZ"),
    "osc-hyperlink": (
        "ZQ" + ESC + "]8;;http://h/" + ESC + BACKSLASH + "QZ",
        "ZQ]8;;http://h/" + BACKSLASH + "QZ",
    ),
    "bel": ("ZQ" + BEL + "QZ", "ZQQZ"),
    "backspace": ("ZQ" + BACKSPACE + "QZ", "ZQQZ"),
    "carriage-return": ("ZQ" + CARRIAGE_RETURN + "QZ", "ZQQZ"),
    "del": ("ZQ" + DEL + "QZ", "ZQQZ"),
    "c1-csi": ("ZQ" + CSI_C1 + "2JQZ", "ZQ2JQZ"),
    "c1-osc": ("ZQ" + OSC_C1 + "0;T" + ST_C1 + "QZ", "ZQ0;TQZ"),
}

#: One value carrying every sequence above plus tag pairs and a bare closing tag.
FILE_TEXT = (
    "ZQ"
    + ESC + "[31m"
    + ESC + "]0;T" + BEL
    + ESC + "]8;;http://h/" + ESC + BACKSLASH
    + "-" + BACKSPACE + CARRIAGE_RETURN + DEL
    + CSI_C1 + "2J"
    + OSC_C1 + "0;T" + ST_C1
    + MARKUP
    + "QZ"
)

#: What the terminal must show for it: controls gone, brackets literal.
SHOWN = "ZQ[31m]0;T]8;;http://h/" + BACKSLASH + "-2J0;T" + MARKUP + "QZ"

#: Raw byte strings that must never be in a capture.
RAW_CONTROLS = {
    "ESC": ESC.encode("utf-8"),
    "BEL": BEL.encode("utf-8"),
    "BACKSPACE": BACKSPACE.encode("utf-8"),
    "CARRIAGE RETURN": CARRIAGE_RETURN.encode("utf-8"),
    "DEL": DEL.encode("utf-8"),
    "C1 CSI": CSI_C1.encode("utf-8"),
    "C1 ST": ST_C1.encode("utf-8"),
    "C1 OSC": OSC_C1.encode("utf-8"),
}


def _is_control(char: str) -> bool:
    code = ord(char)
    return (code < 0x20 and char not in "\t\n") or 0x7F <= code <= 0x9F


def _assert_no_control_bytes(out: str) -> None:
    raw = out.encode("utf-8")
    found = [name for name, needle in RAW_CONTROLS.items() if needle in raw]
    assert not found, (found, raw)
    stray = sorted({f"U+{ord(char):04X}" for char in out if _is_control(char)})
    assert not stray, (stray, raw)


def _flat(out: str) -> str:
    """Drop all whitespace: a table title wraps to the table's width."""
    return "".join(out.split())


def _assert_shown_literally(out: str, *, times: int = 1) -> None:
    _assert_no_control_bytes(out)
    assert _flat(out).count(SHOWN) == times, out


def _run(monkeypatch, tmp_path, *args):
    """Invoke ``unlearning`` in ``tmp_path``; return (result, console capture)."""
    monkeypatch.chdir(tmp_path)
    buffer = StringIO()
    eval_app = typer.Typer()

    @eval_app.callback()
    def _root() -> None:
        """Keep ``unlearning`` a named subcommand, as under ``soup eval``."""

    unlearning_cmd.register(eval_app, Console(file=buffer, color_system=None, width=400))
    result = runner.invoke(eval_app, ["unlearning", *args])
    return result, buffer.getvalue()


def _exit(result, code: int) -> None:
    assert result.exit_code == code, (result.output, repr(result.exception))


def _with_evidence_text(monkeypatch, text: str) -> None:
    """Make every metric of the report carry ``text`` as its evidence line."""
    real = unlearning_eval.run_unlearn_eval

    def _run_eval(**kwargs):
        report = real(**kwargs)
        metrics = tuple(dataclasses.replace(m, evidence=text) for m in report.metrics)
        return dataclasses.replace(report, metrics=metrics)

    monkeypatch.setattr(unlearning_eval, "run_unlearn_eval", _run_eval)


# ---------------------------------------------------------------------------
# The report table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_table_title_drops_each_control_sequence_of_the_run_id(monkeypatch, tmp_path, name):
    given, shown = SEQUENCES[name]
    result, out = _run(monkeypatch, tmp_path, given)
    _exit(result, 0)
    _assert_no_control_bytes(out)
    assert _flat(out).count(shown) == 1, out


def test_table_title_shows_the_run_id_literally(monkeypatch, tmp_path):
    result, out = _run(monkeypatch, tmp_path, FILE_TEXT)
    _exit(result, 0)
    _assert_shown_literally(out)


def test_evidence_column_is_shown_literally(monkeypatch, tmp_path):
    _with_evidence_text(monkeypatch, FILE_TEXT)
    result, out = _run(monkeypatch, tmp_path, "run1")
    _exit(result, 0)
    _assert_shown_literally(out, times=3)


@pytest.mark.parametrize(
    "markup",
    [
        "[red]x[/red]",
        "[link=http://h/]y[/link]",
        "[bold]never closed",
        "closed but never opened[/]",
        "[/red]",
    ],
)
def test_evidence_column_prints_markup_as_literal_text(monkeypatch, tmp_path, markup):
    _with_evidence_text(monkeypatch, "ZQ" + markup + "QZ")
    result, out = _run(monkeypatch, tmp_path, "run1")
    _exit(result, 0)
    assert out.count("ZQ" + markup + "QZ") == 3, out


def test_evidence_column_keeps_ordinary_non_ascii_text(monkeypatch, tmp_path):
    # e-acute, inverted exclamation mark (first code point after the C1 range),
    # a Cyrillic and a CJK letter, and a character outside the BMP.
    text = "caf" + chr(0xE9) + chr(0xA1) + chr(0x416) + chr(0x65E5) + chr(0x1D11E) + "~end"
    _with_evidence_text(monkeypatch, text)
    result, out = _run(monkeypatch, tmp_path, "run1")
    _exit(result, 0)
    assert out.count(text) == 3, out
    _assert_no_control_bytes(out)


# ---------------------------------------------------------------------------
# Input errors
# ---------------------------------------------------------------------------


def test_benchmark_error_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _reject(value):
        raise ValueError(f"unknown benchmark {value}")

    monkeypatch.setattr(unlearning_eval, "validate_benchmark_name", _reject)
    result, out = _run(monkeypatch, tmp_path, "run1", "--benchmark", FILE_TEXT)
    _exit(result, 2)
    assert "Invalid benchmark: unknown benchmark" in out
    _assert_shown_literally(out)


def test_benchmark_error_from_the_real_validator_has_no_control_bytes(monkeypatch, tmp_path):
    # The validator caps the name at a short length and quotes it with repr().
    value = "Z" + ESC + "[31m" + CSI_C1 + DEL + "[b]x[/b]Q"
    result, out = _run(monkeypatch, tmp_path, "run1", "--benchmark", value)
    _exit(result, 2)
    assert "Invalid benchmark: unknown benchmark" in out
    _assert_no_control_bytes(out)
    assert "[b]x[/b]Q" in out, out


def test_invalid_json_error_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _load(path):
        raise json.JSONDecodeError(f"cannot parse {path}", "{", 0)

    monkeypatch.setattr(unlearning_eval, "load_evidence_file", _load)
    result, out = _run(monkeypatch, tmp_path, "run1", "--evidence", FILE_TEXT)
    _exit(result, 2)
    assert "Evidence file is not valid JSON: cannot parse" in out
    _assert_shown_literally(out)


def test_evidence_read_error_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _load(path):
        raise ValueError(f"evidence path must stay under cwd: {path}")

    monkeypatch.setattr(unlearning_eval, "load_evidence_file", _load)
    result, out = _run(monkeypatch, tmp_path, "run1", "--evidence", FILE_TEXT)
    _exit(result, 2)
    assert "Cannot read evidence: evidence path must stay under cwd:" in out
    _assert_shown_literally(out)


def test_eval_failure_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _fail(**kwargs):
        raise ValueError(f"cannot score run {kwargs['run_id']}")

    monkeypatch.setattr(unlearning_eval, "run_unlearn_eval", _fail)
    result, out = _run(monkeypatch, tmp_path, FILE_TEXT)
    _exit(result, 2)
    assert "Eval failed: cannot score run" in out
    _assert_shown_literally(out)


# ---------------------------------------------------------------------------
# --output and --attach-to-registry lines
# ---------------------------------------------------------------------------


def test_wrote_line_shows_the_output_path_literally(monkeypatch, tmp_path):
    monkeypatch.setattr(unlearning_eval, "write_unlearn_report", lambda report, path: path)
    result, out = _run(monkeypatch, tmp_path, "run1", "--output", FILE_TEXT)
    _exit(result, 0)
    assert "Wrote report ->" in out
    _assert_shown_literally(out)


def test_write_error_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _refuse(report, path):
        raise ValueError(f"report path {path} must stay under cwd")

    monkeypatch.setattr(unlearning_eval, "write_unlearn_report", _refuse)
    result, out = _run(monkeypatch, tmp_path, "run1", "--output", FILE_TEXT)
    _exit(result, 2)
    assert "Cannot write report: report path" in out
    _assert_shown_literally(out)


def test_attached_line_shows_the_registry_id_literally(monkeypatch, tmp_path):
    # The stand-ins take the helper's own keyword-only signature, so a call the
    # real attach_artifact would refuse fails here too.
    def _attached(entry_id, *, path, kind, enforce_cwd=True):
        return 1

    monkeypatch.setattr(attach_mod, "attach_artifact", _attached)
    result, out = _run(
        monkeypatch, tmp_path, "run1", "--output", "report.json", "--attach-to-registry", FILE_TEXT
    )
    _exit(result, 0)
    assert "Attached eval_results -> registry" in out
    _assert_shown_literally(out)


def test_attach_failure_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _missing(entry_id, *, path, kind, enforce_cwd=True):
        raise ValueError(f"registry entry not found: {entry_id}")

    monkeypatch.setattr(attach_mod, "attach_artifact", _missing)
    result, out = _run(
        monkeypatch, tmp_path, "run1", "--output", "report.json", "--attach-to-registry", FILE_TEXT
    )
    _exit(result, 1)
    assert "Registry attach failed: registry entry not found:" in out
    _assert_shown_literally(out)


# ---------------------------------------------------------------------------
# What is left on rich.markup.escape
# ---------------------------------------------------------------------------


def test_plain_escape_is_left_only_on_values_from_closed_sets():
    """Benchmark, metric name and verdicts are validated against fixed tuples."""
    tree = ast.parse(Path(unlearning_cmd.__file__).read_text(encoding="utf-8"))
    arguments = sorted(
        {
            ast.unparse(node.args[0])
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "escape"
        }
    )
    assert arguments == ["bench", "m.name", "m.verdict", "report.overall"], arguments
