"""``soup train --diagnose-gate`` prints evidence-file text as text.

The gate reads a JSON evidence file after training and prints the evidence
line of every MAJOR / NOT_RUN mode, or the reason the file was refused. The
evidence lines went through ``rich.markup.escape`` alone, which leaves ESC,
DEL and the C1 range U+0080-U+009F in place; the refusal line interpolated the
exception text into Rich markup with no escaping at all, so a ``[...]`` tag
quoted in the message was read as markup. All three now go through
``soup_cli.utils.terminal.for_terminal``.

The tests call the gate helpers of ``soup_cli.commands.train`` directly, with
the module's console swapped for ``Console(file=StringIO(), color_system=None)``
so Rich emits no styling of its own: any control byte in the capture came from
the value. Control characters are asserted on the UTF-8 bytes of the capture.
Non-ASCII characters are built with ``chr()`` so this file stays ASCII.
"""

from __future__ import annotations

import ast
import json
from io import StringIO
from pathlib import Path

import pytest
import typer
from rich.console import Console

import soup_cli.commands.train as train_mod
from soup_cli.utils.exit_codes import EXIT_USAGE_ERROR

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

#: name -> (text as stored in the file, text the terminal must show)
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

#: verdict -> (a score FailureScore accepts for it, the gate's exit code)
GATED = {"MAJOR": (0.1, 2), "NOT_RUN": (0.0, EXIT_USAGE_ERROR)}


def _is_control(char: str) -> bool:
    code = ord(char)
    return (code < 0x20 and char not in "\t\n") or 0x7F <= code <= 0x9F


def _assert_no_control_bytes(out: str) -> None:
    raw = out.encode("utf-8")
    found = [name for name, needle in RAW_CONTROLS.items() if needle in raw]
    assert not found, (found, raw)
    stray = sorted({f"U+{ord(char):04X}" for char in out if _is_control(char)})
    assert not stray, (stray, raw)


def _capture(monkeypatch) -> StringIO:
    buffer = StringIO()
    monkeypatch.setattr(
        train_mod, "console", Console(file=buffer, color_system=None, width=400)
    )
    return buffer


def _gate(monkeypatch, tmp_path, verdict: str, text: str, *, ensure_ascii: bool = True):
    """Run the gate on one ``format`` entry; return (exit code, console capture)."""
    monkeypatch.chdir(tmp_path)
    score, _ = GATED[verdict]
    entry = {"score": score, "verdict": verdict, "evidence": text}
    (tmp_path / "ev.json").write_text(
        json.dumps({"scores": {"format": entry}}, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    buffer = _capture(monkeypatch)
    with pytest.raises(typer.Exit) as excinfo:
        train_mod._run_diagnose_gate("ev.json", "run1", "base", "adapter")
    return excinfo.value.exit_code, buffer.getvalue()


# ---------------------------------------------------------------------------
# The evidence line of a MAJOR / NOT_RUN mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verdict", sorted(GATED))
@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_gate_evidence_line_drops_each_control_sequence(monkeypatch, tmp_path, verdict, name):
    stored, shown = SEQUENCES[name]
    code, out = _gate(monkeypatch, tmp_path, verdict, stored)
    assert code == GATED[verdict][1], out
    _assert_no_control_bytes(out)
    assert out.count(f"{verdict} format: {shown}\n") == 1, out


@pytest.mark.parametrize("verdict", sorted(GATED))
def test_gate_evidence_line_is_shown_literally(monkeypatch, tmp_path, verdict):
    code, out = _gate(monkeypatch, tmp_path, verdict, FILE_TEXT)
    assert code == GATED[verdict][1], out
    _assert_no_control_bytes(out)
    assert out.count(f"{verdict} format: {SHOWN}\n") == 1, out


@pytest.mark.parametrize("verdict", sorted(GATED))
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
def test_gate_evidence_line_prints_markup_as_literal_text(
    monkeypatch, tmp_path, verdict, markup
):
    code, out = _gate(monkeypatch, tmp_path, verdict, "ZQ" + markup + "QZ")
    assert code == GATED[verdict][1], out
    assert out.count(f"{verdict} format: ZQ{markup}QZ\n") == 1, out


@pytest.mark.parametrize("verdict", sorted(GATED))
def test_gate_evidence_line_keeps_ordinary_non_ascii_text(monkeypatch, tmp_path, verdict):
    # e-acute, inverted exclamation mark (first code point after the C1 range),
    # a Cyrillic and a CJK letter, and a character outside the BMP.
    text = "caf" + chr(0xE9) + chr(0xA1) + chr(0x416) + chr(0x65E5) + chr(0x1D11E) + "~end"
    code, out = _gate(monkeypatch, tmp_path, verdict, text, ensure_ascii=False)
    assert code == GATED[verdict][1], out
    assert out.count(f"{verdict} format: {text}\n") == 1, out
    _assert_no_control_bytes(out)


# ---------------------------------------------------------------------------
# The "--diagnose-gate failed" line
# ---------------------------------------------------------------------------


def test_gate_failure_line_quotes_a_refused_verdict_literally(monkeypatch, tmp_path):
    """The refusal quotes the verdict with repr(); its brackets are not markup."""
    monkeypatch.chdir(tmp_path)
    verdict = "ZQ" + ESC + "]0;T" + BEL + CSI_C1 + MARKUP + "QZ"
    entry = {"score": 1.0, "verdict": verdict, "evidence": "plain"}
    (tmp_path / "ev.json").write_text(
        json.dumps({"scores": {"format": entry}}), encoding="utf-8"
    )
    buffer = _capture(monkeypatch)
    with pytest.raises(typer.Exit) as excinfo:
        train_mod._run_diagnose_gate_or_exit("ev.json", "run1", "base", "adapter")
    out = buffer.getvalue()
    assert excinfo.value.exit_code == 1, out
    assert "--diagnose-gate failed: ValueError: verdict must be one of" in out
    _assert_no_control_bytes(out)
    assert out.count(MARKUP + "QZ'") == 1, out


@pytest.mark.parametrize("error", [ValueError, OSError])
def test_gate_failure_line_shows_the_exception_text_literally(monkeypatch, error):
    def _refuse(evidence_path, run_id, base, adapter):
        raise error(f"cannot use {evidence_path}")

    monkeypatch.setattr(train_mod, "_run_diagnose_gate", _refuse)
    buffer = _capture(monkeypatch)
    with pytest.raises(typer.Exit) as excinfo:
        train_mod._run_diagnose_gate_or_exit(FILE_TEXT, "run1", "base", "adapter")
    out = buffer.getvalue()
    assert excinfo.value.exit_code == 1, out
    _assert_no_control_bytes(out)
    assert out.count(f"--diagnose-gate failed: {error.__name__}: cannot use {SHOWN}\n") == 1, out


@pytest.mark.parametrize("verdict", sorted(GATED))
def test_gate_exit_code_passes_through_the_failure_handler(monkeypatch, tmp_path, verdict):
    monkeypatch.chdir(tmp_path)
    score, code = GATED[verdict]
    entry = {"score": score, "verdict": verdict, "evidence": "plain"}
    (tmp_path / "ev.json").write_text(
        json.dumps({"scores": {"format": entry}}), encoding="utf-8"
    )
    buffer = _capture(monkeypatch)
    with pytest.raises(typer.Exit) as excinfo:
        train_mod._run_diagnose_gate_or_exit("ev.json", "run1", "base", "adapter")
    assert excinfo.value.exit_code == code, buffer.getvalue()
    assert "--diagnose-gate failed" not in buffer.getvalue()


def test_train_runs_the_gate_through_the_failure_handler():
    """``train`` is too heavy to drive here; pin which helper it calls."""
    tree = ast.parse(Path(train_mod.__file__).read_text(encoding="utf-8"))
    train_def = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    called = {
        node.func.id
        for node in ast.walk(train_def)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_run_diagnose_gate_or_exit" in called
    assert "_run_diagnose_gate" not in called
