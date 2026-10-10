"""Text that ``soup diagnose`` did not write reaches the terminal as text.

``soup diagnose`` interpolates strings from outside the program into Rich
markup: the evidence line of every score (from the ``--evidence`` JSON, or on a
live run from a model reply, a dataset row or a probe's exception text), the
run id, the adapter name, the ``--output`` / ``--badge`` paths, the registry
entry id, and the text of the exceptions it reports. Each went through
``rich.markup.escape`` alone, which neutralises ``[...]`` tags and leaves ESC,
DEL and the C1 range U+0080-U+009F in place. Each now goes through
``soup_cli.utils.terminal.for_terminal``.

Every test drives the real ``soup diagnose`` command line, with the module's
console swapped for ``Console(file=StringIO(), color_system=None)`` so Rich
emits no styling of its own: any control byte in the capture came from the
value. Control characters are asserted on the UTF-8 bytes of the capture.
Non-ASCII characters are built with ``chr()`` so this file stays ASCII.
"""

from __future__ import annotations

import ast
import json
import sys
import types
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.commands.diagnose as diagnose_mod
import soup_cli.registry.attach as attach_mod
import soup_cli.utils.citation_faithful as citation_mod
import soup_cli.utils.diagnose.live as live_mod
from soup_cli.cli import app
from soup_cli.utils.diagnose.report import FAILURE_MODES, FailureScore
from soup_cli.utils.diagnose.runner import build_report
from soup_cli.utils.exit_codes import EXIT_USAGE_ERROR
from tests.conftest import strip_ansi

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


def _run(monkeypatch, tmp_path, *args, evidence=None, ensure_ascii=True):
    """Invoke ``soup diagnose`` in ``tmp_path``; return (result, console capture)."""
    monkeypatch.chdir(tmp_path)
    argv = ["diagnose", *args]
    if evidence is not None:
        (tmp_path / "ev.json").write_text(
            json.dumps(evidence, ensure_ascii=ensure_ascii), encoding="utf-8"
        )
        argv += ["--evidence", "ev.json"]
    buffer = StringIO()
    monkeypatch.setattr(
        diagnose_mod, "console", Console(file=buffer, color_system=None, width=400)
    )
    result = runner.invoke(app, argv)
    return result, buffer.getvalue()


def _evidence_for(mode: str, text: str) -> dict:
    return {"scores": {mode: {"score": 1.0, "evidence": text}}}


def _ok(result) -> None:
    assert result.exit_code == 0, (result.output, repr(result.exception))


# ---------------------------------------------------------------------------
# The evidence column of the report table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_evidence_line_drops_each_control_sequence(monkeypatch, tmp_path, name):
    stored, shown = SEQUENCES[name]
    result, out = _run(
        monkeypatch, tmp_path, "run1", evidence=_evidence_for("forgetting", stored)
    )
    _ok(result)
    _assert_no_control_bytes(out)
    assert out.count(shown) == 1, out


def test_evidence_line_of_every_mode_is_shown_literally(monkeypatch, tmp_path):
    evidence = {
        "scores": {mode: {"score": 1.0, "evidence": FILE_TEXT} for mode in FAILURE_MODES}
    }
    result, out = _run(monkeypatch, tmp_path, "run1", evidence=evidence)
    _ok(result)
    _assert_shown_literally(out, times=len(FAILURE_MODES))


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
def test_evidence_line_prints_markup_as_literal_text(monkeypatch, tmp_path, markup):
    result, out = _run(
        monkeypatch, tmp_path, "run1", evidence=_evidence_for("refusal", "ZQ" + markup + "QZ")
    )
    _ok(result)
    assert out.count("ZQ" + markup + "QZ") == 1, out


def test_evidence_line_keeps_ordinary_non_ascii_text(monkeypatch, tmp_path):
    # e-acute, inverted exclamation mark (first code point after the C1 range),
    # a Cyrillic and a CJK letter, and a character outside the BMP.
    text = "caf" + chr(0xE9) + chr(0xA1) + chr(0x416) + chr(0x65E5) + chr(0x1D11E) + "~end"
    result, out = _run(
        monkeypatch,
        tmp_path,
        "run1",
        evidence=_evidence_for("format", text),
        ensure_ascii=False,
    )
    _ok(result)
    assert out.count(text) == 1, out
    _assert_no_control_bytes(out)


def test_evidence_line_of_a_live_report_is_shown_literally(monkeypatch, tmp_path):
    """A live run's evidence quotes model replies, dataset rows and exception text."""

    def _live(**kwargs):
        scores = {
            "memorization": FailureScore(
                mode="memorization", score=1.0, verdict="OK", evidence=FILE_TEXT
            )
        }
        return build_report(
            run_id=kwargs["run_id"],
            base=kwargs["base"],
            adapter="",
            scores=scores,
            soup_version="0",
        )

    monkeypatch.setattr(live_mod, "run_live_diagnose", _live)
    result, out = _run(monkeypatch, tmp_path, "run1", "--base-model", "org/m")
    _ok(result)
    _assert_shown_literally(out)


# ---------------------------------------------------------------------------
# The table title and the overall panel
# ---------------------------------------------------------------------------


def test_table_title_shows_the_adapter_name_literally(monkeypatch, tmp_path):
    result, out = _run(monkeypatch, tmp_path, "run1", "--adapter", FILE_TEXT)
    _ok(result)
    _assert_shown_literally(out)


def test_overall_panel_shows_the_run_id_literally(monkeypatch, tmp_path):
    result, out = _run(monkeypatch, tmp_path, FILE_TEXT, "--adapter", "plain")
    _ok(result)
    _assert_shown_literally(out)


def test_table_title_falls_back_to_the_run_id(monkeypatch, tmp_path):
    result, out = _run(monkeypatch, tmp_path, FILE_TEXT)
    _ok(result)
    _assert_shown_literally(out, times=2)


# ---------------------------------------------------------------------------
# --output, --badge and --attach-to-registry lines
# ---------------------------------------------------------------------------


def test_wrote_line_shows_the_output_path_literally(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose_mod, "write_report", lambda report, path: path)
    result, out = _run(monkeypatch, tmp_path, "run1", "--output", FILE_TEXT)
    _ok(result)
    _assert_shown_literally(out)
    assert "Wrote" in out


def test_output_error_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _refuse(report, path):
        raise ValueError(f"diagnose report path {path} must stay under cwd")

    monkeypatch.setattr(diagnose_mod, "write_report", _refuse)
    result, out = _run(monkeypatch, tmp_path, "run1", "--output", FILE_TEXT)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "cannot write --output: ValueError: diagnose report path" in out
    _assert_shown_literally(out)


def test_badge_line_shows_the_badge_path_literally(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnose_mod, "_write_badge", lambda badge_path, svg: None)
    result, out = _run(monkeypatch, tmp_path, "run1", "--badge", FILE_TEXT)
    _ok(result)
    _assert_shown_literally(out)
    assert "Badge written" in out


def test_badge_error_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _refuse(badge_path, svg):
        raise ValueError(f"badge path {badge_path} must stay under cwd")

    monkeypatch.setattr(diagnose_mod, "_write_badge", _refuse)
    result, out = _run(monkeypatch, tmp_path, "run1", "--badge", FILE_TEXT)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "cannot write --badge: ValueError: badge path" in out
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
    _ok(result)
    assert "Attached" in out
    _assert_shown_literally(out)


def test_attach_error_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _missing(entry_id, *, path, kind, enforce_cwd=True):
        raise ValueError(f"registry entry not found: {entry_id}")

    monkeypatch.setattr(attach_mod, "attach_artifact", _missing)
    result, out = _run(
        monkeypatch, tmp_path, "run1", "--output", "report.json", "--attach-to-registry", FILE_TEXT
    )
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "could not attach to registry: ValueError: registry entry not found:" in out
    _assert_shown_literally(out)


# ---------------------------------------------------------------------------
# Input errors reported before the report is built
# ---------------------------------------------------------------------------


def test_citation_style_error_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _reject(value):
        raise ValueError(f"unknown citation_style {value}")

    monkeypatch.setattr(citation_mod, "validate_citation_style", _reject)
    result, out = _run(monkeypatch, tmp_path, "run1", "--citation-style", FILE_TEXT)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "Invalid --citation-style: unknown citation_style" in out
    _assert_shown_literally(out)


def test_citation_style_error_from_the_real_validator_has_no_control_bytes(
    monkeypatch, tmp_path
):
    # The validator caps the name at 32 characters and quotes it with repr().
    value = "Z" + ESC + "[31m" + CSI_C1 + DEL + "[b]x[/b]Q"
    result, out = _run(monkeypatch, tmp_path, "run1", "--citation-style", value)
    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "Invalid --citation-style: unknown citation_style" in out
    _assert_no_control_bytes(out)
    assert "[b]x[/b]Q" in out, out


def test_tokenizer_error_shows_the_loader_text_literally(monkeypatch, tmp_path):
    """The loader's own message repeats the id as given; the wrapper quotes it too."""

    def _from_pretrained(name, *args, **kwargs):
        raise OSError(f"{name} is not a local folder and is not a valid model identifier")

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=_from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        live_mod,
        "run_live_diagnose",
        lambda **kwargs: pytest.fail("the model must not load after a tokenizer error"),
    )
    result, out = _run(
        monkeypatch, tmp_path, "run1", "--base-model", "org/m", "--tokenizer", FILE_TEXT
    )
    assert result.exit_code == EXIT_USAGE_ERROR, (result.output, repr(result.exception))
    assert "could not load tokenizer" in out
    _assert_shown_literally(out)


def test_live_failure_shows_the_exception_text_literally(monkeypatch, tmp_path):
    def _fail(**kwargs):
        raise RuntimeError(f"row 3 of d.jsonl: {FILE_TEXT}")

    monkeypatch.setattr(live_mod, "run_live_diagnose", _fail)
    result, out = _run(monkeypatch, tmp_path, "run1", "--base-model", "org/m")
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "live diagnose failed: RuntimeError: row 3 of d.jsonl:" in out
    _assert_shown_literally(out)


# ---------------------------------------------------------------------------
# Evidence values quoted in a usage error (printed by Typer, not by the module)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["score", "verdict"])
def test_rejected_evidence_value_is_quoted_with_repr(monkeypatch, tmp_path, field):
    """These messages quote the value with ``repr()`` and Typer prints them as text."""
    value = "Z" + ESC + "]0;T" + BEL + CSI_C1 + DEL + BACKSPACE + "[red]x[/red]Q"
    entry = {"score": 1.0, "evidence": "plain"}
    entry[field] = value
    result, _ = _run(monkeypatch, tmp_path, "run1", evidence={"scores": {"forgetting": entry}})
    assert result.exit_code == 2, (result.output, repr(result.exception))
    for char in (BEL, CSI_C1, DEL, BACKSPACE):
        assert char not in result.output, repr(result.output)
    assert ESC + "]0;T" not in result.output, repr(result.output)
    shown = "".join(strip_ansi(result.output).replace(chr(0x2502), "").split())
    assert "[red]x[/red]Q" in shown, result.output
    assert BACKSLASH + "x1b]0;T" in shown, result.output


# ---------------------------------------------------------------------------
# What is left on rich.markup.escape
# ---------------------------------------------------------------------------


def test_plain_escape_is_left_only_on_text_the_program_wrote():
    """A mode name is a constant and an exception class name is an identifier."""
    tree = ast.parse(Path(diagnose_mod.__file__).read_text(encoding="utf-8"))
    arguments = sorted(
        {
            ast.unparse(node.args[0])
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "escape"
        }
    )
    assert arguments == ["mode", "type(exc).__name__"], arguments
