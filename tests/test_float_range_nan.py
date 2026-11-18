"""#2906: ``nan`` is refused like ``inf`` on every ranged float option.

``typer.Option(min=..., max=...)`` on a float becomes a Click ``FloatRange``.
Click checks the range with ``value < min`` and ``value > max``; both are false
for NaN, so ``FloatRange.convert("nan")`` returned ``nan`` while ``inf`` was
refused.  Six options had such a range, and three of them then acted on NaN
without a word: ``soup data validate --min-valid-fraction nan`` and
``soup draft measure --min-acceptance nan`` turned their gates off, and
``soup data toxicity --threshold nan`` wrote an empty file and exited 0.  The
other three (``data decontaminate``, ``data score``, ``data forge``) refused NaN
only after their input files were read, with exit 1.

A shared callback now refuses NaN after Click's conversion, on the number
(Python reads ``nan`` in any case and with a sign), as the usage error Click
gives ``inf``: same message, same option name, and the usage exit code of each
command (3 for the ``GateCommand`` ``data validate``, Click's 2 elsewhere).

#2997 adds ``data filter --perplexity/--coherence`` and ``tui --refresh``. The
filter options get the ranges their help names (coherence 0.0-1.0, perplexity
at least 0.0), so only NaN needs the callback there; ``inf`` stays a valid
perplexity ("no row is above the limit"). ``--refresh`` has no Click range, so
the callback names a finite number, and the command keeps its own 0.25-10.0
check and message.
"""

from __future__ import annotations

import json
import sys

import pytest
import typer

from soup_cli.cli import app
from soup_cli.utils.typer_compat import UsageError, is_group, is_option
from tests.conftest import make_cli_runner, strip_ansi

pytestmark = pytest.mark.unit

_RANGE = "0.0<=x<=1.0"
_NAN_SPELLINGS = ["nan", "NaN", "NAN", "+nan", "-nan"]

# (id, argv before the option, the option, the usage exit code ``inf`` gets today)
_COMMANDS = [
    ("data-validate", ["data", "validate", "missing.jsonl"], "--min-valid-fraction", 3),
    (
        "draft-measure",
        ["draft", "measure", "--target", "t", "--draft", "d", "--prompts", "p.jsonl"],
        "--min-acceptance",
        2,
    ),
    ("data-toxicity", ["data", "toxicity", "-i", "missing.jsonl"], "--threshold", 2),
    (
        "data-decontaminate",
        ["data", "decontaminate", "-i", "missing.jsonl", "--benchmark-file", "bench.jsonl"],
        "--threshold",
        2,
    ),
    ("data-score", ["data", "score", "-i", "missing.jsonl"], "--threshold", 2),
    ("data-forge", ["data", "forge", "-d", "missing"], "--uncertainty-threshold", 2),
]
_IDS = [case[0] for case in _COMMANDS]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """An empty working directory: no input file exists, so none can be read."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_NO_AUDIT_LOG", "1")
    monkeypatch.setenv("SOUP_TELEMETRY", "0")
    # Keep Click's error on one line of Rich's error panel.
    monkeypatch.setenv("COLUMNS", "400")
    return tmp_path


def _invoke(argv):
    return make_cli_runner().invoke(app, argv)


def _plain(result) -> str:
    """The output without colour, panel borders or line breaks."""
    return " ".join(strip_ansi(result.output).replace("│", " ").split())


def _refusal(option: str, shown: str) -> str:
    return f"Invalid value for '{option}': {shown} is not in the range {_RANGE}."


# --- the six commands --------------------------------------------------------


@pytest.mark.parametrize("spelling", _NAN_SPELLINGS)
@pytest.mark.parametrize(("argv", "option", "code"), [c[1:] for c in _COMMANDS], ids=_IDS)
def test_nan_is_a_usage_error_before_any_input_is_read(workdir, argv, option, code, spelling):
    result = _invoke([*argv, option, spelling])

    assert result.exit_code == code, result.output
    assert _refusal(option, "nan") in _plain(result), result.output
    # Refused while parsing: nothing was read and nothing was written.
    assert list(workdir.iterdir()) == []


@pytest.mark.parametrize("value", ["inf", "-inf", "1.5", "-0.5"])
@pytest.mark.parametrize(("argv", "option", "code"), [c[1:] for c in _COMMANDS], ids=_IDS)
def test_inf_and_out_of_range_values_keep_their_refusal(workdir, argv, option, code, value):
    result = _invoke([*argv, option, value])

    assert result.exit_code == code, result.output
    assert _refusal(option, str(float(value))) in _plain(result), result.output


def _one_of_four_valid(workdir):
    rows = [{"instruction": "Say hi", "output": "hello"}, {"foo": 1}, {"foo": 2}, {"foo": 3}]
    path = workdir / "mixed.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(("value", "code"), [("0.5", 2), ("0.25", 0)])
def test_the_validate_gate_still_reads_a_finite_fraction(workdir, value, code):
    path = _one_of_four_valid(workdir)

    result = _invoke(["data", "validate", path, "--format", "alpaca",
                      "--min-valid-fraction", value])

    assert result.exit_code == code, result.output
    assert ("Validation failed: valid fraction 0.250" in _plain(result)) is (code == 2)


def test_toxicity_with_a_finite_threshold_still_writes_its_rows(workdir):
    rows = [{"instruction": "you idiot", "output": "stupid moron"},
            {"instruction": "hello", "output": "nice day"}]
    (workdir / "tox.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = _invoke(["data", "toxicity", "-i", "tox.jsonl", "-o", "kept.jsonl",
                      "--threshold", "0.0"])

    assert result.exit_code == 0, result.output
    assert len((workdir / "kept.jsonl").read_text(encoding="utf-8").splitlines()) == 2


# --- #2997: data filter and tui ---------------------------------------------

_COHERENCE_REFUSED = [*_NAN_SPELLINGS, "inf", "-inf", "Infinity", "1e999"]
_PERPLEXITY_REFUSED = [*_NAN_SPELLINGS, "-inf", "-Infinity", "-1e999", "-1"]
_PERPLEXITY_ACCEPTED = ["50", "inf", "Infinity", "1e999"]
_REFRESH_REFUSED = [
    (spelling, "nan is not a finite number.") for spelling in _NAN_SPELLINGS
] + [
    (spelling, "--refresh must be between 0.25 and 10.0")
    for spelling in ["inf", "-inf", "Infinity", "1e999"]
]


def _filter_rows(workdir):
    rows = [
        {"text": "The quick brown fox jumps over the lazy dog every single morning."},
        {"text": "asdf qwer zxcv"},
        {"text": "Machine learning models learn patterns from training data."},
    ]
    path = workdir / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("value", _COHERENCE_REFUSED)
def test_filter_refuses_coherence_outside_its_stated_range(workdir, value):
    result = _invoke(["data", "filter", "missing.jsonl", "--coherence", value,
                      "--output", "kept.jsonl"])

    assert result.exit_code == 2, result.output
    assert "--coherence" in _plain(result), result.output
    assert not (workdir / "kept.jsonl").exists()


@pytest.mark.parametrize("value", _PERPLEXITY_REFUSED)
def test_filter_refuses_perplexity_nan_infinities_and_negatives(workdir, value):
    result = _invoke(["data", "filter", "missing.jsonl", "--perplexity", value,
                      "--output", "kept.jsonl"])

    assert result.exit_code == 2, result.output
    assert "--perplexity" in _plain(result), result.output
    assert not (workdir / "kept.jsonl").exists()


@pytest.mark.parametrize("value", _PERPLEXITY_ACCEPTED)
def test_filter_accepts_perplexity_infinity_as_no_limit(workdir, value):
    result = _invoke(["data", "filter", "missing.jsonl", "--perplexity", value,
                      "--output", "kept.jsonl"])

    # Past validation, stopped only by the missing input: inf is not a usage error.
    assert result.exit_code == 1, result.output
    assert "File not found" in _plain(result), result.output


@pytest.mark.parametrize(("value", "kept"), [("0.0", 3), ("0.5", 3), ("1.0", 0)])
def test_filter_keeps_finite_coherence_results(workdir, value, kept):
    path = _filter_rows(workdir)

    result = _invoke(["data", "filter", path, "--coherence", value,
                      "--output", "kept.jsonl"])

    assert result.exit_code == 0, result.output
    assert len((workdir / "kept.jsonl").read_text(encoding="utf-8").splitlines()) == kept


@pytest.mark.parametrize(("value", "shown"), _REFRESH_REFUSED, ids=[c[0] for c in _REFRESH_REFUSED])
def test_tui_refresh_refuses_non_finite_before_the_import(workdir, monkeypatch, value, shown):
    monkeypatch.setitem(sys.modules, "soup_cli.tui_app", None)

    result = _invoke(["tui", "--refresh", value])

    assert result.exit_code == 2, result.output
    text = _plain(result)
    assert "--refresh" in text, result.output
    assert shown in text, result.output
    assert "Textual is not installed" not in text, result.output


@pytest.mark.parametrize("value", ["0.25", "10.0"])
def test_tui_refresh_keeps_finite_bounds(workdir, monkeypatch, value):
    monkeypatch.setitem(sys.modules, "soup_cli.tui_app", None)

    result = _invoke(["tui", "--refresh", value])

    assert result.exit_code == 1, result.output
    assert "Textual is not installed." in _plain(result), result.output


@pytest.mark.parametrize("value", ["0.24", "10.1"])
def test_tui_refresh_keeps_its_bounds_message(workdir, monkeypatch, value):
    monkeypatch.setitem(sys.modules, "soup_cli.tui_app", None)

    result = _invoke(["tui", "--refresh", value])

    assert result.exit_code == 2, result.output
    assert "--refresh must be between 0.25 and 10.0" in _plain(result), result.output


# --- every ranged float option in the app ------------------------------------


def _ranged_float_options():
    """``(command path, option)`` for every option whose type is a ``FloatRange``."""
    found = []
    stack = [((), typer.main.get_command(app))]
    while stack:
        path, command = stack.pop()
        if is_group(command):
            stack.extend((path + (name,), sub) for name, sub in command.commands.items())
            continue
        for param in command.params:
            if is_option(param) and type(param.type).__name__ == "FloatRange":
                found.append((" ".join(path), command, param))
    return found


_RANGED = _ranged_float_options()
_RANGED_IDS = [f"{path} {param.opts[-1]}" for path, _, param in _RANGED]


def test_the_walker_sees_the_ranged_options():
    seen = {(path, param.opts[-1]) for path, _, param in _RANGED}

    assert {
        ("data validate", "--min-valid-fraction"),
        ("draft measure", "--min-acceptance"),
        ("data toxicity", "--threshold"),
        ("data decontaminate", "--threshold"),
        ("data score", "--threshold"),
        ("data forge", "--uncertainty-threshold"),
        ("data filter", "--ppl"),
        ("data filter", "--min-coherence"),
    } <= seen


@pytest.mark.parametrize("spelling", _NAN_SPELLINGS)
@pytest.mark.parametrize(("path", "command", "param"), _RANGED, ids=_RANGED_IDS)
def test_every_ranged_float_option_refuses_nan(path, command, param, spelling):
    with pytest.raises(UsageError) as excinfo:
        param.process_value(typer.Context(command), spelling)

    assert "nan is not in the range" in excinfo.value.format_message()


@pytest.mark.parametrize(("path", "command", "param"), _RANGED, ids=_RANGED_IDS)
def test_every_ranged_float_option_lets_a_finite_value_and_none_through(path, command, param):
    ctx = typer.Context(command)
    low = param.type.min if param.type.min is not None else 0.0

    assert param.process_value(ctx, str(low)) == low
    assert param.process_value(ctx, None) is None
