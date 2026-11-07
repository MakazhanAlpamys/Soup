"""soup reward prints typed values, induced text and reward-file errors as typed, not as markup."""

import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.paths import quote_path
from soup_cli.utils.reward_synth import CalibrationReport
from tests.conftest import strip_ansi

runner = CliRunner()

_CONTROLS = "\x9b\u202e"

# A colon is not a legal file-name character on Windows, and "n:" there is a drive.
_NO_COLON_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32", reason="':' is not valid in a Windows name"
)

# One spelling per way the old code went wrong: a tag, a closing tag with
# nothing to close (a MarkupError), an emoji code, and a tag with control
# characters, so the strip and the escape are pinned separately.
_VALUES = [
    "x[red]n",
    "[/]n",
    pytest.param("n:warning:", id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]
# A file name cannot hold a slash, so no closing tag with nothing to close.
_NAMES = [
    "x[red]n",
    pytest.param("n:warning:", marks=_NO_COLON_ON_WINDOWS, id="emoji-code"),
    f"x[bold]n{_CONTROLS}",
]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    """Rich wraps at the terminal width, which CI sets differently from a laptop."""
    monkeypatch.setattr("soup_cli.commands.reward.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _references(name: str = "refs.jsonl", golds=("42", "7", "13")) -> str:
    lines = "".join(json.dumps({"answer": gold}) + "\n" for gold in golds)
    Path(name).write_text(lines, encoding="utf-8")
    return name


def _verifier(name: str = "verifier.py", body: str = "return [0.0 for _ in completions]") -> str:
    Path(name).write_text(f"def reward_fn(completions, **kwargs):\n    {body}\n", encoding="utf-8")
    return name


def _synth(*args: str):
    return runner.invoke(app, ["reward", "synth", *args])


def _stress(*args: str):
    return runner.invoke(app, ["reward", "stress", *args])


def _report(refused: bool, rejected: tuple) -> CalibrationReport:
    return CalibrationReport(
        kind="numeric", positives=3, negatives=5, pos_accept=0.67, neg_accept=0.0,
        discrimination=0.67, precision=1.0, recall=0.67, refused=refused,
        reason="verifier rejects its own references", rejected_references=rejected,
    )


@pytest.mark.parametrize("value", _VALUES)
def test_an_unknown_kind_is_shown_as_typed(value):
    result = _synth(_references(), "--kind", value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"unknown --kind {value!r}; options:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_an_unknown_attack_is_named_as_typed(value):
    result = _stress("verifiable", "--verifiable-domain", "math", "--attacks", value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"unknown attack kind(s): {_shown(value)}; options:" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_references_outside_cwd_are_shown_as_typed(name):
    result = _synth(f"../{name}.jsonl", "--plan-only")

    assert result.exit_code == 1
    text = _plain(result)
    quoted = quote_path(f"../{name}.jsonl").replace("\u202e", "")
    assert f"references path {quoted} must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_an_output_that_is_not_python_is_shown_as_typed(value):
    result = _synth(_references(), "--kind", "numeric", "-o", value)

    assert result.exit_code == 1
    text = _plain(result)
    assert f"output must be a .py file, got {value!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_reward_file_that_fails_to_load_is_shown_as_typed(value):
    Path("bad.py").write_text(f"raise RuntimeError('boom ' + {value!r})\n", encoding="utf-8")

    result = _stress("bad.py")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"could not load reward target: boom {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_synth_shows_a_report_it_cannot_write_as_typed(name):
    Path(f"{name}.json").mkdir()

    result = _synth(
        _references(), "--kind", "numeric", "-o", "out.py", "--output-report", f"{name}.json"
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "Warning: could not write report:" in text
    assert f"-> {name + '.json'!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_stress_shows_a_report_it_cannot_write_as_typed(name):
    Path(f"{name}.json").mkdir()

    result = _stress(_verifier(), "--output-report", f"{name}.json")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert "Warning: could not write report:" in text
    assert f"-> {name + '.json'!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_plan_shows_the_induced_schema_keys_as_typed(value):
    golds = [json.dumps({value: 1, "zz": 2}), json.dumps({value: 3, "zz": 4})]

    result = _synth(_references(golds=golds), "--plan-only")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"json_schema (type=object, keys={_shown(value)}, zz)" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_emitted_verifier_is_named_as_typed(name):
    result = _synth(_references(), "--kind", "numeric", "-o", f"{name}.py")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"emitted " + re.escape(_shown(name)) + r"\.py", text)
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_rejected_reference_of_an_emitted_verifier_is_shown_as_typed(monkeypatch, value):
    monkeypatch.setattr(
        "soup_cli.utils.reward_synth.calibrate", lambda *a, **k: _report(False, (value,))
    )

    result = _synth(_references(), "--kind", "numeric", "-o", "out.py")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"rejected reference {value!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_a_refusal_shows_the_rejected_references_as_typed(monkeypatch, value):
    monkeypatch.setattr(
        "soup_cli.utils.reward_synth.calibrate", lambda *a, **k: _report(True, (value,))
    )

    result = _synth(_references(), "--kind", "numeric", "-o", "out.py")

    assert result.exit_code == 2
    text = _plain(result)
    assert f"rejected reference: {value!r}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_stress_title_names_the_target_as_typed(name):
    target = _verifier(f"{name}.py")

    result = _stress(target)

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # The reward loader prints the path on a line of its own; only the title is read here.
    titles = [
        line for line in strip_ansi(result.output).splitlines() if "reward stress:" in line
    ]
    assert len(titles) == 1
    assert "reward stress: " + _shown(name) + ".py" in " ".join(titles[0].split())
    assert _not_shown(titles[0])


def test_the_stress_title_keeps_its_bold_style(monkeypatch):
    monkeypatch.setattr(
        "soup_cli.commands.reward.console",
        Console(width=500, force_terminal=True, color_system="standard", legacy_windows=False),
    )

    result = _stress(_verifier("plain.py"))

    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Bold and green (SGR 1;32) on the title text itself, as the markup title always had.
    assert "\x1b[1;32mreward stress: plain.py\x1b[0m" in result.output
