"""soup advise prints typed paths, stored text and errors as typed."""

import json
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import soup_cli.commands.advise as advise_cmd
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
    monkeypatch.setattr("soup_cli.commands.advise.console", Console(width=500, height=100))
    monkeypatch.setenv("COLUMNS", "500")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("SOUP_ADVISE_HISTORY_PATH", str(tmp_path / "history.jsonl"))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    rows = [{"prompt": f"summarize doc {i}", "response": f"tldr {i}"} for i in range(120)]
    (workdir / "data.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


def _shown(value: str) -> str:
    return "".join(ch for ch in value if ch not in _CONTROLS)


def _not_shown(text: str) -> bool:
    return "\x9b" not in text and "\u202e" not in text


def _run(*args: str, data: str = "data.jsonl"):
    return runner.invoke(advise_cmd.app, ["run", data, *args])


def _raising(error):
    def fail(*args, **kwargs):
        raise error

    return fail


@pytest.mark.parametrize("name", _NAMES)
def test_a_missing_dataset_is_shown_as_typed(name):
    result = _run(data=f"{name}.jsonl")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Dataset error: dataset not found: {_shown(name)}.jsonl" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_a_dataset_outside_cwd_is_shown_as_typed(name):
    Path(f"../{name}.jsonl").write_text("{}\n", encoding="utf-8")

    result = _run(data=f"../{name}.jsonl")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Dataset error: dataset path '../{_shown(name)}.jsonl' must stay under cwd" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_analysis_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(advise_cmd, "classify_task", _raising(error(f"analysis said {value}")))

    result = _run()

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Analysis failed: analysis said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_proximity_probe_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(
        advise_cmd, "measure_base_model_proximity", _raising(error(f"proximity said {value}"))
    )

    result = _run("--probe-model", "some-model")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Proximity probe failed: proximity said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_probe_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(advise_cmd, "synth_probe_baselines", _raising(error(f"probe said {value}")))

    result = _run("--probe")

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Probe failed: probe said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_verdict_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(advise_cmd, "build_verdict", _raising(error(f"verdict said {value}")))

    result = _run()

    assert result.exit_code == 1
    text = _plain(result)
    assert f"Verdict build failed: verdict said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError, OSError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_record_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(advise_cmd, "record_verdict", _raising(error(f"record said {value}")))

    result = _run("--record")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"History record failed: record said {_shown(value)}" in text
    assert _not_shown(text)


@pytest.mark.parametrize("name", _NAMES)
def test_the_recorded_history_path_is_shown_as_typed(monkeypatch, tmp_path, name):
    monkeypatch.setenv("SOUP_ADVISE_HISTORY_PATH", str(tmp_path / f"{name}.jsonl"))

    result = _run("--record")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert re.search(r"Recorded to \S*" + re.escape(_shown(name)) + r"\.jsonl", text)
    assert _not_shown(text)


@pytest.mark.parametrize("error", [ValueError, TypeError])
@pytest.mark.parametrize("value", _VALUES)
def test_a_failed_history_read_is_shown_as_typed(monkeypatch, error, value):
    monkeypatch.setattr(advise_cmd, "load_history", _raising(error(f"history said {value}")))

    result = runner.invoke(advise_cmd.app, ["compare"])

    assert result.exit_code == 1
    text = _plain(result)
    assert f"History error: history said {_shown(value)}" in text
    assert _not_shown(text)


def _history_row(project: str, timestamp: str) -> str:
    return json.dumps(
        {
            "timestamp": timestamp,
            "project": project,
            "choice": "SFT",
            "task_category": "summarization",
            "confidence": 0.5,
            "accepted": True,
        }
    )


@pytest.mark.parametrize("value", _VALUES)
def test_the_history_table_shows_the_project_as_typed(tmp_path, value):
    (tmp_path / "history.jsonl").write_text(
        _history_row(f"proj-{value}", "2026-01-02T03:04:05") + "\n", encoding="utf-8"
    )

    result = runner.invoke(advise_cmd.app, ["compare"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"2026-01-02T03:04 \u2502 proj-{_shown(value)} \u2502 SFT \u2502 summarization" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_the_history_table_shows_the_time_as_typed(tmp_path, value):
    # The table keeps the first 16 characters of the stored time.
    when = (value + "-" * 16)[:16]
    (tmp_path / "history.jsonl").write_text(
        _history_row("proj", value + "-" * 40) + "\n", encoding="utf-8"
    )

    result = runner.invoke(advise_cmd.app, ["compare"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    assert f"\u2502 {_shown(when)} \u2502 proj \u2502 SFT \u2502" in text
    assert _not_shown(text)


@pytest.mark.parametrize("value", _VALUES)
def test_explain_shows_the_cached_verdict_as_typed(tmp_path, value):
    cache = tmp_path / "home" / ".soup"
    cache.mkdir()
    (cache / "advise_last.json").write_text(
        json.dumps(
            {
                "choice": "SFT",
                "confidence": 0.5,
                "reason": f"reason {value}",
                "reverse_when": f"flip {value}",
                "task_category": f"task {value}",
                "roi": {},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(advise_cmd.app, ["explain"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = _plain(result)
    shown = _shown(value)
    assert f"Task category: task {shown} Reason: reason {shown} Reverses when: flip {shown}" in text
    assert _not_shown(text)
