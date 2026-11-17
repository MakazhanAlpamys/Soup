"""Compare and leaderboard render external names without altering stored evidence."""

import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import eval as eval_command
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi

SPELLINGS = [
    pytest.param("plain", "plain", id="plain"),
    pytest.param(":rocket:", ":rocket:", id="emoji"),
    pytest.param("[red]name", "[red]name", id="markup"),
    pytest.param(r"name\[red]tag", r"name\[red]tag", id="backslash"),
    pytest.param("name\x1b\x07\x9b\u202e[end]", "name[end]", id="control-and-bidi"),
]

runner = CliRunner()


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "eval.db"))
    monkeypatch.setattr(
        eval_command, "console",
        Console(width=250, height=40, force_terminal=True, color_system="standard", no_color=False),
    )
    db = ExperimentTracker()
    try:
        yield db
    finally:
        db.close()


def _start(tracker, *ids):
    for run_id in ids:
        tracker.start_run({}, "cpu", "test", {}, run_id=run_id)


def _lines(result, exit_code=0):
    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    plain = strip_ansi(result.output)
    assert not any(char in plain for char in ("\x1b", "\x07", "\x9b", "\u202e")), repr(plain)
    cells = str.maketrans("│┃|", "   ")
    return [" ".join(line.translate(cells).split()) for line in plain.splitlines() if line.strip()]


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_compare_shows_run_ids_benchmark_and_regression_literally(tracker, value, shown):
    left, right = f"before-{value}", f"after-{value}"
    bench = f"shared-benchmark-for-both-runs-{value}"
    _start(tracker, left, right)
    tracker.save_eval_result("model", bench, 0.8, {}, run_id=left)
    tracker.save_eval_result("model", bench, 0.6, {}, run_id=right)

    lines = _lines(runner.invoke(app, ["eval", "compare", left, right]))

    assert f"Eval Comparison: before-{shown} vs after-{shown}" in lines
    assert f"shared-benchmark-for-both-runs-{shown} 0.8000 0.6000 -0.2000" in lines
    assert f"Regressions detected: shared-benchmark-for-both-runs-{shown}" in lines


@pytest.mark.parametrize("value,shown", SPELLINGS)
def test_leaderboard_shows_model_and_benchmark_literally(tracker, value, shown):
    tracker.save_eval_result(f"model-{value}", f"bench-{value}", 0.75, {})

    lines = _lines(runner.invoke(app, ["eval", "leaderboard"]))

    assert f"# Model bench-{shown} Avg" in lines
    assert f"1 model-{shown} 0.7500 0.7500" in lines


def test_leaderboard_keeps_shortened_path_and_literal_text(tracker):
    tracker.save_eval_result("root/nested/model-:rocket:[red]name", "bench", 0.75, {})

    lines = _lines(runner.invoke(app, ["eval", "leaderboard"]))

    assert "1 nested/model-:rocket:[red]name 0.7500 0.7500" in lines


@pytest.mark.parametrize("value,shown", SPELLINGS)
@pytest.mark.parametrize("ambiguous", [False, True], ids=["missing", "ambiguous"])
def test_compare_run_lookup_refusals_show_typed_id(tracker, value, shown, ambiguous):
    query = f"prefix-{value}"
    if ambiguous:
        _start(tracker, f"{query}-one", f"{query}-two")
        expected = f"Run id prefix 'prefix-{shown}' matches 2 runs; give more characters"
    else:
        expected = f"Run not found: prefix-{shown}"

    result = runner.invoke(app, ["eval", "compare", query, "other-run"])

    assert _lines(result, 1) == [expected]


@pytest.mark.parametrize("value,_shown", SPELLINGS)
@pytest.mark.parametrize("command", ["compare", "leaderboard"])
def test_stored_score_refusal_displays_literal_context(tracker, value, _shown, command):
    first, second = f"run-{value}", "valid-run"
    model, bench = f"model-{value}", f"bench-{value}"
    _start(tracker, first, second)
    tracker.save_eval_result(model, bench, float("inf"), {}, run_id=first)

    if command == "compare":
        args = ["eval", "compare", first, second]
        context = f"run {first!r}"
    else:
        args = ["eval", "leaderboard"]
        context = f"model {model!r}"
    result = runner.invoke(app, args)

    assert _lines(result, 1) == [f"{context}, benchmark {bench!r}: stored score inf must be finite"]


@pytest.mark.parametrize("value,_shown", SPELLINGS)
def test_leaderboard_exports_preserve_original_bytes(tracker, value, _shown):
    model, bench = f"model-{value}", f"bench-{value}"
    tracker.save_eval_result(model, bench, 0.75, {})
    expected_json = json.dumps(
        [{"model": model, "scores": {bench: 0.75}, "average": 0.75}], indent=2,
    ) + "\n"
    expected_csv = f"model,{bench},average\n{model},0.75,0.7500\n"

    for fmt, expected in (("json", expected_json), ("csv", expected_csv)):
        result = runner.invoke(app, ["eval", "leaderboard", "--format", fmt])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert result.stdout.encode("utf-8") == expected.encode("utf-8")
