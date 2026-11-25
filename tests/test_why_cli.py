"""The why summary distinguishes measured zeroes from absent measurements."""

from contextlib import closing

import pytest

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


@pytest.mark.parametrize("latest", [False, True])
@pytest.mark.parametrize(
    ("initial", "final", "steps"),
    [(0.0, 0.0, 0), (2.5, 0.0, 10), (0.0, 1.5, 10), (2.5, 1.5, 10)],
)
def test_why_preserves_recorded_summary_numbers(
    tmp_path, monkeypatch, latest, initial, final, steps,
):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    with closing(ExperimentTracker()) as tracker:
        run_id = tracker.start_run({}, "cpu", "CPU", {})
        tracker.finish_run(run_id, initial, final, steps, 1.0, "output")
        before = tracker.get_run(run_id)

    result = runner.invoke(app, ["why"] if latest else ["why", run_id])

    assert result.exit_code == 0, result.output
    text = strip_ansi(result.stdout)
    assert f"Steps: {steps}" in text
    assert f"Loss: {initial} → {final}" in text
    with closing(ExperimentTracker()) as tracker:
        assert tracker.get_run(run_id) == before


def test_why_still_marks_absent_summary_numbers(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    with closing(ExperimentTracker()) as tracker:
        run_id = tracker.start_run({}, "cpu", "CPU", {})

    result = runner.invoke(app, ["why", run_id])

    assert result.exit_code == 0, result.output
    text = strip_ansi(result.stdout)
    assert "Steps: -" in text
    assert "Loss: - → -" in text
