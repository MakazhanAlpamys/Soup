"""Every command that looks up a run by id or prefix says when a prefix is ambiguous (#2470).

The tracker returns ``None`` both for an id that names no run and for a prefix
that names several, so each caller asks ``run_lookup_failure`` which one it was.
"""

from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import make_cli_runner, strip_ansi
from tests.ui_helpers import ui_client

runner = make_cli_runner()

AMBIGUOUS = "run_dup_"
AMBIGUOUS_LINE = "Run id prefix 'run_dup_' matches 2 runs; give more characters"
MISSING = "run_gone"
MISSING_LINE = "Run not found: run_gone"
EXISTING = "run_solo_1"


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(path))
    monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("COLUMNS", "500")
    for module in ("runs", "why", "rewind", "registry", "eval"):
        monkeypatch.setattr(f"soup_cli.commands.{module}.console", Console(width=500))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    tracker = ExperimentTracker(db_path=path)
    try:
        for run_id in ("run_dup_1", "run_dup_2", "run_[x]\x9b1", "run_[x]\x9b2", EXISTING):
            tracker.start_run(
                config_dict={}, device="cpu", device_name="CPU", gpu_info={}, run_id=run_id
            )
    finally:
        tracker.close()
    return path


def _commands():
    """(id, argv builder) for every CLI command that prints the lookup failure."""
    return [
        pytest.param(lambda r: ["runs", "show", r], id="runs-show"),
        pytest.param(lambda r: ["runs", "compare", r, EXISTING], id="runs-compare-first"),
        pytest.param(lambda r: ["runs", "compare", EXISTING, r], id="runs-compare-second"),
        pytest.param(lambda r: ["runs", "delete", r], id="runs-delete"),
        pytest.param(lambda r: ["runs", "clean", r], id="runs-clean"),
        pytest.param(lambda r: ["runs", "replay", r], id="runs-replay"),
        pytest.param(lambda r: ["runs", "curriculum-curve", r], id="runs-curriculum-curve"),
        pytest.param(lambda r: ["why", r], id="why"),
        pytest.param(lambda r: ["rewind", r], id="rewind"),
        pytest.param(
            lambda r: ["registry", "push", "--run-id", r, "--name", "demo"], id="registry-push"
        ),
        pytest.param(lambda r: ["eval", "compare", r, EXISTING], id="eval-compare-first"),
        pytest.param(lambda r: ["eval", "compare", EXISTING, r], id="eval-compare-second"),
        pytest.param(
            lambda r: ["eval", "aider", "--model", "m", "--output", "out", "--run-id", r],
            id="eval-aider",
        ),
        pytest.param(
            lambda r: ["eval", "judge", "--target", "responses.jsonl", "--run-id", r],
            id="eval-judge",
        ),
    ]


def _lines(result) -> list[str]:
    return [line.strip() for line in strip_ansi(result.stdout).splitlines() if line.strip()]


@pytest.mark.parametrize("argv", _commands())
def test_ambiguous_prefix_is_named_as_ambiguous(argv):
    result = runner.invoke(app, argv(AMBIGUOUS))

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    lines = _lines(result)
    assert AMBIGUOUS_LINE in lines
    assert not any("Run not found" in line for line in lines)


@pytest.mark.parametrize("argv", _commands())
def test_missing_run_keeps_run_not_found(argv):
    result = runner.invoke(app, argv(MISSING))

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert MISSING_LINE in _lines(result)


@pytest.mark.parametrize(
    "left,right",
    [
        ("run_baseline_v1", "run_candidate_v1"),
        ("run_baseline", "run_candidate_v1"),
        ("run_baseline_v1", "run_candidate"),
        ("run_baseline", "run_candidate"),
    ],
)
def test_eval_compare_loads_scores_for_resolved_runs(db: Path, left: str, right: str) -> None:
    tracker = ExperimentTracker(db_path=db)
    try:
        for run_id, score in (("run_baseline_v1", 0.9), ("run_candidate_v1", 0.7)):
            tracker.start_run({}, "cpu", "CPU", {}, run_id=run_id)
            tracker.save_eval_result("model", "synthetic", score, {}, run_id=run_id)
    finally:
        tracker.close()

    result = runner.invoke(app, ["eval", "compare", left, right])

    assert result.exit_code == 0, result.output
    output = " ".join(strip_ansi(result.output).split())
    assert f"Eval Comparison: {left} vs {right}" in output
    assert "0.9000" in output
    assert "0.7000" in output
    assert "-0.2000" in output


@pytest.mark.parametrize("run_id", [AMBIGUOUS, MISSING])
def test_runs_show_keeps_its_hint_line(run_id):
    result = runner.invoke(app, ["runs", "show", run_id])

    assert result.exit_code == 1
    assert _lines(result)[-1] == "Use soup runs to see all runs."


@pytest.mark.parametrize("path", ["/api/runs/{}", "/api/runs/{}/metrics"])
@pytest.mark.parametrize(
    ("run_id", "detail"),
    [
        (AMBIGUOUS, AMBIGUOUS_LINE),
        (MISSING, "Run not found"),
        ("run_[x]\x9b", "Run id prefix 'run_[x]' matches 2 runs; give more characters"),
    ],
)
def test_ui_run_lookup_404_detail(path, run_id, detail):
    from soup_cli.ui.app import create_app, get_auth_token

    client = ui_client(create_app())
    response = client.get(
        path.format(run_id), headers={"Authorization": f"Bearer {get_auth_token()}"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == detail
