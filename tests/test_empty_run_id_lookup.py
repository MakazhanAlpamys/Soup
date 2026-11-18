"""An empty run id names no run, whatever the number of runs (#2937).

``get_run`` fell back to a prefix lookup whose pattern for ``""`` is ``"%"``: it
matched every run, so with exactly one run ``soup runs delete "" --force``
deleted it and exited 0, and with several it reported an ambiguous prefix.
A whitespace-only id already found nothing (no run id starts with a space) and
is pinned here so that it stays that way.
"""

from pathlib import Path

import pytest
from rich.console import Console

from soup_cli.cli import app
from soup_cli.experiment.tracker import ExperimentTracker
from soup_cli.mcp_server import registry as reg
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()

ONLY = "run_only_1"
NOT_FOUND = "Run not found:"


def _make_db(tmp_path: Path, monkeypatch, run_ids) -> Path:
    path = tmp_path / "experiments.db"
    monkeypatch.setenv("SOUP_DB_PATH", str(path))
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setattr("soup_cli.commands.runs.console", Console(width=500))
    tracker = ExperimentTracker(db_path=path)
    try:
        for run_id in run_ids:
            tracker.start_run(
                config_dict={}, device="cpu", device_name="CPU", gpu_info={}, run_id=run_id
            )
    finally:
        tracker.close()
    return path


@pytest.fixture
def one_run(tmp_path, monkeypatch) -> Path:
    return _make_db(tmp_path, monkeypatch, [ONLY])


@pytest.fixture
def three_runs(tmp_path, monkeypatch) -> Path:
    return _make_db(tmp_path, monkeypatch, ["run_dup_1", "run_dup_2", ONLY])


def _run_ids(db: Path) -> list[str]:
    tracker = ExperimentTracker(db_path=db)
    try:
        return sorted(run["run_id"] for run in tracker.list_runs(limit=100))
    finally:
        tracker.close()


def _text(result) -> str:
    """Output without ANSI escapes, whitespace collapsed to single spaces."""
    return " ".join(strip_ansi(result.output).split())


# --- tracker level ---------------------------------------------------------


@pytest.mark.parametrize("blank", ["", " ", "   "], ids=["empty", "space", "spaces"])
def test_blank_id_finds_no_run(one_run, blank):
    tracker = ExperimentTracker(db_path=one_run)
    try:
        assert tracker.get_run(blank) is None
        assert tracker.count_run_id_matches(blank) == 0
    finally:
        tracker.close()


def test_empty_id_is_not_ambiguous_with_several_runs(three_runs):
    tracker = ExperimentTracker(db_path=three_runs)
    try:
        assert tracker.get_run("") is None
        assert tracker.count_run_id_matches("") == 0
    finally:
        tracker.close()


def test_exact_prefix_and_ambiguous_ids_resolve_as_before(three_runs):
    """Controls: the guard must not change any id that names a real run."""
    tracker = ExperimentTracker(db_path=three_runs)
    try:
        assert tracker.get_run(ONLY)["run_id"] == ONLY
        assert tracker.get_run("run_only")["run_id"] == ONLY
        assert tracker.count_run_id_matches("run_only") == 1
        assert tracker.get_run("run_dup_") is None
        assert tracker.count_run_id_matches("run_dup_") == 2
    finally:
        tracker.close()


# --- CLI -------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "spaces"])
def test_runs_delete_blank_id_keeps_the_only_run(one_run, blank):
    result = runner.invoke(app, ["runs", "delete", blank, "--force"])

    assert result.exit_code == 1, result.output
    assert NOT_FOUND in _text(result)
    assert _run_ids(one_run) == [ONLY]


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["runs", "show", ""], id="runs-show"),
        pytest.param(["runs", "compare", "", ONLY], id="runs-compare-first"),
        pytest.param(["runs", "compare", ONLY, ""], id="runs-compare-second"),
    ],
)
def test_lookup_commands_refuse_empty_id(one_run, argv):
    result = runner.invoke(app, argv)

    assert result.exit_code == 1, result.output
    text = _text(result)
    assert NOT_FOUND in text
    assert "matches" not in text


def test_empty_id_with_several_runs_is_not_found_not_ambiguous(three_runs):
    result = runner.invoke(app, ["runs", "delete", "", "--force"])

    assert result.exit_code == 1, result.output
    text = _text(result)
    assert NOT_FOUND in text
    assert "matches" not in text
    assert _run_ids(three_runs) == sorted(["run_dup_1", "run_dup_2", ONLY])


def test_runs_delete_exact_and_prefix_still_delete(three_runs):
    """Control: a real prefix still resolves and deletes its run."""
    result = runner.invoke(app, ["runs", "delete", "run_only", "--force"])

    assert result.exit_code == 0, result.output
    assert _run_ids(three_runs) == ["run_dup_1", "run_dup_2"]


# --- MCP -------------------------------------------------------------------


def test_mcp_runs_show_refuses_empty_id(one_run):
    # The MCP tool rejects an empty string before any lookup (_require_str).
    with pytest.raises(reg.McpToolError):
        reg.tool_runs_show({"run_id": ""})


def test_mcp_runs_show_refuses_whitespace_id(one_run):
    with pytest.raises(reg.McpToolError, match="run not found"):
        reg.tool_runs_show({"run_id": "   "})
