"""Unmeasured benchmark results must not become measured zeroes (#1718)."""

import sys
import types
from unittest.mock import MagicMock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.commands import eval as eval_command
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (None, None),
        ({"alias": "scoreless"}, None),
        ({"acc,none": 0.0}, 0.0),
        ({"acc,none": 0.0, "acc_norm,none": 0.9}, 0.0),
        ({"acc,none": 0.75}, 0.75),
        ({"other": 0.25}, 0.25),
        ({"other": 0.25, "acc,none": 0.5}, 0.5),
        ({"other": True}, None),
        ({"acc,none": False}, None),
        ({"acc,none": "unavailable", "other": 0.25}, 0.25),
        ({"alias": 123}, None),
        ({}, None),
    ],
)
def test_helper_saves_only_measured_scores(monkeypatch, data, expected):
    tracker = MagicMock()
    monkeypatch.setattr("soup_cli.experiment.tracker.ExperimentTracker", lambda: tracker)
    payload = {"results": {} if data is None else {"requested": data}}

    count = eval_command._save_benchmark_results(payload, "model", ["requested"], "run")

    assert count == (expected is not None)
    if expected is None:
        tracker.save_eval_result.assert_not_called()
    else:
        tracker.save_eval_result.assert_called_once_with(
            model_path="model", benchmark="requested", score=expected, details=data, run_id="run",
        )


@pytest.fixture
def benchmark_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "synthetic.sqlite3"))
    (tmp_path / "model").mkdir()
    monkeypatch.setattr(eval_command, "console", Console(width=200))

    def invoke(payload, names):
        lm_eval = types.ModuleType("lm_eval")
        lm_eval.simple_evaluate = lambda **kwargs: payload
        monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
        result = CliRunner().invoke(
            eval_command.app,
            ["benchmark", "--model", "model", "--benchmarks", names, "--device", "cpu"],
        )
        tracker = ExperimentTracker()
        try:
            rows = tracker.get_eval_results()
        finally:
            tracker.close()
        return result, strip_ansi(result.output), rows

    return invoke


@pytest.mark.parametrize(
    ("data", "reason"),
    [(None, "not found"), ({"alias": "scoreless"}, "no numeric result"),
     ({"other": True}, "no numeric result"), ({"acc,none": False}, "no numeric result")],
)
def test_cli_nothing_measured_is_failure(benchmark_cli, data, reason):
    payload = {"results": {} if data is None else {"requested": data}}
    result, output, rows = benchmark_cli(payload, "requested")

    assert result.exit_code == 1, output
    assert rows == []
    assert f"Skipped requested: {reason}" in output
    assert "Nothing measured; nothing saved" in output
    assert "Results saved to experiment tracker" not in output
    assert "1.0000" not in output and "0.0000" not in output


@pytest.mark.parametrize("score", [0.0, 0.75])
def test_cli_real_scores_are_saved(benchmark_cli, score):
    result, output, rows = benchmark_cli({"results": {"requested": {"acc,none": score}}},
                                         "requested")

    assert result.exit_code == 0, output
    assert [(row["benchmark"], row["score"]) for row in rows] == [("requested", score)]
    assert "Results saved to experiment tracker: 1 of 1" in output


def test_cli_partial_results_name_count_and_skip_group(benchmark_cli):
    payload = {"results": {"zero": {"acc,none": 0.0}, "value": {"acc,none": 0.75},
                           "group": {"alias": "group"}, "subtask": {"acc,none": 1.0}},
               "groups": {"group": {"acc,none": 1.0}}}
    result, output, rows = benchmark_cli(payload, "zero,missing,value,group")

    assert result.exit_code == 0, output
    assert {row["benchmark"]: row["score"] for row in rows} == {"zero": 0.0, "value": 0.75}
    assert "Skipped missing: not found" in output
    assert "Skipped group: no numeric result" in output
    assert "Skipped 2 of 4" in output
    assert "Results saved to experiment tracker: 2 of 4" in output


def test_cli_group_own_numeric_entry_is_measured(benchmark_cli):
    result, output, rows = benchmark_cli(
        {"results": {"group": {"acc,none": 0.5}, "subtask": {"acc,none": 1.0}}}, "group",
    )
    assert result.exit_code == 0, output
    assert [(row["benchmark"], row["score"]) for row in rows] == [("group", 0.5)]


def test_cli_skipped_name_is_terminal_safe(benchmark_cli):
    name = "[red]missing[/red]\x1b\x9b"
    result, output, rows = benchmark_cli({"results": {}}, name)

    assert result.exit_code == 1, output
    assert rows == []
    assert "Skipped [red]missing[/red]: not found" in output
    assert "\x1b" not in output and "\x9b" not in output
