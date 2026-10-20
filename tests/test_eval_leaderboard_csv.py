"""CSV exports preserve evaluation identities and existing output conventions."""

import csv
import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.commands.eval import app
from soup_cli.eval.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    build_leaderboard_from_tracker,
    export_leaderboard,
)
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi


def parse_csv(payload: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(payload, newline="")))


def make_leaderboard(model: str, benchmark: str) -> Leaderboard:
    board = Leaderboard(entries=[
        LeaderboardEntry(model_path=model, benchmark=benchmark, score=0.75),
    ])
    board.compute()
    return board


@pytest.mark.parametrize("model", [
    "models/base,adapter",
    '"quoted-model"',
    "models/first\nsecond",
    "models/first\rsecond",
    "models/first\r\nsecond",
    'models/"base",adapter',
])
def test_csv_model_identity_round_trip(model: str) -> None:
    board = make_leaderboard(model, "custom")
    assert parse_csv(export_leaderboard(board, fmt="csv")) == [
        ["model", "custom", "average"],
        [model, "0.75", "0.7500"],
    ]


@pytest.mark.parametrize("benchmark", [
    "metric,split",
    '"quoted-metric"',
    "metric\npart",
    "metric\rpart",
    "metric\r\npart",
])
def test_csv_benchmark_identity_round_trip(benchmark: str) -> None:
    board = make_leaderboard("models/base", benchmark)
    assert parse_csv(export_leaderboard(board, fmt="csv")) == [
        ["model", benchmark, "average"],
        ["models/base", "0.75", "0.7500"],
    ]


@pytest.mark.parametrize("model,benchmark", [
    ("models/base,adapter", "custom"),
    ("models/base", "metric,split"),
])
@pytest.mark.parametrize("force_color", [False, True])
def test_actual_cli_csv_from_real_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, benchmark: str,
    force_color: bool,
) -> None:
    import soup_cli.experiment.tracker as tracker_module

    database = tmp_path / "synthetic.db"
    tracker = ExperimentTracker(db_path=database)
    tracker.save_eval_result(model, benchmark, 0.75, {})
    tracker.close()
    opened: list[ExperimentTracker] = []

    def open_tracker() -> ExperimentTracker:
        connection = ExperimentTracker(db_path=database)
        opened.append(connection)
        return connection

    monkeypatch.setattr(tracker_module, "ExperimentTracker", open_tracker)
    if force_color:
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.setenv("COLUMNS", "80")
        monkeypatch.setenv("LINES", "24")
    else:
        monkeypatch.delenv("FORCE_COLOR", raising=False)
    try:
        result = CliRunner().invoke(app, ["leaderboard", "--format", "csv"])
        assert result.exit_code == 0, result.exception
        # Normalization must not hide styling added to machine-readable CSV.
        assert result.stdout == strip_ansi(result.stdout)
        assert parse_csv(strip_ansi(result.stdout)) == [
            ["model", benchmark, "average"],
            [model, "0.75", "0.7500"],
        ]
    finally:
        for connection in opened:
            connection.close()


def test_plain_csv_control() -> None:
    board = make_leaderboard("models/base", "custom")
    assert export_leaderboard(board, fmt="csv") == (
        "model,custom,average\nmodels/base,0.75,0.7500"
    )


def test_plain_two_row_csv_is_byte_exact() -> None:
    board = Leaderboard(entries=[
        LeaderboardEntry(model_path="beta", benchmark="custom", score=0.25),
        LeaderboardEntry(model_path="alpha", benchmark="custom", score=0.75),
    ])
    board.compute()
    assert export_leaderboard(board, fmt="csv") == (
        "model,custom,average\n"
        "alpha,0.75,0.7500\n"
        "beta,0.25,0.2500"
    )


def test_json_control_preserves_special_names() -> None:
    board = make_leaderboard('models/"base",adapter', "metric,split")
    assert json.loads(export_leaderboard(board, fmt="json")) == [{
        "model": 'models/"base",adapter',
        "scores": {"metric,split": 0.75},
        "average": 0.75,
    }]


def test_tracker_preserves_names_before_export(tmp_path: Path) -> None:
    tracker = ExperimentTracker(db_path=tmp_path / "synthetic.db")
    try:
        tracker.save_eval_result("models/base,adapter", "metric,split", 0.75, {})
        board = build_leaderboard_from_tracker(tracker)
        assert board.models == {"models/base,adapter": {"metric,split": 0.75}}
    finally:
        tracker.close()


@pytest.mark.parametrize("board,expected", [
    (Leaderboard(), "model,,average"),
    (Leaderboard(models={"model": {}}), "model,,average\nmodel,,0.0000"),
])
def test_empty_csv_conventions_unchanged(board: Leaderboard, expected: str) -> None:
    assert export_leaderboard(board, fmt="csv") == expected


def test_csv_order_missing_cells_and_average_format_unchanged() -> None:
    board = Leaderboard(entries=[
        LeaderboardEntry(model_path="alpha", benchmark="zeta", score=0.25),
        LeaderboardEntry(model_path="beta", benchmark="beta", score=0.5),
        LeaderboardEntry(model_path="alpha", benchmark="beta", score=1.0),
    ])
    board.compute()
    assert export_leaderboard(board, fmt="csv") == (
        "model,beta,zeta,average\n"
        "alpha,1.0,0.25,0.6250\n"
        "beta,0.5,,0.5000"
    )


@pytest.mark.parametrize("lead", ["=", "+", "-", "@", "\t", "\r", "\n"])
def test_formula_like_names_are_prefixed_with_a_quote(lead: str) -> None:
    name = lead + "SUM(1,2)"
    model_rows = parse_csv(export_leaderboard(make_leaderboard(name, "custom"), fmt="csv"))
    assert model_rows == [
        ["model", "custom", "average"],
        ["'" + name, "0.75", "0.7500"],
    ]
    bench_rows = parse_csv(export_leaderboard(make_leaderboard("models/base", name), fmt="csv"))
    assert bench_rows == [
        ["model", "'" + name, "average"],
        ["models/base", "0.75", "0.7500"],
    ]


def test_plain_names_and_negative_score_are_unchanged() -> None:
    # A formula character that is not the first character is left alone, and so is a
    # negative score: only the name cells are prefixed.
    board = Leaderboard(entries=[
        LeaderboardEntry(model_path="models/a=b-c@d+e", benchmark="delta-x", score=-0.5),
    ])
    board.compute()
    assert export_leaderboard(board, fmt="csv") == (
        "model,delta-x,average\nmodels/a=b-c@d+e,-0.5,-0.5000"
    )


def test_spreadsheet_safe_helper() -> None:
    from soup_cli.utils.csv_safe import spreadsheet_safe

    assert spreadsheet_safe("=A1") == "'=A1"
    assert spreadsheet_safe("\nx") == "'\nx"
    assert spreadsheet_safe("plain") == "plain"
    assert spreadsheet_safe("") == ""
    assert spreadsheet_safe(-0.5) == "'-0.5"  # callers keep numbers out of it
