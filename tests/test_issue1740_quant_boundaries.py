"""Count-derived quant-check boundaries must retain their documented verdicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import eval as eval_command
from soup_cli.eval import quant_check

from .conftest import strip_ansi


@pytest.mark.parametrize("before, after, total, expected", [
    (19, 18, 20, "MAJOR"),
    (20, 19, 20, "MAJOR"),
    (47, 46, 50, "MINOR"),
    (50, 49, 50, "MINOR"),
    (40, 39, 40, "MINOR"),
    (100, 99, 100, "OK"),
    (20, 20, 20, "OK"),
    (18, 19, 20, "OK"),
])
def test_count_derived_verdict(before: int, after: int, total: int, expected: str) -> None:
    assert quant_check.classify_delta(after / total - before / total) == expected


@pytest.mark.parametrize("before, after, expected", [
    (95, 91, "MAJOR"),
    (29, 28, "MINOR"),
])
def test_custom_thresholds_keep_their_boundaries(before: int, after: int, expected: str) -> None:
    delta = after / 100 - before / 100
    assert quant_check.classify_delta(delta, minor=0.01, major=0.04) == expected


@pytest.mark.parametrize("delta, expected", [
    (-0.0199999, "OK"),
    (-0.0200001, "MINOR"),
    (-0.0499999, "MINOR"),
    (-0.0500001, "MAJOR"),
])
def test_real_near_boundary_differences_are_not_float_noise(delta: float, expected: str) -> None:
    assert quant_check.classify_delta(delta) == expected


@pytest.mark.parametrize("total, before, after, verdict, exit_code", [
    (20, 19, 18, "MAJOR", 2),
    (20, 20, 19, "MAJOR", 2),
    (50, 47, 46, "MINOR", 0),
    (40, 40, 39, "MINOR", 0),
    (20, 20, 20, "OK", 0),
])
def test_count_boundaries_through_core_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    total: int, before: int, after: int, verdict: str, exit_code: int,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "before").mkdir()
    (tmp_path / "after").mkdir()
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(
        json.dumps({"prompt": str(index), "expected": "ok", "scoring": "exact"}) + "\n"
        for index in range(total)
    ), encoding="utf-8")

    def generator(correct: int) -> Callable[[str], str]:
        return lambda prompt: "ok" if int(prompt) < correct else "wrong"

    before_gen, after_gen = generator(before), generator(after)
    core_row = quant_check.run_quant_check(
        before_gen=before_gen, after_gen=after_gen, tasks_file=str(tasks),
    ).rows[0]
    generators = {"before": before_gen, "after": after_gen}
    monkeypatch.setattr(
        quant_check, "make_model_generator", lambda path: generators[Path(path).name],
    )
    monkeypatch.setattr(eval_command, "console", Console(width=200))
    result = CliRunner().invoke(app, [
        "eval", "quant-check", "--before", "before", "--after", "after",
        "--tasks", "tasks.jsonl", "--format", "json",
    ])
    cli_row = json.loads(strip_ansi(result.output))["rows"][0]
    assert core_row.before == cli_row["before"] == before / total
    assert core_row.after == cli_row["after"] == after / total
    assert core_row.delta == cli_row["delta"] == after / total - before / total
    assert (core_row.verdict, cli_row["verdict"], result.exit_code) == (verdict, verdict, exit_code)
