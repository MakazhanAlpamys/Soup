"""Count-derived deploy measure boundaries must retain their documented verdicts (#1744)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import deploy as deploy_command
from soup_cli.utils import deploy_measure

from .conftest import strip_ansi


def _make_tasks(tmp_path: Path, total: int) -> Path:
    tasks = tmp_path / f"tasks_{total}.jsonl"
    tasks.write_text(
        "".join(
            json.dumps({"prompt": str(index), "expected": "ok", "scoring": "exact"}) + "\n"
            for index in range(total)
        ),
        encoding="utf-8",
    )
    return tasks


def _make_generator(correct: int) -> Callable[[str], str]:
    return lambda prompt: "ok" if int(prompt) < correct else "wrong"


@pytest.mark.parametrize(
    "before, after, total, expected",
    [
        (19, 18, 20, "MAJOR"),
        (20, 19, 20, "MAJOR"),
        (47, 46, 50, "MINOR"),
        (50, 49, 50, "MINOR"),
        (40, 39, 40, "MINOR"),
        (100, 99, 100, "OK"),
        (20, 20, 20, "OK"),
        (18, 19, 20, "OK"),
    ],
)
def test_count_derived_measure_candidate_verdict(
    tmp_path: Path, before: int, after: int, total: int, expected: str
) -> None:
    tasks = _make_tasks(tmp_path, total)
    before_gen = _make_generator(before)
    after_gen = _make_generator(after)

    res = deploy_measure.measure_candidate(
        candidate="4bit",
        tasks_file=str(tasks),
        before_gen=before_gen,
        after_gen=after_gen,
    )
    assert res.verdict == expected

    res_with_score = deploy_measure.measure_candidate(
        candidate="4bit",
        tasks_file=str(tasks),
        after_gen=after_gen,
        before_score=before / total,
    )
    assert res_with_score.verdict == expected


def test_removing_rounding_fails_five_point_boundary(tmp_path: Path) -> None:
    """Removing round(-delta, 9) produces MINOR for 19/20 -> 18/20."""
    tasks = _make_tasks(tmp_path, 20)
    before_gen = _make_generator(19)
    after_gen = _make_generator(18)

    # Compute raw delta
    delta = 18 / 20 - 19 / 20
    assert delta == -0.04999999999999993
    unrounded_drop = -delta
    assert unrounded_drop < deploy_measure.DEFAULT_MAJOR_THRESHOLD  # raw unrounded check fails!

    res = deploy_measure.measure_candidate(
        candidate="4bit",
        tasks_file=str(tasks),
        before_gen=before_gen,
        after_gen=after_gen,
    )
    assert res.verdict == "MAJOR"


def test_removing_rounding_fails_two_point_boundary(tmp_path: Path) -> None:
    """Removing round(-delta, 9) produces OK for 47/50 -> 46/50."""
    tasks = _make_tasks(tmp_path, 50)
    before_gen = _make_generator(47)
    after_gen = _make_generator(46)

    delta = 46 / 50 - 47 / 50
    assert delta == -0.019999999999999907
    unrounded_drop = -delta
    assert unrounded_drop < deploy_measure.DEFAULT_MINOR_THRESHOLD  # raw unrounded check fails!

    res = deploy_measure.measure_candidate(
        candidate="4bit",
        tasks_file=str(tasks),
        before_gen=before_gen,
        after_gen=after_gen,
    )
    assert res.verdict == "MINOR"


@pytest.mark.parametrize(
    "before_score, after_score, expected",
    [
        (19 / 20, 18 / 20, "MAJOR"),
        (47 / 50, 46 / 50, "MINOR"),
        (1.0, 0.9500001, "MINOR"),
        (1.0, 0.9499999, "MAJOR"),
        (1.0, 0.9800001, "OK"),
        (1.0, 0.9799999, "MINOR"),
    ],
)
def test_real_near_boundary_differences(
    before_score: float, after_score: float, expected: str
) -> None:
    delta = after_score - before_score
    assert deploy_measure.classify_delta(delta) == expected


def test_deploy_measure_cli_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise deploy autopilot --measure through CLI with count-derived boundaries."""
    monkeypatch.chdir(tmp_path)
    cache_file = tmp_path / "deploy_autopilot_cache.json"
    monkeypatch.setenv("SOUP_DEPLOY_AUTOPILOT_CACHE", str(cache_file))
    tasks = _make_tasks(tmp_path, 20)
    before_gen = _make_generator(19)
    after_gen = _make_generator(18)

    monkeypatch.setattr(deploy_measure, "_DEPLOY_MEASURE_BEFORE_GEN", before_gen)
    monkeypatch.setattr(deploy_measure, "_DEPLOY_MEASURE_AFTER_FACTORY", lambda _: after_gen)
    monkeypatch.setattr(deploy_command, "console", Console(width=200))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "deploy",
            "autopilot",
            "--target",
            "rtx-4090-24gb",
            "--base",
            "dummy-model",
            "--measure",
            "--tasks",
            str(tasks.name),
            "--measure-candidates",
            "4bit",
        ],
    )
    assert result.exit_code == 0, result.output
    clean_out = strip_ansi(result.output)
    assert "verdict=MAJOR" in clean_out
