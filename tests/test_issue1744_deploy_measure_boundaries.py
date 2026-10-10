"""Count-derived deploy measure boundaries must retain their documented verdicts (#1744)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import deploy_measure
from soup_cli.utils.deploy_measure import measure_candidate


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.mark.parametrize("before, after, total, expected", [
    (19, 18, 20, "MAJOR"),
    (20, 19, 20, "MAJOR"),
    (410, 400, 500, "MINOR"),
    (50, 49, 50, "MINOR"),
    (47, 46, 50, "MINOR"),
    (40, 39, 40, "MINOR"),
    (100, 99, 100, "OK"),
    (20, 20, 20, "OK"),
    (18, 19, 20, "OK"),
])
def test_count_derived_verdict_through_measure_candidate(
    tmp_path: Path, before: int, after: int, total: int, expected: str
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(
        json.dumps({"prompt": f"p{i}", "expected": "ok", "scoring": "exact"}) + "\n"
        for i in range(total)
    ), encoding="utf-8")

    def before_gen(p: str) -> str:
        return "ok" if int(p[1:]) < before else "wrong"

    def after_gen(p: str) -> str:
        return "ok" if int(p[1:]) < after else "wrong"

    res = measure_candidate(
        candidate="4bit",
        tasks_file=str(tasks),
        before_gen=before_gen,
        after_gen=after_gen,
    )
    assert res.verdict == expected
    assert res.before == before / total
    assert res.after == after / total
    assert res.delta == after / total - before / total


def test_count_derived_verdict_with_before_score(tmp_path: Path) -> None:
    total = 20
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(
        json.dumps({"prompt": f"p{i}", "expected": "ok", "scoring": "exact"}) + "\n"
        for i in range(total)
    ), encoding="utf-8")

    def after_gen(p: str) -> str:
        return "ok" if int(p[1:]) < 18 else "wrong"

    res = measure_candidate(
        candidate="4bit",
        tasks_file=str(tasks),
        before_score=19 / 20,
        after_gen=after_gen,
    )
    assert res.verdict == "MAJOR"
    assert res.before == 19 / 20
    assert res.after == 18 / 20


@pytest.mark.parametrize("delta, expected", [
    (-0.0199999, "OK"),
    (-0.0200001, "MINOR"),
    (-0.0499999, "MINOR"),
    (-0.0500001, "MAJOR"),
])
def test_real_near_boundary_differences_are_not_float_noise(
    tmp_path: Path, delta: float, expected: str
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"prompt": "p0", "expected": "ok", "scoring": "exact"}\n', encoding="utf-8")
    res = measure_candidate(
        candidate="4bit",
        tasks_file=str(tasks),
        before_score=1.0 - delta,
        after_gen=lambda p: "ok",
    )
    assert res.verdict == expected


def test_count_derived_five_point_drop_through_deploy_autopilot_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    total = 20
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(
        json.dumps({"prompt": f"p{i}", "expected": "ok", "scoring": "exact"}) + "\n"
        for i in range(total)
    ), encoding="utf-8")

    def before(p: str) -> str:
        return "ok" if int(p[1:]) < 19 else "wrong"

    def after_factory(candidate: str):
        return lambda p: "ok" if int(p[1:]) < 18 else "wrong"

    monkeypatch.setattr(deploy_measure, "_DEPLOY_MEASURE_BEFORE_GEN", before, raising=False)
    monkeypatch.setattr(
        deploy_measure, "_DEPLOY_MEASURE_AFTER_FACTORY", after_factory, raising=False
    )
    monkeypatch.setenv("SOUP_DEPLOY_AUTOPILOT_CACHE", str(tmp_path / "cache.json"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "deploy", "autopilot",
            "--target", "rtx-4090-24gb",
            "--base", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--recipe-out", str(tmp_path / "recipe.yaml"),
            "--script-out", str(tmp_path / "deploy.sh"),
            "--measure",
            "--tasks", str(tasks),
            "--measure-candidates", "4bit",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    clean_output = _strip_ansi(result.output)
    assert "MAJOR" in clean_output
    assert "Recommended: 4bit (verdict=MAJOR" in clean_output


def test_count_derived_two_point_drop_through_deploy_autopilot_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    total = 500
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(
        json.dumps({"prompt": f"p{i}", "expected": "ok", "scoring": "exact"}) + "\n"
        for i in range(total)
    ), encoding="utf-8")

    def before(p: str) -> str:
        return "ok" if int(p[1:]) < 410 else "wrong"

    def after_factory(candidate: str):
        return lambda p: "ok" if int(p[1:]) < 400 else "wrong"

    monkeypatch.setattr(deploy_measure, "_DEPLOY_MEASURE_BEFORE_GEN", before, raising=False)
    monkeypatch.setattr(
        deploy_measure, "_DEPLOY_MEASURE_AFTER_FACTORY", after_factory, raising=False
    )
    monkeypatch.setenv("SOUP_DEPLOY_AUTOPILOT_CACHE", str(tmp_path / "cache.json"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "deploy", "autopilot",
            "--target", "rtx-4090-24gb",
            "--base", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--recipe-out", str(tmp_path / "recipe.yaml"),
            "--script-out", str(tmp_path / "deploy.sh"),
            "--measure",
            "--tasks", str(tasks),
            "--measure-candidates", "4bit",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    clean_output = _strip_ansi(result.output)
    assert "MINOR" in clean_output
    assert "Recommended: 4bit (verdict=MINOR" in clean_output
