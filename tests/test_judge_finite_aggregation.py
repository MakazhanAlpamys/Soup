"""Judge arithmetic must not turn representable finite means into inf or nan."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import eval as eval_command
from soup_cli.eval.judge import (
    JudgeEvaluator,
    JudgeResults,
    JudgeScore,
    _compute_weighted_score,
    load_rubric,
)
from soup_cli.experiment.tracker import ExperimentTracker
from tests.conftest import strip_ansi


def rubric(weights: list[object], maximum: float = 5.0) -> dict:
    return {
        "criteria": [
            {"name": str(index), "description": "synthetic", "weight": weight}
            for index, weight in enumerate(weights)
        ],
        "scale": {"min": 0, "max": maximum},
    }


def score_map(values: list[float]) -> dict[str, float]:
    return {str(index): value for index, value in enumerate(values)}


@pytest.mark.parametrize(("weights", "values", "expected"), [
    ([1e308], [5.0], 5.0),
    ([1e308, 1e308], [5.0, 1.0], 3.0),
    ([1e308, 1e308], [1e-308, 1e-308], 1e-308),
    ([10 ** 400], [5.0], 5.0),
    ([sys.float_info.max] * 2, [sys.float_info.max] * 2, sys.float_info.max),
    ([1e308, 1e308], [-1e308, -1e308], -1e308),
    ([1e308, -1e308, 1.0], [5.0, 5.0, 5.0], 5.0),
    ([1e308, 1e308, -1e308, -1e308], [1.0] * 4, 0.0),
])
def test_representable_weighted_mean_survives_intermediate_overflow(
    weights: list[object], values: list[float], expected: float,
) -> None:
    result = _compute_weighted_score(score_map(values), rubric(weights))
    assert math.isfinite(result)
    assert result == expected


@pytest.mark.parametrize(("weights", "values", "expected"), [
    ([1.0] * 3, [0.1] * 3, 0.10000000000000002),
    ([2.0, 1.0], [5.0, 2.0], 4.0),
    ([2.0, -1.0], [5.0, 3.0], 7.0),
    ([0.0], [5.0], 0.0),
    ([-1.0], [5.0], 0.0),
    ([True], [5.0], 5.0),
    ([False], [5.0], 0.0),
])
def test_existing_finite_weighted_result_is_bit_for_bit_unchanged(
    weights: list[object], values: list[float], expected: float,
) -> None:
    assert _compute_weighted_score(score_map(values), rubric(weights)) == expected


def test_missing_weight_still_defaults_to_one() -> None:
    value = {"criteria": [{"name": "a"}, {"name": "b"}]}
    assert _compute_weighted_score({"a": 3.0, "b": 5.0}, value) == 4.0


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_evaluator_rejects_nonfinite_weight_at_construction(weight: float) -> None:
    with pytest.raises(ValueError, match="Criterion 0 weight must be a finite number"):
        JudgeEvaluator(rubric=rubric([weight]), provider="ollama")


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
def test_private_weighted_helper_refuses_nonfinite_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="Criterion 0 weight must be a finite number"):
        _compute_weighted_score({"0": 5.0}, rubric([weight]))


@pytest.mark.parametrize("weight", [0.0, -1.0])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonpositive_weight_does_not_hide_an_invalid_score(
    weight: float, value: float,
) -> None:
    with pytest.raises(ValueError, match="Weighted judge score must be finite"):
        _compute_weighted_score({"0": value}, rubric([weight]))


@pytest.mark.parametrize("total", [0, -1])
@pytest.mark.parametrize("invalid_first", [False, True])
def test_all_scores_are_checked_before_an_overflow_can_short_circuit(
    total: int, invalid_first: bool,
) -> None:
    weights = [10 ** 400, -(10 ** 400) + total]
    values = [5.0, float("nan")]
    if invalid_first:
        weights.reverse()
        values.reverse()
    with pytest.raises(ValueError, match="Weighted judge score must be finite"):
        _compute_weighted_score(score_map(values), rubric(weights))


def test_changed_weight_is_rechecked_before_a_judge_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = JudgeEvaluator(rubric=rubric([1.0]), provider="ollama")
    calls: list[str] = []
    monkeypatch.setattr(
        evaluator, "_call_llm",
        lambda prompt: calls.append(prompt) or json.dumps({"scores": {"0": 5.0}}),
    )
    evaluator.rubric["criteria"][0]["weight"] = float("nan")
    with pytest.raises(ValueError, match="Criterion 0 weight must be a finite number"):
        evaluator.evaluate("synthetic", "synthetic")
    assert calls == []


@pytest.mark.parametrize(("weights", "values"), [
    ([1e308, -1e308, 1.0], [5.0, 1.0, 0.0]),
    ([1.0], [10 ** 400]),
], ids=["signed-mean-overflow", "unrepresentable-integer-score"])
def test_unrepresentable_weighted_score_fails_explicitly(
    weights: list[object], values: list[float],
) -> None:
    with pytest.raises(ValueError, match="Weighted judge score must be finite"):
        _compute_weighted_score(score_map(values), rubric(weights))


@pytest.mark.parametrize(("values", "expected"), [
    ([1e308, 1e308], 1e308),
    ([sys.float_info.max] * 3, sys.float_info.max),
    ([1e308, 1e308, -1e308, -1e308], 0.0),
    ([0.1] * 3, 0.10000000000000002),
])
def test_all_aggregate_means_are_finite_and_keep_ordinary_rounding(
    values: list[float], expected: float,
) -> None:
    result = JudgeResults(scores=[
        JudgeScore(prompt=str(index), response="synthetic", category="synthetic",
                   scores={"quality": value}, weighted_score=value)
        for index, value in enumerate(values)
    ])
    result.compute()
    assert result.overall_score == expected
    assert result.category_scores == {"synthetic": expected}
    assert result.criteria_averages == {"quality": expected}
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10 ** 400],
                         ids=["nan", "infinity", "huge-integer"])
def test_nonfinite_sdk_aggregate_is_not_returned_as_a_score(value: float) -> None:
    result = JudgeResults(scores=[JudgeScore(prompt="q", response="a", weighted_score=value)])
    with pytest.raises(ValueError, match="Judge average must be finite"):
        result.compute()


@pytest.mark.parametrize(("weights", "values", "rows", "expected"), [
    ([1e308], [5.0], 1, 5.0),
    ([1e308, 1e308], [5.0, 1.0], 1, 3.0),
    ([1.0], [1e308], 2, 1e308),
    ([10 ** 400], [5.0], 1, 5.0),
])
def test_cli_saves_finite_weighted_and_aggregate_scores(
    weights: list[object], values: list[float], rows: int, expected: float,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "responses.jsonl"
    target.write_text("".join(
        json.dumps({"prompt": str(index), "response": "synthetic"}) + "\n"
        for index in range(rows)
    ), encoding="utf-8")
    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(yaml.safe_dump(rubric(weights, max(values))), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        JudgeEvaluator, "_call_llm",
        lambda self, prompt: calls.append(prompt) or json.dumps({"scores": score_map(values)}),
    )
    monkeypatch.setattr(eval_command, "console", Console(width=160))
    tracker = ExperimentTracker(db_path=tmp_path / "synthetic.db")
    tracker.start_run({}, "cpu", "synthetic", {}, run_id="synthetic")
    monkeypatch.setattr("soup_cli.experiment.tracker.ExperimentTracker", lambda: tracker)
    try:
        result = CliRunner().invoke(app, [
            "eval", "judge", "--target", str(target), "--rubric", str(rubric_path),
            "--provider", "ollama", "--model", "synthetic", "--run-id", "synthetic",
        ])
        assert result.exit_code == 0, result.output
        saved = tracker.get_eval_results(run_id="synthetic")
        assert len(saved) == 1
        assert saved[0]["score"] == expected
        details = json.loads(saved[0]["details_json"])
        assert details["category_scores"] == {"default": expected}
        assert details["criteria_averages"] == score_map(values)
        json.dumps(details, allow_nan=False)
        assert len(calls) == rows
    finally:
        tracker.close()


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf"),
                                    "not-numeric", None],
                         ids=["nan", "infinity", "negative-infinity", "text", "null"])
def test_cli_rejects_invalid_weight_before_any_request(
    weight: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "responses.jsonl"
    target.write_text('{"prompt":"q","response":"synthetic"}\n', encoding="utf-8")
    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(yaml.safe_dump(rubric([weight])), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        JudgeEvaluator, "_call_llm",
        lambda self, prompt: calls.append(prompt) or '{"scores":{"0":5}}',
    )
    monkeypatch.setattr(eval_command, "console", Console(width=160))
    result = CliRunner().invoke(app, [
        "eval", "judge", "--target", str(target), "--rubric", str(rubric_path),
        "--provider", "ollama", "--model", "synthetic",
    ])
    output = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 1, output
    assert "Criterion 0 weight must be a finite number" in output
    assert calls == []


def test_generic_loader_keeps_weights_ignored_by_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.trainer.grpo_judge import rubric_criteria

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "rank-rubric.yaml"
    path.write_text(yaml.safe_dump(rubric([float("nan")])), encoding="utf-8")
    assert math.isnan(load_rubric(path)["criteria"][0]["weight"])
    assert rubric_criteria(str(path)) == [("0", "synthetic")]
