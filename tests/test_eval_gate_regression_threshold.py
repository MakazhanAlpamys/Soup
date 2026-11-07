"""Raw SDK tolerances must not disable regression detection with NaN or infinity."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import EvalGateConfig
from soup_cli.eval import gate, quant_check
from tests.conftest import strip_ansi

NONFINITE = [float("nan"), float("inf"), -float("inf")]
NONFINITE_IDS = ["nan", "positive-infinity", "negative-infinity"]
ERROR = "regression_threshold must be finite"


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> gate.EvalSuite:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks.jsonl").write_text(
        '{"prompt":"answer yes","expected":"yes","scoring":"exact"}\n', encoding="utf-8",
    )
    return gate.EvalSuite(suite="raw-tolerance", tasks=[gate.GateTask(
        type="custom", name="control", tasks="tasks.jsonl", scorer="exact", threshold=0.0,
    )])


@pytest.mark.parametrize("threshold", NONFINITE, ids=NONFINITE_IDS)
@pytest.mark.parametrize("baseline", [None, {"control": 1.0}])
def test_sdk_refuses_nonfinite_tolerance_before_generation(
    suite: gate.EvalSuite, threshold: float, baseline: dict[str, float] | None,
) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError) as caught:
        gate.run_gate(
            suite, generate_fn=lambda prompt: calls.append(prompt) or "no",
            baseline=baseline, regression_threshold=threshold,
        )
    assert str(caught.value) == ERROR
    assert calls == []


@pytest.mark.parametrize("baseline", [None, {"control": 1.0}])
def test_sdk_refuses_integer_tolerance_beyond_float_range(
    suite: gate.EvalSuite, baseline: dict[str, float] | None,
) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError) as caught:
        gate.run_gate(
            suite, generate_fn=lambda prompt: calls.append(prompt) or "no",
            baseline=baseline, regression_threshold=10**400,
        )
    assert str(caught.value) == ERROR
    assert calls == []


@pytest.mark.parametrize("baseline,exception,message", [
    ([], TypeError, "baseline must be a mapping"),
    ({"control": True}, TypeError, "baseline score for 'control' must be a number"),
    ({"control": float("nan")}, ValueError, "baseline score: score for 'control' must be finite"),
])
def test_baseline_errors_keep_priority(
    suite: gate.EvalSuite, baseline: dict[str, float] | list[object],
    exception: type[Exception], message: str,
) -> None:
    calls: list[str] = []
    with pytest.raises(exception) as caught:
        gate.run_gate(
            suite, generate_fn=lambda prompt: calls.append(prompt) or "no",
            baseline=baseline, regression_threshold=float("nan"),
        )
    assert str(caught.value).startswith(message)
    assert calls == []


@pytest.mark.parametrize("invalid_tasks,message", [
    ("empty", "eval-gate suite 'raw-tolerance' field 'tasks' must contain at least one task"),
    ("duplicate", "eval-gate suite 'raw-tolerance' repeats task name 'control'"),
])
def test_suite_errors_keep_priority(
    suite: gate.EvalSuite, invalid_tasks: str, message: str,
) -> None:
    if invalid_tasks == "empty":
        suite.tasks.clear()
    else:
        suite.tasks.append(suite.tasks[0].model_copy())
    calls: list[str] = []
    with pytest.raises(ValueError) as caught:
        gate.run_gate(
            suite, generate_fn=lambda prompt: calls.append(prompt) or "no",
            regression_threshold=float("nan"),
        )
    assert str(caught.value) == message
    assert calls == []


@pytest.mark.parametrize("threshold,regression", [
    (0.0, True), (0.5, True), (-0.5, True),
    (1.0, False), (-1.0, False), (2.0, False), (-2.0, False),
    (1e308, False), (-1e308, False), (10**300, False),
])
def test_finite_tolerances_keep_absolute_value_and_strict_boundary(
    suite: gate.EvalSuite, threshold: float, regression: bool,
) -> None:
    calls: list[str] = []
    result = gate.run_gate(
        suite, generate_fn=lambda prompt: calls.append(prompt) or "no",
        baseline={"control": 1.0}, regression_threshold=threshold,
    )
    row = result.task_results[0]
    assert calls == ["answer yes"]
    assert row.error is None
    assert (row.score, row.baseline, row.delta) == (0.0, 1.0, -1.0)
    assert result.regression is regression
    assert result.passed is not regression


@pytest.mark.parametrize("threshold", [-2.0, 0.0, 2.0])
def test_finite_tolerance_without_baseline_preserves_measured_zero(
    suite: gate.EvalSuite, threshold: float,
) -> None:
    result = gate.run_gate(suite, generate_fn=lambda prompt: "no", regression_threshold=threshold)
    row = result.task_results[0]
    assert (row.score, row.baseline, row.delta, row.error) == (0.0, None, None, None)
    assert result.passed
    assert not result.regression


@pytest.mark.parametrize("baseline,output,score", [(0.0, "no", 0.0), (1.0, "yes", 1.0)])
def test_equal_finite_scores_do_not_regress(
    suite: gate.EvalSuite, baseline: float, output: str, score: float,
) -> None:
    result = gate.run_gate(
        suite, generate_fn=lambda prompt: output, baseline={"control": baseline},
        regression_threshold=0.0,
    )
    row = result.task_results[0]
    assert (row.score, row.baseline, row.delta, row.error) == (score, baseline, 0.0, None)
    assert result.passed
    assert not result.regression


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_still_refuses_nonfinite_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    factory_calls: list[str] = []
    monkeypatch.setattr(
        quant_check, "make_model_generator",
        lambda model: factory_calls.append(model) or (lambda prompt: "yes"),
    )
    result = CliRunner().invoke(app, [
        "eval", "gate", "--suite", "missing.yaml", "--model", "synthetic",
        f"--regression-threshold={value}",
    ])
    assert result.exit_code == 3
    assert "regression-threshold must be between" in strip_ansi(result.output)
    assert factory_calls == []


@pytest.mark.parametrize("threshold", NONFINITE, ids=NONFINITE_IDS)
def test_config_still_refuses_nonfinite(threshold: float) -> None:
    with pytest.raises(ValidationError, match="regression_threshold"):
        EvalGateConfig(regression_threshold=threshold)
