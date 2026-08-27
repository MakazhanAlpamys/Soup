"""Tests for Eval-Gated Training (Part B of v0.26.0)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class TestEvalGateConfig:
    def test_default_disabled(self):
        from soup_cli.config.schema import EvalGateConfig

        cfg = EvalGateConfig()
        assert cfg.enabled is False

    def test_enabled_requires_suite(self):
        from soup_cli.config.schema import EvalGateConfig

        with pytest.raises(ValueError, match="suite"):
            EvalGateConfig(enabled=True, suite=None)

    def test_regression_threshold_bounds(self):
        from soup_cli.config.schema import EvalGateConfig

        with pytest.raises(ValueError):
            EvalGateConfig(suite="x.yaml", regression_threshold=-0.1)
        with pytest.raises(ValueError):
            EvalGateConfig(suite="x.yaml", regression_threshold=1.5)

    def test_every_n_epochs_bounds(self):
        from soup_cli.config.schema import EvalGateConfig

        with pytest.raises(ValueError):
            EvalGateConfig(suite="x.yaml", every_n_epochs=0)
        with pytest.raises(ValueError):
            EvalGateConfig(suite="x.yaml", every_n_epochs=101)

    def test_on_regression_literal(self):
        from soup_cli.config.schema import EvalGateConfig

        # Valid
        EvalGateConfig(suite="x.yaml", on_regression="stop")
        EvalGateConfig(suite="x.yaml", on_regression="warn")
        EvalGateConfig(suite="x.yaml", on_regression="continue")
        with pytest.raises(ValueError):
            EvalGateConfig(suite="x.yaml", on_regression="explode")


# ---------------------------------------------------------------------------
# Suite loading
# ---------------------------------------------------------------------------


class TestEvalSuiteLoading:
    def test_load_valid_suite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import load_suite

        suite_path = tmp_path / "gate.yaml"
        suite_path.write_text(
            yaml.safe_dump({
                "suite": "my-gate",
                "tasks": [
                    {"type": "custom", "name": "math_acc",
                     "tasks": "evals/math.jsonl", "scorer": "exact",
                     "threshold": 0.8},
                ],
            }),
            encoding="utf-8",
        )
        suite = load_suite(str(suite_path))
        assert suite.suite == "my-gate"
        assert len(suite.tasks) == 1
        assert suite.tasks[0].name == "math_acc"
        assert suite.tasks[0].threshold == 0.8

    def test_load_rejects_unknown_task_type(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import load_suite

        suite_path = tmp_path / "gate.yaml"
        suite_path.write_text(
            yaml.safe_dump({
                "suite": "bad",
                "tasks": [{"type": "zap", "name": "x", "threshold": 0.5}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_suite(str(suite_path))

    def test_load_rejects_path_traversal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import load_suite

        with pytest.raises(ValueError, match="outside|cwd"):
            load_suite(str(tmp_path.parent / "escape.yaml"))

    def test_load_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import load_suite

        with pytest.raises(FileNotFoundError):
            load_suite(str(tmp_path / "nope.yaml"))


# ---------------------------------------------------------------------------
# Baseline resolution
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_resolve_from_registry(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        from soup_cli.eval.gate import resolve_baseline
        from soup_cli.registry.store import RegistryStore

        store = RegistryStore()
        eid = store.push(name="baseline", tag="v1", base_model="llama",
                         task="sft", run_id=None, config={})
        store.close()

        # Simulate attached eval via ExperimentTracker — if no run, returns {}
        baseline = resolve_baseline(f"registry://{eid}")
        # No evals attached yet, so empty baseline — valid case
        assert baseline == {}

    def test_resolve_from_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import resolve_baseline

        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text('{"math_acc": 0.80, "helpfulness": 7.5}',
                                 encoding="utf-8")
        baseline = resolve_baseline(str(baseline_file))
        assert baseline["math_acc"] == 0.80

    def test_resolve_file_path_traversal_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import resolve_baseline

        outside = tmp_path.parent / "outside_baseline.json"
        outside.write_text("{}", encoding="utf-8")
        try:
            with pytest.raises(ValueError, match="outside|cwd"):
                resolve_baseline(str(outside))
        finally:
            outside.unlink(missing_ok=True)

    def test_resolve_none_returns_empty(self):
        from soup_cli.eval.gate import resolve_baseline

        assert resolve_baseline(None) == {}
        assert resolve_baseline("") == {}

    def test_resolve_missing_registry_entry_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOUP_REGISTRY_DB_PATH", str(tmp_path / "reg.db"))
        from soup_cli.eval.gate import resolve_baseline

        with pytest.raises(ValueError, match="not found|missing"):
            resolve_baseline("registry://nonexistent_entry")


# ---------------------------------------------------------------------------
# Gate execution
# ---------------------------------------------------------------------------


class TestRunGate:
    def _suite_with_custom_task(self, tmp_path):
        tasks_file = tmp_path / "tasks.jsonl"
        tasks_file.write_text(
            '{"prompt": "2+2=", "expected": "4", "scorer": "exact"}\n'
            '{"prompt": "3+3=", "expected": "6", "scorer": "exact"}\n',
            encoding="utf-8",
        )
        return tasks_file

    def test_run_gate_passing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

        tasks_file = self._suite_with_custom_task(tmp_path)
        suite = EvalSuite(
            suite="pass",
            tasks=[GateTask(
                type="custom", name="math", tasks=str(tasks_file),
                scorer="exact", threshold=0.5,
            )],
        )
        # Model that always returns correct answers
        def fake_generate(prompt: str) -> str:
            return {"2+2=": "4", "3+3=": "6"}.get(prompt, "")

        result = run_gate(suite, generate_fn=fake_generate, baseline={})
        assert result.passed is True
        assert result.task_results[0].score == 1.0

    def test_run_gate_failing_on_threshold(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

        tasks_file = self._suite_with_custom_task(tmp_path)
        suite = EvalSuite(
            suite="fail",
            tasks=[GateTask(
                type="custom", name="math", tasks=str(tasks_file),
                scorer="exact", threshold=0.9,
            )],
        )
        # Only one correct
        def fake_generate(prompt: str) -> str:
            return {"2+2=": "4"}.get(prompt, "")

        result = run_gate(suite, generate_fn=fake_generate, baseline={})
        assert result.passed is False
        assert result.task_results[0].passed is False

    def test_run_gate_regression_vs_baseline(self, tmp_path, monkeypatch):
        """Gate fails when score drops more than regression_threshold below baseline."""
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

        tasks_file = self._suite_with_custom_task(tmp_path)
        suite = EvalSuite(
            suite="regress",
            tasks=[GateTask(
                type="custom", name="math", tasks=str(tasks_file),
                scorer="exact", threshold=0.0,
            )],
        )

        def fake_generate(prompt: str) -> str:
            return {"2+2=": "4"}.get(prompt, "")  # 0.5 score

        # Baseline was 1.0, candidate 0.5 → delta -0.5, exceeds 0.05 threshold
        result = run_gate(
            suite, generate_fn=fake_generate,
            baseline={"math": 1.0}, regression_threshold=0.05,
        )
        assert result.passed is False
        assert result.regression is True

    def test_run_gate_no_regression(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import EvalSuite, GateTask, run_gate

        tasks_file = self._suite_with_custom_task(tmp_path)
        suite = EvalSuite(
            suite="ok",
            tasks=[GateTask(
                type="custom", name="math", tasks=str(tasks_file),
                scorer="exact", threshold=0.0,
            )],
        )

        def fake_generate(prompt: str) -> str:
            return {"2+2=": "4", "3+3=": "6"}.get(prompt, "")  # 1.0 score

        # Baseline was 0.95, candidate 1.0 → improvement
        result = run_gate(
            suite, generate_fn=fake_generate,
            baseline={"math": 0.95}, regression_threshold=0.05,
        )
        assert result.passed is True
        assert result.regression is False


# ---------------------------------------------------------------------------
# Callback integration
# ---------------------------------------------------------------------------


class TestCallbackIntegration:
    def test_callback_runs_gate_on_epoch_end(self, tmp_path):
        from soup_cli.eval.gate import EvalSuite, GateResult, GateTaskResult
        from soup_cli.monitoring.callback import SoupTrainerCallback
        display = MagicMock()

        cb = SoupTrainerCallback(
            display=display,
            tracker=None,
            run_id="test_run",
            eval_gate_config=MagicMock(
                enabled=True, every_n_epochs=1, on_regression="warn",
                regression_threshold=0.05, suite="dummy.yaml",
                baseline=None,
            ),
        )
        cb._gate_suite = EvalSuite(suite="s", tasks=[])
        cb._gate_generate_fn = lambda prompt: "x"

        state = MagicMock(epoch=1.0, global_step=100)
        args = MagicMock()
        control = MagicMock(should_training_stop=False)

        # Stub run_gate to return a failing result
        def fake_run_gate(suite, generate_fn, baseline, regression_threshold=0.05):
            return GateResult(
                passed=False, regression=True,
                task_results=[
                    GateTaskResult(
                        name="math", score=0.5, threshold=0.8,
                        baseline=0.9, delta=-0.4, passed=False,
                    ),
                ],
            )

        cb._gate_run_fn = fake_run_gate
        cb.on_epoch_end(args, state, control)
        # on_regression='warn' shouldn't stop
        assert control.should_training_stop is False

    def test_callback_stops_training_on_regression(self, tmp_path):
        from soup_cli.eval.gate import EvalSuite, GateResult, GateTaskResult
        from soup_cli.monitoring.callback import SoupTrainerCallback
        display = MagicMock()

        cb = SoupTrainerCallback(
            display=display,
            tracker=None,
            run_id="test_run",
            eval_gate_config=MagicMock(
                enabled=True, every_n_epochs=1, on_regression="stop",
                regression_threshold=0.05, suite="dummy.yaml",
                baseline=None,
            ),
        )
        cb._gate_suite = EvalSuite(suite="s", tasks=[])
        cb._gate_generate_fn = lambda prompt: "x"

        state = MagicMock(epoch=1.0, global_step=100)
        args = MagicMock()
        control = MagicMock(should_training_stop=False)

        def fake_run_gate(suite, generate_fn, baseline, regression_threshold=0.05):
            return GateResult(
                passed=False, regression=True,
                task_results=[GateTaskResult(
                    name="x", score=0.1, threshold=0.5,
                    baseline=1.0, delta=-0.9, passed=False,
                )],
            )

        cb._gate_run_fn = fake_run_gate
        cb.on_epoch_end(args, state, control)
        assert control.should_training_stop is True

    def test_callback_skips_gate_when_disabled(self):
        from soup_cli.monitoring.callback import SoupTrainerCallback
        display = MagicMock()
        cb = SoupTrainerCallback(
            display=display, tracker=None, run_id="test_run",
            eval_gate_config=None,
        )

        state = MagicMock(epoch=1.0, global_step=100)
        args = MagicMock()
        control = MagicMock(should_training_stop=False)
        cb.on_epoch_end(args, state, control)
        # No gate config → no side-effects, no crash
        assert control.should_training_stop is False

    def test_callback_skips_when_not_epoch_boundary(self):
        """every_n_epochs=2 → must NOT run the gate on epoch 1."""
        from soup_cli.eval.gate import EvalSuite
        from soup_cli.monitoring.callback import SoupTrainerCallback

        display = MagicMock()

        def should_not_run(*a, **k):
            raise AssertionError("gate should not have been run")

        cb = SoupTrainerCallback(
            display=display, tracker=None, run_id="test_run",
            eval_gate_config=MagicMock(
                enabled=True, every_n_epochs=2, on_regression="stop",
                regression_threshold=0.05, suite="x.yaml", baseline=None,
            ),
        )
        cb._gate_suite = EvalSuite(suite="s", tasks=[])
        cb._gate_generate_fn = lambda p: ""
        cb._gate_run_fn = should_not_run
        control = MagicMock(should_training_stop=False)
        cb.on_epoch_end(MagicMock(), MagicMock(epoch=1.0, global_step=100),
                        control)
        assert control.should_training_stop is False

    def test_callback_continue_on_regression_does_not_stop(self):
        """on_regression='continue' → regression is silent, no stop."""
        from soup_cli.eval.gate import EvalSuite, GateResult, GateTaskResult
        from soup_cli.monitoring.callback import SoupTrainerCallback

        display = MagicMock()
        cb = SoupTrainerCallback(
            display=display, tracker=None, run_id="test_run",
            eval_gate_config=MagicMock(
                enabled=True, every_n_epochs=1, on_regression="continue",
                regression_threshold=0.05, suite="x.yaml", baseline=None,
            ),
        )
        cb._gate_suite = EvalSuite(suite="s", tasks=[])
        cb._gate_generate_fn = lambda p: ""
        cb._gate_run_fn = lambda *a, **k: GateResult(
            passed=False, regression=True,
            task_results=[GateTaskResult(
                name="t", score=0.1, threshold=0.5, baseline=1.0,
                delta=-0.9, passed=False,
            )],
        )
        control = MagicMock(should_training_stop=False)
        cb.on_epoch_end(MagicMock(), MagicMock(epoch=1.0, global_step=100),
                        control)
        assert control.should_training_stop is False

    def test_callback_structured_error_triggers_stop(self):
        """Gate errors should stop training under on_regression='stop'."""
        from soup_cli.eval.gate import EvalSuite
        from soup_cli.monitoring.callback import SoupTrainerCallback

        display = MagicMock()
        cb = SoupTrainerCallback(
            display=display, tracker=None, run_id="test_run",
            eval_gate_config=MagicMock(
                enabled=True, every_n_epochs=1, on_regression="stop",
                regression_threshold=0.05, suite="x.yaml", baseline=None,
            ),
        )
        cb._gate_suite = EvalSuite(suite="s", tasks=[])
        cb._gate_generate_fn = lambda p: ""

        def raising_run(*a, **k):
            raise FileNotFoundError("missing tasks file")

        cb._gate_run_fn = raising_run
        control = MagicMock(should_training_stop=False)
        cb.on_epoch_end(MagicMock(), MagicMock(epoch=1.0, global_step=100),
                        control)
        assert control.should_training_stop is True


# ---------------------------------------------------------------------------
# CLI: soup eval gate
# ---------------------------------------------------------------------------


class TestEvalGateCLI:
    def test_eval_gate_requires_suite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["eval", "gate"])
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_eval_gate_missing_suite_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app, ["eval", "gate", "--suite", "nope.yaml"],
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))

    def test_eval_gate_invalid_regression_threshold(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "gate.yaml"
        suite.write_text(yaml.safe_dump({"suite": "s", "tasks": []}),
                         encoding="utf-8")
        result = runner.invoke(
            app, ["eval", "gate", "--suite", str(suite),
                  "--regression-threshold", "2.5"],
        )
        assert result.exit_code != 0, (result.output, repr(result.exception))


# ---------------------------------------------------------------------------
# Train --gate integration
# ---------------------------------------------------------------------------


class TestTrainGateFlag:
    def test_train_gate_flag_accepted(self, tmp_path, monkeypatch):
        """The --gate flag must be visible in `soup train --help`."""
        import re

        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        # Strip ANSI escape sequences — macOS CI runners emit color codes
        # that split "--gate" into non-contiguous characters (see commit msg).
        cleaned = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--gate" in cleaned
        assert "eval-gated" in cleaned
class TestJudgeModelValidation:
    def test_rejects_localhost_prefix_bypass(self):
        from soup_cli.eval.gate import GateTask

        with pytest.raises(ValueError, match="judge_model"):
            GateTask(
                type="judge",
                name="judge",
                threshold=0.5,
                prompts="prompts.jsonl",
                judge_model="http://localhost.attacker.com/model",
            )


# ---------------------------------------------------------------------------
# Judge task score normalization
# ---------------------------------------------------------------------------


class TestJudgeTaskNormalization:
    """Regression tests for judge score normalization against active rubric scale."""

    @pytest.fixture
    def prompts_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        prompts = tmp_path / "prompts.jsonl"
        prompts.write_text('{"prompt": "test prompt"}\n', encoding="utf-8")
        return prompts

    def _run_with_mock_judge(self, prompts_file, overall_score, rubric=None):
        from soup_cli.eval.gate import GateTask, _run_judge_task
        from soup_cli.eval.judge import DEFAULT_RUBRIC, JudgeResults

        task = GateTask(
            type="judge",
            name="quality",
            threshold=0.7,
            prompts=str(prompts_file.name),
            judge_model="ollama://llama3",
        )

        mock_evaluator = MagicMock()
        mock_evaluator.rubric = rubric if rubric is not None else DEFAULT_RUBRIC
        mock_evaluator.evaluate_batch.return_value = JudgeResults(
            scores=[], overall_score=overall_score,
        )

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("soup_cli.eval.judge.JudgeEvaluator", lambda **kw: mock_evaluator)
            return _run_judge_task(task, generate_fn=lambda p: "test response")

    def test_perfect_score_default_rubric_normalizes_to_one(self, prompts_file):
        """A 5.0 score on the default 1-5 rubric must normalize to 1.0, not 0.50."""
        score = self._run_with_mock_judge(prompts_file, overall_score=5.0)
        assert score == pytest.approx(1.0)

    def test_midpoint_score_default_rubric(self, prompts_file):
        """A 3.0 score on the default 1-5 rubric must normalize to 0.50, not 0.30."""
        score = self._run_with_mock_judge(prompts_file, overall_score=3.0)
        assert score == pytest.approx(0.50)

    def test_minimum_score_default_rubric(self, prompts_file):
        """A 1.0 score on the default 1-5 rubric must normalize to 0.0, not 0.10."""
        score = self._run_with_mock_judge(prompts_file, overall_score=1.0)
        assert score == pytest.approx(0.0)

    def test_intermediate_score_default_rubric(self, prompts_file):
        """A 4.0 score on the default 1-5 rubric must normalize to 0.75, not 0.40."""
        score = self._run_with_mock_judge(prompts_file, overall_score=4.0)
        assert score == pytest.approx(0.75)

    def test_custom_rubric_scale_dynamic_normalization(self, prompts_file):
        """A 1-10 custom rubric must normalize dynamically: (8-1)/9 = 7/9."""
        custom_rubric = {"scale": {"min": 1, "max": 10}}
        score = self._run_with_mock_judge(
            prompts_file, overall_score=8.0, rubric=custom_rubric,
        )
        assert score == pytest.approx(7.0 / 9.0)

        score_max = self._run_with_mock_judge(
            prompts_file, overall_score=10.0, rubric=custom_rubric,
        )
        assert score_max == pytest.approx(1.0)

        score_min = self._run_with_mock_judge(
            prompts_file, overall_score=1.0, rubric=custom_rubric,
        )
        assert score_min == pytest.approx(0.0)

    def test_clamping_out_of_bounds_scores(self, prompts_file):
        """Pathological scores outside [min, max] must clamp strictly to [0.0, 1.0]."""
        # Score above max
        score_high = self._run_with_mock_judge(prompts_file, overall_score=7.0)
        assert score_high == pytest.approx(1.0)

        # Score below min
        score_low = self._run_with_mock_judge(prompts_file, overall_score=0.0)
        assert score_low == pytest.approx(0.0)

    def test_degenerate_scale_handled_safely(self, prompts_file):
        """If min == max, division by zero is avoided and score is safely clamped."""
        degenerate_rubric = {"scale": {"min": 5, "max": 5}}
        score = self._run_with_mock_judge(
            prompts_file, overall_score=5.0, rubric=degenerate_rubric,
        )
        assert 0.0 <= score <= 1.0

    def test_end_to_end_gate_passes_for_perfect_judge_score(self, tmp_path, monkeypatch):
        """End-to-end: a perfect 5/5 score must pass a threshold of 0.70."""
        monkeypatch.chdir(tmp_path)
        from soup_cli.eval.gate import EvalSuite, GateTask, run_gate
        from soup_cli.eval.judge import DEFAULT_RUBRIC, JudgeResults

        prompts = tmp_path / "eval_prompts.jsonl"
        prompts.write_text('{"prompt": "Summarize this article"}\n', encoding="utf-8")

        suite = EvalSuite(
            suite="judge_quality_suite",
            tasks=[
                GateTask(
                    type="judge",
                    name="quality_check",
                    threshold=0.70,
                    prompts=str(prompts.name),
                    judge_model="ollama://llama3.1",
                ),
            ],
        )

        mock_evaluator = MagicMock()
        mock_evaluator.rubric = DEFAULT_RUBRIC
        # Perfect 5/5 score from judge
        mock_evaluator.evaluate_batch.return_value = JudgeResults(
            scores=[], overall_score=5.0,
        )

        monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator", lambda **kw: mock_evaluator)

        result = run_gate(suite, generate_fn=lambda prompt: "Summary output")
        assert len(result.task_results) == 1
        task_res = result.task_results[0]
        # Under the old buggy implementation, score was 0.50 and failed threshold 0.70!
        assert task_res.score == pytest.approx(1.0)
        assert task_res.passed is True
        assert result.passed is True
