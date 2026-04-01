"""Tests for the eval platform: custom eval, judge, human eval, leaderboard, compare."""

import json
import re
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.eval.custom import (
    MAX_EVAL_TASKS,
    MAX_REGEX_INPUT_LEN,
    MAX_REGEX_PATTERN_LEN,
    EvalResult,
    EvalResults,
    EvalTask,
    load_eval_tasks,
    score_contains,
    score_exact,
    score_regex,
    score_semantic,
    score_task,
)
from soup_cli.eval.human import (
    ELO_DEFAULT,
    MAX_PROMPTS,
    HumanEvalResults,
    HumanJudgment,
    _expected_score,
    load_prompts,
    load_results,
    run_human_eval_session,
    save_results,
)
from soup_cli.eval.judge import (
    DEFAULT_RUBRIC,
    JudgeEvaluator,
    JudgeResults,
    JudgeScore,
    _build_judge_prompt,
    _compute_weighted_score,
    _parse_judge_response,
    load_rubric,
    validate_judge_api_base,
)
from soup_cli.eval.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    build_leaderboard_from_tracker,
    compare_runs,
    export_leaderboard,
)

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from Rich-formatted output."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


# ═══════════════════════════════════════════════════════════
# Custom Eval — Scoring Functions
# ═══════════════════════════════════════════════════════════


class TestScoreExact:
    def test_match(self):
        assert score_exact("Hello", "hello") is True

    def test_no_match(self):
        assert score_exact("Hello", "World") is False

    def test_whitespace(self):
        assert score_exact("  hello  ", "hello") is True

    def test_empty(self):
        assert score_exact("", "") is True

    def test_case_insensitive(self):
        assert score_exact("ANSWER", "answer") is True


class TestScoreContains:
    def test_contains(self):
        assert score_contains("The answer is 42.", "42") is True

    def test_not_contains(self):
        assert score_contains("The answer is 42.", "99") is False

    def test_case_insensitive(self):
        assert score_contains("Hello World", "hello") is True

    def test_full_match(self):
        assert score_contains("exact", "exact") is True


class TestScoreRegex:
    def test_match(self):
        assert score_regex("The answer is 42", r"\d+") is True

    def test_no_match(self):
        assert score_regex("no numbers here", r"\d+") is False

    def test_invalid_regex(self):
        assert score_regex("test", r"[invalid") is False

    def test_case_insensitive(self):
        assert score_regex("Hello World", r"hello") is True


class TestScoreSemantic:
    def test_identical(self):
        score = score_semantic("hello world", "hello world")
        assert score == 1.0

    def test_partial_overlap(self):
        score = score_semantic("hello world foo", "hello world bar")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        score = score_semantic("foo bar", "baz qux")
        assert score == 0.0

    def test_empty_output(self):
        score = score_semantic("", "hello")
        assert score == 0.0

    def test_empty_expected(self):
        score = score_semantic("hello", "")
        assert score == 0.0


# ═══════════════════════════════════════════════════════════
# Custom Eval — Task Loading
# ═══════════════════════════════════════════════════════════


class TestLoadEvalTasks:
    def test_valid_jsonl(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text(
            '{"prompt": "What is 2+2?", "expected": "4", "category": "math"}\n'
            '{"prompt": "Capital of France?", "expected": "Paris"}\n',
            encoding="utf-8",
        )
        tasks = load_eval_tasks(path)
        assert len(tasks) == 2
        assert tasks[0].prompt == "What is 2+2?"
        assert tasks[0].expected == "4"
        assert tasks[0].category == "math"
        assert tasks[1].category == "default"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_eval_tasks(tmp_path / "nonexistent.jsonl")

    def test_wrong_extension(self, tmp_path):
        path = tmp_path / "tasks.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected .jsonl"):
            load_eval_tasks(path)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_eval_tasks(path)

    def test_missing_prompt(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text('{"expected": "4"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required field"):
            load_eval_tasks(path)

    def test_invalid_scoring(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text(
            '{"prompt": "q", "expected": "a", "scoring": "invalid"}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid scoring"):
            load_eval_tasks(path)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No eval tasks"):
            load_eval_tasks(path)

    def test_max_tasks_exceeded(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        lines = '{"prompt": "q"}\n' * (MAX_EVAL_TASKS + 1)
        path.write_text(lines, encoding="utf-8")
        with pytest.raises(ValueError, match="exceeds maximum"):
            load_eval_tasks(path)

    def test_non_object_json(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object"):
            load_eval_tasks(path)

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text(
            '{"prompt": "q1", "expected": "a1"}\n'
            "\n"
            '{"prompt": "q2", "expected": "a2"}\n',
            encoding="utf-8",
        )
        tasks = load_eval_tasks(path)
        assert len(tasks) == 2

    def test_scoring_types(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        lines = []
        for scoring in ["exact", "contains", "regex", "semantic"]:
            lines.append(json.dumps({
                "prompt": "q", "expected": "a", "scoring": scoring,
            }))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tasks = load_eval_tasks(path)
        assert len(tasks) == 4
        assert tasks[0].scoring == "exact"
        assert tasks[3].scoring == "semantic"

    def test_metadata_preserved(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text(
            '{"prompt": "q", "expected": "a", "difficulty": "hard"}\n',
            encoding="utf-8",
        )
        tasks = load_eval_tasks(path)
        assert tasks[0].metadata == {"difficulty": "hard"}


# ═══════════════════════════════════════════════════════════
# Custom Eval — score_task
# ═══════════════════════════════════════════════════════════


class TestScoreTask:
    def test_exact_match(self):
        task = EvalTask(prompt="q", expected="answer", scoring="exact")
        result = score_task(task, "answer")
        assert result.matched is True
        assert result.score == 1.0

    def test_exact_no_match(self):
        task = EvalTask(prompt="q", expected="answer", scoring="exact")
        result = score_task(task, "wrong")
        assert result.matched is False
        assert result.score == 0.0

    def test_contains_match(self):
        task = EvalTask(prompt="q", expected="42", scoring="contains")
        result = score_task(task, "The answer is 42")
        assert result.matched is True

    def test_regex_match(self):
        task = EvalTask(prompt="q", expected=r"\d+", scoring="regex")
        result = score_task(task, "42")
        assert result.matched is True

    def test_semantic_match(self):
        task = EvalTask(
            prompt="q", expected="hello world", scoring="semantic",
        )
        result = score_task(task, "hello world test")
        assert result.matched is True
        assert 0.0 < result.score <= 1.0


# ═══════════════════════════════════════════════════════════
# Custom Eval — EvalResults
# ═══════════════════════════════════════════════════════════


class TestEvalResults:
    def test_compute_basic(self):
        results = EvalResults(results=[
            EvalResult(
                task=EvalTask(prompt="q1", expected="a", category="math"),
                output="a", score=1.0, matched=True,
            ),
            EvalResult(
                task=EvalTask(prompt="q2", expected="b", category="math"),
                output="x", score=0.0, matched=False,
            ),
            EvalResult(
                task=EvalTask(prompt="q3", expected="c", category="code"),
                output="c", score=1.0, matched=True,
            ),
        ])
        results.compute()
        assert results.total == 3
        assert results.correct == 2
        assert abs(results.accuracy - 2 / 3) < 1e-6
        assert results.category_scores["math"]["total"] == 2
        assert results.category_scores["math"]["correct"] == 1
        assert results.category_scores["code"]["accuracy"] == 1.0

    def test_empty_results(self):
        results = EvalResults(results=[])
        results.compute()
        assert results.total == 0
        assert results.accuracy == 0.0


# ═══════════════════════════════════════════════════════════
# Custom Eval — run_eval with mock generator
# ═══════════════════════════════════════════════════════════


class TestRunEval:
    def test_run_eval_with_mock_generator(self):
        from soup_cli.eval.custom import run_eval

        tasks = [
            EvalTask(prompt="What is 2+2?", expected="4", scoring="exact"),
            EvalTask(prompt="Capital?", expected="Paris", scoring="contains"),
        ]

        def mock_gen(prompt: str) -> str:
            if "2+2" in prompt:
                return "4"
            return "The capital is Paris"

        results = run_eval("dummy_path", tasks, generate_fn=mock_gen)
        assert results.total == 2
        assert results.correct == 2
        assert results.accuracy == 1.0

    def test_run_eval_empty_tasks(self):
        from soup_cli.eval.custom import run_eval

        results = run_eval("dummy", tasks=[], generate_fn=lambda p: "")
        assert results.total == 0
        assert results.accuracy == 0.0

    def test_run_eval_partial_match(self):
        from soup_cli.eval.custom import run_eval

        tasks = [
            EvalTask(prompt="q1", expected="yes", scoring="exact"),
            EvalTask(prompt="q2", expected="no", scoring="exact"),
        ]
        results = run_eval(
            "dummy", tasks, generate_fn=lambda p: "yes",
        )
        assert results.total == 2
        assert results.correct == 1


# ═══════════════════════════════════════════════════════════
# Human Eval — Edge cases
# ═══════════════════════════════════════════════════════════


class TestHumanEvalEdgeCases:
    def test_invalid_winner_treated_as_tie(self):
        """Invalid winner value falls through to tie (else branch)."""
        results = HumanEvalResults()
        results.judgments.append(HumanJudgment(
            prompt="q", response_a="a", response_b="b",
            model_a="m1", model_b="m2", winner="invalid",
        ))
        results.compute_ratings()
        # Both models should be near default (tie behavior)
        assert abs(results.ratings["m1"].rating - ELO_DEFAULT) < 1e-6
        assert results.ratings["m1"].ties == 1

    def test_load_results_malformed_json(self, tmp_path):
        """Corrupted results file raises JSONDecodeError."""
        path = tmp_path / "bad.json"
        path.write_text("not valid json{{{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_results(path)


# ═══════════════════════════════════════════════════════════
# Leaderboard — Edge cases
# ═══════════════════════════════════════════════════════════


class TestLeaderboardEdgeCases:
    def test_export_unknown_format_defaults_to_json(self):
        """Unknown format falls through to JSON export."""
        lb = Leaderboard(entries=[
            LeaderboardEntry(model_path="m1", benchmark="mmlu", score=0.8),
        ])
        lb.compute()
        output = export_leaderboard(lb, fmt="xml")
        # Should still be valid JSON
        data = json.loads(output)
        assert len(data) == 1

    def test_ipv6_loopback_http_ok(self):
        """::1 is a valid loopback address for HTTP."""
        validate_judge_api_base("http://[::1]:8000")


# ═══════════════════════════════════════════════════════════
# Judge — API Base Validation (SSRF)
# ═══════════════════════════════════════════════════════════


class TestValidateJudgeApiBase:
    def test_none_is_ok(self):
        validate_judge_api_base(None)

    def test_https_ok(self):
        validate_judge_api_base("https://api.openai.com")

    def test_localhost_http_ok(self):
        validate_judge_api_base("http://localhost:8000")

    def test_127_http_ok(self):
        validate_judge_api_base("http://127.0.0.1:8000")

    def test_remote_http_blocked(self):
        with pytest.raises(ValueError, match="HTTPS"):
            validate_judge_api_base("http://evil.com")

    def test_0000_http_blocked(self):
        """0.0.0.0 is a bind address, not a safe loopback for clients."""
        with pytest.raises(ValueError, match="HTTPS"):
            validate_judge_api_base("http://0.0.0.0:8000")

    def test_ftp_blocked(self):
        with pytest.raises(ValueError, match="Invalid scheme"):
            validate_judge_api_base("ftp://example.com")

    def test_file_blocked(self):
        with pytest.raises(ValueError, match="Invalid scheme"):
            validate_judge_api_base("file:///etc/passwd")


# ═══════════════════════════════════════════════════════════
# Judge — Rubric Loading
# ═══════════════════════════════════════════════════════════


class TestLoadRubric:
    def test_valid_rubric(self, tmp_path):
        rubric_path = tmp_path / "rubric.yaml"
        rubric_path.write_text(
            "criteria:\n"
            "  - name: quality\n"
            "    description: response quality\n"
            "    weight: 1.0\n"
            "scale:\n"
            "  min: 1\n"
            "  max: 10\n",
            encoding="utf-8",
        )
        rubric = load_rubric(rubric_path)
        assert len(rubric["criteria"]) == 1
        assert rubric["criteria"][0]["name"] == "quality"
        assert rubric["scale"]["max"] == 10

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_rubric(tmp_path / "nonexistent.yaml")

    def test_missing_criteria(self, tmp_path):
        rubric_path = tmp_path / "rubric.yaml"
        rubric_path.write_text("scale:\n  min: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="criteria"):
            load_rubric(rubric_path)

    def test_empty_criteria(self, tmp_path):
        rubric_path = tmp_path / "rubric.yaml"
        rubric_path.write_text("criteria: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty"):
            load_rubric(rubric_path)

    def test_invalid_criterion(self, tmp_path):
        rubric_path = tmp_path / "rubric.yaml"
        rubric_path.write_text(
            "criteria:\n  - name: test\n", encoding="utf-8",
        )
        with pytest.raises(ValueError, match="description"):
            load_rubric(rubric_path)

    def test_non_mapping(self, tmp_path):
        rubric_path = tmp_path / "rubric.yaml"
        rubric_path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_rubric(rubric_path)


# ═══════════════════════════════════════════════════════════
# Judge — Prompt Building & Response Parsing
# ═══════════════════════════════════════════════════════════


class TestJudgePromptAndParsing:
    def test_build_prompt(self):
        prompt = _build_judge_prompt("question", "answer", DEFAULT_RUBRIC)
        assert "question" in prompt
        assert "answer" in prompt
        assert "helpfulness" in prompt
        assert "JSON" in prompt

    def test_parse_valid_response(self):
        response = json.dumps({
            "scores": {"helpfulness": 4, "accuracy": 5, "safety": 3},
            "reasoning": "Good response.",
        })
        scores, reasoning = _parse_judge_response(response, DEFAULT_RUBRIC)
        assert scores["helpfulness"] == 4.0
        assert scores["accuracy"] == 5.0
        assert reasoning == "Good response."

    def test_parse_clamps_scores(self):
        response = json.dumps({
            "scores": {"helpfulness": 10, "accuracy": -1, "safety": 3},
            "reasoning": "Test",
        })
        scores, _ = _parse_judge_response(response, DEFAULT_RUBRIC)
        assert scores["helpfulness"] == 5.0  # clamped to max
        assert scores["accuracy"] == 1.0  # clamped to min

    def test_parse_no_json(self):
        with pytest.raises(ValueError, match="No JSON"):
            _parse_judge_response("no json here", DEFAULT_RUBRIC)

    def test_parse_invalid_json(self):
        with pytest.raises(ValueError, match="No JSON"):
            _parse_judge_response("not {valid json", DEFAULT_RUBRIC)

    def test_compute_weighted_score(self):
        scores = {"helpfulness": 4.0, "accuracy": 5.0, "safety": 3.0}
        result = _compute_weighted_score(scores, DEFAULT_RUBRIC)
        assert abs(result - 4.0) < 1e-6  # (4+5+3)/3 = 4.0

    def test_compute_weighted_custom_weights(self):
        rubric = {
            "criteria": [
                {"name": "a", "description": "x", "weight": 2.0},
                {"name": "b", "description": "y", "weight": 1.0},
            ],
        }
        scores = {"a": 5.0, "b": 2.0}
        result = _compute_weighted_score(scores, rubric)
        expected = (5.0 * 2.0 + 2.0 * 1.0) / 3.0
        assert abs(result - expected) < 1e-6


# ═══════════════════════════════════════════════════════════
# Judge — JudgeEvaluator
# ═══════════════════════════════════════════════════════════


class TestJudgeEvaluator:
    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Invalid provider"):
            JudgeEvaluator(provider="bad")

    def test_ssrf_blocked(self):
        with pytest.raises(ValueError):
            JudgeEvaluator(api_base="ftp://evil.com")

    def test_valid_init(self):
        evaluator = JudgeEvaluator(
            provider="openai", model="gpt-4o-mini",
        )
        assert evaluator.provider == "openai"
        assert evaluator.model == "gpt-4o-mini"

    def test_evaluate_with_mock(self):
        evaluator = JudgeEvaluator(provider="openai")
        mock_response = json.dumps({
            "scores": {"helpfulness": 4, "accuracy": 5, "safety": 4},
            "reasoning": "Solid response.",
        })
        with patch.object(evaluator, "_call_llm", return_value=mock_response):
            score = evaluator.evaluate("test prompt", "test response")
            assert score.weighted_score > 0
            assert score.reasoning == "Solid response."

    def test_evaluate_batch_with_mock(self):
        evaluator = JudgeEvaluator(provider="openai")
        mock_response = json.dumps({
            "scores": {"helpfulness": 3, "accuracy": 4, "safety": 5},
            "reasoning": "Ok.",
        })
        items = [
            {"prompt": "q1", "response": "a1", "category": "cat1"},
            {"prompt": "q2", "response": "a2"},
        ]
        with patch.object(evaluator, "_call_llm", return_value=mock_response):
            results = evaluator.evaluate_batch(items)
            assert len(results.scores) == 2
            results.compute()
            assert results.overall_score > 0
            assert "cat1" in results.category_scores


# ═══════════════════════════════════════════════════════════
# Judge — JudgeResults
# ═══════════════════════════════════════════════════════════


class TestJudgeResults:
    def test_compute_empty(self):
        results = JudgeResults(scores=[])
        results.compute()
        assert results.overall_score == 0.0

    def test_compute_with_scores(self):
        scores = [
            JudgeScore(
                prompt="q1", response="a1",
                scores={"h": 4, "a": 5}, weighted_score=4.5,
                category="cat1",
            ),
            JudgeScore(
                prompt="q2", response="a2",
                scores={"h": 3, "a": 4}, weighted_score=3.5,
                category="cat2",
            ),
        ]
        results = JudgeResults(scores=scores)
        results.compute()
        assert abs(results.overall_score - 4.0) < 1e-6
        assert "cat1" in results.category_scores
        assert "h" in results.criteria_averages


# ═══════════════════════════════════════════════════════════
# Human Eval — Elo Rating
# ═══════════════════════════════════════════════════════════


class TestEloRating:
    def test_expected_score_equal(self):
        assert abs(_expected_score(1500, 1500) - 0.5) < 1e-6

    def test_expected_score_higher(self):
        assert _expected_score(1700, 1500) > 0.5

    def test_expected_score_lower(self):
        assert _expected_score(1300, 1500) < 0.5

    def test_elo_update_winner_a(self):
        results = HumanEvalResults()
        results.judgments.append(HumanJudgment(
            prompt="q", response_a="a", response_b="b",
            model_a="m1", model_b="m2", winner="a",
        ))
        results.compute_ratings()
        assert results.ratings["m1"].rating > ELO_DEFAULT
        assert results.ratings["m2"].rating < ELO_DEFAULT
        assert results.ratings["m1"].wins == 1
        assert results.ratings["m2"].losses == 1

    def test_elo_update_winner_b(self):
        results = HumanEvalResults()
        results.judgments.append(HumanJudgment(
            prompt="q", response_a="a", response_b="b",
            model_a="m1", model_b="m2", winner="b",
        ))
        results.compute_ratings()
        assert results.ratings["m1"].rating < ELO_DEFAULT
        assert results.ratings["m2"].rating > ELO_DEFAULT

    def test_elo_update_tie(self):
        results = HumanEvalResults()
        results.judgments.append(HumanJudgment(
            prompt="q", response_a="a", response_b="b",
            model_a="m1", model_b="m2", winner="tie",
        ))
        results.compute_ratings()
        assert abs(results.ratings["m1"].rating - ELO_DEFAULT) < 1e-6
        assert results.ratings["m1"].ties == 1

    def test_multiple_judgments(self):
        results = HumanEvalResults()
        for _ in range(3):
            results.judgments.append(HumanJudgment(
                prompt="q", response_a="a", response_b="b",
                model_a="m1", model_b="m2", winner="a",
            ))
        results.compute_ratings()
        assert results.ratings["m1"].wins == 3
        assert results.ratings["m1"].rating > ELO_DEFAULT + 30

    def test_to_dict(self):
        results = HumanEvalResults()
        results.judgments.append(HumanJudgment(
            prompt="q", response_a="a", response_b="b",
            model_a="m1", model_b="m2", winner="a",
        ))
        results.compute_ratings()
        data = results.to_dict()
        assert len(data["judgments"]) == 1
        assert "m1" in data["ratings"] or "m2" in data["ratings"]


# ═══════════════════════════════════════════════════════════
# Human Eval — Load/Save
# ═══════════════════════════════════════════════════════════


class TestHumanEvalIO:
    def test_load_prompts(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text(
            '{"prompt": "Hello"}\n'
            '{"prompt": "World", "category": "test"}\n',
            encoding="utf-8",
        )
        prompts = load_prompts(path)
        assert len(prompts) == 2

    def test_load_prompts_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prompts(tmp_path / "nope.jsonl")

    def test_load_prompts_invalid_json(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_prompts(path)

    def test_load_prompts_missing_field(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"text": "hello"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required field"):
            load_prompts(path)

    def test_save_and_load_results(self, tmp_path):
        results = HumanEvalResults()
        results.judgments.append(HumanJudgment(
            prompt="q", response_a="a", response_b="b",
            model_a="m1", model_b="m2", winner="a",
        ))
        results.compute_ratings()

        path = tmp_path / "results.json"
        save_results(results, path)

        loaded = load_results(path)
        assert len(loaded.judgments) == 1
        assert loaded.judgments[0].winner == "a"
        assert loaded.ratings["m1"].rating > ELO_DEFAULT

    def test_load_results_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nope.json")


class TestRunHumanEvalSession:
    def test_with_precomputed_responses(self):
        prompts = [{"prompt": "q1"}, {"prompt": "q2"}]
        results = run_human_eval_session(
            prompts=prompts,
            model_a_name="m1",
            model_b_name="m2",
            responses_a=["resp_a1", "resp_a2"],
            responses_b=["resp_b1", "resp_b2"],
        )
        assert len(results.judgments) == 2
        assert results.judgments[0].response_a == "resp_a1"


# ═══════════════════════════════════════════════════════════
# Leaderboard
# ═══════════════════════════════════════════════════════════


class TestLeaderboard:
    def test_compute(self):
        lb = Leaderboard(entries=[
            LeaderboardEntry(model_path="m1", benchmark="mmlu", score=0.8),
            LeaderboardEntry(model_path="m1", benchmark="gsm8k", score=0.6),
            LeaderboardEntry(model_path="m2", benchmark="mmlu", score=0.9),
        ])
        lb.compute()
        assert len(lb.models) == 2
        assert lb.models["m1"]["mmlu"] == 0.8
        assert lb.models["m2"]["mmlu"] == 0.9

    def test_get_sorted_models_by_avg(self):
        lb = Leaderboard(entries=[
            LeaderboardEntry(model_path="m1", benchmark="mmlu", score=0.8),
            LeaderboardEntry(model_path="m2", benchmark="mmlu", score=0.9),
        ])
        lb.compute()
        sorted_models = lb.get_sorted_models()
        assert sorted_models[0][0] == "m2"  # higher score first

    def test_get_sorted_by_benchmark(self):
        lb = Leaderboard(entries=[
            LeaderboardEntry(model_path="m1", benchmark="mmlu", score=0.9),
            LeaderboardEntry(model_path="m1", benchmark="gsm8k", score=0.3),
            LeaderboardEntry(model_path="m2", benchmark="mmlu", score=0.7),
            LeaderboardEntry(model_path="m2", benchmark="gsm8k", score=0.8),
        ])
        lb.compute()
        # Sort by gsm8k: m2 should be first
        sorted_models = lb.get_sorted_models(sort_by="gsm8k")
        assert sorted_models[0][0] == "m2"

    def test_export_json(self):
        lb = Leaderboard(entries=[
            LeaderboardEntry(model_path="m1", benchmark="mmlu", score=0.8),
        ])
        lb.compute()
        output = export_leaderboard(lb, fmt="json")
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["model"] == "m1"

    def test_export_csv(self):
        lb = Leaderboard(entries=[
            LeaderboardEntry(model_path="m1", benchmark="mmlu", score=0.8),
            LeaderboardEntry(model_path="m1", benchmark="gsm8k", score=0.6),
        ])
        lb.compute()
        output = export_leaderboard(lb, fmt="csv")
        lines = output.strip().split("\n")
        assert len(lines) == 2  # header + 1 model
        assert "mmlu" in lines[0]
        assert "gsm8k" in lines[0]

    def test_empty_leaderboard(self):
        lb = Leaderboard()
        lb.compute()
        assert lb.get_sorted_models() == []


# ═══════════════════════════════════════════════════════════
# Leaderboard — Compare Runs
# ═══════════════════════════════════════════════════════════


class TestCompareRuns:
    def test_compare_basic(self):
        tracker = MagicMock()
        tracker.get_eval_results.side_effect = [
            [{"benchmark": "mmlu", "score": 0.7}],
            [{"benchmark": "mmlu", "score": 0.8}],
        ]
        result = compare_runs(tracker, "run1", "run2")
        assert len(result["comparisons"]) == 1
        assert result["comparisons"][0]["delta"] == pytest.approx(0.1)
        assert result["has_regressions"] is False

    def test_compare_regression(self):
        tracker = MagicMock()
        tracker.get_eval_results.side_effect = [
            [{"benchmark": "mmlu", "score": 0.9}],
            [{"benchmark": "mmlu", "score": 0.7}],
        ]
        result = compare_runs(tracker, "run1", "run2")
        assert result["has_regressions"] is True
        assert "mmlu" in result["regressions"]

    def test_compare_different_benchmarks(self):
        tracker = MagicMock()
        tracker.get_eval_results.side_effect = [
            [{"benchmark": "mmlu", "score": 0.7}],
            [{"benchmark": "gsm8k", "score": 0.8}],
        ]
        result = compare_runs(tracker, "run1", "run2")
        assert len(result["comparisons"]) == 2

    def test_build_leaderboard_from_tracker(self):
        tracker = MagicMock()
        tracker.get_eval_results.return_value = [
            {"model_path": "m1", "benchmark": "mmlu", "score": 0.8,
             "run_id": "r1", "created_at": "2026-01-01"},
        ]
        lb = build_leaderboard_from_tracker(tracker)
        assert len(lb.entries) == 1
        assert lb.models["m1"]["mmlu"] == 0.8


# ═══════════════════════════════════════════════════════════
# Config — EvalConfig
# ═══════════════════════════════════════════════════════════


class TestEvalConfig:
    def test_eval_config_default(self):
        from soup_cli.config.schema import EvalConfig
        config = EvalConfig()
        assert config.auto_eval is False
        assert config.benchmarks is None
        assert config.custom_tasks is None
        assert config.judge is None

    def test_eval_config_with_values(self):
        from soup_cli.config.schema import EvalConfig
        config = EvalConfig(
            auto_eval=True,
            benchmarks=["mmlu", "gsm8k"],
            custom_tasks="eval.jsonl",
            judge={"model": "gpt-4o-mini", "provider": "openai"},
        )
        assert config.auto_eval is True
        assert config.benchmarks == ["mmlu", "gsm8k"]

    def test_soup_config_with_eval(self):
        from soup_cli.config.schema import SoupConfig
        config = SoupConfig(
            base="test-model",
            data={"train": "data.jsonl"},
            eval={"auto_eval": True, "benchmarks": ["mmlu"]},
        )
        assert config.eval is not None
        assert config.eval.auto_eval is True
        assert config.eval.benchmarks == ["mmlu"]

    def test_soup_config_without_eval(self):
        from soup_cli.config.schema import SoupConfig
        config = SoupConfig(
            base="test-model",
            data={"train": "data.jsonl"},
        )
        assert config.eval is None


# ═══════════════════════════════════════════════════════════
# Callback — Auto-eval hook
# ═══════════════════════════════════════════════════════════


class TestCallbackAutoEval:
    def test_auto_eval_not_called_without_config(self):
        from soup_cli.monitoring.callback import SoupTrainerCallback
        display = MagicMock()
        callback = SoupTrainerCallback(display=display)
        callback._run_auto_eval()  # Should be a no-op

    def test_auto_eval_not_called_when_disabled(self):
        from soup_cli.config.schema import EvalConfig
        from soup_cli.monitoring.callback import SoupTrainerCallback
        display = MagicMock()
        eval_config = EvalConfig(auto_eval=False)
        callback = SoupTrainerCallback(
            display=display, eval_config=eval_config,
        )
        callback._run_auto_eval()  # Should be a no-op

    def test_auto_eval_called_when_enabled(self):
        from soup_cli.config.schema import EvalConfig
        from soup_cli.monitoring.callback import SoupTrainerCallback
        display = MagicMock()
        eval_config = EvalConfig(
            auto_eval=True, benchmarks=["mmlu"],
        )
        callback = SoupTrainerCallback(
            display=display,
            eval_config=eval_config,
            output_dir="/tmp/model",
            run_id="test_run",
        )
        with patch("soup_cli.commands.eval.benchmark") as mock_bench:
            callback._run_auto_eval()
            mock_bench.assert_called_once()


# ═══════════════════════════════════════════════════════════
# CLI — eval subcommands
# ═══════════════════════════════════════════════════════════


class TestEvalCLI:
    def test_eval_help(self):
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower()
        assert "custom" in result.output.lower()
        assert "judge" in result.output.lower()
        assert "leaderboard" in result.output.lower()

    def test_eval_benchmark_help(self):
        result = runner.invoke(app, ["eval", "benchmark", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()
        assert "benchmarks" in result.output.lower()

    def test_eval_custom_help(self):
        result = runner.invoke(app, ["eval", "custom", "--help"])
        assert result.exit_code == 0
        assert "tasks" in result.output.lower()

    def test_eval_judge_help(self):
        result = runner.invoke(app, ["eval", "judge", "--help"])
        assert result.exit_code == 0
        assert "target" in result.output.lower()
        assert "provider" in result.output.lower()

    def test_eval_compare_help(self):
        result = runner.invoke(app, ["eval", "compare", "--help"])
        assert result.exit_code == 0

    def test_eval_leaderboard_help(self):
        result = runner.invoke(app, ["eval", "leaderboard", "--help"])
        assert result.exit_code == 0

    def test_eval_human_help(self):
        result = runner.invoke(app, ["eval", "human", "--help"])
        assert result.exit_code == 0
        clean = _strip_ansi(result.output).lower()
        assert "model-a" in clean

    def test_eval_auto_help(self):
        result = runner.invoke(app, ["eval", "auto", "--help"])
        assert result.exit_code == 0
        assert "config" in result.output.lower()

    def test_eval_benchmark_missing_model(self):
        result = runner.invoke(
            app, ["eval", "benchmark", "--model", "nonexistent_path"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_eval_custom_missing_model(self, tmp_path):
        tasks_path = tmp_path / "tasks.jsonl"
        tasks_path.write_text(
            '{"prompt": "q", "expected": "a"}\n', encoding="utf-8",
        )
        result = runner.invoke(
            app, [
                "eval", "custom",
                "--tasks", str(tasks_path),
                "--model", "nonexistent_path",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_eval_judge_missing_target(self):
        result = runner.invoke(
            app, ["eval", "judge", "--target", "nonexistent.jsonl"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_eval_leaderboard_empty(self, tmp_path):
        """Leaderboard with no results should show a message."""
        import os
        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            result = runner.invoke(app, ["eval", "leaderboard"])
            assert result.exit_code == 1
            assert "no eval results" in result.output.lower()

    def test_eval_compare_run_not_found(self, tmp_path):
        """Compare with nonexistent run IDs."""
        import os
        db_path = tmp_path / "test.db"
        with patch.dict(os.environ, {"SOUP_DB_PATH": str(db_path)}):
            result = runner.invoke(
                app, ["eval", "compare", "nonexistent1", "nonexistent2"],
            )
            assert result.exit_code == 1
            assert "not found" in result.output.lower()


# ═══════════════════════════════════════════════════════════
# Security — SSRF, input validation
# ═══════════════════════════════════════════════════════════


class TestSecurity:
    def test_custom_eval_schema_validation(self, tmp_path):
        """Ensure custom eval rejects invalid schemas."""
        path = tmp_path / "bad.jsonl"
        path.write_text("42\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object"):
            load_eval_tasks(path)

    def test_custom_eval_task_cap(self, tmp_path):
        """Ensure eval tasks are capped at MAX_EVAL_TASKS."""
        path = tmp_path / "tasks.jsonl"
        lines = '{"prompt": "q"}\n' * (MAX_EVAL_TASKS + 1)
        path.write_text(lines, encoding="utf-8")
        with pytest.raises(ValueError, match="exceeds maximum"):
            load_eval_tasks(path)

    def test_judge_ssrf_remote_http(self):
        """Remote HTTP is blocked for judge API."""
        with pytest.raises(ValueError, match="HTTPS"):
            validate_judge_api_base("http://malicious-server.com/api")

    def test_judge_ssrf_file_protocol(self):
        """file:// protocol blocked."""
        with pytest.raises(ValueError):
            validate_judge_api_base("file:///etc/passwd")

    def test_leaderboard_read_only(self):
        """Leaderboard queries are read-only (no user input in SQL)."""
        tracker = MagicMock()
        tracker.get_eval_results.return_value = []
        lb = build_leaderboard_from_tracker(tracker)
        assert len(lb.entries) == 0
        # Verify only get_eval_results was called (read-only)
        tracker.get_eval_results.assert_called_once()

    def test_regex_redos_pattern_length_guard(self):
        """Long regex patterns are rejected to prevent ReDoS."""
        long_pattern = "a" * (MAX_REGEX_PATTERN_LEN + 1)
        assert score_regex("test input", long_pattern) is False

    def test_regex_redos_input_truncation(self):
        """Long inputs are truncated before regex matching."""
        # Pattern that matches only at the end
        long_input = "a" * (MAX_REGEX_INPUT_LEN + 100) + "FIND_ME"
        # FIND_ME is beyond the truncation limit, so should not match
        assert score_regex(long_input, "FIND_ME") is False
        # But within the limit, it should match
        short_input = "a" * 100 + "FIND_ME"
        assert score_regex(short_input, "FIND_ME") is True

    def test_judge_api_key_not_leaked_to_ollama(self):
        """OpenAI API key should NOT be auto-loaded for non-openai providers."""
        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret-key"}):
            evaluator = JudgeEvaluator(provider="ollama")
            assert evaluator.api_key == ""

    def test_judge_api_key_loaded_for_openai(self):
        """OpenAI API key SHOULD be auto-loaded for openai provider."""
        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret-key"}):
            evaluator = JudgeEvaluator(provider="openai")
            assert evaluator.api_key == "sk-secret-key"

    def test_judge_api_key_not_leaked_to_server(self):
        """OpenAI API key should NOT be auto-loaded for server provider."""
        import os
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-secret-key"}):
            evaluator = JudgeEvaluator(
                provider="server",
                api_base="http://localhost:8000",
            )
            assert evaluator.api_key == ""

    def test_human_prompts_cap(self, tmp_path):
        """Human eval prompts are capped at MAX_PROMPTS."""
        path = tmp_path / "prompts.jsonl"
        lines = '{"prompt": "q"}\n' * (MAX_PROMPTS + 1)
        path.write_text(lines, encoding="utf-8")
        with pytest.raises(ValueError, match="exceeds maximum"):
            load_prompts(path)
