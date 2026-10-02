"""Tests for Issue #1433: expect_chosen_preferred_over_rejected_by_judge requires a judge.

Without a judge configured, the expectation must not silently pass every row with 1.0.
Supports --judge on CLI, args: {judge: ...} in suite YAML, and explicit advisory: True opt-in.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.expectations import (
    build_pairwise_judge_fn,
    expect_chosen_preferred_over_rejected_by_judge,
    parse_suite_yaml,
)


def _plain(text: str) -> str:
    import re

    cleaned = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    return " ".join(cleaned.split())


def _write_prefs(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "prefs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _write_suite(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "suite.yaml"
    p.write_text(content, encoding="utf-8")
    return p


SAMPLE_ROWS = [
    {"prompt": "What is 2+2?", "chosen": "banana", "rejected": "4"},
    {"prompt": "Capital of France?", "chosen": "", "rejected": "Paris"},
]

CLEAN_ROWS = [
    {"prompt": "What is 2+2?", "chosen": "4", "rejected": "banana"},
    {"prompt": "Capital of France?", "chosen": "Paris", "rejected": "London"},
]


class TestExpectJudgeDirectFunction:
    def test_missing_judge_fails_by_default(self) -> None:
        result = expect_chosen_preferred_over_rejected_by_judge(SAMPLE_ROWS)
        assert result.passed is False
        assert result.num_rows_checked == 2
        assert result.num_violations == 2
        assert any("no judge configured" in d for d in result.details)

    def test_advisory_mode_passes(self) -> None:
        result = expect_chosen_preferred_over_rejected_by_judge(SAMPLE_ROWS, advisory=True)
        assert result.passed is True
        assert result.num_rows_checked == 2
        assert result.num_violations == 0

    def test_callable_judge_invoked(self) -> None:
        calls = []

        def counting_judge(row: dict) -> float:
            calls.append(row["prompt"])
            return 1.0

        result = expect_chosen_preferred_over_rejected_by_judge(
            CLEAN_ROWS, judge_fn=counting_judge, threshold=0.7
        )
        assert result.passed is True
        assert len(calls) == 2
        assert calls == ["What is 2+2?", "Capital of France?"]

    def test_callable_judge_below_threshold_fails(self) -> None:
        result = expect_chosen_preferred_over_rejected_by_judge(
            CLEAN_ROWS, judge_fn=lambda r: 0.0, threshold=0.7
        )
        assert result.passed is False
        assert result.num_violations == 2
        assert any("< 0.700" in d for d in result.details)

    def test_invalid_advisory_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="advisory must be bool"):
            expect_chosen_preferred_over_rejected_by_judge(CLEAN_ROWS, advisory="yes")  # type: ignore[arg-type]


class TestExpectSuiteParse:
    def test_suite_with_judge_arg_parses(self) -> None:
        yaml_text = (
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args:\n"
            "      judge: ollama://llama3.1\n"
            "      threshold: 0.8\n"
        )
        spec = parse_suite_yaml(yaml_text)
        assert spec.expectations[0].args["judge"] == "ollama://llama3.1"
        assert spec.expectations[0].args["threshold"] == 0.8

    def test_suite_with_invalid_judge_scheme_refused_at_parse(self) -> None:
        yaml_text = (
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args:\n"
            "      judge: ftp://unsupported.host/model\n"
        )
        with pytest.raises(ValueError, match="unsupported scheme"):
            parse_suite_yaml(yaml_text)

    def test_suite_with_non_string_judge_refused_at_parse(self) -> None:
        yaml_text = (
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args:\n"
            "      judge: 12345\n"
        )
        with pytest.raises(TypeError, match="judge must be a string"):
            parse_suite_yaml(yaml_text)

    def test_suite_with_empty_judge_refused_at_parse(self) -> None:
        yaml_text = (
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args:\n"
            "      judge: '   '\n"
        )
        with pytest.raises(ValueError, match="judge must be non-empty"):
            parse_suite_yaml(yaml_text)

    def test_suite_with_advisory_opt_in_parses(self) -> None:
        yaml_text = (
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args:\n"
            "      advisory: true\n"
        )
        spec = parse_suite_yaml(yaml_text)
        assert spec.expectations[0].args["advisory"] is True


class TestBuildPairwiseJudgeFn:
    def test_callable_passes_through(self) -> None:
        def stub_judge(row: dict) -> float:
            return 0.8

        built = build_pairwise_judge_fn(stub_judge)
        assert built({"prompt": "p"}) == 0.8

    def test_pairwise_judge_evaluator_dispatch(self) -> None:
        calls = []

        class StubEvaluator:
            def compare_pair(self, prompt: str, resp_a: str, resp_b: str) -> int:
                calls.append((prompt, resp_a, resp_b))
                # When ans_a is resp_a, ans_a (position 0) wins.
                # When ans_a is resp_b (swapped), ans_a (position 1) wins.
                return 0 if resp_a == "ans_a" else 1

        judge_fn = build_pairwise_judge_fn(StubEvaluator())
        score = judge_fn({"prompt": "q", "chosen": "ans_a", "rejected": "ans_b"})
        assert score == 1.0
        assert len(calls) == 2  # swap=True checks both (A, B) and (B, A)

    def test_pairwise_judge_loss(self) -> None:
        class RejectWinsEvaluator:
            def compare_pair(self, prompt: str, resp_a: str, resp_b: str) -> int:
                # ans_b (rejected) always wins
                return 0 if resp_a == "ans_b" else 1

        judge_fn = build_pairwise_judge_fn(RejectWinsEvaluator())
        score = judge_fn({"prompt": "q", "chosen": "ans_a", "rejected": "ans_b"})
        assert score == 0.0

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="judge must be a URL string"):
            build_pairwise_judge_fn(12345)


class TestCliExpectJudge:
    def test_unconfigured_judge_exits_2_with_violation_details(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data = _write_prefs(tmp_path, SAMPLE_ROWS)
        suite = _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args: {threshold: 1.0}\n",
        )
        result = CliRunner().invoke(app, ["expect", str(data), str(suite)])
        out = _plain(result.output)
        assert result.exit_code == 2
        assert "FAIL" in out
        assert "no judge configured" in out

    def test_cli_advisory_mode_exits_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data = _write_prefs(tmp_path, SAMPLE_ROWS)
        suite = _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args: {threshold: 1.0, advisory: true}\n",
        )
        result = CliRunner().invoke(app, ["expect", str(data), str(suite)])
        out = _plain(result.output)
        assert result.exit_code == 0
        assert "PASS" in out

    def test_cli_flag_judge_scores_and_exits_0_on_clean_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data = _write_prefs(tmp_path, CLEAN_ROWS)
        suite = _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args: {threshold: 0.7}\n",
        )
        calls = []

        class PerfectJudge:
            def compare_pair(self, prompt: str, resp_a: str, resp_b: str) -> int:
                calls.append((prompt, resp_a, resp_b))
                return 0 if resp_a in ("4", "Paris") else 1

        with mock.patch("soup_cli.utils.expectations.build_pairwise_judge_fn") as mock_build:
            mock_build.return_value = build_pairwise_judge_fn(PerfectJudge())
            result = CliRunner().invoke(
                app, ["expect", str(data), str(suite), "--judge", "ollama://llama3.1"]
            )
        out = _plain(result.output)
        assert result.exit_code == 0
        assert "PASS" in out
        assert len(calls) == 4  # 2 rows * 2 calls each for swap-debiasing

    def test_cli_flag_judge_fails_when_judge_prefers_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data = _write_prefs(tmp_path, SAMPLE_ROWS)
        suite = _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args: {threshold: 0.7}\n",
        )

        class HarshJudge:
            def compare_pair(self, prompt: str, resp_a: str, resp_b: str) -> int:
                return 1 if resp_a in ("banana", "") else 0

        with mock.patch("soup_cli.utils.expectations.build_pairwise_judge_fn") as mock_build:
            mock_build.return_value = build_pairwise_judge_fn(HarshJudge())
            result = CliRunner().invoke(
                app, ["expect", str(data), str(suite), "--judge", "ollama://llama3.1"]
            )
        out = _plain(result.output)
        assert result.exit_code == 2
        assert "FAIL" in out

    def test_cli_invalid_judge_url_exits_3_before_running_suite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data = _write_prefs(tmp_path, SAMPLE_ROWS)
        suite = _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n",
        )
        result = CliRunner().invoke(
            app, ["expect", str(data), str(suite), "--judge", "ftp://bad-url"]
        )
        assert result.exit_code == 3
        assert "unsupported scheme" in _plain(result.output)

    def test_suite_judge_arg_is_used_at_run_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_prefs(tmp_path, CLEAN_ROWS)
        _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args: {judge: 'ollama://llama3.1', threshold: 0.7}\n",
        )
        calls = []

        def fake_compare(self, prompt, resp_a, resp_b):
            calls.append((self.provider, self.model))
            return 0 if resp_a in ("4", "Paris") else 1

        monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator.compare_pair", fake_compare)
        result = CliRunner().invoke(app, ["expect", "prefs.jsonl", "suite.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert calls == [("ollama", "llama3.1")] * 4  # 2 rows, both orders

    def test_advisory_pass_says_no_judge_ran(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_prefs(tmp_path, CLEAN_ROWS)
        _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args: {advisory: true}\n",
        )
        result = CliRunner().invoke(app, ["expect", "prefs.jsonl", "suite.yaml"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "advisory" in " ".join(_plain(result.output).split()).lower()

    @pytest.mark.parametrize("value", ["'false'", "'no'", "1"])
    def test_non_bool_advisory_refused_at_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_prefs(tmp_path, CLEAN_ROWS)
        _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            f"    args: {{advisory: {value}}}\n",
        )
        result = CliRunner().invoke(app, ["expect", "prefs.jsonl", "suite.yaml"])
        assert result.exit_code == 3, (result.output, repr(result.exception))
        assert "advisory must be a boolean" in " ".join(_plain(result.output).split())

    def test_cli_judge_takes_precedence_over_suite_judge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_prefs(tmp_path, CLEAN_ROWS)
        _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args: {judge: 'ollama://suite-model', threshold: 0.7}\n",
        )
        calls = []

        def fake_compare(self, prompt, resp_a, resp_b):
            calls.append((self.provider, self.model))
            return 0 if resp_a in ("4", "Paris") else 1

        monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator.compare_pair", fake_compare)
        result = CliRunner().invoke(
            app,
            ["expect", "prefs.jsonl", "suite.yaml", "--judge", "ollama://cli-model"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert calls == [("ollama", "cli-model")] * 4

    def test_unreachable_judge_exits_1_without_evaluating_remaining_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_prefs(tmp_path, CLEAN_ROWS)
        _write_suite(
            tmp_path,
            "expectations:\n"
            "  - name: expect_chosen_preferred_over_rejected_by_judge\n"
            "    args: {judge: 'ollama://llama3.1', threshold: 0.7}\n",
        )
        from soup_cli.eval.judge import JudgeUnavailableError

        call_count = 0

        def dead_judge(self, prompt, resp_a, resp_b):
            nonlocal call_count
            call_count += 1
            raise JudgeUnavailableError("connection refused", url="http://localhost:11434")

        monkeypatch.setattr("soup_cli.eval.judge.JudgeEvaluator.compare_pair", dead_judge)
        result = CliRunner().invoke(app, ["expect", "prefs.jsonl", "suite.yaml"])
        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert call_count == 1
        assert "judge unavailable" in _plain(result.output).lower()
