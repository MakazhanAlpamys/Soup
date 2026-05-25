"""v0.69.0 Part B — Expectations suite for chat data."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import expectations


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Allowlist + immutability
# -----------------------------------------------------------------------------


class TestSupportedExpectations:
    def test_exact(self) -> None:
        assert expectations.SUPPORTED_EXPECTATIONS == frozenset(
            {
                "expect_no_pii",
                "expect_token_length_between",
                "expect_no_refusal_pattern",
                "expect_chosen_preferred_over_rejected_by_judge",
            }
        )

    def test_is_frozenset(self) -> None:
        assert isinstance(expectations.SUPPORTED_EXPECTATIONS, frozenset)

    def test_immutable(self) -> None:
        with pytest.raises(AttributeError):
            expectations.SUPPORTED_EXPECTATIONS.add(  # type: ignore[attr-defined]
                "evil"
            )


# -----------------------------------------------------------------------------
# validate_expectation_name
# -----------------------------------------------------------------------------


class TestValidateExpectationName:
    def test_happy(self) -> None:
        assert (
            expectations.validate_expectation_name("expect_no_pii") == "expect_no_pii"
        )

    def test_case_insensitive(self) -> None:
        assert (
            expectations.validate_expectation_name("EXPECT_NO_PII") == "expect_no_pii"
        )

    def test_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown expectation"):
            expectations.validate_expectation_name("expect_perfection")

    def test_non_string(self) -> None:
        with pytest.raises(TypeError):
            expectations.validate_expectation_name(42)

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            expectations.validate_expectation_name(True)

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            expectations.validate_expectation_name("")

    def test_null_byte(self) -> None:
        with pytest.raises(ValueError, match="null"):
            expectations.validate_expectation_name("expect\x00_no_pii")

    def test_oversize(self) -> None:
        with pytest.raises(ValueError):
            expectations.validate_expectation_name("x" * 200)


# -----------------------------------------------------------------------------
# ExpectationResult frozen
# -----------------------------------------------------------------------------


class TestExpectationResult:
    def test_happy(self) -> None:
        r = expectations.ExpectationResult(
            name="expect_no_pii",
            passed=True,
            num_rows_checked=10,
            num_violations=0,
            details=(),
        )
        assert r.passed is True

    def test_frozen(self) -> None:
        r = expectations.ExpectationResult(
            name="expect_no_pii",
            passed=True,
            num_rows_checked=10,
            num_violations=0,
            details=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.passed = False  # type: ignore[misc]

    def test_details_must_be_tuple(self) -> None:
        with pytest.raises(TypeError):
            expectations.ExpectationResult(
                name="expect_no_pii",
                passed=True,
                num_rows_checked=10,
                num_violations=0,
                details=["x"],  # type: ignore[arg-type]
            )

    def test_violations_non_negative(self) -> None:
        with pytest.raises(ValueError):
            expectations.ExpectationResult(
                name="expect_no_pii",
                passed=True,
                num_rows_checked=10,
                num_violations=-1,
                details=(),
            )

    def test_invalid_name(self) -> None:
        with pytest.raises(ValueError):
            expectations.ExpectationResult(
                name="bogus",
                passed=True,
                num_rows_checked=0,
                num_violations=0,
                details=(),
            )


# -----------------------------------------------------------------------------
# expect_no_pii
# -----------------------------------------------------------------------------


class TestExpectNoPii:
    def test_clean(self) -> None:
        rows = [
            {"text": "Hello, how are you today?"},
            {"text": "This is a normal sentence."},
        ]
        result = expectations.expect_no_pii(rows)
        assert result.passed is True
        assert result.num_violations == 0
        assert result.num_rows_checked == 2

    def test_email_flagged(self) -> None:
        rows = [{"text": "Email me at evil@example.com please."}]
        result = expectations.expect_no_pii(rows)
        assert result.passed is False
        assert result.num_violations >= 1

    def test_phone_flagged(self) -> None:
        rows = [{"text": "Call me at 555-123-4567 anytime."}]
        result = expectations.expect_no_pii(rows)
        assert result.passed is False

    def test_empty_rows(self) -> None:
        result = expectations.expect_no_pii([])
        assert result.passed is True
        assert result.num_rows_checked == 0

    def test_messages_field(self) -> None:
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello, my email is evil@e.com"},
                ]
            }
        ]
        result = expectations.expect_no_pii(rows)
        assert result.passed is False

    def test_non_list_raises(self) -> None:
        with pytest.raises(TypeError):
            expectations.expect_no_pii("rows")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# expect_token_length_between
# -----------------------------------------------------------------------------


class TestExpectTokenLengthBetween:
    def test_clean(self) -> None:
        rows = [{"text": "a " * 50}, {"text": "b " * 60}]
        result = expectations.expect_token_length_between(rows, min_tokens=10, max_tokens=200)
        assert result.passed is True

    def test_below_min(self) -> None:
        rows = [{"text": "short"}]
        result = expectations.expect_token_length_between(rows, min_tokens=50, max_tokens=200)
        assert result.passed is False
        assert result.num_violations == 1

    def test_above_max(self) -> None:
        rows = [{"text": "a " * 500}]
        result = expectations.expect_token_length_between(rows, min_tokens=10, max_tokens=50)
        assert result.passed is False
        assert result.num_violations == 1

    def test_invalid_min(self) -> None:
        with pytest.raises(ValueError):
            expectations.expect_token_length_between([], min_tokens=-1, max_tokens=100)

    def test_invalid_max(self) -> None:
        with pytest.raises(ValueError):
            expectations.expect_token_length_between([], min_tokens=10, max_tokens=0)

    def test_min_greater_than_max(self) -> None:
        with pytest.raises(ValueError):
            expectations.expect_token_length_between([], min_tokens=200, max_tokens=10)

    def test_bool_min(self) -> None:
        with pytest.raises(TypeError):
            expectations.expect_token_length_between([], min_tokens=True, max_tokens=100)

    def test_bool_max(self) -> None:
        with pytest.raises(TypeError):
            expectations.expect_token_length_between([], min_tokens=1, max_tokens=False)


# -----------------------------------------------------------------------------
# expect_no_refusal_pattern
# -----------------------------------------------------------------------------


class TestExpectNoRefusalPattern:
    def test_clean(self) -> None:
        rows = [{"output": "Here is the recipe you asked for."}]
        result = expectations.expect_no_refusal_pattern(rows)
        assert result.passed is True

    def test_refusal_detected(self) -> None:
        rows = [{"output": "I cannot help with that request."}]
        result = expectations.expect_no_refusal_pattern(rows)
        assert result.passed is False

    def test_assistant_refusal_in_messages(self) -> None:
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "How do I do X?"},
                    {"role": "assistant", "content": "Sorry, but I refuse to discuss that."},
                ]
            }
        ]
        result = expectations.expect_no_refusal_pattern(rows)
        assert result.passed is False

    def test_empty(self) -> None:
        result = expectations.expect_no_refusal_pattern([])
        assert result.passed is True


# -----------------------------------------------------------------------------
# expect_chosen_preferred_over_rejected_by_judge
# -----------------------------------------------------------------------------


class TestChosenPreferredByJudge:
    def test_clean(self) -> None:
        rows = [
            {"prompt": "Q", "chosen": "good", "rejected": "bad"},
            {"prompt": "Q2", "chosen": "great", "rejected": "terrible"},
        ]

        def judge(row: dict) -> float:
            # Chosen always wins; score in [0, 1].
            return 0.9

        result = expectations.expect_chosen_preferred_over_rejected_by_judge(
            rows, judge_fn=judge, threshold=0.7
        )
        assert result.passed is True
        assert result.num_violations == 0

    def test_violations(self) -> None:
        rows = [{"prompt": "Q", "chosen": "good", "rejected": "bad"}]

        def judge(row: dict) -> float:
            return 0.3  # below threshold

        result = expectations.expect_chosen_preferred_over_rejected_by_judge(
            rows, judge_fn=judge, threshold=0.7
        )
        assert result.passed is False
        assert result.num_violations == 1

    def test_missing_chosen_rejected_fields(self) -> None:
        rows = [{"prompt": "Q"}]

        def judge(row: dict) -> float:
            return 0.9

        # Rows missing chosen+rejected are flagged as violations.
        result = expectations.expect_chosen_preferred_over_rejected_by_judge(
            rows, judge_fn=judge, threshold=0.7
        )
        assert result.passed is False

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            expectations.expect_chosen_preferred_over_rejected_by_judge(
                [], judge_fn=lambda r: 0.9, threshold=1.5
            )

    def test_non_callable_judge(self) -> None:
        with pytest.raises(TypeError):
            expectations.expect_chosen_preferred_over_rejected_by_judge(
                [], judge_fn="not callable", threshold=0.5  # type: ignore[arg-type]
            )

    def test_bool_threshold(self) -> None:
        with pytest.raises(TypeError):
            expectations.expect_chosen_preferred_over_rejected_by_judge(
                [], judge_fn=lambda r: 0.5, threshold=True
            )

    def test_nan_threshold(self) -> None:
        with pytest.raises(ValueError):
            expectations.expect_chosen_preferred_over_rejected_by_judge(
                [], judge_fn=lambda r: 0.5, threshold=float("nan")
            )

    def test_judge_exception_treated_as_violation(self) -> None:
        rows = [{"prompt": "Q", "chosen": "c", "rejected": "r"}]

        def bad_judge(row: dict) -> float:
            raise RuntimeError("kaboom")

        result = expectations.expect_chosen_preferred_over_rejected_by_judge(
            rows, judge_fn=bad_judge, threshold=0.5
        )
        assert result.passed is False
        assert result.num_violations == 1


# -----------------------------------------------------------------------------
# Suite spec + runner
# -----------------------------------------------------------------------------


class TestSuiteSpec:
    def test_happy(self) -> None:
        spec = expectations.parse_suite_spec(
            {
                "expectations": [
                    {"name": "expect_no_pii"},
                    {
                        "name": "expect_token_length_between",
                        "args": {"min_tokens": 10, "max_tokens": 1000},
                    },
                ]
            }
        )
        assert len(spec.expectations) == 2

    def test_non_dict(self) -> None:
        with pytest.raises(TypeError):
            expectations.parse_suite_spec(["nope"])  # type: ignore[arg-type]

    def test_missing_expectations(self) -> None:
        with pytest.raises(ValueError):
            expectations.parse_suite_spec({})

    def test_empty_expectations(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            expectations.parse_suite_spec({"expectations": []})

    def test_unknown_expectation(self) -> None:
        with pytest.raises(ValueError):
            expectations.parse_suite_spec(
                {"expectations": [{"name": "expect_perfection"}]}
            )

    def test_args_must_be_dict(self) -> None:
        with pytest.raises(TypeError):
            expectations.parse_suite_spec(
                {
                    "expectations": [
                        {"name": "expect_no_pii", "args": "not a dict"}
                    ]
                }
            )

    def test_too_many_expectations(self) -> None:
        raw = {
            "expectations": [
                {"name": "expect_no_pii"} for _ in range(expectations._MAX_SUITE_LEN + 1)
            ]
        }
        with pytest.raises(ValueError, match="exceeds"):
            expectations.parse_suite_spec(raw)


class TestRunSuite:
    def test_all_pass(self) -> None:
        spec = expectations.parse_suite_spec(
            {"expectations": [{"name": "expect_no_pii"}]}
        )
        rows = [{"text": "all clean here"}]
        report = expectations.run_suite(rows, spec)
        assert report.passed is True
        assert len(report.results) == 1

    def test_one_fails(self) -> None:
        spec = expectations.parse_suite_spec(
            {"expectations": [{"name": "expect_no_pii"}]}
        )
        rows = [{"text": "email me at a@b.com"}]
        report = expectations.run_suite(rows, spec)
        assert report.passed is False

    def test_token_length_args(self) -> None:
        spec = expectations.parse_suite_spec(
            {
                "expectations": [
                    {
                        "name": "expect_token_length_between",
                        "args": {"min_tokens": 10, "max_tokens": 1000},
                    }
                ]
            }
        )
        rows = [{"text": "a " * 500}]
        report = expectations.run_suite(rows, spec)
        assert report.passed is True

    def test_chosen_preferred_arg(self) -> None:
        spec = expectations.parse_suite_spec(
            {
                "expectations": [
                    {
                        "name": "expect_chosen_preferred_over_rejected_by_judge",
                        "args": {"threshold": 0.5},
                    }
                ]
            }
        )
        rows = [{"prompt": "Q", "chosen": "good", "rejected": "bad"}]
        # No judge fn supplied = neutral pass (operator must inject for live).
        report = expectations.run_suite(rows, spec)
        # Default judge returns 1.0 (no judge → assume chosen wins).
        assert report.passed is True

    def test_report_frozen(self) -> None:
        spec = expectations.parse_suite_spec(
            {"expectations": [{"name": "expect_no_pii"}]}
        )
        report = expectations.run_suite([], spec)
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.passed = False  # type: ignore[misc]


# -----------------------------------------------------------------------------
# YAML loaders
# -----------------------------------------------------------------------------


class TestLoadSuiteYaml:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        path = _write(
            tmp_path / "suite.yaml",
            "expectations:\n  - name: expect_no_pii\n",
        )
        spec = expectations.load_suite_yaml(str(path))
        assert len(spec.expectations) == 1

    def test_outside_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(outside / "s.yaml", "expectations:\n  - name: expect_no_pii\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError, match="cwd"):
            expectations.load_suite_yaml(str(outside / "s.yaml"))

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_symlink_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        target = _write(tmp_path / "real.yaml", "expectations:\n  - name: expect_no_pii\n")
        link = tmp_path / "link.yaml"
        os.symlink(str(target), str(link))
        with pytest.raises(ValueError, match="symlink"):
            expectations.load_suite_yaml(str(link))

    def test_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            expectations.load_suite_yaml(str(tmp_path / "nope.yaml"))


# -----------------------------------------------------------------------------
# CLI: `soup expect`
# -----------------------------------------------------------------------------


class TestSoupExpectCli:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["expect", "--help"])
        assert result.exit_code == 0, result.output
        assert "expect" in result.output.lower()

    def test_all_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        data = _write(tmp_path / "data.jsonl", '{"text": "clean output here"}\n')
        suite = _write(
            tmp_path / "suite.yaml", "expectations:\n  - name: expect_no_pii\n"
        )
        runner = CliRunner()
        result = runner.invoke(app, ["expect", str(data), str(suite)])
        assert result.exit_code == 0, result.output

    def test_failure_exits_3(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        data = _write(
            tmp_path / "data.jsonl", '{"text": "email me at evil@e.com"}\n'
        )
        suite = _write(
            tmp_path / "suite.yaml", "expectations:\n  - name: expect_no_pii\n"
        )
        runner = CliRunner()
        result = runner.invoke(app, ["expect", str(data), str(suite)])
        assert result.exit_code == 3

    def test_outside_cwd_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(outside / "d.jsonl", '{"text":"x"}\n')
        sub = tmp_path / "sub"
        sub.mkdir()
        suite = _write(sub / "suite.yaml", "expectations:\n  - name: expect_no_pii\n")
        monkeypatch.chdir(sub)
        runner = CliRunner()
        result = runner.invoke(
            app, ["expect", str(outside / "d.jsonl"), str(suite)]
        )
        assert result.exit_code != 0


# -----------------------------------------------------------------------------
# Source wiring
# -----------------------------------------------------------------------------


class TestSourceWiring:
    def test_cli_registers_expect(self) -> None:
        root = Path(__file__).resolve().parent.parent
        cli = (root / "soup_cli" / "cli.py").read_text(encoding="utf-8")
        assert 'name="expect"' in cli or "from soup_cli.commands import expect" in cli

    def test_no_heavy_top_level_imports(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = (root / "soup_cli" / "utils" / "expectations.py").read_text(encoding="utf-8")
        for forbidden in (
            "\nimport torch",
            "\nimport transformers",
            "\nimport peft",
            "\nimport yaml",
        ):
            assert forbidden not in src

    def test_version_bumped(self) -> None:
        from soup_cli import __version__

        assert __version__ == "0.69.0"
