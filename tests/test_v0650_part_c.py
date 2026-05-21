"""v0.65.0 Part C — Capability auto-suite tests.

Pre-bundled MMLU-Pro / GPQA / BBEH / AIME / MATH-500 / HumanEval+ /
SWE-bench-Verified entries with sensible lm-eval-harness task ids.
Profile selector ``full | fast | math | code``.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from soup_cli.utils.capability_suite import (
    CAPABILITY_BENCHMARKS,
    PROFILES,
    CapabilityBenchmark,
    get_benchmark_spec,
    list_benchmarks,
    list_suites,
    resolve_suite,
    validate_benchmark_name,
    validate_suite_name,
)


class TestBenchmarks:
    def test_known_benchmarks_present(self):
        for name in ("mmlu-pro", "gpqa", "bbeh", "aime",
                     "math-500", "humaneval-plus", "swe-bench-verified"):
            assert name in CAPABILITY_BENCHMARKS

    def test_frozenset_immutable(self):
        assert isinstance(CAPABILITY_BENCHMARKS, frozenset)
        with pytest.raises(AttributeError):
            CAPABILITY_BENCHMARKS.add("evil")  # type: ignore[attr-defined]


class TestValidateBenchmarkName:
    def test_happy(self):
        assert validate_benchmark_name("mmlu-pro") == "mmlu-pro"

    def test_case_insensitive(self):
        assert validate_benchmark_name("MMLU-Pro") == "mmlu-pro"

    def test_unknown(self):
        with pytest.raises(ValueError, match="unknown"):
            validate_benchmark_name("not-real")

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_benchmark_name("")

    def test_null_byte(self):
        with pytest.raises(ValueError, match="null"):
            validate_benchmark_name("mmlu\x00")

    def test_non_string(self):
        with pytest.raises(TypeError):
            validate_benchmark_name(42)  # type: ignore[arg-type]

    def test_bool(self):
        with pytest.raises(TypeError):
            validate_benchmark_name(True)  # type: ignore[arg-type]

    def test_oversize(self):
        with pytest.raises(ValueError, match="long"):
            validate_benchmark_name("a" * 65)


class TestGetBenchmarkSpec:
    def test_known(self):
        spec = get_benchmark_spec("mmlu-pro")
        assert isinstance(spec, CapabilityBenchmark)
        assert spec.name == "mmlu-pro"
        assert spec.lm_eval_task

    def test_unknown(self):
        with pytest.raises(KeyError):
            get_benchmark_spec("not-real")

    def test_frozen(self):
        spec = get_benchmark_spec("mmlu-pro")
        with pytest.raises(Exception):
            spec.name = "x"  # type: ignore[misc]

    def test_list_returns_sorted(self):
        names = list_benchmarks()
        assert list(names) == sorted(names)


class TestSuites:
    def test_profiles_immutable(self):
        with pytest.raises(TypeError):
            PROFILES["evil"] = ("x",)  # type: ignore[index]

    def test_full_includes_all(self):
        full = set(resolve_suite("full"))
        all_benchmarks = {get_benchmark_spec(n) for n in CAPABILITY_BENCHMARKS}
        assert full == all_benchmarks

    def test_fast_is_subset(self):
        fast = set(b.name for b in resolve_suite("fast"))
        assert len(fast) >= 2
        assert fast.issubset(CAPABILITY_BENCHMARKS)

    def test_math_profile(self):
        math_suite = [b.name for b in resolve_suite("math")]
        # Should include AIME and/or MATH-500.
        assert "aime" in math_suite or "math-500" in math_suite

    def test_code_profile(self):
        code_suite = [b.name for b in resolve_suite("code")]
        assert "humaneval-plus" in code_suite or "swe-bench-verified" in code_suite

    def test_list_suites_sorted(self):
        names = list_suites()
        assert list(names) == sorted(names)
        assert "full" in names


class TestValidateSuiteName:
    def test_happy(self):
        assert validate_suite_name("fast") == "fast"

    def test_case_insensitive(self):
        assert validate_suite_name("FAST") == "fast"

    def test_unknown(self):
        with pytest.raises(ValueError, match="unknown"):
            validate_suite_name("evil")

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_suite_name("")

    def test_null_byte(self):
        with pytest.raises(ValueError, match="null"):
            validate_suite_name("fast\x00")

    def test_bool(self):
        with pytest.raises(TypeError):
            validate_suite_name(True)  # type: ignore[arg-type]

    def test_non_string(self):
        with pytest.raises(TypeError):
            validate_suite_name(42)  # type: ignore[arg-type]

    def test_oversize(self):
        with pytest.raises(ValueError, match="long"):
            validate_suite_name("a" * 33)


class TestResolveSuite:
    def test_returns_tuple_of_benchmarks(self):
        result = resolve_suite("fast")
        assert isinstance(result, tuple)
        for b in result:
            assert isinstance(b, CapabilityBenchmark)

    def test_unknown_suite(self):
        with pytest.raises(ValueError):
            resolve_suite("not-real")

    def test_full_distinct(self):
        result = resolve_suite("full")
        names = [b.name for b in result]
        assert len(names) == len(set(names))


class TestCapabilityBenchmark:
    def test_frozen(self):
        b = CapabilityBenchmark(
            name="mmlu-pro", lm_eval_task="mmlu_pro",
            category="knowledge", default_fewshot=5,
        )
        with pytest.raises(Exception):
            b.name = "x"  # type: ignore[misc]

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="name"):
            CapabilityBenchmark(
                name="", lm_eval_task="x",
                category="knowledge", default_fewshot=5,
            )

    def test_invalid_task(self):
        with pytest.raises(ValueError, match="lm_eval_task"):
            CapabilityBenchmark(
                name="x", lm_eval_task="",
                category="knowledge", default_fewshot=5,
            )

    def test_invalid_fewshot(self):
        with pytest.raises(ValueError, match="fewshot"):
            CapabilityBenchmark(
                name="x", lm_eval_task="x",
                category="knowledge", default_fewshot=-1,
            )

    def test_bool_fewshot(self):
        with pytest.raises(ValueError, match="fewshot"):
            CapabilityBenchmark(
                name="x", lm_eval_task="x",
                category="knowledge",
                default_fewshot=True,  # type: ignore[arg-type]
            )

    def test_null_byte_name(self):
        with pytest.raises(ValueError, match="null"):
            CapabilityBenchmark(
                name="x\x00", lm_eval_task="t",
                category="knowledge", default_fewshot=5,
            )

    def test_null_byte_task(self):
        with pytest.raises(ValueError, match="null"):
            CapabilityBenchmark(
                name="x", lm_eval_task="t\x00",
                category="knowledge", default_fewshot=5,
            )


class TestCapabilityCli:
    def test_help_listed(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "capability" in result.output.lower()

    def test_capability_help(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, ["capability", "--help"])
        assert result.exit_code == 0

    def test_capability_unknown_suite(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "capability", "test_run", "--suite", "evil",
        ])
        assert result.exit_code != 0

    def test_capability_fast_smoke(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "cap.json"
        runner = CliRunner()
        result = runner.invoke(app, [
            "capability", "test_run", "--suite", "fast",
            "--output", str(out),
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert out.exists()


class TestSourceWiring:
    def test_no_heavy_imports(self):
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "soup_cli" / "utils" / "capability_suite.py"
        text = src.read_text(encoding="utf-8")
        forbidden_imports = (
            "import torch\n",
            "import transformers\n",
            "import lm_eval\n",
            "from torch",
            "from transformers",
            "from lm_eval",
        )
        for forbidden in forbidden_imports:
            assert forbidden not in text, f"Found heavy top-level: {forbidden!r}"
