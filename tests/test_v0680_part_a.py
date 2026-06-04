"""v0.68.0 Part A — ``soup compile`` (DSPy/GEPA prompt-program compiler).

Tests for ``soup_cli/utils/prompt_compile.py`` + ``soup_cli/commands/compile_cmd.py``.

Coverage:
- Closed ``SUPPORTED_PROMPT_OPTIMIZERS`` allowlist (frozenset, immutable)
- ``validate_prompt_optimizer`` happy + bool/null-byte/oversize/empty/non-string/unknown
- ``validate_max_iters`` bounds + bool rejection
- ``validate_program_path`` + ``validate_eval_suite_path`` containment + symlink reject
- Frozen ``CompilePlan`` + ``CompileResult`` + FrozenInstanceError
- ``build_compile_plan`` factory
- ``run_compile`` deferred-live stub raises NotImplementedError w/ v0.68.1 marker
- CLI smoke: help + plan-only + unknown optimizer + missing program
- Source-grep: no heavy top-level imports (torch / transformers / dspy)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import prompt_compile

        assert hasattr(prompt_compile, "SUPPORTED_PROMPT_OPTIMIZERS")
        assert hasattr(prompt_compile, "validate_prompt_optimizer")
        assert hasattr(prompt_compile, "validate_max_iters")
        assert hasattr(prompt_compile, "validate_program_path")
        assert hasattr(prompt_compile, "validate_eval_suite_path")
        assert hasattr(prompt_compile, "CompilePlan")
        assert hasattr(prompt_compile, "CompileResult")
        assert hasattr(prompt_compile, "build_compile_plan")
        assert hasattr(prompt_compile, "run_compile")

    def test_allowlist_is_frozenset(self) -> None:
        from soup_cli.utils.prompt_compile import SUPPORTED_PROMPT_OPTIMIZERS

        assert isinstance(SUPPORTED_PROMPT_OPTIMIZERS, frozenset)
        # Allowlist should cover the canonical DSPy / GEPA / TextGrad set.
        assert "bootstrap_fewshot" in SUPPORTED_PROMPT_OPTIMIZERS
        assert "mipro" in SUPPORTED_PROMPT_OPTIMIZERS
        assert "copro" in SUPPORTED_PROMPT_OPTIMIZERS
        assert "gepa" in SUPPORTED_PROMPT_OPTIMIZERS
        assert "textgrad" in SUPPORTED_PROMPT_OPTIMIZERS

    def test_allowlist_immutable(self) -> None:
        from soup_cli.utils.prompt_compile import SUPPORTED_PROMPT_OPTIMIZERS

        with pytest.raises(AttributeError):
            SUPPORTED_PROMPT_OPTIMIZERS.add("evil")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# validate_prompt_optimizer
# ---------------------------------------------------------------------------


class TestValidatePromptOptimizer:
    def test_happy(self) -> None:
        from soup_cli.utils.prompt_compile import validate_prompt_optimizer

        assert validate_prompt_optimizer("mipro") == "mipro"

    def test_case_insensitive(self) -> None:
        from soup_cli.utils.prompt_compile import validate_prompt_optimizer

        assert validate_prompt_optimizer("MIPRO") == "mipro"
        assert validate_prompt_optimizer("Gepa") == "gepa"

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_prompt_optimizer

        with pytest.raises(TypeError):
            validate_prompt_optimizer(True)  # type: ignore[arg-type]

    def test_non_string_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_prompt_optimizer

        with pytest.raises(TypeError):
            validate_prompt_optimizer(42)  # type: ignore[arg-type]

    def test_empty_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_prompt_optimizer

        with pytest.raises(ValueError):
            validate_prompt_optimizer("")

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_prompt_optimizer

        with pytest.raises(ValueError):
            validate_prompt_optimizer("mipro\x00")

    def test_oversize_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_prompt_optimizer

        with pytest.raises(ValueError):
            validate_prompt_optimizer("a" * 33)

    def test_unknown_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_prompt_optimizer

        with pytest.raises(ValueError, match="unknown"):
            validate_prompt_optimizer("evil-optimizer")


# ---------------------------------------------------------------------------
# validate_max_iters
# ---------------------------------------------------------------------------


class TestValidateMaxIters:
    def test_happy(self) -> None:
        from soup_cli.utils.prompt_compile import validate_max_iters

        assert validate_max_iters(10) == 10

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_max_iters

        with pytest.raises(TypeError):
            validate_max_iters(True)  # type: ignore[arg-type]

    def test_non_int_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_max_iters

        with pytest.raises(TypeError):
            validate_max_iters(3.14)  # type: ignore[arg-type]

    def test_zero_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_max_iters

        with pytest.raises(ValueError):
            validate_max_iters(0)

    def test_negative_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_max_iters

        with pytest.raises(ValueError):
            validate_max_iters(-1)

    def test_overcap_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import (
            MAX_COMPILE_ITERS,
            validate_max_iters,
        )

        with pytest.raises(ValueError):
            validate_max_iters(MAX_COMPILE_ITERS + 1)

    def test_max_boundary_accepted(self) -> None:
        from soup_cli.utils.prompt_compile import (
            MAX_COMPILE_ITERS,
            validate_max_iters,
        )

        assert validate_max_iters(MAX_COMPILE_ITERS) == MAX_COMPILE_ITERS


# ---------------------------------------------------------------------------
# validate_program_path + validate_eval_suite_path
# ---------------------------------------------------------------------------


class TestValidateProgramPath:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.prompt_compile import validate_program_path

        monkeypatch.chdir(tmp_path)
        prog = tmp_path / "program.py"
        prog.write_text("# dspy program\n", encoding="utf-8")
        assert validate_program_path(str(prog)).endswith("program.py")

    def test_outside_cwd_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.prompt_compile import validate_program_path

        outside = tmp_path / "outside"
        outside.mkdir()
        prog = outside / "program.py"
        prog.write_text("pass\n", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError):
            validate_program_path(str(prog))

    def test_extension_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.prompt_compile import validate_program_path

        monkeypatch.chdir(tmp_path)
        bad = tmp_path / "program.txt"
        bad.write_text("# wrong\n", encoding="utf-8")
        with pytest.raises(ValueError, match="\\.py"):
            validate_program_path(str(bad))

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import validate_program_path

        with pytest.raises(ValueError):
            validate_program_path("a\x00b.py")

    @pytest.mark.skipif(
        not hasattr(__import__("os"), "symlink"), reason="POSIX symlink only"
    )
    def test_symlink_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from soup_cli.utils.prompt_compile import validate_program_path

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real.py"
        target.write_text("# real\n", encoding="utf-8")
        link = tmp_path / "link.py"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not creatable")
        with pytest.raises(ValueError, match="symlink"):
            validate_program_path(str(link))


class TestValidateEvalSuitePath:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.prompt_compile import validate_eval_suite_path

        monkeypatch.chdir(tmp_path)
        suite = tmp_path / "eval.json"
        suite.write_text("[]", encoding="utf-8")
        assert validate_eval_suite_path(str(suite)).endswith("eval.json")

    def test_outside_cwd_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.prompt_compile import validate_eval_suite_path

        outside = tmp_path / "outside"
        outside.mkdir()
        bad = outside / "suite.json"
        bad.write_text("[]", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError):
            validate_eval_suite_path(str(bad))


# ---------------------------------------------------------------------------
# CompilePlan / CompileResult
# ---------------------------------------------------------------------------


class TestCompilePlan:
    def test_frozen(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.prompt_compile import CompilePlan

        monkeypatch.chdir(tmp_path)
        prog = tmp_path / "p.py"
        prog.write_text("pass\n", encoding="utf-8")
        suite = tmp_path / "s.json"
        suite.write_text("[]", encoding="utf-8")

        plan = CompilePlan(
            program_path=str(prog),
            eval_suite_path=str(suite),
            optimizer="mipro",
            max_iters=8,
            output_path="out.py",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.optimizer = "gepa"  # type: ignore[misc]

    def test_invalid_optimizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.prompt_compile import CompilePlan

        monkeypatch.chdir(tmp_path)
        prog = tmp_path / "p.py"
        prog.write_text("pass\n", encoding="utf-8")
        suite = tmp_path / "s.json"
        suite.write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError):
            CompilePlan(
                program_path=str(prog),
                eval_suite_path=str(suite),
                optimizer="evil",
                max_iters=8,
                output_path="out.py",
            )


class TestCompileResult:
    def test_frozen(self) -> None:
        from soup_cli.utils.prompt_compile import CompileResult

        result = CompileResult(
            program_text="# compiled",
            score=0.85,
            iterations=5,
            converged=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.score = 0.9  # type: ignore[misc]

    def test_invalid_score(self) -> None:
        from soup_cli.utils.prompt_compile import CompileResult

        with pytest.raises(ValueError):
            CompileResult(
                program_text="# x", score=float("nan"), iterations=1, converged=True
            )

    def test_negative_iterations_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import CompileResult

        with pytest.raises(ValueError):
            CompileResult(
                program_text="# x", score=0.5, iterations=-1, converged=True
            )

    def test_bool_iterations_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import CompileResult

        with pytest.raises(TypeError):
            CompileResult(
                program_text="# x",
                score=0.5,
                iterations=True,  # type: ignore[arg-type]
                converged=True,
            )

    def test_non_bool_converged_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import CompileResult

        with pytest.raises(TypeError):
            CompileResult(
                program_text="# x",
                score=0.5,
                iterations=1,
                converged="yes",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# build_compile_plan
# ---------------------------------------------------------------------------


class TestBuildCompilePlan:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.prompt_compile import build_compile_plan

        monkeypatch.chdir(tmp_path)
        prog = tmp_path / "p.py"
        prog.write_text("pass\n", encoding="utf-8")
        suite = tmp_path / "s.json"
        suite.write_text("[]", encoding="utf-8")

        plan = build_compile_plan(
            program_path=str(prog),
            eval_suite_path=str(suite),
            optimizer="mipro",
            max_iters=4,
            output_path="out.py",
        )
        assert plan.optimizer == "mipro"
        assert plan.max_iters == 4


# ---------------------------------------------------------------------------
# run_compile deferred stub
# ---------------------------------------------------------------------------


class TestRunCompileDeferred:
    def test_live_missing_dep_friendly_importerror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # v0.71.13 #225: the runner is live; with dspy absent the real
        # branch raises a friendly ImportError naming the [compile] extra.
        from soup_cli.utils.prompt_compile import build_compile_plan, run_compile

        monkeypatch.chdir(tmp_path)
        prog = tmp_path / "p.py"
        prog.write_text("program = 1\n", encoding="utf-8")
        suite = tmp_path / "s.json"
        suite.write_text("[]", encoding="utf-8")
        plan = build_compile_plan(
            program_path=str(prog),
            eval_suite_path=str(suite),
            optimizer="mipro",
            max_iters=4,
            output_path="out.py",
        )
        with pytest.raises(ImportError, match=r"soup-cli\[compile\]"):
            run_compile(plan)

    def test_non_plan_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import run_compile

        with pytest.raises(TypeError):
            run_compile("not-a-plan")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCli:
    def test_help(self) -> None:
        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["compile", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "compile" in result.output.lower()

    def test_plan_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        prog = tmp_path / "p.py"
        prog.write_text("pass\n", encoding="utf-8")
        suite = tmp_path / "s.json"
        suite.write_text("[]", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compile",
                str(prog),
                "--eval",
                str(suite),
                "--optimizer",
                "mipro",
                "--plan-only",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_unknown_optimizer_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        prog = tmp_path / "p.py"
        prog.write_text("pass\n", encoding="utf-8")
        suite = tmp_path / "s.json"
        suite.write_text("[]", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compile",
                str(prog),
                "--eval",
                str(suite),
                "--optimizer",
                "evil",
                "--plan-only",
            ],
        )
        assert result.exit_code == 2

    def test_live_missing_dep_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """v0.71.13 #225: live runner with dspy absent -> friendly exit 2."""
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        prog = tmp_path / "p.py"
        prog.write_text("program = 1\n", encoding="utf-8")
        suite = tmp_path / "s.json"
        suite.write_text("[]", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compile",
                str(prog),
                "--eval",
                str(suite),
                "--optimizer",
                "mipro",
            ],
        )
        assert result.exit_code == 2, (result.output, repr(result.exception))


# ---------------------------------------------------------------------------
# Source-grep regression guards
# ---------------------------------------------------------------------------


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "prompt_compile.py"
        )
        text = path.read_text(encoding="utf-8")
        # Heavy / optional deps must be lazy-imported.
        for token in (
            "\nimport torch",
            "\nimport transformers",
            "\nimport dspy",
        ):
            assert token not in text

    def test_cli_registered(self) -> None:
        from soup_cli.cli import app

        names = [c.name for c in app.registered_commands]
        assert "compile" in names

    def test_uses_atomic_write_helper(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "commands" / "compile_cmd.py"
        )
        # If this file is missing the test must fail loudly — Part A CLI is a
        # shipped artefact, not optional. Skipping would hide a regression.
        assert path.exists(), f"missing CLI command module: {path}"
        text = path.read_text(encoding="utf-8")
        util = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli" / "utils" / "prompt_compile.py"
        )
        combined = text + util.read_text(encoding="utf-8")
        assert "atomic_write_text" in combined
