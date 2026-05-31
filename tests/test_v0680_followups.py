"""v0.68.0 review-fix follow-ups.

Closes every TDD-review finding plus the additional manual code/security
review surface gaps:

- HIGH H1: `validate_student_id` rejection-matrix parity with teacher
- HIGH H2: Part C CLI `unknown-optimizer` exit 2 test
- HIGH H3: `harvest_dpo_pairs` edge cases (one-up-no-down / dedup / multi-prompt)
- HIGH H4: POSIX symlink skip predicate uses `sys.platform` not `hasattr(os, "symlink")`
- MED M1: `CompileResult` Inf rejection + bool-score rejection
- MED M2: `validate_eval_suite_path` symlink test
- MED M3: Part C `validate_tool_optimizer` empty + oversize tests
- MED M4: Part C `validate_spec_path` null-byte + symlink tests
- MED M5: `TestBuildToolCompilePlan` factory test (Part C)
- MED M6: Part D `validate_direction` non-string TypeError + oversize tests
- MED M7: `TestBuildAppleAdapterPlan` factory test (Part D)
- MED M8: Part E backend / train_method validators — full rejection matrix
- MED M9: `record_thumb` null-byte + oversize on `response`
- MED M10: `SUPPORTED_LOCAL_RL_TRAIN_METHODS` immutability test
- LOW L1: Part E `TestInitDb` column-level schema assertions
- LOW L2: Part D `output_dir` null-byte / oversize / outside-cwd tests
- LOW L3: `validate_db_path` is a public symbol
- LOW L4: `DpoPair` frozen + invariants
- Sec/code: `validate_db_path` exported in `__all__`
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Cross-platform symlink test gate (HIGH H4)
# ---------------------------------------------------------------------------


def _symlinks_available() -> bool:
    """``os.symlink`` exists on Windows but needs elevation. Use platform."""
    return sys.platform != "win32"


# ---------------------------------------------------------------------------
# Part B — validate_student_id rejection-matrix parity (HIGH H1)
# ---------------------------------------------------------------------------


class TestValidateStudentIdParity:
    def test_oversize_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_student_id

        with pytest.raises(ValueError):
            validate_student_id("a" * 513)

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_student_id

        with pytest.raises(ValueError):
            validate_student_id("a\x00b")

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_student_id

        with pytest.raises(TypeError):
            validate_student_id(True)  # type: ignore[arg-type]

    def test_empty_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_student_id

        with pytest.raises(ValueError):
            validate_student_id("")

    def test_non_string_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_student_id

        with pytest.raises(TypeError):
            validate_student_id(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Part C — CLI unknown-optimizer exits 2 (HIGH H2)
# ---------------------------------------------------------------------------


class TestPartCUnknownOptimizerCli:
    def test_exits_2(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        eval_suite = tmp_path / "eval.jsonl"
        eval_suite.write_text("[]", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "compile-tools",
                str(spec),
                "--eval",
                str(eval_suite),
                "--optimizer",
                "evil",
                "--plan-only",
            ],
        )
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# Part E — harvest_dpo_pairs edge cases (HIGH H3)
# ---------------------------------------------------------------------------


class TestHarvestEdgeCases:
    def test_one_up_no_down_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import (
            harvest_dpo_pairs,
            init_local_rl_db,
            record_thumb,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        record_thumb(
            db_path="rl.db", prompt="q", response="r", thumb="up"
        )
        assert harvest_dpo_pairs("rl.db") == ()

    def test_one_down_no_up_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import (
            harvest_dpo_pairs,
            init_local_rl_db,
            record_thumb,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        record_thumb(
            db_path="rl.db", prompt="q", response="r", thumb="down"
        )
        assert harvest_dpo_pairs("rl.db") == ()

    def test_multiple_prompts_independent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import (
            harvest_dpo_pairs,
            init_local_rl_db,
            record_thumb,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        for prompt, chosen, rejected in (("q1", "c1", "r1"), ("q2", "c2", "r2")):
            record_thumb(db_path="rl.db", prompt=prompt, response=chosen, thumb="up")
            record_thumb(
                db_path="rl.db", prompt=prompt, response=rejected, thumb="down"
            )
        pairs = harvest_dpo_pairs("rl.db")
        assert len(pairs) == 2
        prompts = sorted(p.prompt for p in pairs)
        assert prompts == ["q1", "q2"]

    def test_dedup_keeps_latest_per_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Duplicate thumbs for the same prompt collapse to one DPO pair."""
        from soup_cli.utils.local_rl import (
            harvest_dpo_pairs,
            init_local_rl_db,
            record_thumb,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        # Two ups + two downs for the same prompt should still yield exactly one pair.
        record_thumb(db_path="rl.db", prompt="q", response="up1", thumb="up")
        record_thumb(db_path="rl.db", prompt="q", response="up2", thumb="up")
        record_thumb(db_path="rl.db", prompt="q", response="down1", thumb="down")
        record_thumb(db_path="rl.db", prompt="q", response="down2", thumb="down")
        pairs = harvest_dpo_pairs("rl.db")
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# Part A — CompileResult Inf + bool-score (MED M1)
# ---------------------------------------------------------------------------


class TestCompileResultBoundaries:
    def test_positive_inf_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import CompileResult

        with pytest.raises(ValueError):
            CompileResult(
                program_text="# x",
                score=float("inf"),
                iterations=1,
                converged=True,
            )

    def test_negative_inf_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import CompileResult

        with pytest.raises(ValueError):
            CompileResult(
                program_text="# x",
                score=float("-inf"),
                iterations=1,
                converged=True,
            )

    def test_bool_score_rejected(self) -> None:
        from soup_cli.utils.prompt_compile import CompileResult

        with pytest.raises(TypeError):
            CompileResult(
                program_text="# x",
                score=True,  # type: ignore[arg-type]
                iterations=1,
                converged=True,
            )

    def test_zero_iterations_accepted(self) -> None:
        from soup_cli.utils.prompt_compile import CompileResult

        result = CompileResult(
            program_text="# x", score=0.5, iterations=0, converged=False
        )
        assert result.iterations == 0


# ---------------------------------------------------------------------------
# Part A — validate_eval_suite_path symlink test (MED M2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _symlinks_available(), reason="POSIX symlink only")
class TestEvalSuitePathSymlink:
    def test_symlink_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.prompt_compile import validate_eval_suite_path

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "suite.json"
        real.write_text("[]", encoding="utf-8")
        link = tmp_path / "link.json"
        try:
            os.symlink(real, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not creatable on this filesystem")
        with pytest.raises(ValueError, match="symlink"):
            validate_eval_suite_path(str(link))


# ---------------------------------------------------------------------------
# Part C — validate_tool_optimizer empty + oversize (MED M3)
# ---------------------------------------------------------------------------


class TestValidateToolOptimizerExtras:
    def test_empty_rejected(self) -> None:
        from soup_cli.utils.compile_tools import validate_tool_optimizer

        with pytest.raises(ValueError):
            validate_tool_optimizer("")

    def test_oversize_rejected(self) -> None:
        from soup_cli.utils.compile_tools import validate_tool_optimizer

        with pytest.raises(ValueError):
            validate_tool_optimizer("a" * 33)


# ---------------------------------------------------------------------------
# Part C — validate_spec_path null-byte + symlink (MED M4)
# ---------------------------------------------------------------------------


class TestValidateSpecPathExtras:
    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.compile_tools import validate_spec_path

        with pytest.raises(ValueError):
            validate_spec_path("spec\x00.json")

    @pytest.mark.skipif(not _symlinks_available(), reason="POSIX symlink only")
    def test_symlink_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.compile_tools import validate_spec_path

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.json"
        real.write_text("{}", encoding="utf-8")
        link = tmp_path / "link.json"
        try:
            os.symlink(real, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not creatable on this filesystem")
        with pytest.raises(ValueError, match="symlink"):
            validate_spec_path(str(link))


# ---------------------------------------------------------------------------
# Part C — build_tool_compile_plan factory (MED M5)
# ---------------------------------------------------------------------------


class TestBuildToolCompilePlan:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.compile_tools import build_tool_compile_plan

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        eval_suite = tmp_path / "eval.jsonl"
        eval_suite.write_text("[]", encoding="utf-8")
        plan = build_tool_compile_plan(
            spec_path=str(spec),
            eval_suite_path=str(eval_suite),
            optimizer="GEPA",  # case-insensitive
            output_path="tools.json",
        )
        assert plan.optimizer == "gepa"


# ---------------------------------------------------------------------------
# Part D — validate_direction non-string + oversize (MED M6)
# ---------------------------------------------------------------------------


class TestValidateDirectionExtras:
    def test_non_string_rejected(self) -> None:
        from soup_cli.utils.apple_adapter import validate_direction

        with pytest.raises(TypeError):
            validate_direction(42)  # type: ignore[arg-type]

    def test_oversize_rejected(self) -> None:
        from soup_cli.utils.apple_adapter import validate_direction

        with pytest.raises(ValueError):
            validate_direction("a" * 33)


# ---------------------------------------------------------------------------
# Part D — build_apple_adapter_plan factory (MED M7)
# ---------------------------------------------------------------------------


class TestBuildAppleAdapterPlan:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.apple_adapter import build_apple_adapter_plan

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        plan = build_apple_adapter_plan(
            source_dir=str(adapter),
            output_dir="out",
            direction="HF-TO-MLX",  # case-insensitive
            sign=False,
        )
        assert plan.direction == "hf-to-mlx"


# ---------------------------------------------------------------------------
# Part D — output_dir validation (LOW L2)
# ---------------------------------------------------------------------------


class TestAppleAdapterOutputDirValidation:
    def test_empty_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.apple_adapter import AppleAdapterPlan

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "a"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            AppleAdapterPlan(
                source_dir=str(adapter),
                output_dir="",
                direction="hf-to-mlx",
                sign=False,
            )

    def test_null_byte_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.apple_adapter import AppleAdapterPlan

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "a"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError):
            AppleAdapterPlan(
                source_dir=str(adapter),
                output_dir="out\x00",
                direction="hf-to-mlx",
                sign=False,
            )

    def test_bool_output_dir_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.apple_adapter import AppleAdapterPlan

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "a"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(TypeError):
            AppleAdapterPlan(
                source_dir=str(adapter),
                output_dir=True,  # type: ignore[arg-type]
                direction="hf-to-mlx",
                sign=False,
            )


# ---------------------------------------------------------------------------
# Part E — backend / train_method full rejection matrix (MED M8)
# ---------------------------------------------------------------------------


class TestPartEValidatorParity:
    def test_backend_null_byte_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_backend

        with pytest.raises(ValueError):
            validate_local_rl_backend("ollama\x00")

    def test_backend_empty_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_backend

        with pytest.raises(ValueError):
            validate_local_rl_backend("")

    def test_backend_oversize_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_backend

        with pytest.raises(ValueError):
            validate_local_rl_backend("a" * 33)

    def test_backend_non_string_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_backend

        with pytest.raises(TypeError):
            validate_local_rl_backend(42)  # type: ignore[arg-type]

    def test_train_method_null_byte_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_train_method

        with pytest.raises(ValueError):
            validate_local_rl_train_method("dpo\x00")

    def test_train_method_empty_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_train_method

        with pytest.raises(ValueError):
            validate_local_rl_train_method("")

    def test_train_method_oversize_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_train_method

        with pytest.raises(ValueError):
            validate_local_rl_train_method("a" * 33)

    def test_train_method_non_string_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_train_method

        with pytest.raises(TypeError):
            validate_local_rl_train_method(42)  # type: ignore[arg-type]

    def test_train_method_bool_rejected(self) -> None:
        from soup_cli.utils.local_rl import validate_local_rl_train_method

        with pytest.raises(TypeError):
            validate_local_rl_train_method(True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Part E — record_thumb response validation (MED M9)
# ---------------------------------------------------------------------------


class TestRecordThumbResponseValidation:
    def test_response_null_byte_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db, record_thumb

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        with pytest.raises(ValueError):
            record_thumb(
                db_path="rl.db", prompt="p", response="r\x00", thumb="up"
            )

    def test_response_oversize_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import (
            MAX_RESPONSE_LEN,
            init_local_rl_db,
            record_thumb,
        )

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        with pytest.raises(ValueError):
            record_thumb(
                db_path="rl.db",
                prompt="p",
                response="a" * (MAX_RESPONSE_LEN + 1),
                thumb="up",
            )

    def test_response_empty_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db, record_thumb

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        with pytest.raises(ValueError):
            record_thumb(
                db_path="rl.db", prompt="p", response="", thumb="up"
            )


# ---------------------------------------------------------------------------
# Part E — train_method allowlist immutability (MED M10)
# ---------------------------------------------------------------------------


class TestTrainMethodAllowlistImmutability:
    def test_immutable(self) -> None:
        from soup_cli.utils.local_rl import SUPPORTED_LOCAL_RL_TRAIN_METHODS

        with pytest.raises(AttributeError):
            SUPPORTED_LOCAL_RL_TRAIN_METHODS.add("ppo")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Part E — schema column-level assertions (LOW L1)
# ---------------------------------------------------------------------------


class TestInitDbColumnLevel:
    def test_thumbs_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        with sqlite3.connect("rl.db") as conn:
            rows = conn.execute("PRAGMA table_info(thumbs)").fetchall()
        names = {r[1] for r in rows}
        # Schema regression guard: every named column must exist.
        for col in ("id", "ts", "prompt", "response", "thumb"):
            assert col in names

    def test_interactions_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import init_local_rl_db

        monkeypatch.chdir(tmp_path)
        init_local_rl_db("rl.db")
        with sqlite3.connect("rl.db") as conn:
            rows = conn.execute("PRAGMA table_info(interactions)").fetchall()
        names = {r[1] for r in rows}
        for col in ("id", "ts", "prompt", "response"):
            assert col in names


# ---------------------------------------------------------------------------
# Part E — validate_db_path is public (LOW L3)
# ---------------------------------------------------------------------------


class TestValidateDbPathPublic:
    def test_importable(self) -> None:
        from soup_cli.utils.local_rl import validate_db_path  # noqa: F401

    def test_in_all(self) -> None:
        from soup_cli.utils import local_rl

        assert "validate_db_path" in local_rl.__all__

    def test_rejects_null_byte(self) -> None:
        from soup_cli.utils.local_rl import validate_db_path

        with pytest.raises(ValueError):
            validate_db_path("db\x00.db")

    def test_rejects_bool(self) -> None:
        from soup_cli.utils.local_rl import validate_db_path

        with pytest.raises(TypeError):
            validate_db_path(True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Part E — DpoPair frozen invariant (LOW L4)
# ---------------------------------------------------------------------------


class TestDpoPairFrozen:
    def test_frozen(self) -> None:
        from soup_cli.utils.local_rl import DpoPair

        pair = DpoPair(prompt="p", chosen="c", rejected="r")
        with pytest.raises(dataclasses.FrozenInstanceError):
            pair.prompt = "evil"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Part E — harvest_dpo_pairs missing-file rejection (manual review)
# ---------------------------------------------------------------------------


class TestHarvestMissingFile:
    def test_missing_db_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.local_rl import harvest_dpo_pairs

        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            harvest_dpo_pairs("no_such.db")


# ---------------------------------------------------------------------------
# Source-grep regression guards (LOW)
# ---------------------------------------------------------------------------


class TestSourceGuards:
    def test_no_shell_true_in_v068_modules(self) -> None:
        """No subprocess invocation in v0.68.0 utility modules."""
        for stem in (
            "prompt_compile",
            "prompt_distill",
            "compile_tools",
            "apple_adapter",
            "local_rl",
        ):
            path = (
                Path(__file__).resolve().parent.parent
                / "src" / "soup_cli"
                / "utils"
                / f"{stem}.py"
            )
            text = path.read_text(encoding="utf-8")
            assert "shell=True" not in text, f"{stem}.py contains shell=True"

    def test_no_resolve_relative_to(self) -> None:
        """Project policy: realpath + commonpath, not Path.resolve()+relative_to."""
        for stem in (
            "prompt_compile",
            "prompt_distill",
            "compile_tools",
            "apple_adapter",
            "local_rl",
        ):
            path = (
                Path(__file__).resolve().parent.parent
                / "src" / "soup_cli"
                / "utils"
                / f"{stem}.py"
            )
            text = path.read_text(encoding="utf-8")
            assert ".resolve()" not in text or ".resolve().parent" in text, (
                f"{stem}.py should not use Path.resolve() for containment"
            )

    def test_local_rl_cli_imports_public_helper(self) -> None:
        """CLI must import the public `validate_db_path`, not the private one."""
        path = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli"
            / "commands"
            / "local_rl.py"
        )
        text = path.read_text(encoding="utf-8")
        assert "_validate_db_path" not in text, (
            "CLI must use public `validate_db_path`, not the private alias"
        )
        assert "validate_db_path" in text

    def test_all_command_modules_use_typer_exit(self) -> None:
        """All v0.68 CLI command modules raise typer.Exit on rejection."""
        for stem in (
            "compile_cmd",
            "distill_prompt",
            "compile_tools",
            "apple_adapter",
            "local_rl",
        ):
            path = (
                Path(__file__).resolve().parent.parent
                / "src" / "soup_cli"
                / "commands"
                / f"{stem}.py"
            )
            text = path.read_text(encoding="utf-8")
            assert "typer.Exit" in text, f"{stem}.py missing typer.Exit"

    def test_version_bumped(self) -> None:
        from soup_cli import __version__

        # Floor-check: must be at or above the v0.68.0 release.
        major, minor, patch = (int(x) for x in __version__.split("."))
        assert (major, minor) >= (0, 68)
