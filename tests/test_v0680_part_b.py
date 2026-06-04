"""v0.68.0 Part B — ``soup distill-prompt``.

Distill prompt-heavy traces (large-prompt GPT-5 / Claude calls) into a small
FT plan. Bridge between prompt-engineering and FT worlds. Composes with the
v0.70 Part B cross-tokenizer KD when that ships.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import prompt_distill

        assert hasattr(prompt_distill, "SUPPORTED_DISTILL_STRATEGIES")
        assert hasattr(prompt_distill, "validate_distill_strategy")
        assert hasattr(prompt_distill, "validate_teacher_id")
        assert hasattr(prompt_distill, "validate_student_id")
        assert hasattr(prompt_distill, "validate_traces_path")
        assert hasattr(prompt_distill, "DistillPromptPlan")
        assert hasattr(prompt_distill, "build_distill_prompt_plan")
        assert hasattr(prompt_distill, "prepare_distill_dataset")


class TestAllowlist:
    def test_frozenset(self) -> None:
        from soup_cli.utils.prompt_distill import SUPPORTED_DISTILL_STRATEGIES

        assert isinstance(SUPPORTED_DISTILL_STRATEGIES, frozenset)
        assert "sft" in SUPPORTED_DISTILL_STRATEGIES
        assert "preference" in SUPPORTED_DISTILL_STRATEGIES
        assert "kl" in SUPPORTED_DISTILL_STRATEGIES

    def test_immutable(self) -> None:
        from soup_cli.utils.prompt_distill import SUPPORTED_DISTILL_STRATEGIES

        with pytest.raises(AttributeError):
            SUPPORTED_DISTILL_STRATEGIES.add("evil")  # type: ignore[attr-defined]


class TestValidateStrategy:
    def test_happy(self) -> None:
        from soup_cli.utils.prompt_distill import validate_distill_strategy

        assert validate_distill_strategy("sft") == "sft"

    def test_case_insensitive(self) -> None:
        from soup_cli.utils.prompt_distill import validate_distill_strategy

        assert validate_distill_strategy("SFT") == "sft"

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_distill_strategy

        with pytest.raises(TypeError):
            validate_distill_strategy(True)  # type: ignore[arg-type]

    def test_unknown_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_distill_strategy

        with pytest.raises(ValueError, match="unknown"):
            validate_distill_strategy("evil")

    def test_empty_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_distill_strategy

        with pytest.raises(ValueError):
            validate_distill_strategy("")

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_distill_strategy

        with pytest.raises(ValueError):
            validate_distill_strategy("sft\x00")


class TestValidateModelId:
    def test_teacher_happy(self) -> None:
        from soup_cli.utils.prompt_distill import validate_teacher_id

        assert validate_teacher_id("anthropic/claude-3-5-sonnet") == "anthropic/claude-3-5-sonnet"

    def test_teacher_oversize_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_teacher_id

        with pytest.raises(ValueError):
            validate_teacher_id("a" * 513)

    def test_teacher_null_byte_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_teacher_id

        with pytest.raises(ValueError):
            validate_teacher_id("model\x00")

    def test_teacher_bool_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_teacher_id

        with pytest.raises(TypeError):
            validate_teacher_id(True)  # type: ignore[arg-type]

    def test_teacher_empty_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_teacher_id

        with pytest.raises(ValueError):
            validate_teacher_id("")

    def test_student_happy(self) -> None:
        from soup_cli.utils.prompt_distill import validate_student_id

        assert validate_student_id("meta-llama/Llama-3.2-1B") == "meta-llama/Llama-3.2-1B"


class TestValidateTracesPath:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.prompt_distill import validate_traces_path

        monkeypatch.chdir(tmp_path)
        p = tmp_path / "traces.jsonl"
        p.write_text('{"prompt":"x","response":"y"}\n', encoding="utf-8")
        assert validate_traces_path(str(p)).endswith("traces.jsonl")

    def test_outside_cwd_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.prompt_distill import validate_traces_path

        outside = tmp_path / "outside"
        outside.mkdir()
        p = outside / "t.jsonl"
        p.write_text("[]", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError):
            validate_traces_path(str(p))

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import validate_traces_path

        with pytest.raises(ValueError):
            validate_traces_path("t\x00.jsonl")

    @pytest.mark.skipif(
        not hasattr(__import__("os"), "symlink"), reason="POSIX symlink only"
    )
    def test_symlink_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from soup_cli.utils.prompt_distill import validate_traces_path

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "r.jsonl"
        real.write_text("[]", encoding="utf-8")
        link = tmp_path / "link.jsonl"
        try:
            os.symlink(real, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not creatable")
        with pytest.raises(ValueError, match="symlink"):
            validate_traces_path(str(link))


class TestDistillPromptPlan:
    def test_frozen(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.prompt_distill import DistillPromptPlan

        monkeypatch.chdir(tmp_path)
        traces = tmp_path / "traces.jsonl"
        traces.write_text("[]", encoding="utf-8")

        plan = DistillPromptPlan(
            traces_path=str(traces),
            teacher="anthropic/claude-3-5-sonnet",
            student="meta-llama/Llama-3.2-1B",
            strategy="sft",
            output_path="distilled.jsonl",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.strategy = "preference"  # type: ignore[misc]

    def test_invalid_strategy_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.prompt_distill import DistillPromptPlan

        monkeypatch.chdir(tmp_path)
        traces = tmp_path / "traces.jsonl"
        traces.write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError):
            DistillPromptPlan(
                traces_path=str(traces),
                teacher="t",
                student="s",
                strategy="evil",
                output_path="o.jsonl",
            )


class TestBuildPlan:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.prompt_distill import build_distill_prompt_plan

        monkeypatch.chdir(tmp_path)
        traces = tmp_path / "traces.jsonl"
        traces.write_text("[]", encoding="utf-8")
        plan = build_distill_prompt_plan(
            traces_path=str(traces),
            teacher="t/x",
            student="s/y",
            strategy="sft",
            output_path="o.jsonl",
        )
        assert plan.strategy == "sft"


class TestPrepareDataset:
    def test_live_writes_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # v0.71.13 #226: live dataset preparation with an injected teacher.
        from soup_cli.utils.prompt_distill import (
            build_distill_prompt_plan,
            prepare_distill_dataset,
        )

        monkeypatch.chdir(tmp_path)
        traces = tmp_path / "traces.jsonl"
        traces.write_text(
            json.dumps({"prompt": "q"}) + "\n", encoding="utf-8"
        )
        plan = build_distill_prompt_plan(
            traces_path=str(traces),
            teacher="t/x",
            student="s/y",
            strategy="sft",
            output_path="o.jsonl",
        )
        n = prepare_distill_dataset(plan, teacher_fn=lambda p: {"text": "T"})
        assert n == 1
        assert (tmp_path / "o.jsonl").is_file()

    def test_non_plan_rejected(self) -> None:
        from soup_cli.utils.prompt_distill import prepare_distill_dataset

        with pytest.raises(TypeError):
            prepare_distill_dataset({})  # type: ignore[arg-type]


class TestCli:
    def test_help(self) -> None:
        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["distill-prompt", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_plan_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        traces = tmp_path / "traces.jsonl"
        traces.write_text(json.dumps({"prompt": "x"}) + "\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "distill-prompt",
                "--traces",
                str(traces),
                "--teacher",
                "anthropic/claude-3-5-sonnet",
                "--student",
                "meta-llama/Llama-3.2-1B",
                "--strategy",
                "sft",
                "--plan-only",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_unknown_strategy_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        traces = tmp_path / "traces.jsonl"
        traces.write_text("[]", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "distill-prompt",
                "--traces",
                str(traces),
                "--teacher",
                "t",
                "--student",
                "s",
                "--strategy",
                "evil",
                "--plan-only",
            ],
        )
        assert result.exit_code == 2

    def test_live_writes_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # v0.71.13 #226: live runner writes a distilled dataset (provider
        # mocked so the test never touches the network).
        import soup_cli.utils.prompt_distill as pd
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            pd, "_build_provider_fn", lambda *a, **k: (lambda p: {"text": "R"})
        )
        traces = tmp_path / "traces.jsonl"
        traces.write_text(
            json.dumps({"prompt": "x"}) + "\n", encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "distill-prompt",
                "--traces",
                str(traces),
                "--teacher",
                "t",
                "--student",
                "s",
                "--strategy",
                "sft",
                "--output",
                "o.jsonl",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "o.jsonl").is_file()


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent
            / "src" / "soup_cli"
            / "utils"
            / "prompt_distill.py"
        )
        text = path.read_text(encoding="utf-8")
        for token in ("\nimport torch", "\nimport transformers", "\nimport anthropic"):
            assert token not in text

    def test_cli_registered(self) -> None:
        from soup_cli.cli import app

        names = [c.name for c in app.registered_commands]
        assert "distill-prompt" in names
