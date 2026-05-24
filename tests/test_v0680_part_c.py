"""v0.68.0 Part C — ``soup compile-tools``.

Generate tool schemas + descriptions optimized via TextGrad-style textual
gradients. CI runs on OpenAPI / MCP schema changes. Composes with v0.46
Agent Forge.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import compile_tools

        assert hasattr(compile_tools, "SUPPORTED_TOOL_OPTIMIZERS")
        assert hasattr(compile_tools, "validate_tool_optimizer")
        assert hasattr(compile_tools, "validate_spec_path")
        assert hasattr(compile_tools, "ToolCompilePlan")
        assert hasattr(compile_tools, "build_tool_compile_plan")
        assert hasattr(compile_tools, "run_tool_compile")


class TestAllowlist:
    def test_frozenset(self) -> None:
        from soup_cli.utils.compile_tools import SUPPORTED_TOOL_OPTIMIZERS

        assert isinstance(SUPPORTED_TOOL_OPTIMIZERS, frozenset)
        assert "textgrad" in SUPPORTED_TOOL_OPTIMIZERS
        assert "gepa" in SUPPORTED_TOOL_OPTIMIZERS

    def test_immutable(self) -> None:
        from soup_cli.utils.compile_tools import SUPPORTED_TOOL_OPTIMIZERS

        with pytest.raises(AttributeError):
            SUPPORTED_TOOL_OPTIMIZERS.add("x")  # type: ignore[attr-defined]


class TestValidateOptimizer:
    def test_happy(self) -> None:
        from soup_cli.utils.compile_tools import validate_tool_optimizer

        assert validate_tool_optimizer("textgrad") == "textgrad"

    def test_case_insensitive(self) -> None:
        from soup_cli.utils.compile_tools import validate_tool_optimizer

        assert validate_tool_optimizer("GEPA") == "gepa"

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.compile_tools import validate_tool_optimizer

        with pytest.raises(TypeError):
            validate_tool_optimizer(True)  # type: ignore[arg-type]

    def test_unknown_rejected(self) -> None:
        from soup_cli.utils.compile_tools import validate_tool_optimizer

        with pytest.raises(ValueError, match="unknown"):
            validate_tool_optimizer("evil")

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.compile_tools import validate_tool_optimizer

        with pytest.raises(ValueError):
            validate_tool_optimizer("textgrad\x00")


class TestValidateSpecPath:
    def test_json_happy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.compile_tools import validate_spec_path

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        assert validate_spec_path(str(spec)).endswith("spec.json")

    def test_yaml_happy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.compile_tools import validate_spec_path

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.yaml"
        spec.write_text("openapi: 3.0\n", encoding="utf-8")
        assert validate_spec_path(str(spec)).endswith("spec.yaml")

    def test_invalid_extension_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.compile_tools import validate_spec_path

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.txt"
        spec.write_text("garbage", encoding="utf-8")
        with pytest.raises(ValueError, match="extension"):
            validate_spec_path(str(spec))

    def test_outside_cwd_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.compile_tools import validate_spec_path

        outside = tmp_path / "outside"
        outside.mkdir()
        bad = outside / "spec.json"
        bad.write_text("{}", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError):
            validate_spec_path(str(bad))


class TestToolCompilePlan:
    def test_frozen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.compile_tools import ToolCompilePlan

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        eval_suite = tmp_path / "eval.jsonl"
        eval_suite.write_text("[]", encoding="utf-8")

        plan = ToolCompilePlan(
            spec_path=str(spec),
            eval_suite_path=str(eval_suite),
            optimizer="textgrad",
            output_path="tools.json",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.optimizer = "gepa"  # type: ignore[misc]

    def test_invalid_optimizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.compile_tools import ToolCompilePlan

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        eval_suite = tmp_path / "eval.jsonl"
        eval_suite.write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError):
            ToolCompilePlan(
                spec_path=str(spec),
                eval_suite_path=str(eval_suite),
                optimizer="evil",
                output_path="tools.json",
            )


class TestRunToolCompileDeferred:
    def test_raises_v068_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.compile_tools import (
            build_tool_compile_plan,
            run_tool_compile,
        )

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        eval_suite = tmp_path / "eval.jsonl"
        eval_suite.write_text("[]", encoding="utf-8")
        plan = build_tool_compile_plan(
            spec_path=str(spec),
            eval_suite_path=str(eval_suite),
            optimizer="textgrad",
            output_path="tools.json",
        )
        with pytest.raises(NotImplementedError, match="v0.68.1"):
            run_tool_compile(plan)

    def test_non_plan_rejected(self) -> None:
        from soup_cli.utils.compile_tools import run_tool_compile

        with pytest.raises(TypeError):
            run_tool_compile("not-a-plan")  # type: ignore[arg-type]


class TestCli:
    def test_help(self) -> None:
        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["compile-tools", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_plan_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps({"openapi": "3.0"}), encoding="utf-8")
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
                "--plan-only",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_live_exits_3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        spec = tmp_path / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        eval_suite = tmp_path / "eval.jsonl"
        eval_suite.write_text("[]", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["compile-tools", str(spec), "--eval", str(eval_suite)],
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent
            / "soup_cli"
            / "utils"
            / "compile_tools.py"
        )
        text = path.read_text(encoding="utf-8")
        for token in (
            "\nimport torch",
            "\nimport transformers",
            "\nimport dspy",
            "\nimport textgrad",
        ):
            assert token not in text

    def test_cli_registered(self) -> None:
        from soup_cli.cli import app

        names = [c.name for c in app.registered_commands]
        assert "compile-tools" in names
