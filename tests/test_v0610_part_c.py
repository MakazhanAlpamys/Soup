"""Tests for v0.61.0 Part C — `soup edit set` (ROME / MEMIT / AlphaEdit).

Coverage:
- ``SUPPORTED_EDIT_METHODS`` + ``validate_edit_method``.
- ``EditRequest`` / ``EditPlan`` frozen dataclasses.
- ``parse_edit_subject_target`` parser (free-text subject + target).
- ``build_edit_plan`` schema-only orchestrator.
- ``apply_edit`` deferred stub.
- ``soup edit set`` CLI smoke.
"""

from __future__ import annotations

import dataclasses

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


class TestModuleSurface:
    def test_imports(self):
        from soup_cli.utils.knowledge_edit import (
            SUPPORTED_EDIT_METHODS,
            EditPlan,
            EditRequest,
            apply_edit,
            build_edit_plan,
            parse_edit_subject_target,
            validate_edit_method,
        )
        assert callable(validate_edit_method)
        assert callable(parse_edit_subject_target)
        assert callable(build_edit_plan)
        assert callable(apply_edit)
        assert dataclasses.is_dataclass(EditRequest)
        assert dataclasses.is_dataclass(EditPlan)
        assert isinstance(SUPPORTED_EDIT_METHODS, frozenset)

    def test_supported_methods_exact(self):
        from soup_cli.utils.knowledge_edit import SUPPORTED_EDIT_METHODS

        assert SUPPORTED_EDIT_METHODS == frozenset({"rome", "memit", "alphaedit"})


class TestValidateEditMethod:
    def test_happy_path(self):
        from soup_cli.utils.knowledge_edit import validate_edit_method

        for name in ("rome", "memit", "alphaedit"):
            assert validate_edit_method(name) == name

    def test_case_insensitive(self):
        from soup_cli.utils.knowledge_edit import validate_edit_method

        assert validate_edit_method("ROME") == "rome"
        assert validate_edit_method("MEMIT") == "memit"

    def test_unknown_rejected(self):
        from soup_cli.utils.knowledge_edit import validate_edit_method

        with pytest.raises(ValueError, match="unknown"):
            validate_edit_method("ftedit")

    def test_bool_rejected(self):
        from soup_cli.utils.knowledge_edit import validate_edit_method

        with pytest.raises(TypeError):
            validate_edit_method(True)

    def test_non_string_rejected(self):
        from soup_cli.utils.knowledge_edit import validate_edit_method

        with pytest.raises(TypeError):
            validate_edit_method(42)

    def test_null_byte_rejected(self):
        from soup_cli.utils.knowledge_edit import validate_edit_method

        with pytest.raises(ValueError):
            validate_edit_method("rome\x00")

    def test_oversize_rejected(self):
        from soup_cli.utils.knowledge_edit import validate_edit_method

        with pytest.raises(ValueError):
            validate_edit_method("a" * 100)


class TestParseEditSubjectTarget:
    def test_happy_path(self):
        from soup_cli.utils.knowledge_edit import parse_edit_subject_target

        subject, target = parse_edit_subject_target(
            subject="Paris is the capital of France",
            target="Lyon",
        )
        assert subject == "Paris is the capital of France"
        assert target == "Lyon"

    def test_empty_subject_rejected(self):
        from soup_cli.utils.knowledge_edit import parse_edit_subject_target

        with pytest.raises(ValueError, match="subject"):
            parse_edit_subject_target(subject="", target="Lyon")

    def test_empty_target_rejected(self):
        from soup_cli.utils.knowledge_edit import parse_edit_subject_target

        with pytest.raises(ValueError, match="target"):
            parse_edit_subject_target(subject="Paris is the capital", target="")

    def test_null_byte_rejected(self):
        from soup_cli.utils.knowledge_edit import parse_edit_subject_target

        with pytest.raises(ValueError):
            parse_edit_subject_target(subject="Paris\x00", target="Lyon")

    def test_oversize_rejected(self):
        from soup_cli.utils.knowledge_edit import parse_edit_subject_target

        with pytest.raises(ValueError):
            parse_edit_subject_target(
                subject="x" * 5000,
                target="Lyon",
            )

    def test_bool_rejected(self):
        from soup_cli.utils.knowledge_edit import parse_edit_subject_target

        with pytest.raises(TypeError):
            parse_edit_subject_target(subject=True, target="Lyon")  # type: ignore


class TestBuildEditPlan:
    def test_happy_path(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        plan = build_edit_plan(
            base="meta-llama/Llama-3.1-8B-Instruct",
            method="rome",
            subject="Paris is the capital of France",
            target="Lyon",
        )
        assert plan.method == "rome"
        assert plan.base == "meta-llama/Llama-3.1-8B-Instruct"
        assert plan.subject == "Paris is the capital of France"
        assert plan.target == "Lyon"
        assert plan.layer is not None

    def test_custom_layer(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        plan = build_edit_plan(
            base="meta-llama/Llama-3.1-8B-Instruct",
            method="rome",
            subject="x",
            target="y",
            layer=17,
        )
        assert plan.layer == 17

    def test_layer_bool_rejected(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        with pytest.raises(TypeError):
            build_edit_plan(
                base="b", method="rome", subject="s", target="t",
                layer=True,  # type: ignore
            )

    def test_layer_negative_rejected(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        with pytest.raises(ValueError):
            build_edit_plan(
                base="b", method="rome", subject="s", target="t",
                layer=-1,
            )

    def test_layer_oversize_rejected(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        with pytest.raises(ValueError):
            build_edit_plan(
                base="b", method="rome", subject="s", target="t",
                layer=10000,
            )

    def test_unknown_method_rejected(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        with pytest.raises(ValueError):
            build_edit_plan(
                base="b", method="zzz", subject="s", target="t",
            )

    def test_empty_base_rejected(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        with pytest.raises(ValueError):
            build_edit_plan(
                base="", method="rome", subject="s", target="t",
            )

    def test_null_byte_base_rejected(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        with pytest.raises(ValueError):
            build_edit_plan(
                base="b\x00", method="rome", subject="s", target="t",
            )


class TestEditPlanFrozen:
    def test_frozen(self):
        from soup_cli.utils.knowledge_edit import build_edit_plan

        plan = build_edit_plan(
            base="b", method="rome", subject="s", target="t",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.method = "memit"  # type: ignore


class TestApplyEdit:
    def test_deferred(self):
        from soup_cli.utils.knowledge_edit import apply_edit, build_edit_plan

        plan = build_edit_plan(
            base="b", method="rome", subject="s", target="t",
        )
        with pytest.raises(NotImplementedError, match="v0.61.1"):
            apply_edit(plan)

    def test_unknown_method_in_plan_short_circuits(self):
        from soup_cli.utils.knowledge_edit import apply_edit

        # apply_edit defends against direct (non-builder) plan construction
        # by re-validating the method before raising NotImplementedError.
        class _BadPlan:
            method = "zzz"
            base = "b"
            subject = "s"
            target = "t"
            layer = 5

        with pytest.raises((TypeError, ValueError)):
            apply_edit(_BadPlan())  # type: ignore


class TestCli:
    def test_edit_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["edit", "--help"])
        assert result.exit_code == 0, result.output

    def test_edit_set_help(self):
        runner = CliRunner()
        result = runner.invoke(app, ["edit", "set", "--help"])
        assert result.exit_code == 0, result.output

    def test_edit_set_plan_only(self):
        runner = CliRunner()
        result = runner.invoke(app, [
            "edit", "set",
            "--base", "meta-llama/Llama-3.1-8B-Instruct",
            "--method", "rome",
            "--subject", "Paris is the capital of France",
            "--target", "Lyon",
            "--plan-only",
        ])
        # Plan-only mode prints the plan and exits 0 without invoking
        # the deferred apply_edit kernel.
        assert result.exit_code == 0, result.output
        assert "rome" in result.output.lower()
        assert "lyon" in result.output.lower()

    def test_edit_set_unknown_method_rejected(self):
        runner = CliRunner()
        result = runner.invoke(app, [
            "edit", "set",
            "--base", "test",
            "--method", "zzz",
            "--subject", "s",
            "--target", "t",
            "--plan-only",
        ])
        assert result.exit_code != 0

    def test_edit_set_apply_deferred(self):
        runner = CliRunner()
        result = runner.invoke(app, [
            "edit", "set",
            "--base", "test",
            "--method", "rome",
            "--subject", "Paris is the capital",
            "--target", "Lyon",
        ])
        # Without --plan-only, the CLI invokes apply_edit which raises
        # NotImplementedError. Exit code 3 (deferred, NOT validation
        # failure which is exit 2). Review HIGH H5 — distinguishes
        # "not yet shipped" from "validation rejection".
        assert result.exit_code == 3
        assert "v0.61.1" in result.output

    def test_edit_set_registry_id_null_byte_rejected(self):
        """Review HIGH H4 — null-byte in --registry-id."""
        runner = CliRunner()
        result = runner.invoke(app, [
            "edit", "set",
            "--base", "test",
            "--method", "rome",
            "--subject", "Paris is the capital",
            "--target", "Lyon",
            "--registry-id", "evil\x00id",
            "--plan-only",
        ])
        assert result.exit_code == 2
        assert "null" in result.output.lower() or "registry-id" in result.output.lower()

    def test_edit_set_registry_id_oversize_rejected(self):
        """Review HIGH H4 — oversized --registry-id."""
        runner = CliRunner()
        result = runner.invoke(app, [
            "edit", "set",
            "--base", "test",
            "--method", "rome",
            "--subject", "Paris is the capital",
            "--target", "Lyon",
            "--registry-id", "x" * 300,
            "--plan-only",
        ])
        assert result.exit_code == 2
