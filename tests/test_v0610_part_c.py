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

        # v0.62.0 Part E added "grace" to the allowlist (codebook edit).
        assert SUPPORTED_EDIT_METHODS == frozenset(
            {"rome", "memit", "alphaedit", "grace"}
        )


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
    def test_live_dispatch(self, monkeypatch):
        # v0.71.9 #194 — apply_edit now runs the live kernel (mocked here).
        import soup_cli.utils.edit_kernels as ek
        import soup_cli.utils.live_eval as live_eval
        from soup_cli.utils.knowledge_edit import EditResult, apply_edit, build_edit_plan

        plan = build_edit_plan(
            base="b", method="rome", subject="s", target="t",
        )
        monkeypatch.setattr(
            live_eval, "load_model_and_tokenizer",
            lambda *a, **k: ("M", "T", "cpu"),
        )
        monkeypatch.setattr(ek, "measure_target_prob", lambda *a, **k: 0.0)
        monkeypatch.setattr(
            ek, "run_edit_kernel",
            lambda *a, **k: ek.EditKernelResult(
                method="rome", layer=5, norm_delta=0.3, layers_edited=(5,),
            ),
        )
        result = apply_edit(plan)
        assert isinstance(result, EditResult)
        assert result.method == "rome"

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

    def test_edit_set_apply_failure_exit2(self, monkeypatch):
        # v0.71.9 #194 — apply_edit is live; a load/edit failure surfaces a
        # friendly "Edit failed" error with exit 2 (NOT an unhandled crash).
        import soup_cli.utils.knowledge_edit as ke

        monkeypatch.setattr(
            ke, "apply_edit",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom-load")),
        )
        runner = CliRunner()
        result = runner.invoke(app, [
            "edit", "set",
            "--base", "test",
            "--method", "rome",
            "--subject", "Paris is the capital",
            "--target", "Lyon",
            "--no-governor",
            "--device", "cpu",
        ])
        assert result.exit_code == 2, result.output
        assert "failed" in result.output.lower()

    def test_edit_set_apply_success(self, monkeypatch):
        # v0.71.9 #194 — successful live edit renders the result panel + exit 0.
        import soup_cli.utils.knowledge_edit as ke
        from soup_cli.utils.knowledge_edit import EditResult

        monkeypatch.setattr(
            ke, "apply_edit",
            lambda *a, **k: EditResult(
                method="rome", layer=5, norm_delta=0.42, layers_edited=(5,),
                output_dir=None, target_prob_before=0.01, target_prob_after=0.9,
                governed=False,
            ),
        )
        runner = CliRunner()
        result = runner.invoke(app, [
            "edit", "set",
            "--base", "test",
            "--method", "rome",
            "--subject", "Paris is the capital",
            "--target", "Lyon",
            "--no-governor",
        ])
        assert result.exit_code == 0, result.output
        assert "applied" in result.output.lower()

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
