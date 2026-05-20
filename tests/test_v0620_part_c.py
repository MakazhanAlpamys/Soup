"""Tests for v0.62.0 Part C — `soup steer` (CAA / ITI / RepE control vectors).

Schema-only release: validators + frozen dataclass + CLI surface ship now,
live forward-hook + decode-time intervention land in v0.62.1.
"""

from __future__ import annotations

import dataclasses

import pytest

# ---------- Module surface ----------


class TestModuleSurface:
    def test_imports(self):
        from soup_cli.utils.steering import (
            SUPPORTED_STEERING_METHODS,
            SteeringMethodSpec,
            apply_steering,
            build_steering_vector,
            get_steering_method_spec,
            validate_steering_method,
        )
        assert callable(validate_steering_method)
        assert callable(get_steering_method_spec)
        assert callable(apply_steering)
        assert callable(build_steering_vector)
        assert dataclasses.is_dataclass(SteeringMethodSpec)
        assert isinstance(SUPPORTED_STEERING_METHODS, frozenset)

    def test_methods_exact(self):
        from soup_cli.utils.steering import SUPPORTED_STEERING_METHODS

        assert SUPPORTED_STEERING_METHODS == frozenset({"caa", "iti", "repe"})

    def test_metadata_mapping_proxy(self):
        from types import MappingProxyType

        from soup_cli.utils.steering import _STEERING_METHOD_METADATA  # type: ignore

        assert isinstance(_STEERING_METHOD_METADATA, MappingProxyType)


# ---------- validate_steering_method ----------


class TestValidateMethod:
    def test_happy(self):
        from soup_cli.utils.steering import validate_steering_method

        for name in ("caa", "iti", "repe"):
            assert validate_steering_method(name) == name

    def test_case_insensitive(self):
        from soup_cli.utils.steering import validate_steering_method

        assert validate_steering_method("CAA") == "caa"
        assert validate_steering_method("ItI") == "iti"

    def test_bool_rejected(self):
        from soup_cli.utils.steering import validate_steering_method

        with pytest.raises(TypeError):
            validate_steering_method(True)

    def test_non_string_rejected(self):
        from soup_cli.utils.steering import validate_steering_method

        with pytest.raises(TypeError):
            validate_steering_method(42)

    def test_empty_rejected(self):
        from soup_cli.utils.steering import validate_steering_method

        with pytest.raises(ValueError):
            validate_steering_method("")

    def test_null_byte_rejected(self):
        from soup_cli.utils.steering import validate_steering_method

        with pytest.raises(ValueError):
            validate_steering_method("caa\x00")

    def test_oversize_rejected(self):
        from soup_cli.utils.steering import validate_steering_method

        with pytest.raises(ValueError):
            validate_steering_method("c" * 100)

    def test_unknown_rejected(self):
        from soup_cli.utils.steering import validate_steering_method

        with pytest.raises(ValueError, match="steering"):
            validate_steering_method("nonsense")


# ---------- validate_steering_name ----------


class TestValidateName:
    def test_happy(self):
        from soup_cli.utils.steering import validate_steering_name

        for name in ("safety-v1", "helpfulness_2024", "tone-formal"):
            assert validate_steering_name(name) == name

    def test_bool_rejected(self):
        from soup_cli.utils.steering import validate_steering_name

        with pytest.raises(TypeError):
            validate_steering_name(True)

    def test_non_string_rejected(self):
        from soup_cli.utils.steering import validate_steering_name

        with pytest.raises(TypeError):
            validate_steering_name(1)

    def test_empty_rejected(self):
        from soup_cli.utils.steering import validate_steering_name

        with pytest.raises(ValueError):
            validate_steering_name("")

    def test_null_byte_rejected(self):
        from soup_cli.utils.steering import validate_steering_name

        with pytest.raises(ValueError):
            validate_steering_name("safety\x00")

    def test_oversize_rejected(self):
        from soup_cli.utils.steering import validate_steering_name

        with pytest.raises(ValueError):
            validate_steering_name("x" * 200)

    def test_invalid_chars_rejected(self):
        from soup_cli.utils.steering import validate_steering_name

        # Path separators / spaces / shell metacharacters rejected so
        # the name can be safely embedded in CLI args and filenames.
        for bad in ("foo/bar", "foo bar", "foo;rm", "foo$x", "../escape"):
            with pytest.raises(ValueError):
                validate_steering_name(bad)


# ---------- validate_steering_strength ----------


class TestValidateStrength:
    def test_happy(self):
        from soup_cli.utils.steering import validate_steering_strength

        assert validate_steering_strength(0.5) == 0.5
        assert validate_steering_strength(-1.0) == -1.0
        assert validate_steering_strength(0.0) == 0.0

    def test_int_coerced(self):
        from soup_cli.utils.steering import validate_steering_strength

        assert validate_steering_strength(1) == 1.0

    def test_bool_rejected(self):
        from soup_cli.utils.steering import validate_steering_strength

        with pytest.raises(TypeError):
            validate_steering_strength(True)

    def test_non_finite_rejected(self):
        import math

        from soup_cli.utils.steering import validate_steering_strength

        with pytest.raises(ValueError):
            validate_steering_strength(math.nan)
        with pytest.raises(ValueError):
            validate_steering_strength(math.inf)

    def test_out_of_bounds_rejected(self):
        from soup_cli.utils.steering import validate_steering_strength

        with pytest.raises(ValueError):
            validate_steering_strength(100.0)
        with pytest.raises(ValueError):
            validate_steering_strength(-100.0)


# ---------- get_steering_method_spec ----------


class TestSpec:
    def test_caa_spec(self):
        from soup_cli.utils.steering import get_steering_method_spec

        spec = get_steering_method_spec("caa")
        assert spec.name == "caa"
        assert spec.live_wired is False
        assert spec.needs_contrastive_pairs is True

    def test_iti_spec(self):
        from soup_cli.utils.steering import get_steering_method_spec

        spec = get_steering_method_spec("iti")
        assert spec.name == "iti"
        assert spec.needs_attention_heads is True

    def test_repe_spec(self):
        from soup_cli.utils.steering import get_steering_method_spec

        spec = get_steering_method_spec("repe")
        assert spec.name == "repe"

    def test_unknown_raises(self):
        from soup_cli.utils.steering import get_steering_method_spec

        with pytest.raises(ValueError):
            get_steering_method_spec("nonsense")

    def test_frozen(self):
        from soup_cli.utils.steering import get_steering_method_spec

        spec = get_steering_method_spec("caa")
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "mutated"  # type: ignore[misc]


# ---------- Deferred-live stubs ----------


class TestDeferredStubs:
    def test_apply_steering_deferred(self):
        from soup_cli.utils.steering import apply_steering

        with pytest.raises(NotImplementedError, match="v0.62.1"):
            apply_steering("caa")

    def test_apply_steering_validates_first(self):
        from soup_cli.utils.steering import apply_steering

        # Unknown method rejected BEFORE the deferred-live raise.
        with pytest.raises(ValueError):
            apply_steering("nonsense")

    def test_build_steering_vector_deferred(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(NotImplementedError, match="v0.62.1"):
            build_steering_vector(method="caa", name="safety-v1")

    def test_build_steering_vector_validates_method_first(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(ValueError):
            build_steering_vector(method="nonsense", name="safety-v1")

    def test_build_steering_vector_validates_name_first(self):
        from soup_cli.utils.steering import build_steering_vector

        with pytest.raises(ValueError):
            build_steering_vector(method="caa", name="bad/path")


# ---------- Registry artifact kind ----------


class TestRegistryArtifactKind:
    def test_steering_vector_in_valid_kinds(self):
        from soup_cli.registry.store import _VALID_KINDS

        assert "steering_vector" in _VALID_KINDS


# ---------- CLI plumbing ----------


class TestCLI:
    def test_cli_help_lists_steer(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "steer" in result.stdout.lower()

    def test_steer_help(self):
        from typer.testing import CliRunner

        from soup_cli.commands.steer import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_steer_train_help(self):
        from typer.testing import CliRunner

        from soup_cli.commands.steer import app

        runner = CliRunner()
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0

    def test_steer_apply_help(self):
        from typer.testing import CliRunner

        from soup_cli.commands.steer import app

        runner = CliRunner()
        result = runner.invoke(app, ["apply", "--help"])
        assert result.exit_code == 0

    def test_steer_list_help(self):
        from typer.testing import CliRunner

        from soup_cli.commands.steer import app

        runner = CliRunner()
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0

    def test_steer_train_unknown_method_rejected(self):
        from typer.testing import CliRunner

        from soup_cli.commands.steer import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "train",
            "--base", "meta-llama/Llama-3.1-8B-Instruct",
            "--method", "nonsense",
            "--name", "safety-v1",
            "--pairs", "./data/pairs.jsonl",
            "--plan-only",
        ])
        # CLI either exits 2 (validation) or 1 (plan rejected); not 0.
        assert result.exit_code != 0

    def test_steer_train_plan_only(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from soup_cli.commands.steer import app

        monkeypatch.chdir(tmp_path)
        # Create a tiny pairs JSONL to pass the path-containment check.
        pairs = tmp_path / "pairs.jsonl"
        pairs.write_text(
            '{"positive": "be safe", "negative": "be harmful"}\n', encoding="utf-8"
        )
        runner = CliRunner()
        result = runner.invoke(app, [
            "train",
            "--base", "meta-llama/Llama-3.1-8B-Instruct",
            "--method", "caa",
            "--name", "safety-v1",
            "--pairs", "pairs.jsonl",
            "--plan-only",
        ])
        # plan-only succeeds with friendly deferred-live panel.
        assert result.exit_code == 0
        assert "v0.62" in result.stdout or "caa" in result.stdout.lower()


# ---------- soup serve --steer flag plumbing ----------


class TestServeSteerFlag:
    def test_serve_help_mentions_steer(self):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        # Even if the flag is not yet alive, --help should expose it so a
        # YAML / shell pipeline can reference it.
        assert result.exit_code == 0
        assert "steer" in result.stdout.lower()
