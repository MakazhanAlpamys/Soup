"""v0.69.0 Part C — `soup data gen magpie` synthetic generator."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import magpie


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Provider allowlist
# -----------------------------------------------------------------------------


class TestSupportedProviders:
    def test_exact(self) -> None:
        assert magpie.SUPPORTED_MAGPIE_PROVIDERS == frozenset(
            {"ollama", "anthropic", "vllm"}
        )

    def test_immutable(self) -> None:
        with pytest.raises(AttributeError):
            magpie.SUPPORTED_MAGPIE_PROVIDERS.add("evil")  # type: ignore[attr-defined]


class TestValidateProvider:
    def test_happy(self) -> None:
        assert magpie.validate_magpie_provider("ollama") == "ollama"
        assert magpie.validate_magpie_provider("ANTHROPIC") == "anthropic"

    def test_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown magpie provider"):
            magpie.validate_magpie_provider("openai")

    def test_non_string(self) -> None:
        with pytest.raises(TypeError):
            magpie.validate_magpie_provider(42)

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            magpie.validate_magpie_provider(True)

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            magpie.validate_magpie_provider("")

    def test_null_byte(self) -> None:
        with pytest.raises(ValueError, match="null"):
            magpie.validate_magpie_provider("ollama\x00")


# -----------------------------------------------------------------------------
# Target rows + base model validators
# -----------------------------------------------------------------------------


class TestValidateTargetRows:
    def test_happy(self) -> None:
        assert magpie.validate_target_rows(100) == 100

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            magpie.validate_target_rows(0)

    def test_negative(self) -> None:
        with pytest.raises(ValueError):
            magpie.validate_target_rows(-5)

    def test_overcap(self) -> None:
        with pytest.raises(ValueError):
            magpie.validate_target_rows(magpie._MAX_TARGET_ROWS + 1)

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            magpie.validate_target_rows(True)

    def test_non_int(self) -> None:
        with pytest.raises(TypeError):
            magpie.validate_target_rows(1.5)


class TestValidateBaseModel:
    def test_happy(self) -> None:
        assert (
            magpie.validate_base_model("meta-llama/Llama-3.1-8B-Instruct")
            == "meta-llama/Llama-3.1-8B-Instruct"
        )

    def test_empty(self) -> None:
        with pytest.raises(ValueError):
            magpie.validate_base_model("")

    def test_null_byte(self) -> None:
        with pytest.raises(ValueError, match="null"):
            magpie.validate_base_model("meta\x00llama")

    def test_oversize(self) -> None:
        with pytest.raises(ValueError):
            magpie.validate_base_model("x" * 1024)

    def test_non_string(self) -> None:
        with pytest.raises(TypeError):
            magpie.validate_base_model(42)

    def test_bool_rejected(self) -> None:
        with pytest.raises(TypeError):
            magpie.validate_base_model(True)


# -----------------------------------------------------------------------------
# MagpieConfig frozen dataclass
# -----------------------------------------------------------------------------


class TestMagpieConfig:
    def test_happy(self) -> None:
        cfg = magpie.MagpieConfig(
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            provider="ollama",
            target_rows=100,
            quality_filter=True,
        )
        assert cfg.target_rows == 100
        assert cfg.quality_filter is True

    def test_frozen(self) -> None:
        cfg = magpie.MagpieConfig(
            base_model="m",
            provider="ollama",
            target_rows=10,
            quality_filter=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.target_rows = 999  # type: ignore[misc]

    def test_invalid_provider_propagates(self) -> None:
        with pytest.raises(ValueError):
            magpie.MagpieConfig(
                base_model="m",
                provider="openai",
                target_rows=10,
                quality_filter=False,
            )

    def test_invalid_target_propagates(self) -> None:
        with pytest.raises(ValueError):
            magpie.MagpieConfig(
                base_model="m",
                provider="ollama",
                target_rows=0,
                quality_filter=False,
            )

    def test_invalid_base_propagates(self) -> None:
        with pytest.raises(ValueError):
            magpie.MagpieConfig(
                base_model="",
                provider="ollama",
                target_rows=10,
                quality_filter=False,
            )

    def test_quality_filter_must_be_bool(self) -> None:
        with pytest.raises(TypeError):
            magpie.MagpieConfig(
                base_model="m",
                provider="ollama",
                target_rows=10,
                quality_filter="yes",  # type: ignore[arg-type]
            )


# -----------------------------------------------------------------------------
# build_magpie_config factory
# -----------------------------------------------------------------------------


class TestBuildMagpieConfig:
    def test_happy(self) -> None:
        cfg = magpie.build_magpie_config(
            base="meta-llama/Llama-3.1-8B-Instruct",
            provider="OLLAMA",
            target=50,
        )
        assert cfg.provider == "ollama"
        assert cfg.target_rows == 50
        assert cfg.quality_filter is True  # default

    def test_quality_filter_off(self) -> None:
        cfg = magpie.build_magpie_config(
            base="m",
            provider="ollama",
            target=10,
            quality_filter=False,
        )
        assert cfg.quality_filter is False


# -----------------------------------------------------------------------------
# Deferred live runner
# -----------------------------------------------------------------------------


class TestRunMagpie:
    def test_live_runner_writes(self, tmp_path, monkeypatch) -> None:
        # v0.71.6 #232: run_magpie is now live (was a v0.69.1 deferred stub).
        monkeypatch.chdir(tmp_path)
        cfg = magpie.MagpieConfig(
            base_model="m",
            provider="ollama",
            target_rows=1,
            quality_filter=False,
        )
        result = magpie.run_magpie(
            cfg,
            output_path="out.jsonl",
            generate_fn=lambda prompt: (
                "Answer.<|im_end|>" if "assistant" in prompt else "Question?<|im_end|>"
            ),
        )
        assert result.rows_kept == 1
        assert (tmp_path / "out.jsonl").is_file()

    def test_validates_config_type(self) -> None:
        with pytest.raises(TypeError):
            magpie.run_magpie({"base": "m"}, output_path="out.jsonl")  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# CLI: `soup data gen magpie`
# -----------------------------------------------------------------------------


class TestMagpieCli:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(app, ["data", "gen-magpie", "--help"])
        assert result.exit_code == 0, result.output
        assert "magpie" in result.output.lower()

    def test_plan_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "gen-magpie",
                "--base",
                "meta-llama/Llama-3.1-8B-Instruct",
                "--provider",
                "ollama",
                "--target",
                "10",
                "--plan-only",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ollama" in result.output.lower() or "magpie" in result.output.lower()

    def test_unknown_provider(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "gen-magpie",
                "--base",
                "m",
                "--provider",
                "openai",
                "--target",
                "10",
                "--plan-only",
            ],
        )
        assert result.exit_code == 2

    def test_invalid_target(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "gen-magpie",
                "--base",
                "m",
                "--provider",
                "ollama",
                "--target",
                "0",
                "--plan-only",
            ],
        )
        assert result.exit_code == 2

    def test_live_requires_output(self) -> None:
        # v0.71.6 #232: live run now needs --output (no more deferred exit-3).
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "gen-magpie",
                "--base",
                "m",
                "--provider",
                "ollama",
                "--target",
                "10",
            ],
        )
        assert result.exit_code == 2
        assert "0.69.1" not in result.output


# -----------------------------------------------------------------------------
# Source wiring
# -----------------------------------------------------------------------------


class TestSourceWiring:
    def test_no_heavy_imports(self) -> None:
        root = Path(__file__).resolve().parent.parent
        src = (root / "src" / "soup_cli" / "utils" / "magpie.py").read_text(encoding="utf-8")
        for forbidden in (
            "\nimport torch",
            "\nimport transformers",
            "\nimport anthropic",
        ):
            assert forbidden not in src

    def test_cli_registered(self) -> None:
        root = Path(__file__).resolve().parent.parent
        cli = (root / "src" / "soup_cli" / "commands" / "data.py").read_text(encoding="utf-8")
        assert "magpie" in cli.lower() or "gen-magpie" in cli

    def test_version_bumped(self) -> None:
        from soup_cli import __version__

        major_minor = tuple(int(x) for x in __version__.split(".")[:2])
        assert major_minor >= (0, 69)
