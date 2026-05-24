"""v0.68.0 Part D — ``soup apple-adapter``.

HF / PEFT <-> MLX <-> Apple Foundation Models adapter conversion + signing.
Extends v0.25 MLX backend, reuses v0.60 Part B Merkle-root signing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from typer.testing import CliRunner


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import apple_adapter

        assert hasattr(apple_adapter, "SUPPORTED_ADAPTER_DIRECTIONS")
        assert hasattr(apple_adapter, "validate_direction")
        assert hasattr(apple_adapter, "validate_source_adapter")
        assert hasattr(apple_adapter, "AppleAdapterPlan")
        assert hasattr(apple_adapter, "build_apple_adapter_plan")
        assert hasattr(apple_adapter, "convert_apple_adapter")


class TestAllowlist:
    def test_frozenset(self) -> None:
        from soup_cli.utils.apple_adapter import SUPPORTED_ADAPTER_DIRECTIONS

        assert isinstance(SUPPORTED_ADAPTER_DIRECTIONS, frozenset)
        assert "hf-to-mlx" in SUPPORTED_ADAPTER_DIRECTIONS
        assert "mlx-to-hf" in SUPPORTED_ADAPTER_DIRECTIONS
        assert "hf-to-apple" in SUPPORTED_ADAPTER_DIRECTIONS
        assert "mlx-to-apple" in SUPPORTED_ADAPTER_DIRECTIONS

    def test_immutable(self) -> None:
        from soup_cli.utils.apple_adapter import SUPPORTED_ADAPTER_DIRECTIONS

        with pytest.raises(AttributeError):
            SUPPORTED_ADAPTER_DIRECTIONS.add("x")  # type: ignore[attr-defined]


class TestValidateDirection:
    def test_happy(self) -> None:
        from soup_cli.utils.apple_adapter import validate_direction

        assert validate_direction("hf-to-mlx") == "hf-to-mlx"

    def test_case_insensitive(self) -> None:
        from soup_cli.utils.apple_adapter import validate_direction

        assert validate_direction("HF-TO-MLX") == "hf-to-mlx"

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.apple_adapter import validate_direction

        with pytest.raises(TypeError):
            validate_direction(True)  # type: ignore[arg-type]

    def test_unknown_rejected(self) -> None:
        from soup_cli.utils.apple_adapter import validate_direction

        with pytest.raises(ValueError, match="unknown"):
            validate_direction("evil")

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.apple_adapter import validate_direction

        with pytest.raises(ValueError):
            validate_direction("hf-to-mlx\x00")

    def test_empty_rejected(self) -> None:
        from soup_cli.utils.apple_adapter import validate_direction

        with pytest.raises(ValueError):
            validate_direction("")


class TestValidateSourceAdapter:
    def test_happy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from soup_cli.utils.apple_adapter import validate_source_adapter

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        assert validate_source_adapter(str(adapter)).endswith("adapter")

    def test_outside_cwd_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.apple_adapter import validate_source_adapter

        outside = tmp_path / "outside"
        outside.mkdir()
        adapter = outside / "ad"
        adapter.mkdir()
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError):
            validate_source_adapter(str(adapter))

    def test_non_directory_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.apple_adapter import validate_source_adapter

        monkeypatch.chdir(tmp_path)
        f = tmp_path / "not_a_dir.json"
        f.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="directory"):
            validate_source_adapter(str(f))

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.apple_adapter import validate_source_adapter

        with pytest.raises(ValueError):
            validate_source_adapter("source\x00")


class TestAppleAdapterPlan:
    def test_frozen(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.apple_adapter import AppleAdapterPlan

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

        plan = AppleAdapterPlan(
            source_dir=str(adapter),
            output_dir="out",
            direction="hf-to-mlx",
            sign=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.sign = True  # type: ignore[misc]

    def test_invalid_direction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.apple_adapter import AppleAdapterPlan

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError):
            AppleAdapterPlan(
                source_dir=str(adapter),
                output_dir="out",
                direction="evil",
                sign=False,
            )

    def test_sign_must_be_bool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.apple_adapter import AppleAdapterPlan

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(TypeError):
            AppleAdapterPlan(
                source_dir=str(adapter),
                output_dir="out",
                direction="hf-to-mlx",
                sign="yes",  # type: ignore[arg-type]
            )


class TestConvertDeferred:
    def test_raises_v068_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        plan = build_apple_adapter_plan(
            source_dir=str(adapter),
            output_dir="out",
            direction="hf-to-mlx",
            sign=False,
        )
        with pytest.raises(NotImplementedError, match="v0.68.1"):
            convert_apple_adapter(plan)

    def test_non_plan_rejected(self) -> None:
        from soup_cli.utils.apple_adapter import convert_apple_adapter

        with pytest.raises(TypeError):
            convert_apple_adapter({})  # type: ignore[arg-type]


class TestCli:
    def test_help(self) -> None:
        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["apple-adapter", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_plan_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "apple-adapter",
                str(adapter),
                "--direction",
                "hf-to-mlx",
                "--output",
                "out",
                "--plan-only",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_unknown_direction_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "apple-adapter",
                str(adapter),
                "--direction",
                "evil",
                "--output",
                "out",
                "--plan-only",
            ],
        )
        assert result.exit_code == 2

    def test_live_exits_3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "apple-adapter",
                str(adapter),
                "--direction",
                "hf-to-mlx",
                "--output",
                "out",
            ],
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        path = (
            Path(__file__).resolve().parent.parent
            / "soup_cli"
            / "utils"
            / "apple_adapter.py"
        )
        text = path.read_text(encoding="utf-8")
        for token in (
            "\nimport torch",
            "\nimport transformers",
            "\nimport mlx",
            "\nimport safetensors",
        ):
            assert token not in text

    def test_cli_registered(self) -> None:
        from soup_cli.cli import app

        names = [c.name for c in app.registered_commands]
        assert "apple-adapter" in names
