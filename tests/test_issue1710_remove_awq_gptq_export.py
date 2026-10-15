"""Tests for #1710 — Removal of deprecated AWQ and GPTQ export and quantize formats.

AWQ and GPTQ export are removed one release after the #338 deprecation notice.
Both formats are now refused with a non-zero exit code naming the remaining
supported formats (gguf, onnx, tensorrt, bitnet, tq1_0), their helper functions
are gone, and the [awq] and [gptq] extras are removed from pyproject.toml.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.export import SUPPORTED_FORMATS
from soup_cli.commands.quantize import VALID_FORMATS

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX_RE = re.compile(r"[\u2500-\u257f]")

REMOVED = ("awq", "gptq")
REMAINING_FORMATS = ("gguf", "onnx", "tensorrt", "bitnet", "tq1_0")


def _plain(text: str) -> str:
    """ANSI- and border-stripped, whitespace-collapsed CLI output."""
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


class TestExportRefusesRemovedFormats:
    @pytest.mark.parametrize("fmt", REMOVED)
    def test_format_not_in_supported_formats(self, fmt: str) -> None:
        assert fmt not in SUPPORTED_FORMATS

    def test_remaining_formats_stay_supported(self) -> None:
        for fmt in REMAINING_FORMATS:
            assert fmt in SUPPORTED_FORMATS

    @pytest.mark.parametrize("fmt", REMOVED)
    def test_export_cli_refuses_format_and_names_remaining(
        self, fmt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FORCE_COLOR", "1")
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(app, ["export", "--model", str(model_dir), "--format", fmt])

        assert result.exit_code != 0
        plain = _plain(result.output)
        assert f"unsupported format: {fmt}" in plain.lower()
        for remaining in REMAINING_FORMATS:
            assert remaining in plain

    def test_helpers_removed_from_export_module(self) -> None:
        import soup_cli.commands.export as export_mod

        assert not hasattr(export_mod, "_export_awq")
        assert not hasattr(export_mod, "_export_gptq")
        assert not hasattr(export_mod, "DEPRECATED_FORMATS")
        assert not hasattr(export_mod, "deprecated_format_notice")


class TestQuantizeRefusesRemovedFormats:
    @pytest.mark.parametrize("fmt", REMOVED)
    def test_format_not_in_valid_formats(self, fmt: str) -> None:
        assert fmt not in VALID_FORMATS

    @pytest.mark.parametrize("fmt", REMOVED)
    def test_quantize_cli_refuses_format(
        self, fmt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FORCE_COLOR", "1")
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(app, ["quantize", str(model_dir), "--to", fmt])

        assert result.exit_code != 0
        plain = _plain(result.output)
        assert "--to must be one of" in plain
        allowed_list = plain.split("must be one of")[1].split(";")[0]
        assert fmt not in allowed_list
        for remaining in ("gguf", "onnx", "tensorrt"):
            assert remaining in allowed_list


class TestPyprojectExtrasRemoved:
    def test_awq_and_gptq_extras_not_in_pyproject(self) -> None:
        pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")
        match = re.search(
            r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[|\Z)",
            pyproject_text,
            re.M | re.S,
        )
        assert match is not None
        extras_block = match.group(1)
        for line in extras_block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            extra_name = line.split("=")[0].strip()
            assert extra_name not in REMOVED, f"Extra {extra_name} still found in pyproject.toml"
