"""#1710 — Remove deprecated AWQ and GPTQ export and quantize formats.

Follow-up to #338 and #1690. AWQ and GPTQ export formats and extras are removed
one release after the deprecation notice. The CLI refuses both formats with a
non-zero exit code and names the formats that stay: gguf, onnx, tensorrt, bitnet, tq1_0.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from typer.testing import CliRunner

import soup_cli.commands.export as export_mod
import soup_cli.commands.quantize as quantize_mod
from soup_cli.cli import app
from soup_cli.config import deprecation

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_BOX_RE = re.compile(r"[\u2500-\u257f]")

REMOVED = ("awq", "gptq")
STAYING_EXPORT = ("gguf", "onnx", "tensorrt", "bitnet", "tq1_0")
VALID_QUANTIZE = ("gguf", "onnx", "tensorrt")


def _plain(text: str) -> str:
    """ANSI- and border-stripped, whitespace-collapsed CLI output."""
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


class TestExportRefusesRemovedFormats:
    @pytest.mark.parametrize("fmt", REMOVED)
    def test_export_refuses_removed_format(self, fmt: str, tmp_path: pathlib.Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        result = CliRunner().invoke(app, ["export", "--model", str(model_dir), "--format", fmt])

        assert result.exit_code == 1, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert f"unsupported format: {fmt}" in output.lower()
        assert f"--format {fmt} was removed in v0.77." in output
        for remaining in STAYING_EXPORT:
            assert remaining in output

    @pytest.mark.parametrize("fmt", REMOVED)
    def test_export_refusal_names_the_removal_release(
        self, fmt: str, tmp_path: pathlib.Path
    ) -> None:
        """The sentence names the release that removed the format, in full."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        result = CliRunner().invoke(app, ["export", "--model", str(model_dir), "--format", fmt])

        assert result.exit_code == 1, (result.output, repr(result.exception))
        assert f"--format {fmt} was removed in v0.77.0." in _plain(result.output)

    @pytest.mark.parametrize("fmt", REMOVED)
    def test_a_later_deprecation_deadline_does_not_move_the_removal(
        self, fmt: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The next deprecation moves ``DEPRECATED_VALUE_REJECTION_VERSION``; the removal
        stays dated to the release that made it."""
        monkeypatch.setattr(deprecation, "DEPRECATED_VALUE_REJECTION_VERSION", "0.78")
        monkeypatch.setattr(export_mod, "DEPRECATED_VALUE_REJECTION_VERSION", "0.78", raising=False)
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        result = CliRunner().invoke(app, ["export", "--model", str(model_dir), "--format", fmt])

        assert result.exit_code == 1, (result.output, repr(result.exception))
        output = _plain(result.output)
        assert f"--format {fmt} was removed in v0.77.0." in output
        assert "0.78" not in output

    @pytest.mark.parametrize("fmt", REMOVED)
    def test_export_refusal_does_not_mention_deprecation(
        self, fmt: str, tmp_path: pathlib.Path
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        result = CliRunner().invoke(app, ["export", "--model", str(model_dir), "--format", fmt])

        output = _plain(result.output).lower()
        assert "deprecated" not in output

    def test_export_help_omits_removed_formats(self) -> None:
        result = CliRunner().invoke(app, ["export", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        output = _plain(result.output).lower()
        assert "awq and gptq are deprecated" not in output
        assert "--format awq" not in output
        assert "--format gptq" not in output
        assert "--bits" not in output
        assert "--group-size" not in output
        assert "--calibration-samples" not in output


class TestQuantizeRefusesRemovedFormats:
    @pytest.mark.parametrize("fmt", REMOVED)
    def test_quantize_refuses_removed_format(self, fmt: str) -> None:
        result = CliRunner().invoke(app, ["quantize", "./out", "--to", fmt])

        assert result.exit_code == 2
        output = _plain(result.output)
        assert "--to must be one of" in output
        for valid in VALID_QUANTIZE:
            assert valid in output
        assert "deprecated" not in output.lower()

    @pytest.mark.parametrize("fmt", REMOVED)
    def test_quantize_refuses_uppercase_removed_format(self, fmt: str) -> None:
        result = CliRunner().invoke(app, ["quantize", "./out", "--to", fmt.upper()])

        assert result.exit_code == 2
        output = _plain(result.output)
        assert "--to must be one of" in output

    def test_quantize_help_omits_removed_formats(self) -> None:
        result = CliRunner().invoke(app, ["quantize", "--help"])
        assert result.exit_code == 0
        output = _plain(result.output).lower()
        assert "awq and gptq are deprecated" not in output
        assert "gptq | awq" not in output


class TestCodebaseRemovals:
    def test_supported_formats_excludes_awq_and_gptq(self) -> None:
        for fmt in REMOVED:
            assert fmt not in export_mod.SUPPORTED_FORMATS

    def test_valid_formats_excludes_awq_and_gptq(self) -> None:
        for fmt in REMOVED:
            assert fmt not in quantize_mod.VALID_FORMATS

    def test_deprecated_helpers_and_tables_removed(self) -> None:
        assert not hasattr(export_mod, "DEPRECATED_FORMATS")
        assert not hasattr(export_mod, "REPLACEMENT_FORMATS")
        assert not hasattr(export_mod, "deprecated_format_notice")
        assert not hasattr(export_mod, "_warn_deprecated_format")
        assert not hasattr(export_mod, "_export_awq")
        assert not hasattr(export_mod, "_export_gptq")
        assert not hasattr(export_mod, "_standardize_gptq_shard_name")
        assert not hasattr(export_mod, "_validate_output_path")
        assert not hasattr(export_mod, "_validate_calibration_path")


class TestPackagingRemovals:
    def test_pyproject_excludes_awq_and_gptq_extras(self) -> None:
        text = PYPROJECT.read_text(encoding="utf-8")
        table = re.search(r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[)", text, re.M | re.S)
        assert table is not None
        code = "\n".join(
            line for line in table.group(1).splitlines() if not line.lstrip().startswith("#")
        )
        extras = re.findall(r"^([A-Za-z0-9_-]+)\s*=", code, re.M)
        for fmt in REMOVED:
            assert fmt not in extras
            assert f"auto{fmt}" not in code
