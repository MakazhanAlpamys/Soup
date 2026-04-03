"""Tests for ONNX and TensorRT-LLM export — config, validation, CLI."""

from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

from soup_cli.commands.export import SUPPORTED_FORMATS

# ─── Format Support Tests ────────────────────────────────────────────────


class TestExportFormats:
    """Test that new export formats are registered."""

    def test_gguf_format_supported(self):
        assert "gguf" in SUPPORTED_FORMATS

    def test_onnx_format_supported(self):
        assert "onnx" in SUPPORTED_FORMATS

    def test_tensorrt_format_supported(self):
        assert "tensorrt" in SUPPORTED_FORMATS

    def test_format_count(self):
        """Should support exactly 5 export formats (gguf, onnx, tensorrt, awq, gptq)."""
        assert len(SUPPORTED_FORMATS) == 5


# ─── ONNX Export CLI Tests ──────────────────────────────────────────────


class TestOnnxExportCLI:
    """Test ONNX export via CLI."""

    def test_onnx_export_missing_model_path(self, tmp_path):
        """soup export --format onnx should fail if model path doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["export", "--model", str(tmp_path / "nonexistent"), "--format", "onnx"]
        )
        assert result.exit_code != 0

    def test_onnx_export_format_in_help(self):
        """Export help should mention onnx format."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["export", "--help"])
        assert "onnx" in result.output.lower()

    def test_tensorrt_export_format_in_help(self):
        """Export help should mention tensorrt format."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["export", "--help"])
        assert "tensorrt" in result.output.lower()


# ─── ONNX Export Function Tests ─────────────────────────────────────────


class TestOnnxExportFunction:
    """Test _export_onnx logic."""

    def test_export_onnx_calls_main_export(self, tmp_path):
        """_export_onnx should call optimum's main_export."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_main_export = MagicMock()
        with mock_patch(
            "soup_cli.commands.export.main_export",
            mock_main_export,
            create=True,
        ):
            # Need to patch the import inside the function
            import soup_cli.commands.export as export_mod
            original = getattr(export_mod, "main_export", None)
            try:
                # Inject mock at module level for the lazy import
                with mock_patch.object(
                    export_mod, "_export_onnx",
                    wraps=export_mod._export_onnx,
                ):
                    # Patch the actual import
                    with mock_patch.dict("sys.modules", {
                        "optimum": MagicMock(),
                        "optimum.exporters": MagicMock(),
                        "optimum.exporters.onnx": MagicMock(
                            main_export=mock_main_export
                        ),
                    }):
                        export_mod._export_onnx(model_dir, str(tmp_path / "out"), None)
                        mock_main_export.assert_called_once()
            finally:
                if original is not None:
                    export_mod.main_export = original

    def test_export_onnx_default_output_path(self, tmp_path):
        """Default output path should be model_name + _onnx suffix."""
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()

        mock_main_export = MagicMock()
        with mock_patch.dict("sys.modules", {
            "optimum": MagicMock(),
            "optimum.exporters": MagicMock(),
            "optimum.exporters.onnx": MagicMock(main_export=mock_main_export),
        }):
            import soup_cli.commands.export as export_mod

            export_mod._export_onnx(model_dir, None, None)
            call_kwargs = mock_main_export.call_args[1]
            assert "my_model_onnx" in call_kwargs["output"]


# ─── TensorRT Export Function Tests ──────────────────────────────────────


class TestTensorrtExportFunction:
    """Test _export_tensorrt logic."""

    def test_tensorrt_export_calls_subprocess(self, tmp_path):
        """TensorRT export should call subprocess for checkpoint conversion."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with mock_patch.dict("sys.modules", {
            "optimum": MagicMock(),
            "optimum.exporters": MagicMock(),
            "optimum.exporters.onnx": MagicMock(),
            "tensorrt_llm": MagicMock(),
        }), mock_patch("subprocess.run", return_value=mock_result) as mock_run:
            import soup_cli.commands.export as export_mod

            export_mod._export_tensorrt(
                model_dir, str(tmp_path / "trt_out"), None
            )
            # Should call subprocess at least twice (checkpoint + build)
            assert mock_run.call_count >= 2


# ─── Unsupported Format Test ─────────────────────────────────────────────


class TestExportUnsupportedFormat:
    """Test that unsupported formats are rejected."""

    def test_invalid_format_rejected(self, tmp_path):
        """Export with unsupported format should exit with error."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["export", "--model", str(model_dir), "--format", "safetensors"],
        )
        assert result.exit_code != 0
        assert "Unsupported format" in result.output
