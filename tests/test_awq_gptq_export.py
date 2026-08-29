"""Tests for AWQ and GPTQ export — config, validation, CLI."""

import builtins
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import pytest
from click.exceptions import Exit as ClickExit

from soup_cli.commands.export import SUPPORTED_FORMATS


def _mock_import(awq_mock=None, gptq_mock=None):
    """Create a side_effect for builtins.__import__ that intercepts awq/auto_gptq."""
    real_import = builtins.__import__

    def custom_import(name, *args, **kwargs):
        if name == "awq" and awq_mock is not None:
            return awq_mock
        if name == "auto_gptq" and gptq_mock is not None:
            return gptq_mock
        return real_import(name, *args, **kwargs)

    return custom_import


# ─── Format Support Tests ────────────────────────────────────────────────


class TestExportFormatsExtended:
    """Test that AWQ and GPTQ formats are registered."""

    def test_awq_format_supported(self):
        assert "awq" in SUPPORTED_FORMATS

    def test_gptq_format_supported(self):
        assert "gptq" in SUPPORTED_FORMATS

    def test_format_count(self):
        """v0.53.1 — 7 prior formats + torchao + gguf-ud = 9."""
        assert len(SUPPORTED_FORMATS) == 9

    def test_all_formats_present(self):
        """v0.53.1 — adds torchao + gguf-ud to the v0.52.0 baseline."""
        expected = {
            "gguf", "onnx", "tensorrt", "awq", "gptq",
            "bitnet", "tq1_0",
            "torchao", "gguf-ud",
        }
        assert set(SUPPORTED_FORMATS) == expected


# ─── AWQ Export CLI Tests ─────────────────────────────────────────────────


class TestAwqExportCLI:
    """Test AWQ export via CLI."""

    def test_awq_export_missing_model_path(self, tmp_path):
        """soup export --format awq should fail if model path doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["export", "--model", str(tmp_path / "nonexistent"), "--format", "awq"]
        )
        assert result.exit_code != 0

    def test_awq_format_in_help(self):
        """Export help should mention awq format."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["export", "--help"])
        assert "awq" in result.output.lower()

    def test_gptq_format_in_help(self):
        """Export help should mention gptq format."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["export", "--help"])
        assert "gptq" in result.output.lower()

    def test_calibration_help_names_both_required_formats(self):
        """The shared option must not imply AWQ can omit calibration data."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        result = CliRunner().invoke(app, ["export", "--help"])

        assert "Required for AWQ and GPTQ" in result.output


# ─── GPTQ Export CLI Tests ────────────────────────────────────────────────


class TestGptqExportCLI:
    """Test GPTQ export via CLI."""

    def test_gptq_export_missing_model_path(self, tmp_path):
        """soup export --format gptq should fail if model path doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app, ["export", "--model", str(tmp_path / "nonexistent"), "--format", "gptq"]
        )
        assert result.exit_code != 0


# ─── AWQ Export Function Tests ────────────────────────────────────────────


class TestAwqExportFunction:
    """Test _export_awq logic."""

    def test_export_awq_import_error(self, tmp_path, monkeypatch):
        """AWQ export should print friendly error when autoawq not installed."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "export", "--model", str(model_dir), "--format", "awq",
                "--calibration-data", str(cal_file),
            ],
        )
        assert result.exit_code != 0
        assert "autoawq" in result.output.lower() or "not installed" in result.output.lower()

    def test_export_awq_no_calibration_data_raises(self, tmp_path, capsys):
        """AWQ must not delegate missing calibration data to AutoAWQ's implicit
        large dataset download, and must fail before importing or loading a model."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        real_import = builtins.__import__

        def reject_awq_import(name, *args, **kwargs):
            if name == "awq":
                pytest.fail("AWQ was imported before calibration validation")
            return real_import(name, *args, **kwargs)

        import soup_cli.commands.export as export_mod

        with mock_patch.object(builtins, "__import__", side_effect=reject_awq_import):
            with pytest.raises(ClickExit):
                export_mod._export_awq(
                    model_dir, None, None, bits=4, group_size=128,
                    calibration_data=None,
                )
        output = capsys.readouterr().out
        assert "AWQ export requires --calibration-data" in output
        assert "default calibration dataset" in output

    def test_export_awq_empty_calibration_file_raises(self, tmp_path):
        """An empty or malformed calibration file fails before model loading."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text("\nnot json\n", encoding="utf-8")

        real_import = builtins.__import__

        def reject_awq_import(name, *args, **kwargs):
            if name == "awq":
                pytest.fail("AWQ was imported before calibration validation")
            return real_import(name, *args, **kwargs)

        import soup_cli.commands.export as export_mod

        with mock_patch.object(builtins, "__import__", side_effect=reject_awq_import):
            with mock_patch(
                "soup_cli.commands.export._validate_calibration_path",
                return_value=cal_file,
            ):
                with pytest.raises(ClickExit):
                    export_mod._export_awq(
                        model_dir, None, None, bits=4, group_size=128,
                        calibration_data=str(cal_file),
                    )

    def test_export_awq_calls_quantize(self, tmp_path):
        """AWQ export should pass the explicit calibration rows to AutoAWQ."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        mock_model = MagicMock()
        mock_awq_class = MagicMock()
        mock_awq_class.from_pretrained = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        awq_mod = MagicMock()
        awq_mod.AutoAWQForCausalLM = mock_awq_class

        import soup_cli.commands.export as export_mod

        out_path = tmp_path / "out"
        with mock_patch.object(builtins, "__import__", side_effect=_mock_import(awq_mock=awq_mod)):
            with mock_patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                with mock_patch(
                    "soup_cli.commands.export._validate_output_path",
                    return_value=out_path,
                ):
                    with mock_patch(
                        "soup_cli.commands.export._validate_calibration_path",
                        return_value=cal_file,
                    ):
                        export_mod._export_awq(
                            model_dir, str(out_path), None,
                            bits=4, group_size=128,
                            calibration_data=str(cal_file),
                        )
        mock_awq_class.from_pretrained.assert_called_once()
        mock_model.quantize.assert_called_once_with(
            mock_tokenizer,
            quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4},
            calib_data=["sample"],
        )
        mock_model.save_quantized.assert_called_once()

    def test_export_awq_default_output_path(self, tmp_path):
        """Default AWQ output path should be model_name + _awq suffix."""
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        mock_model = MagicMock()
        mock_awq_class = MagicMock()
        mock_awq_class.from_pretrained = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        awq_mod = MagicMock()
        awq_mod.AutoAWQForCausalLM = mock_awq_class

        import soup_cli.commands.export as export_mod

        with mock_patch.object(builtins, "__import__", side_effect=_mock_import(awq_mock=awq_mod)):
            with mock_patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                with mock_patch(
                    "soup_cli.commands.export._validate_calibration_path",
                    return_value=cal_file,
                ):
                    export_mod._export_awq(
                        model_dir, None, None, bits=4, group_size=128,
                        calibration_data=str(cal_file),
                    )
                    save_call = mock_model.save_quantized.call_args
                    output_path = save_call[0][0]
                    assert "my_model_awq" in output_path

    def test_export_awq_with_calibration_data(self, tmp_path):
        """AWQ export with calibration data should load and pass it."""
        import json

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        cal_file = tmp_path / "cal.jsonl"
        lines = [json.dumps({"text": f"sample {i}"}) for i in range(10)]
        cal_file.write_text("\n".join(lines), encoding="utf-8")

        mock_model = MagicMock()
        mock_awq_class = MagicMock()
        mock_awq_class.from_pretrained = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        awq_mod = MagicMock()
        awq_mod.AutoAWQForCausalLM = mock_awq_class

        import soup_cli.commands.export as export_mod

        out_path = tmp_path / "out"
        with mock_patch.object(builtins, "__import__", side_effect=_mock_import(awq_mock=awq_mod)):
            with mock_patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                with mock_patch(
                    "soup_cli.commands.export._validate_calibration_path",
                    return_value=cal_file,
                ):
                    with mock_patch(
                        "soup_cli.commands.export._validate_output_path",
                        return_value=out_path,
                    ):
                        export_mod._export_awq(
                            model_dir, str(out_path), None,
                            bits=4, group_size=128,
                            calibration_data=str(cal_file),
                        )
                        quant_call = mock_model.quantize.call_args
                        assert quant_call is not None
                        assert "calib_data" in quant_call.kwargs

    def test_export_awq_invalid_bits(self, tmp_path):
        """Invalid bits value should raise ClickExit."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        import soup_cli.commands.export as export_mod

        with pytest.raises(ClickExit):
            export_mod._export_awq(
                model_dir, None, None, bits=3, group_size=128,
                calibration_data=None,
            )

    def test_export_awq_calibration_path_traversal(self, tmp_path):
        """Calibration data path should be confined to cwd."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        import soup_cli.commands.export as export_mod

        with pytest.raises(ClickExit):
            export_mod._export_awq(
                model_dir, None, None, bits=4, group_size=128,
                calibration_data=str(Path("C:/Windows/System32/drivers/etc/hosts")),
            )


# ─── GPTQ Export Function Tests ───────────────────────────────────────────


class TestGptqExportFunction:
    """Test _export_gptq logic."""

    def test_export_gptq_import_error(self, tmp_path, monkeypatch):
        """GPTQ export should print friendly error when auto-gptq not installed."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        real_import = builtins.__import__

        def custom_import(name, *args, **kwargs):
            if name == "auto_gptq":
                raise ModuleNotFoundError("No module named 'auto_gptq'", name="auto_gptq")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", custom_import)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "export", "--model", str(model_dir), "--format", "gptq",
                "--calibration-data", str(cal_file),
            ],
        )
        assert result.exit_code != 0
        assert "auto-gptq" in result.output.lower() or "not installed" in result.output.lower()

    def test_export_gptq_no_calibration_data_raises(self, tmp_path, capsys):
        """GPTQ has no built-in fallback dataset: no --calibration-data must raise,
        never fall through to model.quantize(tokenizer) (auto-gptq expects examples,
        not a tokenizer, and rejects one with 'object is not iterable')."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_gptq_class = MagicMock()
        gptq_mod = MagicMock()
        gptq_mod.AutoGPTQForCausalLM = mock_gptq_class
        gptq_mod.BaseQuantizeConfig = MagicMock

        import soup_cli.commands.export as export_mod

        with mock_patch.object(
            builtins, "__import__", side_effect=_mock_import(gptq_mock=gptq_mod)
        ):
            with pytest.raises(ClickExit):
                export_mod._export_gptq(
                    model_dir, None, None, bits=4, group_size=128,
                    calibration_data=None,
                )
        mock_gptq_class.from_pretrained.assert_not_called()
        assert "--calibration-data" in capsys.readouterr().out

    def test_export_gptq_empty_calibration_file_raises(self, tmp_path):
        """A --calibration-data file that yields zero usable samples (empty,
        or only malformed JSON lines) must raise the same way as no file at
        all, before the model is ever loaded."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text("\n\nnot json\n", encoding="utf-8")

        mock_gptq_class = MagicMock()
        gptq_mod = MagicMock()
        gptq_mod.AutoGPTQForCausalLM = mock_gptq_class
        gptq_mod.BaseQuantizeConfig = MagicMock

        import soup_cli.commands.export as export_mod

        with mock_patch.object(
            builtins, "__import__", side_effect=_mock_import(gptq_mock=gptq_mod)
        ):
            with mock_patch(
                "soup_cli.commands.export._validate_calibration_path",
                return_value=cal_file,
            ):
                with pytest.raises(ClickExit):
                    export_mod._export_gptq(
                        model_dir, None, None, bits=4, group_size=128,
                        calibration_data=str(cal_file),
                    )
        mock_gptq_class.from_pretrained.assert_not_called()

    def test_export_gptq_calls_quantize(self, tmp_path):
        """GPTQ export should call AutoGPTQForCausalLM methods."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        mock_model = MagicMock()
        mock_gptq_class = MagicMock()
        mock_gptq_class.from_pretrained = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        gptq_mod = MagicMock()
        gptq_mod.AutoGPTQForCausalLM = mock_gptq_class
        gptq_mod.BaseQuantizeConfig = MagicMock

        import soup_cli.commands.export as export_mod

        out_path = tmp_path / "out"
        out_path.mkdir()
        with mock_patch.object(
            builtins, "__import__", side_effect=_mock_import(gptq_mock=gptq_mod)
        ):
            with mock_patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                with mock_patch(
                    "soup_cli.commands.export._validate_output_path",
                    return_value=out_path,
                ):
                    with mock_patch(
                        "soup_cli.commands.export._validate_calibration_path",
                        return_value=cal_file,
                    ):
                        export_mod._export_gptq(
                            model_dir, str(out_path), None,
                            bits=4, group_size=128, calibration_data=str(cal_file),
                        )
                        mock_gptq_class.from_pretrained.assert_called_once()
                        mock_model.quantize.assert_called_once()
                        mock_model.save_quantized.assert_called_once()

    def test_export_gptq_default_output_path(self, tmp_path):
        """Default GPTQ output path should be model_name + _gptq suffix."""
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        mock_model = MagicMock()
        mock_gptq_class = MagicMock()
        mock_gptq_class.from_pretrained = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        gptq_mod = MagicMock()
        gptq_mod.AutoGPTQForCausalLM = mock_gptq_class
        gptq_mod.BaseQuantizeConfig = MagicMock

        import soup_cli.commands.export as export_mod

        with mock_patch.object(
            builtins, "__import__", side_effect=_mock_import(gptq_mock=gptq_mod)
        ):
            with mock_patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                with mock_patch(
                    "soup_cli.commands.export._validate_calibration_path",
                    return_value=cal_file,
                ):
                    export_mod._export_gptq(
                        model_dir, None, None, bits=4, group_size=128,
                        calibration_data=str(cal_file),
                    )
                    save_call = mock_model.save_quantized.call_args
                    output_path = save_call[0][0]
                    assert "my_model_gptq" in output_path

    def test_export_gptq_invalid_bits(self, tmp_path):
        """Invalid bits value should raise ClickExit."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        import soup_cli.commands.export as export_mod

        with pytest.raises(ClickExit):
            export_mod._export_gptq(
                model_dir, None, None, bits=3, group_size=128,
                calibration_data=None,
            )

    def test_export_gptq_calibration_path_traversal(self, tmp_path):
        """Calibration data path should be confined to cwd."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        import soup_cli.commands.export as export_mod

        with pytest.raises(ClickExit):
            export_mod._export_gptq(
                model_dir, None, None, bits=4, group_size=128,
                calibration_data=str(Path("C:/Windows/System32/drivers/etc/hosts")),
            )


# ─── GPTQ Shard Name Regression (issue #338, P2) ─────────────────────────


class TestGptqStandardShardName:
    """auto_gptq's save_quantized names its shard gptq_model-<bits>bit-<group>g.safetensors,
    which AutoModelForCausalLM.from_pretrained does not look for."""

    def test_standardize_renames_shard_to_model_safetensors(self, tmp_path):
        from soup_cli.commands.export import _standardize_gptq_shard_name

        out = tmp_path / "out"
        out.mkdir()
        (out / "gptq_model-4bit-128g.safetensors").write_bytes(b"weights")

        _standardize_gptq_shard_name(out, bits=4, group_size=128)

        assert (out / "model.safetensors").exists()
        assert not (out / "gptq_model-4bit-128g.safetensors").exists()

    def test_standardize_does_not_overwrite_existing_standard_name(self, tmp_path):
        from soup_cli.commands.export import _standardize_gptq_shard_name

        out = tmp_path / "out"
        out.mkdir()
        (out / "model.safetensors").write_bytes(b"already-standard")
        (out / "gptq_model-4bit-128g.safetensors").write_bytes(b"weights")

        _standardize_gptq_shard_name(out, bits=4, group_size=128)

        assert (out / "model.safetensors").read_bytes() == b"already-standard"

    def test_standardize_no_op_when_shard_absent(self, tmp_path):
        from soup_cli.commands.export import _standardize_gptq_shard_name

        out = tmp_path / "out"
        out.mkdir()

        _standardize_gptq_shard_name(out, bits=4, group_size=128)

        assert not (out / "model.safetensors").exists()

    def test_export_gptq_end_to_end_leaves_standard_name(self, tmp_path):
        """Full _export_gptq run: save_quantized's mock writes the real auto_gptq
        shard name, and the standard model.safetensors must exist afterward."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")
        out_path = tmp_path / "out"
        out_path.mkdir()

        def fake_save_quantized(path):
            Path(path, "gptq_model-4bit-128g.safetensors").write_bytes(b"weights")

        mock_model = MagicMock()
        mock_model.save_quantized.side_effect = fake_save_quantized
        mock_gptq_class = MagicMock()
        mock_gptq_class.from_pretrained = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()

        gptq_mod = MagicMock()
        gptq_mod.AutoGPTQForCausalLM = mock_gptq_class
        gptq_mod.BaseQuantizeConfig = MagicMock

        import soup_cli.commands.export as export_mod

        with mock_patch.object(
            builtins, "__import__", side_effect=_mock_import(gptq_mock=gptq_mod)
        ):
            with mock_patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ):
                with mock_patch(
                    "soup_cli.commands.export._validate_output_path",
                    return_value=out_path,
                ):
                    with mock_patch(
                        "soup_cli.commands.export._validate_calibration_path",
                        return_value=cal_file,
                    ):
                        export_mod._export_gptq(
                            model_dir, str(out_path), None,
                            bits=4, group_size=128, calibration_data=str(cal_file),
                        )

        assert (out_path / "model.safetensors").exists()
        assert not (out_path / "gptq_model-4bit-128g.safetensors").exists()


# ─── Import-Error Masking Regression (issue #338, ImportError swallowing) ──


class TestAwqGptqImportErrorSurfacesRealCause:
    """except ImportError must not print a fixed 'not installed' string when the
    package itself imports but a transitive import inside it fails for an
    unrelated reason, which used to discard the real exception."""

    def test_gptq_transitive_import_failure_message(self, tmp_path, capsys):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        real_import = builtins.__import__

        def custom_import(name, *args, **kwargs):
            if name == "auto_gptq":
                raise ImportError("libfoo.so: cannot open shared object file", name="libfoo")
            return real_import(name, *args, **kwargs)

        import soup_cli.commands.export as export_mod

        with mock_patch.object(builtins, "__import__", side_effect=custom_import):
            with mock_patch(
                "soup_cli.commands.export._validate_calibration_path",
                return_value=cal_file,
            ):
                with pytest.raises(ClickExit):
                    export_mod._export_gptq(
                        model_dir, None, None, bits=4, group_size=128,
                        calibration_data=str(cal_file),
                    )
        output = capsys.readouterr().out
        assert "libfoo.so" in output
        assert "not installed" not in output.lower()

    def test_awq_transitive_import_failure_message(self, tmp_path, capsys):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        real_import = builtins.__import__

        def custom_import(name, *args, **kwargs):
            if name == "awq":
                raise ImportError("libbar.so: cannot open shared object file", name="libbar")
            return real_import(name, *args, **kwargs)

        import soup_cli.commands.export as export_mod

        with mock_patch.object(builtins, "__import__", side_effect=custom_import):
            with mock_patch(
                "soup_cli.commands.export._validate_calibration_path",
                return_value=cal_file,
            ):
                with pytest.raises(ClickExit):
                    export_mod._export_awq(
                        model_dir, None, None, bits=4, group_size=128,
                        calibration_data=str(cal_file),
                    )
        output = capsys.readouterr().out
        assert "libbar.so" in output
        assert "not installed" not in output.lower()

    def test_gptq_real_missing_package_still_shows_install_hint(self, tmp_path, capsys):
        """When auto-gptq is genuinely absent (exc.name == 'auto_gptq'), the
        friendly install hint must still show; this must not regress."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text('{"text": "sample"}\n', encoding="utf-8")

        real_import = builtins.__import__

        def custom_import(name, *args, **kwargs):
            if name == "auto_gptq":
                raise ModuleNotFoundError("No module named 'auto_gptq'", name="auto_gptq")
            return real_import(name, *args, **kwargs)

        import soup_cli.commands.export as export_mod

        with mock_patch.object(builtins, "__import__", side_effect=custom_import):
            with mock_patch(
                "soup_cli.commands.export._validate_calibration_path",
                return_value=cal_file,
            ):
                with pytest.raises(ClickExit):
                    export_mod._export_gptq(
                        model_dir, None, None, bits=4, group_size=128,
                        calibration_data=str(cal_file),
                    )
        output = capsys.readouterr().out
        assert "not installed" in output.lower()


# ─── CLI Integration Tests ───────────────────────────────────────────────


class TestAwqGptqCLIArgs:
    """Test new CLI arguments are declared in export function signature."""

    def test_bits_flag_exists(self):
        """Export function should have a --bits parameter."""
        import inspect

        from soup_cli.commands.export import export

        sig = inspect.signature(export)
        assert "bits" in sig.parameters

    def test_group_size_flag_exists(self):
        """Export function should have a --group-size parameter."""
        import inspect

        from soup_cli.commands.export import export

        sig = inspect.signature(export)
        assert "group_size" in sig.parameters

    def test_calibration_data_flag_exists(self):
        """Export function should have a --calibration-data parameter."""
        import inspect

        from soup_cli.commands.export import export

        sig = inspect.signature(export)
        assert "calibration_data" in sig.parameters

    def test_calibration_samples_flag_exists(self):
        """Export function should have a --calibration-samples parameter."""
        import inspect

        from soup_cli.commands.export import export

        sig = inspect.signature(export)
        assert "calibration_samples" in sig.parameters


# ─── Security Tests ──────────────────────────────────────────────────────


class TestAwqGptqSecurity:
    """Security validation for AWQ/GPTQ export."""

    def test_calibration_data_path_traversal_awq(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "export", "--model", str(model_dir),
                "--format", "awq",
                "--calibration-data", "C:/Windows/System32/drivers/etc/hosts",
            ],
        )
        assert result.exit_code != 0

    def test_calibration_data_path_traversal_gptq(self, tmp_path):
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "export", "--model", str(model_dir),
                "--format", "gptq",
                "--calibration-data", "C:/Windows/System32/drivers/etc/hosts",
            ],
        )
        assert result.exit_code != 0


# ─── Calibration Data Loader Tests ────────────────────────────────────────


class TestCalibrationLoader:
    """Test _load_calibration_texts function."""

    def test_load_text_field(self, tmp_path):
        import json

        from soup_cli.commands.export import _load_calibration_texts

        cal_file = tmp_path / "cal.jsonl"
        lines = [json.dumps({"text": f"line {i}"}) for i in range(5)]
        cal_file.write_text("\n".join(lines), encoding="utf-8")

        texts = _load_calibration_texts(cal_file, max_samples=10)
        assert len(texts) == 5
        assert texts[0] == "line 0"

    def test_load_max_samples(self, tmp_path):
        import json

        from soup_cli.commands.export import _load_calibration_texts

        cal_file = tmp_path / "cal.jsonl"
        lines = [json.dumps({"text": f"line {i}"}) for i in range(100)]
        cal_file.write_text("\n".join(lines), encoding="utf-8")

        texts = _load_calibration_texts(cal_file, max_samples=10)
        assert len(texts) == 10

    def test_load_none_returns_empty(self):
        from soup_cli.commands.export import _load_calibration_texts

        assert _load_calibration_texts(None) == []

    def test_load_skips_empty_lines(self, tmp_path):
        import json

        from soup_cli.commands.export import _load_calibration_texts

        cal_file = tmp_path / "cal.jsonl"
        content = json.dumps({"text": "hello"}) + "\n\n" + json.dumps({"text": "world"})
        cal_file.write_text(content, encoding="utf-8")

        texts = _load_calibration_texts(cal_file, max_samples=10)
        assert len(texts) == 2

    def test_load_concatenates_non_text_fields(self, tmp_path):
        import json

        from soup_cli.commands.export import _load_calibration_texts

        cal_file = tmp_path / "cal.jsonl"
        cal_file.write_text(
            json.dumps({"prompt": "hello", "response": "world"}),
            encoding="utf-8",
        )

        texts = _load_calibration_texts(cal_file, max_samples=10)
        assert len(texts) == 1
        assert "hello" in texts[0]
        assert "world" in texts[0]


# ─── Path Validation Tests ───────────────────────────────────────────────


class TestCalibrationPathValidation:
    """Test _validate_calibration_path function."""

    def test_none_returns_none(self):
        from soup_cli.commands.export import _validate_calibration_path

        assert _validate_calibration_path(None) is None

    def test_nonexistent_file_raises(self):
        """Nonexistent calibration file should raise ClickExit."""
        from soup_cli.commands.export import _validate_calibration_path

        nope = Path.cwd() / "nonexistent_cal_data_test_xyzzy.jsonl"
        with pytest.raises(ClickExit):
            _validate_calibration_path(str(nope))

    def test_valid_path_returns_path(self):
        """Valid path under cwd should return resolved Path."""
        from soup_cli.commands.export import _validate_calibration_path

        cal_file = Path.cwd() / "test_cal_temp_xyzzy.jsonl"
        try:
            cal_file.write_text("{}", encoding="utf-8")
            result = _validate_calibration_path(str(cal_file))
            assert result is not None
            assert result.name == "test_cal_temp_xyzzy.jsonl"
        finally:
            if cal_file.exists():
                cal_file.unlink()

    def test_path_outside_cwd_raises(self):
        from soup_cli.commands.export import _validate_calibration_path

        with pytest.raises(ClickExit):
            _validate_calibration_path("C:/Windows/System32/drivers/etc/hosts")


# ─── Output Path Validation Tests ────────────────────────────────────────


class TestOutputPathValidation:
    """Test _validate_output_path function."""

    def test_none_returns_none(self):
        from soup_cli.commands.export import _validate_output_path

        assert _validate_output_path(None) is None

    def test_valid_path_returns_path(self):
        """Valid path under cwd should return resolved Path."""
        from soup_cli.commands.export import _validate_output_path

        out = Path.cwd() / "test_output_xyzzy"
        result = _validate_output_path(str(out))
        assert result is not None
        assert result.name == "test_output_xyzzy"

    def test_path_outside_cwd_raises(self):
        """Path outside cwd should raise ClickExit."""
        from soup_cli.commands.export import _validate_output_path

        # Use a path guaranteed to be outside cwd on all platforms
        outside = Path(tempfile.gettempdir()).resolve()
        cwd = Path.cwd().resolve()
        # Only test if temp dir is actually outside cwd
        try:
            outside.relative_to(cwd)
            pytest.skip("tempdir is under cwd, cannot test outside path")
        except ValueError:
            pass
        with pytest.raises(ClickExit):
            _validate_output_path(str(outside / "evil_output"))

    def test_path_traversal_rejected(self, tmp_path, monkeypatch):
        """Path outside cwd via relative traversal should be rejected."""
        from soup_cli.commands.export import _validate_output_path

        # Set cwd to a subdirectory so ../evil is outside
        sub = tmp_path / "deep" / "nested"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)
        with pytest.raises(ClickExit):
            _validate_output_path(str(tmp_path / "evil"))


# ─── Optional Dependency Tests ────────────────────────────────────────────


class TestOptionalDeps:
    """Test that optional deps are declared in pyproject.toml."""

    def _read_pyproject(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        return pyproject.read_text(encoding="utf-8")

    def test_awq_optional_dep(self):
        content = self._read_pyproject()
        assert "awq" in content
        assert "autoawq" in content

    def test_gptq_optional_dep(self):
        content = self._read_pyproject()
        assert "gptq" in content
        assert "auto-gptq" in content
