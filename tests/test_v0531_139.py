"""v0.53.1 #139 — export_advanced_gguf live wiring tests.

We mock subprocess invocations so tests run without a real llama.cpp build.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExportAdvancedGguf:
    def test_not_implemented_error_gone(self):
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        # Old stub had no args. New live function should accept kwargs.
        # Calling without args should now raise TypeError, not
        # NotImplementedError.
        with pytest.raises(TypeError):
            export_advanced_gguf()  # type: ignore[call-arg]

    def test_unknown_flavour_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        with pytest.raises(ValueError):
            export_advanced_gguf(
                model_dir=str(model),
                output_path=str(tmp_path / "out.gguf"),
                flavour="EvilQ",
                calibration_data=None,
                llama_cpp_dir=str(tmp_path / "llama"),
            )

    def test_outside_cwd_model_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        monkeypatch.chdir(tmp_path)
        outside = tmp_path.parent / "outside_model_gguf"
        outside.mkdir(exist_ok=True)
        with pytest.raises(ValueError):
            export_advanced_gguf(
                model_dir=str(outside),
                output_path=str(tmp_path / "out.gguf"),
                flavour="UD-Q4_K_XL",
                calibration_data=None,
                llama_cpp_dir=str(tmp_path / "llama"),
            )

    def test_outside_cwd_output_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        with pytest.raises(ValueError):
            export_advanced_gguf(
                model_dir=str(model),
                output_path=str(tmp_path.parent / "out.gguf"),
                flavour="UD-Q4_K_XL",
                calibration_data=None,
                llama_cpp_dir=str(tmp_path / "llama"),
            )

    def test_ud_requires_calibration(self, tmp_path, monkeypatch):
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        llama = tmp_path / "llama"
        llama.mkdir()
        # UD ladder requires calibration data
        with pytest.raises(ValueError, match="calibration"):
            export_advanced_gguf(
                model_dir=str(model),
                output_path=str(tmp_path / "out.gguf"),
                flavour="UD-Q4_K_XL",
                calibration_data=None,
                llama_cpp_dir=str(llama),
            )

    def test_apple_arm_no_calibration_ok(self, tmp_path, monkeypatch):
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        # Write a real safetensors-ish marker so the model dir looks usable
        (model / "config.json").write_text("{}", encoding="utf-8")
        llama = tmp_path / "llama"
        llama.mkdir()
        # Convert script presence
        (llama / "convert_hf_to_gguf.py").write_text("# fake", encoding="utf-8")
        out_path = tmp_path / "out.gguf"

        def fake_quant(**kwargs):
            # Simulate llama-quantize writing the output file
            Path(kwargs["output_path"]).write_bytes(b"FAKEGGUF")

        with patch("soup_cli.utils.gguf_quant._run_convert_to_f16") as mock_conv, \
             patch("soup_cli.utils.gguf_quant._run_quantize_binary",
                   side_effect=fake_quant) as mock_quant:
            mock_conv.return_value = None
            export_advanced_gguf(
                model_dir=str(model),
                output_path=str(out_path),
                flavour="Q4_0_4_4",
                calibration_data=None,
                llama_cpp_dir=str(llama),
            )
            mock_quant.assert_called_once()
            # No imatrix call for Apple/ARM
            assert mock_quant.call_args.kwargs.get("imatrix_path") is None
            assert out_path.is_file()

    def test_ud_with_calibration_invokes_imatrix(self, tmp_path, monkeypatch):
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        calib = tmp_path / "calib.jsonl"
        calib.write_text(
            '{"text": "hello world"}\n{"text": "another sample"}\n',
            encoding="utf-8",
        )
        llama = tmp_path / "llama"
        llama.mkdir()
        (llama / "convert_hf_to_gguf.py").write_text("# fake", encoding="utf-8")

        out_path = tmp_path / "out.gguf"

        def fake_quant(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"FAKEGGUF")

        with patch("soup_cli.utils.gguf_quant._run_convert_to_f16") as mock_conv, \
             patch("soup_cli.utils.gguf_quant._run_imatrix") as mock_imat, \
             patch("soup_cli.utils.gguf_quant._run_quantize_binary",
                   side_effect=fake_quant) as mock_quant:
            mock_conv.return_value = None
            mock_imat.return_value = None
            export_advanced_gguf(
                model_dir=str(model),
                output_path=str(out_path),
                flavour="UD-Q4_K_XL",
                calibration_data=str(calib),
                llama_cpp_dir=str(llama),
            )
            mock_imat.assert_called_once()
            mock_quant.assert_called_once()
            assert out_path.is_file()

    def test_missing_llama_cpp_dir(self, tmp_path, monkeypatch):
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        # L3: narrowed from (FileNotFoundError, ValueError) — impl raises
        # ValueError from `_enforce_under_cwd_and_no_symlink` (the dir
        # doesn't exist on disk yet) OR FileNotFoundError. Accept both but
        # keep the union tight (no RuntimeError).
        with pytest.raises((FileNotFoundError, ValueError)):
            export_advanced_gguf(
                model_dir=str(model),
                output_path=str(tmp_path / "out.gguf"),
                flavour="Q4_0_4_4",
                calibration_data=None,
                llama_cpp_dir=str(tmp_path / "no_llama"),
            )

    def test_missing_convert_script(self, tmp_path, monkeypatch):
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        llama = tmp_path / "llama"
        llama.mkdir()  # no convert_hf_to_gguf.py
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
            export_advanced_gguf(
                model_dir=str(model),
                output_path=str(tmp_path / "out.gguf"),
                flavour="Q4_0_4_4",
                calibration_data=None,
                llama_cpp_dir=str(llama),
            )

    def test_calibration_symlink_rejected(self, tmp_path, monkeypatch):
        if sys.platform == "win32":
            pytest.skip("symlink rejection POSIX-only")
        from soup_cli.utils.gguf_quant import export_advanced_gguf

        monkeypatch.chdir(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        real = tmp_path / "real_calib.jsonl"
        real.write_text('{"text":"x"}\n', encoding="utf-8")
        link = tmp_path / "link_calib.jsonl"
        link.symlink_to(real)
        llama = tmp_path / "llama"
        llama.mkdir()
        with pytest.raises(ValueError, match="symlink"):
            export_advanced_gguf(
                model_dir=str(model),
                output_path=str(tmp_path / "out.gguf"),
                flavour="UD-Q4_K_XL",
                calibration_data=str(link),
                llama_cpp_dir=str(llama),
            )

    def test_run_imatrix_argv_shape(self, tmp_path):
        """Verify _run_imatrix builds a list-args subprocess call (no shell)."""
        from soup_cli.utils.gguf_quant import _run_imatrix

        # Drop a fake binary so the resolver finds it
        fake_bin = tmp_path / "llama-imatrix"
        fake_bin.write_bytes(b"#!/bin/sh\nexit 0\n")
        try:
            os.chmod(fake_bin, 0o755)
        except OSError:
            pass

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _run_imatrix(
                llama_cpp_dir=str(tmp_path),
                f16_path=str(tmp_path / "f16.gguf"),
                calib_data=str(tmp_path / "calib.txt"),
                imatrix_out=str(tmp_path / "imatrix.dat"),
            )
            assert mock_run.called
            args, kwargs = mock_run.call_args
            # First positional must be a list (no shell=True)
            assert isinstance(args[0], list)
            assert kwargs.get("shell") is not True
            # All argv elements must be strings
            for arg in args[0]:
                assert isinstance(arg, str)

    def test_run_quantize_argv_shape(self, tmp_path):
        from soup_cli.utils.gguf_quant import _run_quantize_binary

        fake_bin = tmp_path / "llama-quantize"
        fake_bin.write_bytes(b"#!/bin/sh\nexit 0\n")
        try:
            os.chmod(fake_bin, 0o755)
        except OSError:
            pass

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _run_quantize_binary(
                llama_cpp_dir=str(tmp_path),
                f16_path=str(tmp_path / "f16.gguf"),
                output_path=str(tmp_path / "out.gguf"),
                flavour="UD-Q4_K_XL",
                imatrix_path=str(tmp_path / "imatrix.dat"),
            )
            args, kwargs = mock_run.call_args
            assert isinstance(args[0], list)
            assert kwargs.get("shell") is not True
            # UD- prefix must be stripped before passing to llama-quantize
            assert "Q4_K_XL" in args[0]
            assert "UD-Q4_K_XL" not in args[0]


# --- H2 (TDD review): _prepare_calibration_text direct coverage ----------


class TestPrepareCalibrationText:
    def test_jsonl_with_text_field(self, tmp_path):
        from soup_cli.utils.gguf_quant import _prepare_calibration_text

        src = tmp_path / "calib.jsonl"
        src.write_text(
            '{"text": "hello world"}\n{"text": "foo bar"}\n',
            encoding="utf-8",
        )
        out = _prepare_calibration_text(str(src), tmp_path)
        content = out.read_text(encoding="utf-8")
        assert "hello world" in content
        assert "foo bar" in content

    def test_jsonl_with_prompt_alias(self, tmp_path):
        from soup_cli.utils.gguf_quant import _prepare_calibration_text

        src = tmp_path / "calib.jsonl"
        src.write_text('{"prompt": "via prompt"}\n', encoding="utf-8")
        out = _prepare_calibration_text(str(src), tmp_path)
        assert "via prompt" in out.read_text(encoding="utf-8")

    def test_jsonl_with_content_alias(self, tmp_path):
        from soup_cli.utils.gguf_quant import _prepare_calibration_text

        src = tmp_path / "calib.jsonl"
        src.write_text('{"content": "via content"}\n', encoding="utf-8")
        out = _prepare_calibration_text(str(src), tmp_path)
        assert "via content" in out.read_text(encoding="utf-8")

    def test_null_bytes_stripped(self, tmp_path):
        from soup_cli.utils.gguf_quant import _prepare_calibration_text

        src = tmp_path / "calib.jsonl"
        # JSON-escape the null byte; json.loads turns \u0000 into a real NUL
        src.write_text(
            '{"text": "ev\\u0000il"}\n', encoding="utf-8"
        )
        out = _prepare_calibration_text(str(src), tmp_path)
        content = out.read_text(encoding="utf-8")
        assert "\x00" not in content
        assert "evil" in content  # null byte stripped, neighbours preserved

    def test_newlines_collapsed_to_spaces(self, tmp_path):
        from soup_cli.utils.gguf_quant import _prepare_calibration_text

        src = tmp_path / "calib.jsonl"
        src.write_text('{"text": "line1\\nline2"}\n', encoding="utf-8")
        out = _prepare_calibration_text(str(src), tmp_path)
        content = out.read_text(encoding="utf-8")
        # Each row should be a single line, so we should see "line1 line2"
        # on one row + the trailing newline from the writer
        lines = [ln for ln in content.splitlines() if ln]
        assert len(lines) == 1
        assert "line1 line2" in lines[0]

    def test_raw_text_fallback(self, tmp_path):
        from soup_cli.utils.gguf_quant import _prepare_calibration_text

        # Input must NOT be named `calib.txt` because the helper writes
        # its output to `<staged_dir>/calib.txt`. Use a different name.
        src = tmp_path / "raw_input.txt"
        src.write_text(
            "this is not json\nand neither is this\n", encoding="utf-8"
        )
        out = _prepare_calibration_text(str(src), tmp_path)
        content = out.read_text(encoding="utf-8")
        assert "this is not json" in content
        assert "neither is this" in content

    def test_zero_usable_rows_raises(self, tmp_path):
        from soup_cli.utils.gguf_quant import _prepare_calibration_text

        src = tmp_path / "calib.jsonl"
        # All rows are JSON but lack a usable text field
        src.write_text(
            '{"unrelated": 1}\n{"other_field": "x"}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="0 usable rows"):
            _prepare_calibration_text(str(src), tmp_path)

    def test_missing_calib_file_raises(self, tmp_path):
        from soup_cli.utils.gguf_quant import _prepare_calibration_text

        with pytest.raises(FileNotFoundError):
            _prepare_calibration_text(str(tmp_path / "nope.jsonl"), tmp_path)
