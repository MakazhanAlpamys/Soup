"""Tests for the ASR manifest reader used by ``soup infer --task asr``.

Covers the named error for a manifest that cannot be decoded as UTF-8
(#3177): exit code 1, the manifest path named in the message, the read
loop (and not only ``open``) covered, and Rich tags in the file name
printed as text rather than markup.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.conftest import strip_ansi


def _out(result) -> str:
    """ANSI-strip and fold CliRunner output for substring assertions."""
    return " ".join(strip_ansi(result.output).split())


class TestReadAsrRowsManifest:
    def test_read_asr_rows_names_file_for_utf16_manifest(self, tmp_path, capsys):
        import typer

        import soup_cli.commands.infer as infer_mod

        p = tmp_path / "manifest.jsonl"
        p.write_text(json.dumps({"audio": "clip.wav"}) + "\n", encoding="utf-16")
        with pytest.raises(typer.Exit) as exc_info:
            infer_mod._read_asr_rows(p)
        assert exc_info.value.exit_code == 1
        out = capsys.readouterr().out
        # Rich soft-wraps the path in narrow CI terminals; folding is cosmetic.
        assert "manifest.jsonl" in out.replace("\n", "")
        assert "UTF-8" in out

    def test_read_asr_rows_covers_loop_for_bad_byte_on_second_line(
        self, tmp_path, capsys
    ):
        # UTF-8 manifest, one valid row, then a byte that cannot be decoded:
        # the decode error must surface from the read loop, not only open().
        import typer

        import soup_cli.commands.infer as infer_mod

        p = tmp_path / "manifest.jsonl"
        p.write_bytes(
            json.dumps({"audio": "clip.wav"}).encode("utf-8") + b"\n" + b"\xff\n"
        )
        with pytest.raises(typer.Exit) as exc_info:
            infer_mod._read_asr_rows(p)
        assert exc_info.value.exit_code == 1
        out = capsys.readouterr().out
        assert "manifest.jsonl" in out.replace("\n", "")
        assert "UTF-8" in out

    def test_infer_asr_cli_names_file_and_writes_no_output_for_utf16_manifest(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.cli import app

        p = tmp_path / "manifest.jsonl"
        p.write_text(json.dumps({"audio": "clip.wav"}) + "\n", encoding="utf-16")
        out_path = tmp_path / "out.jsonl"
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, [
            "infer", "--task", "asr", "--model", "openai/whisper-tiny",
            "--input", "manifest.jsonl", "--output", "out.jsonl",
            "--device", "cpu",
        ])
        assert result.exit_code == 1, result.output
        text = _out(result)
        assert "manifest.jsonl" in text
        assert "UTF-8" in text
        # A named message, not the raw codec traceback.
        assert "UnicodeDecodeError" not in text
        assert not out_path.exists()

    def test_infer_asr_cli_prints_rich_tag_in_file_name_as_text(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.cli import app

        # A file name that is also Rich markup must print as literal text.
        p = tmp_path / "[bold]manifest.jsonl"
        p.write_text(json.dumps({"audio": "clip.wav"}) + "\n", encoding="utf-16")
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(app, [
            "infer", "--task", "asr", "--model", "openai/whisper-tiny",
            "--input", "[bold]manifest.jsonl", "--output", "out.jsonl",
            "--device", "cpu",
        ])
        assert result.exit_code == 1, result.output
        text = _out(result)
        assert "[bold]manifest.jsonl" in text
        assert "UTF-8" in text
