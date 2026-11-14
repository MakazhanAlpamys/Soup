"""Tests for soup diff — model comparison command."""

import json
from unittest.mock import patch

import pytest

from tests.conftest import strip_ansi


def _plain(output: str) -> str:
    return " ".join(strip_ansi(output).split())


class TestCollectPrompts:
    """Test prompt collection from files and arguments."""

    def test_collect_from_text_file(self, tmp_path):
        """Should read plain text prompts from a file."""
        from soup_cli.commands.diff import _collect_prompts

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("What is AI?\nExplain gravity.\nHello world.\n")

        result = _collect_prompts(str(prompts_file), None)
        assert len(result) == 3
        assert result[0] == "What is AI?"
        assert result[1] == "Explain gravity."

    def test_collect_from_jsonl_file(self, tmp_path):
        """Should read prompts from JSONL with 'prompt' field."""
        from soup_cli.commands.diff import _collect_prompts

        prompts_file = tmp_path / "prompts.jsonl"
        lines = [
            json.dumps({"prompt": "What is AI?"}),
            json.dumps({"prompt": "Explain gravity."}),
        ]
        prompts_file.write_text("\n".join(lines))

        result = _collect_prompts(str(prompts_file), None)
        assert len(result) == 2
        assert result[0] == "What is AI?"

    def test_collect_from_args(self):
        """Should collect prompts from CLI arguments."""
        from soup_cli.commands.diff import _collect_prompts

        result = _collect_prompts(None, ["Hello!", "How are you?"])
        assert len(result) == 2
        assert result[0] == "Hello!"

    def test_collect_combined(self, tmp_path):
        """Should combine prompts from file and args."""
        from soup_cli.commands.diff import _collect_prompts

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("From file\n")

        result = _collect_prompts(str(prompts_file), ["From args"])
        assert len(result) == 2

    def test_collect_empty(self):
        """Should return empty list if no prompts."""
        from soup_cli.commands.diff import _collect_prompts

        result = _collect_prompts(None, None)
        assert result == []

    def test_collect_skips_empty_lines(self, tmp_path):
        """Should skip empty lines in prompt files."""
        from soup_cli.commands.diff import _collect_prompts

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("Line one\n\n\nLine two\n\n")

        result = _collect_prompts(str(prompts_file), None)
        assert len(result) == 2

    def test_collect_from_jsonl_with_utf8_bom(self, tmp_path):
        """Should parse JSONL prompts even when file starts with a UTF-8 BOM."""
        from soup_cli.commands.diff import _collect_prompts

        prompts_file = tmp_path / "prompts_bom.jsonl"
        content = json.dumps({"prompt": "Prompt with BOM"}) + "\n"
        prompts_file.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

        result = _collect_prompts(str(prompts_file), None)
        assert result == ["Prompt with BOM"]

    def test_collect_from_missing_file_or_directory(self, tmp_path):
        """Should return empty list for missing file or directory."""
        from soup_cli.commands.diff import _collect_prompts

        assert _collect_prompts(str(tmp_path / "missing.jsonl"), None) == []
        assert _collect_prompts(str(tmp_path), None) == []


class TestComputeMetrics:
    """Test comparison metrics computation."""

    def test_identical_responses(self):
        """Identical responses should have 100% overlap."""
        from soup_cli.commands.diff import _compute_metrics

        metrics = _compute_metrics("hello world", "hello world")
        assert metrics["word_overlap"] == pytest.approx(1.0)
        assert metrics["len_a"] == metrics["len_b"]

    def test_completely_different(self):
        """Completely different responses should have 0% overlap."""
        from soup_cli.commands.diff import _compute_metrics

        metrics = _compute_metrics("hello world", "foo bar")
        assert metrics["word_overlap"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap should be between 0 and 1."""
        from soup_cli.commands.diff import _compute_metrics

        metrics = _compute_metrics("hello world today", "hello world tomorrow")
        assert 0 < metrics["word_overlap"] < 1

    def test_empty_responses(self):
        """Empty responses should not crash."""
        from soup_cli.commands.diff import _compute_metrics

        metrics = _compute_metrics("", "")
        assert metrics["len_a"] == 0
        assert metrics["len_b"] == 0
        assert metrics["word_overlap"] == pytest.approx(0.0)

    def test_one_empty(self):
        """One empty response should have 0% overlap."""
        from soup_cli.commands.diff import _compute_metrics

        metrics = _compute_metrics("hello world", "")
        assert metrics["word_overlap"] == pytest.approx(0.0)

    def test_word_counts(self):
        """Should correctly count words."""
        from soup_cli.commands.diff import _compute_metrics

        metrics = _compute_metrics("one two three", "a b")
        assert metrics["words_a"] == 3
        assert metrics["words_b"] == 2


class TestDisplaySummary:
    """Test summary display."""

    def test_display_summary_no_crash(self):
        """Display summary should not crash with valid data."""
        from soup_cli.commands.diff import _display_summary

        results = [
            {
                "prompt": "test",
                "response_a": "hello",
                "response_b": "world",
                "metrics": {
                    "len_a": 5, "len_b": 5,
                    "words_a": 1, "words_b": 1,
                    "word_overlap": 0.0,
                },
            }
        ]
        # Should not raise
        _display_summary(results, "model_a", "model_b")

    def test_display_summary_empty(self):
        """Display summary should handle empty results."""
        from soup_cli.commands.diff import _display_summary

        _display_summary([], "model_a", "model_b")


class TestDiffCLI:
    """Test diff CLI command."""

    def test_model_a_not_found(self):
        """Should fail if model A doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "diff",
            "--model-a", "/nonexistent/model_a",
            "--model-b", ".",
            "--prompt", "test",
        ])
        assert result.exit_code != 0

    def test_model_b_not_found(self, tmp_path):
        """Should fail if model B doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "diff",
            "--model-a", str(tmp_path),
            "--model-b", "/nonexistent/model_b",
            "--prompt", "test",
        ])
        assert result.exit_code != 0

    def test_no_prompts_error(self, tmp_path):
        """Should fail if no prompts provided."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "diff",
            "--model-a", str(tmp_path),
            "--model-b", str(tmp_path),
        ])
        assert result.exit_code != 0

    def test_prompts_file_not_found(self, tmp_path, monkeypatch):
        """Should exit 1 with full message before Diff Plan when prompts file is missing."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        model_a.mkdir()
        model_b.mkdir()
        monkeypatch.chdir(tmp_path)

        with patch(
            "soup_cli.commands.diff._load_model",
            side_effect=AssertionError("_load_model should not be called"),
        ):
            result = CliRunner().invoke(app, [
                "diff",
                "--model-a", str(model_a),
                "--model-b", str(model_b),
                "--prompts", "missing.jsonl",
            ])
        plain = _plain(result.output)
        assert result.exit_code == 1
        assert plain == "Prompts file not found: missing.jsonl"
        assert "Diff Plan" not in plain

    def test_prompts_file_not_found_with_prompt_option(self, tmp_path, monkeypatch):
        """Should refuse missing --prompts even when --prompt is also supplied."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        model_a.mkdir()
        model_b.mkdir()
        monkeypatch.chdir(tmp_path)

        with patch(
            "soup_cli.commands.diff._load_model",
            side_effect=AssertionError("_load_model should not be called"),
        ):
            result = CliRunner().invoke(app, [
                "diff",
                "--model-a", str(model_a),
                "--model-b", str(model_b),
                "--prompts", "missing.jsonl",
                "--prompt", "hello",
            ])
        plain = _plain(result.output)
        assert result.exit_code == 1
        assert plain == "Prompts file not found: missing.jsonl"
        assert "Diff Plan" not in plain

    def test_prompts_is_directory_error(self, tmp_path, monkeypatch):
        """Should exit 1 cleanly when --prompts points to a directory."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        prompts_dir = tmp_path / "prompts_dir"
        model_a.mkdir()
        model_b.mkdir()
        prompts_dir.mkdir()
        monkeypatch.chdir(tmp_path)

        with patch(
            "soup_cli.commands.diff._load_model",
            side_effect=AssertionError("_load_model should not be called"),
        ):
            result = CliRunner().invoke(app, [
                "diff",
                "--model-a", str(model_a),
                "--model-b", str(model_b),
                "--prompts", "prompts_dir",
            ])
        plain = _plain(result.output)
        assert result.exit_code == 1
        assert plain == "Prompts path is not a file: prompts_dir"
        assert "Diff Plan" not in plain

    def test_prompts_file_empty_error(self, tmp_path, monkeypatch):
        """Should exit 1 with full message when --prompts file yields no rows."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        model_a.mkdir()
        model_b.mkdir()
        (tmp_path / "empty.jsonl").write_text("\n   \n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        with patch(
            "soup_cli.commands.diff._load_model",
            side_effect=AssertionError("_load_model should not be called"),
        ):
            result = CliRunner().invoke(app, [
                "diff",
                "--model-a", str(model_a),
                "--model-b", str(model_b),
                "--prompts", "empty.jsonl",
            ])
        plain = _plain(result.output)
        assert result.exit_code == 1
        assert plain == "No prompts found in file: empty.jsonl"
        assert "Diff Plan" not in plain

    def test_prompts_file_empty_with_prompt_option_error(self, tmp_path, monkeypatch):
        """Should refuse an empty --prompts file even when --prompt is also supplied."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        model_a.mkdir()
        model_b.mkdir()
        (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        with patch(
            "soup_cli.commands.diff._load_model",
            side_effect=AssertionError("_load_model should not be called"),
        ):
            result = CliRunner().invoke(app, [
                "diff",
                "--model-a", str(model_a),
                "--model-b", str(model_b),
                "--prompts", "empty.jsonl",
                "--prompt", "hello",
            ])
        plain = _plain(result.output)
        assert result.exit_code == 1
        assert plain == "No prompts found in file: empty.jsonl"
        assert "Diff Plan" not in plain

    def test_prompts_file_non_utf8_error(self, tmp_path, monkeypatch):
        """Should exit 1 with a named message when --prompts file is not valid UTF-8."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        model_a.mkdir()
        model_b.mkdir()
        (tmp_path / "bad.jsonl").write_bytes(b"\xff\xfe\x00not utf-8\n")
        monkeypatch.chdir(tmp_path)

        with patch(
            "soup_cli.commands.diff._load_model",
            side_effect=AssertionError("_load_model should not be called"),
        ):
            result = CliRunner().invoke(app, [
                "diff",
                "--model-a", str(model_a),
                "--model-b", str(model_b),
                "--prompts", "bad.jsonl",
            ])
        plain = _plain(result.output)
        assert result.exit_code == 1
        assert plain == (
            "Cannot read prompts file bad.jsonl: "
            "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"
        )
        assert "Diff Plan" not in plain

    def test_prompts_file_unreadable_os_error(self, tmp_path, monkeypatch):
        """Should exit 1 with a named message when reading --prompts raises OSError."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        model_a.mkdir()
        model_b.mkdir()
        (tmp_path / "locked.jsonl").write_text("hi\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        with (
            patch(
                "soup_cli.commands.diff._collect_prompts",
                side_effect=PermissionError("Permission denied"),
            ),
            patch(
                "soup_cli.commands.diff._load_model",
                side_effect=AssertionError("_load_model should not be called"),
            ),
        ):
            result = CliRunner().invoke(app, [
                "diff",
                "--model-a", str(model_a),
                "--model-b", str(model_b),
                "--prompts", "locked.jsonl",
            ])
        plain = _plain(result.output)
        assert result.exit_code == 1
        assert plain == "Cannot read prompts file locked.jsonl: Permission denied"
        assert "Diff Plan" not in plain

    def test_prompts_file_with_utf8_bom_cli(self, tmp_path, monkeypatch):
        """BOM-prefixed JSONL row must come out as prompt text, not raw JSON line."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        model_a.mkdir()
        model_b.mkdir()
        bom_file = tmp_path / "bom.jsonl"
        bom_file.write_bytes(
            b"\xef\xbb\xbf" + json.dumps({"prompt": "Hello BOM prompt"}).encode("utf-8") + b"\n"
        )
        monkeypatch.chdir(tmp_path)

        with (
            patch("soup_cli.commands.diff._load_model", return_value=(None, None)),
            patch("soup_cli.commands.diff._generate", return_value="stub reply"),
        ):
            result = CliRunner().invoke(app, [
                "diff",
                "--model-a", str(model_a),
                "--model-b", str(model_b),
                "--prompts", "bom.jsonl",
                "--device", "cpu",
            ])
        plain = _plain(result.output)
        assert result.exit_code == 0, plain
        assert "Prompts: 1" in plain
        assert "Hello BOM prompt" in plain
        assert '"prompt"' not in plain

    def test_prompts_file_combined_with_prompt_option_cli(self, tmp_path, monkeypatch):
        """Both --prompts file rows and inline --prompt values reach Diff Plan and output."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        model_a = tmp_path / "model_a"
        model_b = tmp_path / "model_b"
        model_a.mkdir()
        model_b.mkdir()
        (tmp_path / "prompts.txt").write_text("From file prompt\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        with (
            patch("soup_cli.commands.diff._load_model", return_value=(None, None)),
            patch("soup_cli.commands.diff._generate", return_value="stub reply"),
        ):
            result = CliRunner().invoke(app, [
                "diff",
                "--model-a", str(model_a),
                "--model-b", str(model_b),
                "--prompts", "prompts.txt",
                "--prompt", "From inline prompt",
                "--device", "cpu",
            ])
        plain = _plain(result.output)
        assert result.exit_code == 0, plain
        assert "Prompts: 2" in plain
        assert "From file prompt" in plain
        assert "From inline prompt" in plain
