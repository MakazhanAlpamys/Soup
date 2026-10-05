"""Tests for soup diff — model comparison command."""

import json

import pytest


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
                    "len_a": 5,
                    "len_b": 5,
                    "words_a": 1,
                    "words_b": 1,
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
        result = runner.invoke(
            app,
            [
                "diff",
                "--model-a",
                "/nonexistent/model_a",
                "--model-b",
                ".",
                "--prompt",
                "test",
            ],
        )
        assert result.exit_code != 0

    def test_model_b_not_found(self, tmp_path):
        """Should fail if model B doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "diff",
                "--model-a",
                str(tmp_path),
                "--model-b",
                "/nonexistent/model_b",
                "--prompt",
                "test",
            ],
        )
        assert result.exit_code != 0

    def test_no_prompts_error(self, tmp_path):
        """Should fail if no prompts provided."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "diff",
                "--model-a",
                str(tmp_path),
                "--model-b",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0
