"""Tests for soup diff — model comparison command."""

import json

import pytest


@pytest.mark.parametrize('field', ['model_a', 'model_b', 'base_a', 'base_b'])
def test_reward_model_diff_refused_before_device_or_prompt_work(tmp_path, field):
    from soup_cli.commands.diff import diff
    from tests.reward_guard_helpers import assert_reward_guard

    root = tmp_path / 'reward'
    root.mkdir()
    (root / 'reward_model.json').write_text('broken')
    kwargs = dict(model_a='ordinary-a', model_b='ordinary-b', base_a=None, base_b=None)
    kwargs[field] = str(root)
    assert_reward_guard(diff, kwargs, 'path_a = Path(model_a)')


@pytest.mark.parametrize('field', ['model_path', 'base_model', 'metadata'])
def test_reward_diff_loader_refused_before_heavy_imports(tmp_path, field):
    from soup_cli.commands.diff import _load_model
    from tests.reward_guard_helpers import assert_reward_guard

    root = tmp_path / 'reward'
    root.mkdir()
    (root / 'reward_model.json').write_text('broken')
    kwargs = dict(model_path='ordinary', base_model=None, device='cpu')
    if field == 'metadata':
        adapter = tmp_path / 'adapter'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text(json.dumps({'model': str(root)}))
        kwargs['model_path'] = str(adapter)
    else:
        kwargs[field] = str(root)
    assert_reward_guard(
        _load_model, kwargs, 'from transformers import AutoModelForCausalLM, AutoTokenizer',
    )


def test_ordinary_diff_models_reach_existing_work():
    from soup_cli.commands.diff import _load_model, diff
    from tests.reward_guard_helpers import assert_reward_guard

    assert_reward_guard(diff, dict(model_a='ordinary-a', model_b='ordinary-b'),
                        'path_a = Path(model_a)', refused=False)
    assert_reward_guard(
        _load_model, dict(model_path='ordinary', base_model=None, device='cpu'),
        'from transformers import AutoModelForCausalLM, AutoTokenizer', refused=False,
    )


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
