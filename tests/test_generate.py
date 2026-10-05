"""Tests for soup data generate — synthetic data generation."""

import json


class TestParseJsonArray:
    """Test JSON array parsing from LLM output."""

    def test_parse_clean_json_array(self):
        """Should parse a clean JSON array."""
        from soup_cli.commands.generate import _parse_json_array

        content = json.dumps(
            [
                {"instruction": "What is AI?", "input": "", "output": "AI is..."},
                {"instruction": "Explain ML", "input": "", "output": "ML is..."},
            ]
        )
        result = _parse_json_array(content)
        assert len(result) == 2
        assert result[0]["instruction"] == "What is AI?"

    def test_parse_json_with_markdown_fences(self):
        """Should strip markdown code fences."""
        from soup_cli.commands.generate import _parse_json_array

        content = '```json\n[{"instruction": "test", "input": "", "output": "ok"}]\n```'
        result = _parse_json_array(content)
        assert len(result) == 1
        assert result[0]["instruction"] == "test"

    def test_parse_json_with_extra_text(self):
        """Should extract JSON array from surrounding text."""
        from soup_cli.commands.generate import _parse_json_array

        content = (
            'Here are the examples:\n[{"instruction": "a", "input": "", "output": "b"}]\nDone!'
        )
        result = _parse_json_array(content)
        assert len(result) == 1

    def test_parse_empty_content(self):
        """Should return empty list for empty content."""
        from soup_cli.commands.generate import _parse_json_array

        result = _parse_json_array("")
        assert result == []

    def test_parse_jsonl_fallback(self):
        """Should fall back to line-by-line JSON parsing."""
        from soup_cli.commands.generate import _parse_json_array

        content = (
            '{"instruction": "a", "input": "", "output": "b"}\n'
            '{"instruction": "c", "input": "", "output": "d"}'
        )
        result = _parse_json_array(content)
        assert len(result) == 2

    def test_parse_invalid_json(self):
        """Should return empty list for completely invalid JSON."""
        from soup_cli.commands.generate import _parse_json_array

        result = _parse_json_array("this is not json at all")
        assert result == []

    def test_parse_filters_non_dicts(self):
        """Should filter out non-dict items."""
        from soup_cli.commands.generate import _parse_json_array

        content = '[{"instruction": "a", "input": "", "output": "b"}, "string", 42]'
        result = _parse_json_array(content)
        assert len(result) == 1


class TestValidateExample:
    """Test example validation."""

    def test_validate_alpaca_valid(self):
        """Valid alpaca format should pass."""
        from soup_cli.commands.generate import _validate_example

        example = {"instruction": "test", "input": "", "output": "ok"}
        assert _validate_example(example, "alpaca") is True

    def test_validate_alpaca_missing_fields(self):
        """Alpaca missing required fields should fail."""
        from soup_cli.commands.generate import _validate_example

        assert _validate_example({"instruction": "test"}, "alpaca") is False
        assert _validate_example({"output": "test"}, "alpaca") is False

    def test_validate_sharegpt_valid(self):
        """Valid sharegpt format should pass."""
        from soup_cli.commands.generate import _validate_example

        example = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        assert _validate_example(example, "sharegpt") is True

    def test_validate_sharegpt_too_few(self):
        """Sharegpt with fewer than 2 messages should fail."""
        from soup_cli.commands.generate import _validate_example

        example = {"conversations": [{"from": "human", "value": "Hi"}]}
        assert _validate_example(example, "sharegpt") is False

    def test_validate_chatml_valid(self):
        """Valid chatml format should pass."""
        from soup_cli.commands.generate import _validate_example

        example = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        assert _validate_example(example, "chatml") is True

    def test_validate_chatml_empty(self):
        """Chatml with empty messages should fail."""
        from soup_cli.commands.generate import _validate_example

        assert _validate_example({"messages": []}, "chatml") is False

    def test_validate_unknown_format(self):
        """Unknown format should fail."""
        from soup_cli.commands.generate import _validate_example

        assert _validate_example({"data": "test"}, "unknown") is False


class TestBuildGenerationPrompt:
    """Test generation prompt building."""

    def test_prompt_includes_format_spec(self):
        """Prompt should include format specification."""
        from soup_cli.commands.generate import _build_generation_prompt

        result = _build_generation_prompt("Create math questions", 5, "alpaca", [])
        assert "instruction" in result
        assert "output" in result
        assert "5" in result

    def test_prompt_includes_seed_examples(self):
        """Prompt should include seed examples when provided."""
        from soup_cli.commands.generate import _build_generation_prompt

        seeds = [{"instruction": "example", "input": "", "output": "test"}]
        result = _build_generation_prompt("Create data", 3, "alpaca", seeds)
        assert "example" in result
        assert "seed" in result.lower()

    def test_prompt_sharegpt_format(self):
        """Prompt should describe sharegpt format correctly."""
        from soup_cli.commands.generate import _build_generation_prompt

        result = _build_generation_prompt("Create chats", 3, "sharegpt", [])
        assert "conversations" in result

    def test_prompt_chatml_format(self):
        """Prompt should describe chatml format correctly."""
        from soup_cli.commands.generate import _build_generation_prompt

        result = _build_generation_prompt("Create chats", 3, "chatml", [])
        assert "messages" in result


class TestRowToText:
    """Test row to text conversion for dedup."""

    def test_row_to_text_basic(self):
        """Should concatenate all values."""
        from soup_cli.commands.generate import _row_to_text

        row = {"instruction": "What is AI?", "output": "AI is..."}
        text = _row_to_text(row)
        assert "What is AI?" in text
        assert "AI is..." in text

    def test_row_to_text_empty_values(self):
        """Should skip empty values."""
        from soup_cli.commands.generate import _row_to_text

        row = {"instruction": "test", "input": "", "output": "ok"}
        text = _row_to_text(row)
        assert "test" in text


class TestGenerateCLI:
    """Test CLI integration for generate command."""

    def test_invalid_format_rejected(self):
        """Should reject invalid format."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "test",
                "--format",
                "invalid_format",
                "--count",
                "1",
            ],
        )
        assert result.exit_code != 0

    def test_invalid_provider_rejected(self):
        """Should reject invalid provider."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "data",
                "generate",
                "--prompt",
                "test",
                "--provider",
                "invalid_provider",
                "--count",
                "1",
            ],
        )
        assert result.exit_code != 0
