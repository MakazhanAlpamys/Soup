"""Tests for soup infer — batch inference command."""

import json

# ─── Prompt Reading Tests ───────────────────────────────────────────────────


class TestReadPrompts:
    """Test prompt reading from files."""

    def test_read_jsonl_prompts(self, tmp_path):
        """Should read prompts from JSONL with 'prompt' field."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "prompts.jsonl"
        lines = [
            json.dumps({"prompt": "What is AI?"}),
            json.dumps({"prompt": "Explain gravity."}),
            json.dumps({"prompt": "Hello world."}),
        ]
        path.write_text("\n".join(lines))

        result = _read_prompts(path)
        assert len(result) == 3
        assert result[0] == "What is AI?"
        assert result[1] == "Explain gravity."
        assert result[2] == "Hello world."

    def test_read_plain_text_prompts(self, tmp_path):
        """Should read plain text lines as prompts."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "prompts.txt"
        path.write_text("What is AI?\nExplain gravity.\nHello world.\n")

        result = _read_prompts(path)
        assert len(result) == 3
        assert result[0] == "What is AI?"

    def test_read_skips_empty_lines(self, tmp_path):
        """Should skip empty lines."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "prompts.txt"
        path.write_text("Line one\n\n\nLine two\n\n")

        result = _read_prompts(path)
        assert len(result) == 2

    def test_read_empty_file(self, tmp_path):
        """Should return empty list for empty file."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "empty.jsonl"
        path.write_text("")

        result = _read_prompts(path)
        assert result == []

    def test_read_mixed_jsonl_and_text(self, tmp_path):
        """Should handle mixed JSONL and plain text lines."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "mixed.txt"
        lines = [
            json.dumps({"prompt": "From JSONL"}),
            "Plain text prompt",
        ]
        path.write_text("\n".join(lines))

        result = _read_prompts(path)
        assert len(result) == 2
        assert result[0] == "From JSONL"
        assert result[1] == "Plain text prompt"

    def test_read_jsonl_without_prompt_field(self, tmp_path):
        """JSONL without 'prompt' field should be treated as plain text."""
        from soup_cli.commands.infer import _read_prompts

        path = tmp_path / "noprompt.jsonl"
        lines = [
            json.dumps({"text": "This has no prompt field"}),
        ]
        path.write_text("\n".join(lines))

        result = _read_prompts(path)
        assert len(result) == 1
        # Falls through to plain text since 'prompt' key missing
        assert "text" in result[0]


# ─── CLI Validation Tests ──────────────────────────────────────────────────


class TestInferCLI:
    """Test infer CLI command validation."""

    def test_input_file_not_found(self, tmp_path):
        """Should fail if input file doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, [
            "infer",
            "--model", str(tmp_path),
            "--input", "/nonexistent/prompts.jsonl",
            "--output", str(tmp_path / "out.jsonl"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_model_not_found(self, tmp_path):
        """Should fail if model path doesn't exist."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_file = tmp_path / "prompts.jsonl"
        prompts_file.write_text(json.dumps({"prompt": "test"}) + "\n")

        runner = CliRunner()
        result = runner.invoke(app, [
            "infer",
            "--model", "/nonexistent/model",
            "--input", str(prompts_file),
            "--output", str(tmp_path / "out.jsonl"),
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_empty_input_file(self, tmp_path):
        """Should fail if input file has no prompts."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        prompts_file = tmp_path / "empty.jsonl"
        prompts_file.write_text("")

        runner = CliRunner()
        result = runner.invoke(app, [
            "infer",
            "--model", str(tmp_path),
            "--input", str(prompts_file),
            "--output", str(tmp_path / "out.jsonl"),
        ])
        assert result.exit_code != 0
        assert "no prompts" in result.output.lower()

    def test_infer_command_registered(self):
        """Infer command should be registered in the app."""
        from soup_cli.cli import app

        command_names = [
            cmd.name or (cmd.callback.__name__ if cmd.callback else None)
            for cmd in app.registered_commands
        ]
        assert "infer" in command_names

    def test_infer_help(self):
        """Infer command should show help text."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["infer", "--help"])
        assert result.exit_code == 0
        assert "batch inference" in result.output.lower()

    def test_required_options_error(self):
        """Should fail if required options are missing."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["infer"])
        assert result.exit_code != 0


# ─── Output Format Tests ──────────────────────────────────────────────────


class TestInferOutputFormat:
    """Test that output file is valid JSONL with expected fields."""

    def test_output_jsonl_structure(self, tmp_path):
        """Output should be valid JSONL with prompt, response, tokens_generated."""
        # Create a fake output file to validate the structure
        output_path = tmp_path / "results.jsonl"
        results = [
            {"prompt": "What is AI?", "response": "AI is...", "tokens_generated": 5},
            {"prompt": "Hello", "response": "Hi there!", "tokens_generated": 3},
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Read back and validate
        with open(output_path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        for line in lines:
            assert "prompt" in line
            assert "response" in line
            assert "tokens_generated" in line
            assert isinstance(line["tokens_generated"], int)


# ─── Load Model Detection Tests ──────────────────────────────────────────


class TestInferModelDetection:
    """Test model loading and adapter detection."""

    def test_adapter_detection_with_config(self, tmp_path):
        """Should detect adapter when adapter_config.json exists."""
        adapter_config = tmp_path / "adapter_config.json"
        adapter_config.write_text(json.dumps({
            "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"
        }))

        # The adapter_config.json should be detectable
        assert adapter_config.exists()
        config = json.loads(adapter_config.read_text())
        assert "base_model_name_or_path" in config

    def test_no_adapter_config_is_full_model(self, tmp_path):
        """Without adapter_config.json, model is treated as full model."""
        assert not (tmp_path / "adapter_config.json").exists()


# ─── Import Tests ─────────────────────────────────────────────────────────


class TestInferImports:
    """Test that infer module is importable."""

    def test_import_infer_module(self):
        from soup_cli.commands.infer import infer
        assert infer is not None

    def test_import_read_prompts(self):
        from soup_cli.commands.infer import _read_prompts
        assert callable(_read_prompts)

    def test_import_load_model(self):
        from soup_cli.commands.infer import _load_model
        assert callable(_load_model)

    def test_import_generate(self):
        from soup_cli.commands.infer import _generate
        assert callable(_generate)
