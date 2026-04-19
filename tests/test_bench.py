"""Tests for soup bench CLI command."""

from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def test_bench_model_not_found():
    """soup bench with nonexistent model should fail gracefully."""
    result = runner.invoke(app, ["bench", "nonexistent_model_path"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_bench_custom_prompts(tmp_path, monkeypatch):
    """Test using custom prompts from a text file and JSONL."""
    monkeypatch.chdir(tmp_path)

    dummy_model = tmp_path / "dummy_model"
    dummy_model.mkdir()

    # Text file
    prompts_txt = tmp_path / "prompts.txt"
    prompts_txt.write_text("Custom prompt 1\nCustom prompt 2\n")

    # JSONL file
    prompts_jsonl = tmp_path / "prompts.jsonl"
    prompts_jsonl.write_text('{"prompt": "JSON prompt 1"}\n{"prompt": "JSON prompt 2"}\n')

    # Path traversal
    outside_file = tmp_path.parent / "outside.txt"
    outside_file.write_text("Outside\n")

    from unittest.mock import patch

    with patch("soup_cli.commands.infer._load_model") as mock_load, \
         patch("soup_cli.commands.infer._generate") as mock_generate:

        mock_load.return_value = ("mock_model", "mock_tokenizer")
        mock_generate.return_value = (None, 10)

        # Test 1: TXT -- verify exit, output, and that actual prompts were passed to _generate
        result = runner.invoke(app, ["bench", str(dummy_model), "--prompts-file", "prompts.txt"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Running 2 test inferences" in result.output

        used_contents = [
            call.args[2][0]["content"]
            for call in mock_generate.call_args_list
        ]
        assert "Custom prompt 1" in used_contents
        assert "Custom prompt 2" in used_contents

        mock_generate.reset_mock()

        # Test 2: JSONL -- verify JSON prompt field was extracted and used
        result = runner.invoke(app, ["bench", str(dummy_model), "--prompts-file", "prompts.jsonl"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "Running 2 test inferences" in result.output

        used_contents = [
            call.args[2][0]["content"]
            for call in mock_generate.call_args_list
        ]
        assert "JSON prompt 1" in used_contents
        assert "JSON prompt 2" in used_contents

        # Test 3: Path outside CWD -- security check
        result = runner.invoke(
            app, ["bench", str(dummy_model), "--prompts-file", str(outside_file)]
        )
        assert result.exit_code == 1
        assert "Security Error" in result.output


def test_bench_happy_path(tmp_path, monkeypatch):
    """Test full benchmark end-to-end happy path without actual models."""
    monkeypatch.chdir(tmp_path)

    dummy_model = tmp_path / "dummy_model"
    dummy_model.mkdir()

    from unittest.mock import patch

    with patch("soup_cli.commands.infer._load_model") as mock_load, \
         patch("soup_cli.commands.infer._generate") as mock_generate, \
         patch("torch.cuda.is_available") as mock_is_available, \
         patch("torch.cuda.reset_peak_memory_stats"), \
         patch("torch.cuda.max_memory_allocated") as mock_max_memory, \
         patch("soup_cli.utils.gpu.detect_device") as mock_detect_device:

        mock_load.return_value = ("mock_model", "mock_tokenizer")
        mock_generate.return_value = ("mock response", 128)
        mock_is_available.return_value = True
        mock_max_memory.return_value = 4 * 1024**3
        mock_detect_device.return_value = ("cuda", 0)

        result = runner.invoke(app, ["bench", str(dummy_model)])

        assert result.exit_code == 0, result.output
        assert "Inference Benchmark Results" in result.output
        assert "TPS (Avg)" in result.output
        assert "128 tokens" in result.output


def test_bench_cpu_warning(tmp_path, monkeypatch):
    """Test that CPU warning is shown on non-CUDA devices."""
    monkeypatch.chdir(tmp_path)

    dummy_model = tmp_path / "dummy_model"
    dummy_model.mkdir()

    from unittest.mock import patch

    with patch("soup_cli.commands.infer._load_model") as mock_load, \
         patch("soup_cli.commands.infer._generate") as mock_generate, \
         patch("torch.cuda.is_available") as mock_is_available, \
         patch("soup_cli.utils.gpu.detect_device") as mock_detect_device:

        mock_load.return_value = ("mock_model", "mock_tokenizer")
        mock_generate.return_value = ("mock response", 10)
        mock_is_available.return_value = False
        mock_detect_device.return_value = ("cpu", None)

        result = runner.invoke(app, ["bench", str(dummy_model)])

        assert result.exit_code == 0, result.output
        assert "Running on CPU" in result.output
        assert "Inference Benchmark Results" in result.output
