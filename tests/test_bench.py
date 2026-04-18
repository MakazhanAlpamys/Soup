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
    
    with patch("soup_cli.commands.bench._load_model") as mock_load, \
         patch("soup_cli.commands.bench._generate") as mock_generate:
        
        mock_load.return_value = ("mock_model", "mock_tokenizer")
        mock_generate.return_value = (None, 10)
        
        # Test 1: TXT
        result = runner.invoke(app, ["bench", str(dummy_model), "--prompts-file", "prompts.txt"])
        assert result.exit_code == 0
        assert "Running 3 test inferences" in result.output
        
        # Test 2: JSONL
        result = runner.invoke(app, ["bench", str(dummy_model), "--prompts-file", "prompts.jsonl"])
        assert result.exit_code == 0
        assert "Running 3 test inferences" in result.output
        
        # Test 3: Path outside CWD
        result = runner.invoke(
            app, ["bench", str(dummy_model), "--prompts-file", str(outside_file)]
        )
        assert result.exit_code == 1
        assert "Security Error" in result.output
