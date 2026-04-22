"""Tests for soup cost command."""

import json

from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


def test_cost_with_config_file(tmp_path):
    """Test basic cost table output."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  epochs: 3\n"
        "  batch_size: 4\n"
        "  quantization: 4bit\n"
        "  lora:\n"
        "    r: 64\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file)])
    assert result.exit_code == 0
    assert "Training Cost Estimate" in result.output
    assert "Provider" in result.output
    assert "RunPod" in result.output


def test_cost_json_output(tmp_path):
    """Test JSON output for automation."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  batch_size: 4\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file), "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "total_cost" in data[0]
    assert "provider" in data[0]


def test_cost_with_gpu_filter(tmp_path):
    """Test filtering by specific GPU."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  batch_size: 4\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file), "--gpu", "H100"])
    assert result.exit_code == 0
    assert "H100" in result.output
    # Assuming the mock data only has RTX 4090 and we filtered to H100
    assert "RTX 4090" not in result.output


def test_cost_with_unknown_gpu(tmp_path):
    """Test filtering by unknown GPU."""
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: meta-llama/Llama-3.1-8B-Instruct\n"
        "task: sft\n"
        "data:\n"
        "  train: ./data/train.jsonl\n"
        "  max_length: 2048\n"
        "training:\n"
        "  batch_size: 4\n"
        "output: ./output\n"
    )
    result = runner.invoke(app, ["cost", "--config", str(config_file), "--gpu", "nonexistent"])
    assert result.exit_code == 1
    assert "No matching GPUs found" in result.output


def test_cost_missing_config():
    """Test cost fails gracefully when config doesn't exist."""
    result = runner.invoke(app, ["cost", "--config", "nonexistent.yaml"])
    assert result.exit_code != 0
