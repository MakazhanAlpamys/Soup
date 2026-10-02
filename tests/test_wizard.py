"""Tests for soup wizard and soup init --wizard smart setup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch as mock_patch

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import SoupConfig

runner = CliRunner()


def test_wizard_command_registered():
    """soup wizard --help should succeed and describe the command."""
    result = runner.invoke(app, ["wizard", "--help"])
    assert result.exit_code == 0
    assert "Smart interactive setup wizard" in result.output


def test_init_wizard_flag_registered():
    """soup init --help should list the --wizard flag."""
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    assert "--wizard" in result.output or "-w" in result.output


def test_wizard_interactive_chat_with_dataset(tmp_path: Path):
    """Smart wizard with chat goal and real dataset generates validated SFT config."""
    data_file = tmp_path / "chat_data.jsonl"
    data_rows = [
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"},
            ]
        }
        for _ in range(5)
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in data_rows) + "\n",
        encoding="utf-8",
    )

    output_yaml = tmp_path / "soup.yaml"
    with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "1",  # chat goal
        str(data_file),
    ]):
        result = runner.invoke(app, ["wizard", "--output", str(output_yaml)])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert output_yaml.exists()
    content = output_yaml.read_text(encoding="utf-8")
    assert "base: meta-llama/Llama-3.1-8B-Instruct" in content
    assert "task: sft" in content
    assert "format: chatml" in content

    # Verify Pydantic schema validation passes on the generated output
    import yaml

    loaded_dict = yaml.safe_load(content)
    validated_cfg = SoupConfig.model_validate(loaded_dict)
    assert validated_cfg.task == "sft"
    assert validated_cfg.data.format == "chatml"


def test_wizard_interactive_reasoning(tmp_path: Path):
    """Smart wizard with reasoning goal generates GRPO config."""
    data_file = tmp_path / "math_data.jsonl"
    data_rows = [
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 is 4."},
            ]
        }
        for _ in range(3)
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in data_rows) + "\n",
        encoding="utf-8",
    )

    output_yaml = tmp_path / "soup.yaml"
    with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=[
        "Qwen/Qwen2.5-7B-Instruct",
        "2",  # reasoning goal
        str(data_file),
    ]):
        result = runner.invoke(app, ["wizard", "--output", str(output_yaml)])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    content = output_yaml.read_text(encoding="utf-8")
    assert "task: grpo" in content


def test_wizard_interactive_alignment(tmp_path: Path):
    """Smart wizard with alignment goal generates DPO config."""
    data_file = tmp_path / "pref_data.jsonl"
    data_rows = [
        {
            "prompt": "Explain photosynthesis",
            "chosen": "Photosynthesis is the process...",
            "rejected": "Plants eat sunlight.",
        }
        for _ in range(3)
    ]
    data_file.write_text(
        "\n".join(json.dumps(row) for row in data_rows) + "\n",
        encoding="utf-8",
    )

    output_yaml = tmp_path / "soup.yaml"
    with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "3",  # alignment goal
        str(data_file),
    ]):
        result = runner.invoke(app, ["wizard", "--output", str(output_yaml)])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    content = output_yaml.read_text(encoding="utf-8")
    assert "task: dpo" in content
    assert "format: dpo" in content


def test_wizard_nonexistent_dataset_fallback(tmp_path: Path):
    """When dataset does not exist locally yet, wizard generates safe plan config."""
    output_yaml = tmp_path / "soup.yaml"
    missing_data = tmp_path / "missing_data.jsonl"

    with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "1",  # chat goal
        str(missing_data),
    ]):
        result = runner.invoke(app, ["wizard", "--output", str(output_yaml)])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert output_yaml.exists()
    content = output_yaml.read_text(encoding="utf-8")
    assert "task: sft" in content

    import yaml

    loaded_dict = yaml.safe_load(content)
    validated_cfg = SoupConfig.model_validate(loaded_dict)
    assert validated_cfg.task == "sft"


def test_init_wizard_flag_equivalent(tmp_path: Path):
    """soup init --wizard should execute the smart wizard path."""
    output_yaml = tmp_path / "soup.yaml"
    missing_data = tmp_path / "data.jsonl"

    with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "1",
        str(missing_data),
    ]):
        result = runner.invoke(app, ["init", "--wizard", "--output", str(output_yaml)])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert output_yaml.exists()
    content = output_yaml.read_text(encoding="utf-8")
    assert "task: sft" in content


def test_wizard_overwrite_protection(tmp_path: Path):
    """Output overwrite should be protected unless user confirms or passes --force."""
    target = tmp_path / "soup.yaml"
    target.write_text("base: existing_content", encoding="utf-8")

    # Denied with 'n'
    result = runner.invoke(app, ["wizard", "--output", str(target)], input="n\n")
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == "base: existing_content"

    # Confirmed with --force
    missing_data = tmp_path / "data.jsonl"
    with mock_patch("soup_cli.commands.init.Prompt.ask", side_effect=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "1",
        str(missing_data),
    ]):
        result_force = runner.invoke(
            app, ["wizard", "--output", str(target), "--force"]
        )
    assert result_force.exit_code == 0
    assert target.read_text(encoding="utf-8") != "base: existing_content"

