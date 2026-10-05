"""Tests for soup init command."""

from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import TEMPLATES

runner = CliRunner()


def test_init_chat_template(tmp_path: Path):
    """Init with chat template should create a valid config file."""
    output = tmp_path / "soup.yaml"
    result = runner.invoke(app, ["init", "--template", "chat", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    content = output.read_text()
    assert "Llama-3.1-8B-Instruct" in content
    assert "sft" in content


def test_init_code_template(tmp_path: Path):
    """Init with code template should create a valid config file."""
    output = tmp_path / "soup.yaml"
    result = runner.invoke(app, ["init", "--template", "code", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    content = output.read_text()
    assert "CodeLlama" in content


def test_init_medical_template(tmp_path: Path):
    """Init with medical template should create a valid config file."""
    output = tmp_path / "soup.yaml"
    result = runner.invoke(app, ["init", "--template", "medical", "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists()
    content = output.read_text()
    assert "medical" in content.lower() or "Llama" in content


def test_init_unknown_template():
    """Unknown template should fail."""
    result = runner.invoke(app, ["init", "--template", "nonexistent"])
    assert result.exit_code == 1
    assert "Unknown template" in result.output


def test_init_overwrite_denied(tmp_path: Path):
    """If output exists and user denies overwrite, should exit."""
    output = tmp_path / "soup.yaml"
    output.write_text("existing content")
    # typer.confirm will get "n" from stdin
    runner.invoke(app, ["init", "--template", "chat", "--output", str(output)], input="n\n")
    # Should exit without overwriting
    assert output.read_text() == "existing content"


def test_init_overwrite_confirmed(tmp_path: Path):
    """If output exists and user confirms overwrite, should create new file."""
    output = tmp_path / "soup.yaml"
    output.write_text("old content")
    result = runner.invoke(
        app, ["init", "--template", "chat", "--output", str(output)], input="y\n"
    )
    assert result.exit_code == 0
    content = output.read_text()
    assert "old content" not in content
    assert "Llama" in content


def test_all_templates_exist():
    """All expected templates should be registered."""
    assert "chat" in TEMPLATES
    assert "code" in TEMPLATES
    assert "medical" in TEMPLATES


def test_templates_are_valid_yaml():
    """Each template should be parseable YAML with expected keys."""
    import yaml

    for name, text in TEMPLATES.items():
        data = yaml.safe_load(text)
        assert "base" in data, f"Template '{name}' missing 'base' key"
        assert "data" in data, f"Template '{name}' missing 'data' key"
        assert "training" in data, f"Template '{name}' missing 'training' key"
