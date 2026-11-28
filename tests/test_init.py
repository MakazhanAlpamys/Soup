"""Tests for soup init command."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import TEMPLATES
from soup_cli.templates import load_template
from tests.conftest import strip_ansi

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
    runner.invoke(
        app, ["init", "--template", "chat", "--output", str(output)], input="n\n"
    )
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


@pytest.mark.parametrize("name", ["soup.yaml", "[red]soup.yaml", "[/].yaml"])
def test_init_reports_literal_output_path_after_writing(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    output = Path(name)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(app, ["init", "--template", "chat", "--output", name])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert output.read_text(encoding="utf-8") == load_template("chat")
    text = " ".join(strip_ansi(result.output).split())
    assert f"Config saved to {output}" in text
    assert f"soup train --config {output}" in text


@pytest.mark.parametrize(
    ("template", "visible"),
    [
        ("[red]missing", "[red]missing"),
        ("[bold]missing", "[bold]missing"),
        ("bad\x1b\x07\x9b\u202ename", "badname"),
    ],
)
def test_init_unknown_template_is_literal_and_terminal_safe(
    tmp_path, monkeypatch, template, visible
):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "--template", template])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    text = strip_ansi(result.output)
    assert f"Unknown template: {visible}" in text
    assert "Available:" in text
    assert not any(char in text for char in ("\x1b", "\x07", "\x9b", "\u202e"))
    assert not Path("soup.yaml").exists()


def test_init_strips_controls_only_from_displayed_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = Path("[bold]config\x9b\u202e.yaml")

    result = runner.invoke(app, ["init", "--template", "chat", "--output", str(output)])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert output.read_text(encoding="utf-8") == load_template("chat")
    text = " ".join(strip_ansi(result.output).split())
    assert "Config saved to [bold]config.yaml" in text
    assert "soup train --config [bold]config.yaml" in text
    assert "\x9b" not in text and "\u202e" not in text


@pytest.mark.parametrize(
    ("data_present", "train_relpath"),
    [
        (False, None),
        (True, None),
        (False, "./custom_data/chat_train.jsonl"),
        (True, "./custom_data/chat_train.jsonl"),
    ],
)
def test_init_chat_template_missing_dataset_step(
    tmp_path: Path, monkeypatch, data_present: bool, train_relpath: str | None
):
    """When the chat template's local training file is absent, explain how to provide it."""
    monkeypatch.chdir(tmp_path)
    output_dir = Path("custom")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "soup.yaml"

    if train_relpath is not None:
        base_chat = load_template("chat")
        custom_chat = base_chat.replace("./data/train.jsonl", train_relpath)
        monkeypatch.setattr(
            "soup_cli.commands.init.load_template",
            lambda t: custom_chat if t == "chat" else load_template(t),
        )
        expected_path = train_relpath
        expected_template = custom_chat
    else:
        expected_path = "./data/train.jsonl"
        expected_template = load_template("chat")

    if data_present:
        data_file = Path(expected_path)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_text('{"text": "example"}\n', encoding="utf-8")

    result = runner.invoke(
        app,
        ["init", "--template", "chat", "--output", str(output)],
        env={"COLUMNS": "250"},
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert output.exists()
    assert output.read_text(encoding="utf-8") == expected_template

    text = " ".join(strip_ansi(result.output).split())
    assert f"Config saved to {output}" in text
    assert f"soup train --config {output}" in text

    if data_present:
        assert expected_path not in text
        assert "data.train" not in text
    else:
        assert expected_path in text
        assert "data.train" in text

