"""Recipe arguments are data when rendered, and stay unchanged on disk."""

from pathlib import Path

import pytest

from soup_cli.cli import app
from soup_cli.recipes.catalog import get_recipe
from soup_cli.utils.terminal import strip_control
from tests.conftest import make_cli_runner, strip_ansi

runner = make_cli_runner()


@pytest.mark.parametrize("command", ["show", "use"])
@pytest.mark.parametrize(
    "name", ["missing[/]recipe", "missing[bold]recipe", "bad\x1b]0;title\x07", "missing:warning:"],
)
def test_unknown_recipe_is_rendered_literally(command, name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["recipes", command, name])

    assert result.exit_code == 1
    text = strip_ansi(result.stdout)
    assert f"Recipe not found: {strip_control(name)}" in text
    assert "\x1b]0;" not in result.stdout
    assert "MarkupError" not in text
    assert not (tmp_path / "soup.yaml").exists()


@pytest.mark.parametrize("output", ["café[red].yaml", "cfg[/]tail.yaml", "plain.yaml"])
def test_recipe_output_is_literal_and_bytes_are_unchanged(output, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    recipe = get_recipe("llama3.1-8b-sft")

    result = runner.invoke(app, ["recipes", "use", "llama3.1-8b-sft", "-o", output, "-y"])

    assert result.exit_code == 0, result.output
    text = strip_ansi(result.stdout)
    assert f"written to {output}" in text
    assert f"Next: soup train --config {output}" in text
    assert path.read_bytes() == recipe.yaml_str.encode("utf-8")


def test_refused_recipe_output_keeps_the_path_literal(tmp_path, monkeypatch):
    working = tmp_path / "work"
    working.mkdir()
    monkeypatch.chdir(working)
    output = "../outside[red].yaml"

    result = runner.invoke(app, ["recipes", "use", "llama3.1-8b-sft", "-o", output, "-y"])

    assert result.exit_code == 1
    expected = f"Output path is outside the current directory: {Path(output)}"
    assert expected in strip_ansi(result.stdout)
    assert not (tmp_path / "outside[red].yaml").exists()


@pytest.mark.parametrize("name", ["existing[red].yaml", "existing\u202e[red].yaml"])
def test_existing_recipe_output_confirmation_preserves_literal_path(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    output = Path(name)
    output.write_text("keep this", encoding="utf-8")

    result = runner.invoke(
        app, ["recipes", "use", "llama3.1-8b-sft", "-o", str(output)], input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert f"File '{strip_control(output)}' already exists. Overwrite?" in strip_ansi(result.stdout)
    assert "\u202e" not in result.stdout
    assert output.read_text(encoding="utf-8") == "keep this"
