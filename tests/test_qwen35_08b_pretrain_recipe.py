"""CPU-only configuration checks for the Qwen3.5 0.8B Base recipe (#275)."""

from dataclasses import asdict

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.recipes.catalog import RECIPES
from tests.conftest import strip_ansi

NAME = "qwen3.5-0.8b-pretrain"
BASE = "Qwen/Qwen3.5-0.8B-Base"


@pytest.mark.parametrize("size", ["2", "4"])
def test_recipe_inherits_all_pretraining_settings(size: str) -> None:
    recipe = RECIPES[NAME]
    sibling = RECIPES[f"qwen3.5-{size}b-pretrain"]
    assert recipe.model == BASE
    assert recipe.size == "0.8B"
    assert "Apache-2.0" in recipe.description

    actual = yaml.safe_load(recipe.yaml_str)
    expected = yaml.safe_load(sibling.yaml_str)
    assert actual.pop("base") == BASE
    assert expected.pop("base") == f"Qwen/Qwen3.5-{size}B-Base"
    assert actual == expected

    excluded = {"model", "size", "tags", "description", "yaml_str"}
    assert {k: v for k, v in asdict(recipe).items() if k not in excluded} == {
        k: v for k, v in asdict(sibling).items() if k not in excluded
    }


def test_show_and_use_produce_the_08b_base_config(tmp_path, monkeypatch) -> None:
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    shown = runner.invoke(app, ["recipes", "show", NAME])
    assert shown.exit_code == 0, (shown.output, repr(shown.exception))
    assert BASE in strip_ansi(shown.output)

    used = runner.invoke(app, ["recipes", "use", NAME, "--yes"])
    assert used.exit_code == 0, (used.output, repr(used.exception))
    text = (tmp_path / "soup.yaml").read_text(encoding="utf-8")
    assert yaml.safe_load(text) == yaml.safe_load(RECIPES[NAME].yaml_str)
    config = load_config_from_string(text)
    assert (config.base, config.task, config.modality) == (BASE, "pretrain", "text")
