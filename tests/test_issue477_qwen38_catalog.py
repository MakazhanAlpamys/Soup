"""Regression coverage for Qwen3.8-27B text recipes (#477, #275)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.recipes.catalog import RECIPES, get_recipe, search_recipes

from .conftest import strip_ansi

runner = CliRunner()
BASE = "Qwen/Qwen3.8-27B"
DPO_RECIPE = "qwen3.8-27b-dpo"


def test_qwen38_27b_recipe_has_expected_text_sft_shape() -> None:
    recipe = get_recipe("qwen3.8-27b-sft")

    assert recipe is not None
    assert recipe.model == BASE
    assert recipe.task == "sft"
    assert recipe.size == "27B"
    assert {"qwen", "qwen3.8", "sft", "text"}.issubset(recipe.tags)

    config = load_config_from_string(recipe.yaml_str)
    assert config.base == recipe.model
    assert config.task == "sft"
    assert config.modality == "text"
    assert config.data.max_length == 4096
    assert config.training.quantization == "4bit"
    assert config.training.lora.r == 16
    assert config.training.lora.alpha == 32
    assert config.training.lora.target_modules == "auto"


def test_qwen38_27b_recipe_is_searchable() -> None:
    matches = search_recipes(query="qwen3.8")
    assert get_recipe("qwen3.8-27b-sft") in matches


def test_qwen38_27b_dpo_metadata() -> None:
    recipe = get_recipe(DPO_RECIPE)
    assert recipe is not None
    assert recipe.model == BASE
    assert recipe.task == "dpo"
    assert recipe.size == "27B"
    assert {"qwen", "qwen3.8", "dpo", "alignment", "preference", "text"} <= set(recipe.tags)
    assert recipe in search_recipes(query="qwen3.8", task="dpo", size="27b")


def test_qwen38_27b_dpo_inherits_qwen35_27b_settings() -> None:
    config = load_config_from_string(RECIPES[DPO_RECIPE].yaml_str)
    donor = load_config_from_string(RECIPES["qwen3.5-27b-dpo"].yaml_str)

    assert config.base == BASE
    assert config.model_dump(exclude={"base"}) == donor.model_dump(exclude={"base"})


@pytest.mark.parametrize(("name", "task"), [("qwen3.8-27b-sft", "sft"), (DPO_RECIPE, "dpo")])
def test_qwen38_27b_recipe_show_and_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    task: str,
) -> None:
    show_result = runner.invoke(app, ["recipes", "show", name])
    assert show_result.exit_code == 0
    assert BASE in show_result.output
    # #633: "modality: text" spans three Pygments tokens, so escapes land
    # inside it whenever colour is enabled. The model id above is one token and
    # survives raw, which is why only this line broke.
    assert "modality: text" in strip_ansi(show_result.output)

    monkeypatch.chdir(tmp_path)
    use_result = runner.invoke(
        app,
        ["recipes", "use", name, "--yes"],
    )
    assert use_result.exit_code == 0
    content = (tmp_path / "soup.yaml").read_text(encoding="utf-8")
    config = load_config_from_string(content)
    assert config.base == BASE
    assert config.task == task
    assert config.modality == "text"
    assert content == RECIPES[name].yaml_str
