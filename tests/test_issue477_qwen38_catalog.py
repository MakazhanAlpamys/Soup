"""Regression coverage for #477's Qwen3.8-27B text-only SFT recipe."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.recipes.catalog import get_recipe, search_recipes

from .conftest import strip_ansi

runner = CliRunner()


def test_qwen38_27b_recipe_has_expected_text_sft_shape() -> None:
    recipe = get_recipe("qwen3.8-27b-sft")

    assert recipe is not None
    assert recipe.model == "Qwen/Qwen3.8-27B"
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
    assert [recipe.model for recipe in matches] == ["Qwen/Qwen3.8-27B"]


def test_qwen38_27b_recipe_show_and_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    show_result = runner.invoke(app, ["recipes", "show", "qwen3.8-27b-sft"])
    assert show_result.exit_code == 0
    assert "Qwen/Qwen3.8-27B" in show_result.output
    # #633: "modality: text" spans three Pygments tokens, so escapes land
    # inside it whenever colour is enabled. The model id above is one token and
    # survives raw, which is why only this line broke.
    assert "modality: text" in strip_ansi(show_result.output)

    monkeypatch.chdir(tmp_path)
    use_result = runner.invoke(
        app,
        ["recipes", "use", "qwen3.8-27b-sft", "--yes"],
    )
    assert use_result.exit_code == 0
    content = (tmp_path / "soup.yaml").read_text(encoding="utf-8")
    assert "base: Qwen/Qwen3.8-27B" in content
    assert "modality: text" in content
