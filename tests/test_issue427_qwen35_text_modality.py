"""Regression coverage for #427's deliberate Qwen3.5 text-only recipes."""

from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.recipes.catalog import RECIPES

EXPECTED_QWEN35_TEXT_RECIPES = {
    "qwen3.5-0.8b-sft",
    "qwen3.5-2b-sft",
    "qwen3.5-4b-sft",
    "qwen3.5-4b-pretrain",
    "qwen3.5-9b-sft",
    "qwen3.5-9b-grpo",
    "qwen3.5-27b-sft",
    "qwen3.5-35b-a3b-sft",
    "qwen3.5-35b-a3b-dpo",
    "qwen3.5-122b-a10b-sft",
    "qwen3.5-397b-a17b-sft",
    # Qwen3.6 exposes the same qwen3_5 / qwen3_5_moe architecture on the Hub.
    "qwen3.6-27b-sft",
    "qwen3.6-35b-a3b-sft",
    # Qwen3.8-27B exposes the same qwen3_5 text tower.
    "qwen3.8-27b-sft",
}


def _qwen35_family_recipe_names() -> tuple[str, ...]:
    """Select every catalog recipe covered by the measured architecture decision."""
    prefixes = ("Qwen/Qwen3.5-", "Qwen/Qwen3.6-", "Qwen/Qwen3.8-")
    return tuple(name for name, recipe in RECIPES.items() if recipe.model.startswith(prefixes))


QWEN35_TEXT_RECIPES = _qwen35_family_recipe_names()


def test_qwen35_family_audit_covers_every_current_recipe() -> None:
    """A newly catalogued sibling must join the explicit-modality contract."""
    assert set(QWEN35_TEXT_RECIPES) == EXPECTED_QWEN35_TEXT_RECIPES


@pytest.mark.parametrize("name", QWEN35_TEXT_RECIPES)
def test_qwen35_family_recipes_explicitly_select_language_tower(name: str) -> None:
    """Do not let the decoder-only decision fall back to the schema default."""
    recipe = RECIPES[name]
    raw = yaml.safe_load(recipe.yaml_str)
    config = load_config_from_string(recipe.yaml_str)

    assert raw["modality"] == "text"
    assert config.modality == "text"
