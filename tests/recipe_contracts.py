"""Catalog controls explicitly cover both supported and documented refused rows."""

import re

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from soup_cli.recipes.catalog import RECIPES, RecipeMeta


def load_catalog_config(recipe: RecipeMeta) -> SoupConfig | None:
    """An unavailable row must really refuse; no skip or unchecked exception."""
    if recipe.unavailable_reason is not None:
        with pytest.raises(ValueError, match=re.escape(recipe.unavailable_reason)):
            load_config_from_string(recipe.yaml_str)
        return None
    return load_config_from_string(recipe.yaml_str)


def load_shipped_config(name: str, text: str) -> SoupConfig | None:
    """Keep whole shipped-config sweeps strict, including unavailable recipes."""
    if name.startswith("recipe:"):
        recipe = RECIPES[name.removeprefix("recipe:")]
        assert text == recipe.yaml_str
        return load_catalog_config(recipe)
    return load_config_from_string(text)
