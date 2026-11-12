"""Catalog controls explicitly cover both supported and documented refused rows."""

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from soup_cli.recipes.catalog import RECIPES, RecipeMeta


def load_catalog_config(recipe: RecipeMeta) -> SoupConfig | None:
    """An unavailable row must really refuse; no skip or unchecked exception."""
    if recipe.unavailable_reason is not None:
        with pytest.raises(ValueError) as caught:
            load_config_from_string(recipe.yaml_str)
        message = str(caught.value)
        reason = recipe.unavailable_reason
        # Route refusals: the reason appears verbatim in the message. A review-gated
        # row (kimi-k2-sft, #2104 review round 2) appends its own reservation sentence
        # to the refusal text, so there the refusal — pydantic's "Value error, " wrapper
        # stripped — must appear verbatim inside the reason. Anything else fails.
        detail = message.split("Value error, ", 1)[-1]
        if reason not in message and detail not in reason:
            pytest.fail(
                f"{reason!r} does not match the config refusal {message!r}"
            )
        return None
    return load_config_from_string(recipe.yaml_str)


def load_shipped_config(name: str, text: str) -> SoupConfig | None:
    """Keep whole shipped-config sweeps strict, including unavailable recipes."""
    if name.startswith("recipe:"):
        recipe = RECIPES[name.removeprefix("recipe:")]
        assert text == recipe.yaml_str
        return load_catalog_config(recipe)
    return load_config_from_string(text)
