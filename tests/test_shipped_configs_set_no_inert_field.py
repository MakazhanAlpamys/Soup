"""#1814 — no shipped recipe or template sets a field that does nothing.

An ad hoc check on the v0.76.0 tree loaded every recipe and found none that sets
a field in ``KNOWN_UNCONSUMED`` (the fields read by nothing), a staged field
(warned at load in v0.76, refused at load as of v0.77) or a deprecated value
(the same). This makes that check a test, so a recipe cannot start shipping a
setting that a user copying it would get no effect from, or a refusal they did
not write.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import replace
from unittest import mock

import pytest
import yaml

from soup_cli.config import deprecation, loader
from soup_cli.config.deprecation import (
    DEPRECATED_VALUE_REJECTION_VERSION,
    SoupConfigDeprecationWarning,
)
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig, remap_root_level_misplaced_keys
from soup_cli.config.staged_fields import (
    STAGED_FIELD_REJECTION_VERSION,
    find_staged_config_fields,
)
from soup_cli.recipes.catalog import RECIPES
from soup_cli.templates import list_templates, load_template
from tests.test_issue748_config_fields_reach_a_consumer import KNOWN_UNCONSUMED


def _shipped() -> list[tuple[str, str]]:
    configs = [(f"recipe:{name}", meta.yaml_str) for name, meta in RECIPES.items()]
    configs += [(f"template:{name}", load_template(name)) for name in list_templates()]
    return configs


def inert_fields_set(text: str, *, name: str = "") -> list[str]:
    """Every field ``text`` sets that is unconsumed, staged, or a deprecated value."""
    loaded = yaml.safe_load(text) or {}
    raw = remap_root_level_misplaced_keys(loaded)
    found = []
    for section in ("training", "data"):
        for field in (raw.get(section) or {}):
            if f"{section}.{field}" in KNOWN_UNCONSUMED:
                found.append(f"{section}.{field} (read by nothing)")
    # A recipe documented as unavailable (#2130) must refuse with its own reason.
    reason = None
    if name.startswith("recipe:"):
        recipe = RECIPES[name.removeprefix("recipe:")]
        assert text == recipe.yaml_str
        reason = recipe.unavailable_reason
    # The shipped text must load at all. As of v0.77 a staged field or a deprecated
    # value is refused right here (#808, #759, #795), so the refusal is recorded;
    # so is a recipe documented as unavailable that loads after all.
    try:
        load_config_from_string(text)
    except ValueError as exc:
        if reason is None or reason not in str(exc):
            found.append(f"refused at load: {exc}")
    else:
        if reason is not None:
            found.append(f"loads although documented as unavailable: {reason}")
    # Each item is also named on its own, as v0.76 reported them: the staged-field
    # scan does not depend on the switch, and under the warning setting a deprecated
    # value warns rather than raising, so every one is listed, not only the first.
    # The warnings are recorded from the same construction the loader performs.
    with mock.patch.object(
        deprecation, "DEPRECATED_VALUE_SEVERITY", "warn"
    ), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if reason is not None:
            # A documented refusal still validates nested fields and emits their
            # warnings before the root Kimi guard. Keep inspecting all raw fields.
            with pytest.raises(ValueError, match=re.escape(reason)):
                SoupConfig(**loaded)
            config = None
        else:
            config = SoupConfig(**loaded)
    found += [f"{item.path} (staged)" for item in find_staged_config_fields(raw, config=config)]
    found += [
        f"deprecated value: {w.message}"
        for w in caught if issubclass(w.category, SoupConfigDeprecationWarning)
    ]
    return sorted(set(found))


def test_the_catalog_is_not_empty():
    # A loop over nothing passes; the catalog has 169 recipes since pixtral-dpo was dropped (#1867).
    assert len(RECIPES) >= 169
    assert len(list_templates()) >= 10


def test_the_enumeration_covers_both_catalogs_exactly():
    # #1948 review: the checks above read the catalogs, while the parametrized
    # test gets its cases from _shipped(). Pin that the two are the same set, so an
    # empty enumeration or a lost branch (recipes or templates) fails here by name.
    # There are no intentional exclusions.
    enumerated = [name for name, _ in _shipped()]
    expected = {f"recipe:{name}" for name in RECIPES} | {
        f"template:{name}" for name in list_templates()
    }
    assert enumerated, "the shipped-config enumeration is empty"
    assert len(enumerated) == len(set(enumerated)), "an id is enumerated twice"
    missing = sorted(expected - set(enumerated))
    extra = sorted(set(enumerated) - expected)
    assert not missing, f"shipped configs the test never checks: {missing[:10]}"
    assert not extra, f"enumerated ids that are not shipped configs: {extra[:10]}"
    assert all(text for _, text in _shipped()), "a shipped config enumerated with no text"


@pytest.mark.parametrize("name,text", _shipped(), ids=[n for n, _ in _shipped()])
def test_a_shipped_config_sets_no_inert_field(name, text):
    assert inert_fields_set(text, name=name) == [], name


@pytest.mark.parametrize("name", ["kimi-k2-sft", "kimi-k2-thinking-grpo"])
@pytest.mark.parametrize("section,field,value,expected", [
    ("training", "warmup_auto", True, "training.warmup_auto (read by nothing)"),
    ("training", "forgetting_stop", True, "training.forgetting_stop (staged)"),
    ("data", "remove_unused_columns", True, "deprecated value:"),
])
def test_refused_recipes_still_check_every_inert_field(
    monkeypatch, name, section, field, value, expected,
):
    recipe = RECIPES[name]
    raw = yaml.safe_load(recipe.yaml_str)
    raw.setdefault(section, {})[field] = value
    text = yaml.safe_dump(raw)
    monkeypatch.setitem(RECIPES, name, replace(recipe, yaml_str=text))
    found = inert_fields_set(text, name=f"recipe:{name}")
    assert any(item.startswith(expected) for item in found), found


class TestTheCheckCanFail:
    """Each of the three detectors, shown to fire on a config that trips it."""

    _BASE = "base: org/m\ntask: sft\ndata:\n  train: d.jsonl\n"

    def test_an_unconsumed_field_is_named(self):
        text = self._BASE + "training:\n  warmup_auto: true\n"
        assert "training.warmup_auto (read by nothing)" in inert_fields_set(text)

    # Each detector is pinned under both settings of the switch it describes, set
    # here, so the test does not depend on the release's constant: the item is named
    # either way, and the load refusal is recorded only under "error".

    @pytest.mark.parametrize("severity", ["warn", "error"])
    def test_a_staged_field_is_named(self, monkeypatch, severity):
        monkeypatch.setattr(loader, "STAGED_FIELD_SEVERITY", severity)
        text = self._BASE + "training:\n  forgetting_stop: true\n"
        found = inert_fields_set(text)
        assert "training.forgetting_stop (staged)" in found, found
        refusal = (
            "refused at load: training.forgetting_stop is accepted but read by nothing; "
            f"refused as of v{STAGED_FIELD_REJECTION_VERSION}. Delete the line."
        )
        assert (refusal in found) is (severity == "error"), found

    @pytest.mark.parametrize("severity", ["warn", "error"])
    def test_a_deprecated_value_is_named(self, monkeypatch, severity):
        monkeypatch.setattr(deprecation, "DEPRECATED_VALUE_SEVERITY", severity)
        text = self._BASE.replace("  train: d.jsonl\n",
                                  "  train: d.jsonl\n  remove_unused_columns: true\n")
        found = inert_fields_set(text)
        assert any(item.startswith("deprecated value:") for item in found), found
        refusal = (
            "refused at load: data -> remove_unused_columns: Value error, "
            "data.remove_unused_columns: true never had an effect (no trainer reads it; "
            "the trainers that set it pass false). Delete the line. "
            f"Refused as of Soup v{DEPRECATED_VALUE_REJECTION_VERSION}."
        )
        assert (refusal in found) is (severity == "error"), found
