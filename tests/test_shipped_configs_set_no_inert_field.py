"""#1814 — no shipped recipe or template sets a field that does nothing.

An ad hoc check on the v0.76.0 tree loaded every recipe and found none that sets
a field in ``KNOWN_UNCONSUMED`` (the fields read by nothing), a staged field
(warned at load, refused at its deadline) or a deprecated value. This makes that
check a test, so a recipe cannot start shipping a setting that a user copying it
would get no effect from, or a warning they did not write.
"""

from __future__ import annotations

import warnings

import pytest
import yaml

from soup_cli.config.deprecation import SoupConfigDeprecationWarning
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig, remap_root_level_misplaced_keys
from soup_cli.config.staged_fields import find_staged_config_fields
from soup_cli.recipes.catalog import RECIPES
from soup_cli.templates import list_templates, load_template
from tests.test_issue748_config_fields_reach_a_consumer import KNOWN_UNCONSUMED


def _shipped() -> list[tuple[str, str]]:
    configs = [(f"recipe:{name}", meta.yaml_str) for name, meta in RECIPES.items()]
    configs += [(f"template:{name}", load_template(name)) for name in list_templates()]
    return configs


def inert_fields_set(text: str) -> list[str]:
    """Every field ``text`` sets that is unconsumed, staged, or a deprecated value."""
    loaded = yaml.safe_load(text) or {}
    raw = remap_root_level_misplaced_keys(loaded)
    found = []
    for section in ("training", "data"):
        for name in (raw.get(section) or {}):
            if f"{section}.{name}" in KNOWN_UNCONSUMED:
                found.append(f"{section}.{name} (read by nothing)")
    config = load_config_from_string(text)  # the shipped text must load at all
    # The loader turns deprecation warnings into console lines (#759), so they are
    # recorded here from the same construction it performs.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SoupConfig(**loaded)
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
    assert inert_fields_set(text) == [], name


class TestTheCheckCanFail:
    """Each of the three detectors, shown to fire on a config that trips it."""

    _BASE = "base: org/m\ntask: sft\ndata:\n  train: d.jsonl\n"

    def test_an_unconsumed_field_is_named(self):
        text = self._BASE + "training:\n  warmup_auto: true\n"
        assert "training.warmup_auto (read by nothing)" in inert_fields_set(text)

    def test_a_staged_field_is_named(self):
        text = self._BASE + "training:\n  forgetting_stop: true\n"
        assert "training.forgetting_stop (staged)" in inert_fields_set(text)

    def test_a_deprecated_value_is_named(self):
        text = self._BASE.replace("  train: d.jsonl\n",
                                  "  train: d.jsonl\n  remove_unused_columns: true\n")
        found = inert_fields_set(text)
        assert any(item.startswith("deprecated value:") for item in found), found
