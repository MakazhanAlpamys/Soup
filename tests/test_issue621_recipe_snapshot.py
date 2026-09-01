"""#621 — pin every recipe's *resolved* config, not its YAML source text.

162 recipes across the catalog declare training fields that happen to equal
the schema default (``dpo_beta: 0.1``, etc.). Deleting one of those lines is
an equivalent mutation — the resolved ``SoupConfig`` is byte-identical either
way — so no test should be built to kill it, and none here is.

The real exposure is the other direction: nothing pins what a recipe
*resolves to*, so a schema-default change silently retunes every recipe that
relies on that default, and the 36-ish recipes (per field) that omit the line
change behaviour with no red test anywhere. This snapshots the resolved
``SoupConfig.model_dump()`` for every recipe against a committed fixture, so
that kind of drift fails with a diff instead of shipping silently.

Regenerate the fixture with ``python scripts/generate_recipe_snapshot.py``
after a deliberate change — never automatically. The comparison is between
parsed Python objects (``json.load`` output vs. ``model_dump()``), not raw
file bytes, so it isn't sensitive to the CRLF/LF drift a byte-exact fixture
would be on a Windows checkout (the failure mode #580's review hit).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "recipe_config_snapshots.json"
)


def _load_fixture() -> dict[str, dict]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    """Flatten a nested dict to ``{"training.lora.r": 16, ...}`` for readable diffs."""
    out: dict[str, object] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, path))
        else:
            out[path] = value
    return out


def _diff(expected: dict, actual: dict) -> dict[str, tuple[object, object]]:
    """Paths present in either side whose values differ."""
    flat_expected = _flatten(expected)
    flat_actual = _flatten(actual)
    paths = set(flat_expected) | set(flat_actual)
    return {
        path: (flat_expected.get(path, "<missing>"), flat_actual.get(path, "<missing>"))
        for path in paths
        if flat_expected.get(path, "<missing>") != flat_actual.get(path, "<missing>")
    }


def _resolve(name: str) -> dict:
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.recipes.catalog import RECIPES

    return load_config_from_string(RECIPES[name].yaml_str).model_dump(mode="json")


class TestFixtureCoversExactlyTheCurrentCatalog:
    def test_fixture_and_catalog_have_the_same_recipe_names(self):
        """A recipe added or removed without regenerating fails here first,
        with a clear "which names" message, rather than as a KeyError deep
        in a parametrized test."""
        from soup_cli.recipes.catalog import RECIPES

        fixture_names = set(_load_fixture())
        catalog_names = set(RECIPES)
        assert fixture_names == catalog_names, (
            f"missing from fixture: {sorted(catalog_names - fixture_names)}; "
            f"stale in fixture: {sorted(fixture_names - catalog_names)}"
        )


class TestEveryRecipeMatchesItsCommittedSnapshot:
    @pytest.mark.parametrize("name", sorted(_load_fixture()))
    def test_recipe_resolves_to_its_snapshot(self, name: str):
        expected = _load_fixture()[name]
        actual = _resolve(name)
        diff = _diff(expected, actual)
        assert not diff, (
            f"recipe {name!r} no longer resolves to its committed snapshot. "
            f"If this is a deliberate change, review the diff and regenerate "
            f"with scripts/generate_recipe_snapshot.py. Differing fields "
            f"(field: (snapshot, resolved)): {diff}"
        )


class TestTheSnapshotIsNotVacuous:
    """Demonstrates the comparison actually discriminates, rather than
    merely asserting it does. Mutates a resolved value in-memory (no schema
    edit needed) and confirms the same diff logic the real test uses catches
    it — the failure mode this issue exists to prevent."""

    def test_a_changed_field_value_is_detected(self):
        name = "llama3.1-8b-dpo"
        fixture = _load_fixture()[name]
        mutated = _resolve(name)
        mutated["training"]["dpo_beta"] = mutated["training"]["dpo_beta"] + 1.0

        diff = _diff(fixture, mutated)
        assert diff == {"training.dpo_beta": (0.1, 1.1)}

    def test_an_unmutated_resolve_has_no_diff(self):
        """Negative control: same recipe, no mutation, must be silent."""
        name = "llama3.1-8b-dpo"
        fixture = _load_fixture()[name]
        actual = _resolve(name)

        assert _diff(fixture, actual) == {}
