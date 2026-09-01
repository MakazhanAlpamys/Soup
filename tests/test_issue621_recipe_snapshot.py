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

import functools
import json
from pathlib import Path

import pytest

from soup_cli.recipes.catalog import RECIPES

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "recipe_config_snapshots.json"
)
_REGENERATE_HINT = "regenerate with scripts/generate_recipe_snapshot.py"


def _load_fixture() -> dict[str, dict]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _live_snapshots() -> dict[str, dict]:
    """The same resolution `scripts/generate_recipe_snapshot.py` uses to
    build the committed fixture, called here instead of reimplemented.

    Two independent copies of `model_dump(mode="json")` over
    `load_config_from_string(...)` — one in this file, one in the script —
    is exactly the shape this repo has been bitten by before (#372, #392):
    a fix or a behavior change lands in one and the other silently drifts,
    at which point the fixture becomes unregenerable and nobody learns
    until a release. Cached so the whole catalog resolves once per test
    session rather than once per parametrized recipe.
    """
    from scripts.generate_recipe_snapshot import build_snapshot

    return build_snapshot()


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
    return _live_snapshots()[name]


class TestFixtureCoversExactlyTheCurrentCatalog:
    def test_fixture_and_catalog_have_the_same_recipe_names(self):
        """A recipe added or removed without regenerating fails here first,
        with a clear "which names" message, rather than as a KeyError deep
        in a parametrized test."""
        fixture_names = set(_load_fixture())
        catalog_names = set(RECIPES)
        assert fixture_names == catalog_names, (
            f"missing from fixture: {sorted(catalog_names - fixture_names)}; "
            f"stale in fixture: {sorted(fixture_names - catalog_names)}; "
            f"{_REGENERATE_HINT}"
        )


class TestEveryRecipeMatchesItsCommittedSnapshot:
    @pytest.mark.parametrize("name", sorted(RECIPES))
    def test_recipe_resolves_to_its_snapshot(self, name: str):
        fixture = _load_fixture()
        assert name in fixture, (
            f"{name!r} is in the catalog but missing from the snapshot fixture; "
            f"{_REGENERATE_HINT}"
        )
        diff = _diff(fixture[name], _resolve(name))
        assert not diff, (
            f"recipe {name!r} no longer resolves to its committed snapshot. "
            f"If this is a deliberate change, review the diff and {_REGENERATE_HINT}. "
            f"Differing fields (field: (snapshot, resolved)): {diff}"
        )


class TestTheSnapshotIsNotVacuous:
    """Demonstrates the comparison actually discriminates, rather than
    merely asserting it does. Exercises `_diff` directly on synthetic
    dicts, not a live recipe — an unrelated schema addition reddens every
    real per-recipe test already, and coupling these controls to the same
    live data would redden them too, making "did the guard break, or did
    the data move?" unanswerable from the log."""

    def test_a_changed_field_value_is_detected(self):
        expected = {"training": {"dpo_beta": 0.1, "epochs": 3}}
        actual = {"training": {"dpo_beta": 0.2, "epochs": 3}}

        assert _diff(expected, actual) == {"training.dpo_beta": (0.1, 0.2)}

    def test_identical_dicts_have_no_diff(self):
        """Negative control: same values, must be silent."""
        expected = {"training": {"dpo_beta": 0.1, "epochs": 3}}
        actual = {"training": {"dpo_beta": 0.1, "epochs": 3}}

        assert _diff(expected, actual) == {}
