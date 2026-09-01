"""#621 / #638 — pin every recipe's *resolved* config, not its YAML source
text, without storing the same ~296 schema defaults 162 times over.

162 recipes across the catalog declare training fields that happen to equal
the schema default (``dpo_beta: 0.1``, etc.). Deleting one of those lines is
an equivalent mutation — the resolved ``SoupConfig`` is byte-identical either
way — so no test should be built to kill it, and none here is (#621).

The real exposure is the other direction: nothing pins what a recipe
*resolves to*, so a schema-default change silently retunes every recipe that
relies on that default, with no red test anywhere. #621's original fixture
pinned this by storing each recipe's full resolved config — but that meant
162 copies of the same ~296 defaults (49,586 lines, 1.55 MB), and a single
schema-default change reddened all 149-odd recipes that didn't override it,
identically to how an *added* field reddened all 162 — the same wall of red
for two different classes of change (#638).

This fixture now stores the shared defaults ONCE, as ``baseline`` — a
minimal resolved config (placeholder base/task/data, everything else at
schema default) — and per recipe only the fields that differ from it
(``recipes``). Two consequences:

- A schema-default change (or an added field) moves the ONE committed
  baseline value. ``TestBaselineMatchesTheLiveSchema`` is the only place
  that shows up — not once per recipe.
- Per-recipe tests compare each recipe's delta *against the live baseline*,
  not the committed one, so they stay silent for every recipe that doesn't
  touch the changed field. The one case this doesn't erase: a recipe that
  redundantly pins a value equal to the OLD default starts appearing in its
  delta once the default moves out from under it — because what used to be
  a no-op pin just started doing something. That is a real, if small, class
  of failure, called out here rather than rounded down to "one failure,
  full stop" — see ``TestEveryRecipeMatchesItsCommittedDelta``.

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

from scripts.generate_recipe_snapshot import _delta, build_baseline, flatten
from soup_cli.recipes.catalog import RECIPES

_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "recipe_config_snapshots.json"
)
_REGENERATE_HINT = "regenerate with scripts/generate_recipe_snapshot.py"
_MISSING = object()


@functools.lru_cache(maxsize=1)
def _load_fixture() -> dict:
    data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "baseline" in data and "recipes" in data, (
        f"fixture must be a dict with 'baseline' and 'recipes' keys; {_REGENERATE_HINT}"
    )
    return data


# Forces a collection-time failure (not a per-test runtime error) if the
# fixture is missing or malformed — matches #621's original property that a
# deleted fixture is loud immediately, not 162 identical errors.
_load_fixture()


@functools.lru_cache(maxsize=1)
def _live_baseline() -> dict[str, object]:
    return build_baseline()


def _resolve_flat(name: str) -> dict[str, object]:
    from soup_cli.config.loader import load_config_from_string

    return flatten(load_config_from_string(RECIPES[name].yaml_str).model_dump(mode="json"))


def _diff(expected: dict, actual: dict) -> dict[str, tuple[object, object]]:
    """Paths present in either side whose values differ.

    Compares against an internal sentinel, not the string ``"<missing>"`` —
    a real config value equal to that string must never compare as
    spuriously equal to an absent one. The sentinel only ever appears in
    the returned tuples as a stand-in for display.
    """
    paths = set(expected) | set(actual)
    out: dict[str, tuple[object, object]] = {}
    for path in paths:
        e = expected.get(path, _MISSING)
        a = actual.get(path, _MISSING)
        if e != a:
            out[path] = (
                "<missing>" if e is _MISSING else e,
                "<missing>" if a is _MISSING else a,
            )
    return out


class TestBaselineMatchesTheLiveSchema:
    """The one place a schema-default change (or a newly added field) shows
    up. Everything downstream is checked relative to this baseline, not the
    live schema directly, so a shift here is exactly the signal that "the
    shared defaults moved" — not one identical copy of that signal per
    recipe that happens not to override the changed field."""

    def test_baseline_matches_committed_snapshot(self):
        fixture = _load_fixture()
        diff = _diff(fixture["baseline"], _live_baseline())
        assert not diff, (
            f"the shared schema-default baseline has changed. If this is "
            f"deliberate, review the diff and {_REGENERATE_HINT}. Differing "
            f"fields (field: (snapshot, resolved)): {diff}"
        )


class TestFixtureCoversExactlyTheCurrentCatalog:
    def test_fixture_and_catalog_have_the_same_recipe_names(self):
        """A recipe added or removed without regenerating fails here first,
        with a clear "which names" message, rather than as a KeyError deep
        in a parametrized test."""
        fixture_names = set(_load_fixture()["recipes"])
        catalog_names = set(RECIPES)
        assert fixture_names == catalog_names, (
            f"missing from fixture: {sorted(catalog_names - fixture_names)}; "
            f"stale in fixture: {sorted(fixture_names - catalog_names)}; "
            f"{_REGENERATE_HINT}"
        )


class TestEveryRecipeMatchesItsCommittedDelta:
    """Compares each recipe's delta against the LIVE baseline, not the
    committed one — a schema-default change alone must not move this test,
    only ``TestBaselineMatchesTheLiveSchema`` above. The one case that still
    reaches here: a recipe whose own declared value happens to equal the OLD
    default stops being a no-op once the default moves, and starts
    appearing in its delta — correctly, since it is now load-bearing where
    it previously was not."""

    @pytest.mark.parametrize("name", sorted(RECIPES))
    def test_recipe_delta_matches_its_snapshot(self, name: str):
        fixture = _load_fixture()
        assert name in fixture["recipes"], (
            f"{name!r} is in the catalog but missing from the snapshot fixture; "
            f"{_REGENERATE_HINT}"
        )
        live_delta = _delta(_resolve_flat(name), _live_baseline())
        diff = _diff(fixture["recipes"][name], live_delta)
        assert not diff, (
            f"recipe {name!r} no longer resolves to its committed delta. "
            f"If this is a deliberate change, review the diff and {_REGENERATE_HINT}. "
            f"Differing fields (field: (snapshot, resolved)): {diff}"
        )


class TestTheSnapshotIsNotVacuous:
    """Demonstrates the comparison actually discriminates, rather than
    merely asserting it does. Exercises `_diff` directly on synthetic
    dicts, not a live recipe — an unrelated schema addition reddens the
    baseline test already, and coupling these controls to the same live
    data would redden them too, making "did the guard break, or did the
    data move?" unanswerable from the log."""

    def test_a_changed_field_value_is_detected(self):
        expected = {"training.dpo_beta": 0.1, "training.epochs": 3}
        actual = {"training.dpo_beta": 0.2, "training.epochs": 3}

        assert _diff(expected, actual) == {"training.dpo_beta": (0.1, 0.2)}

    def test_identical_dicts_have_no_diff(self):
        """Negative control: same values, must be silent."""
        expected = {"training.dpo_beta": 0.1, "training.epochs": 3}
        actual = {"training.dpo_beta": 0.1, "training.epochs": 3}

        assert _diff(expected, actual) == {}
