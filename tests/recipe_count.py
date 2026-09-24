"""The recipe count, written down once (#1016).

Adding a recipe changes ``len(RECIPES)``. The milestone assertions that pin the
Python dictionary compare against ``EXPECTED_RECIPE_COUNT`` here instead of
carrying their own literal, so that is the one number to edit in ``tests/``.
The documentation sites that state the count are listed, and checked against
the catalog, in ``tests/test_recipe_count_is_synced.py``.
"""

from __future__ import annotations

EXPECTED_RECIPE_COUNT = 175


def recipe_count_hint(actual: int) -> str:
    """The failure message a contributor adding a recipe sees."""
    return (
        f"RECIPES has {actual} entries but tests/recipe_count.py pins "
        f"EXPECTED_RECIPE_COUNT = {EXPECTED_RECIPE_COUNT}. If you added or removed a "
        "recipe on purpose, set EXPECTED_RECIPE_COUNT to the new count, then run "
        "python scripts/sync_recipe_count.py to update the documentation sites "
        "that state it (tests/test_recipe_count_is_synced.py lists them, and names "
        "each file and line that still holds the old number)."
    )
