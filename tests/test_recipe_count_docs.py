import re
from pathlib import Path
from typing import List, Tuple

from soup_cli.recipes.catalog import RECIPES


def test_recipe_count_in_docs() -> None:
    """Assert that the recipe count in documentation matches the actual number of recipes.

    This prevents the count drifting silently out of date when recipes are added.
    The four existing test_catalog_size_is_N assertions are left alone because they
    document the historical release growth and catch backward drift, but this test
    specifically guards the user-facing documentation numbers.
    """
    actual_count = len(RECIPES)
    root = Path(__file__).parent.parent

    # Each site is (file_path_relative_to_root, regex_pattern)
    sites: List[Tuple[str, str]] = [
        ("src/soup_cli/recipes/catalog.py", r"Recipe catalog \((\d+) recipes\)"),
        ("CONTRIBUTING.md", r"\((\d+) recipes\)"),
        ("docs/commands.md", r"List all (\d+) ready-made"),
        ("docs/serving-and-export.md", r"(\d+) ready-made recipes"),
        ("docs/serving-and-export.md", r"\((\d+) recipes\)"),
    ]

    for rel_path, pattern in sites:
        file_path = root / rel_path
        assert file_path.exists(), f"Doc file {rel_path} does not exist."
        content = file_path.read_text(encoding="utf-8")

        matches = re.findall(pattern, content)
        assert (
            matches
        ), f"Pattern '{pattern}' stopped matching in {rel_path}. If the wording changed, update the test."

        for match in matches:
            found_count = int(match)
            assert found_count == actual_count, (
                f"Recipe count mismatch in {rel_path}: found {found_count}, expected {actual_count}. "
                "Did you add a recipe and forget to update the docs?"
            )
