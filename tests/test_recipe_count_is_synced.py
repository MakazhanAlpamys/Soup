"""Repo-wide ratchet: recipe count in documentation must match the catalog.

When new recipes land in the catalog, the recipe count is stated across multiple
documentation and module docstring sites. When two recipe PRs merge sequentially,
Git's 3-way merge conflicts loudly on Python test files, but silently auto-merges
Markdown documentation lines, leaving documentation counts quietly out of date.

This test derives the expected count dynamically from `len(RECIPES)` and scans every
declared documentation site in `DOC_SITES` to guarantee that documentation and code never drift.

Note on existing test assertions:
The existing `test_catalog_size_is_N` assertions in `test_recipes.py`,
`test_v07124.py`, `test_v07130.py`, and `test_v07132.py` are deliberately left
alone as release milestone checkpoints for the Python dictionary. This file
specifically guards documentation and docstring synchronization against silent
Git auto-merge drift (#453).
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import Sequence

from soup_cli.recipes.catalog import RECIPES

ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DocCountSite:
    """A target documentation site declaring a recipe count."""

    rel_path: str
    pattern: str


@dataclass(frozen=True)
class DocCountMatch:
    """A single matched recipe count found at a specific file and line."""

    rel_path: str
    lineno: int
    pattern: str
    count: int


@dataclass(frozen=True)
class SyncVerdict:
    """The result of auditing recipe counts across documentation sites."""

    expected_count: int
    matches: tuple[DocCountMatch, ...]
    mismatches: tuple[DocCountMatch, ...]
    missing_patterns: tuple[DocCountSite, ...]

    @property
    def is_synced(self) -> bool:
        """True if all declared patterns matched and every count equals expected."""
        return len(self.mismatches) == 0 and len(self.missing_patterns) == 0

    def format_diagnostic(self) -> str:
        """Generate a human-readable diagnostic report for CI failure messages."""
        lines: list[str] = []
        if self.missing_patterns:
            lines.append("Missing pattern matches (reworded or deleted doc lines):")
            for site in self.missing_patterns:
                lines.append(f"  * {site.rel_path} with pattern {site.pattern!r}")
        if self.mismatches:
            lines.append(
                f"Out-of-sync recipe counts (catalog has {self.expected_count} recipes):"
            )
            for m in self.mismatches:
                lines.append(
                    f"  * {m.rel_path}:{m.lineno} declares {m.count} (pattern: {m.pattern!r})"
                )
        return "\n".join(lines)


#: Declared documentation and module sites that state the recipe count.
DOC_SITES: tuple[DocCountSite, ...] = (
    DocCountSite(
        "src/soup_cli/recipes/catalog.py", r"#\s*Recipe catalog\s*\((\d+)\s*recipes\)"
    ),
    DocCountSite("CONTRIBUTING.md", r"\((\d+)\s+recipes\)"),
    DocCountSite(
        "docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"
    ),
    DocCountSite(
        "docs/serving-and-export.md",
        r"templates or\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes",
    ),
    DocCountSite(
        "docs/serving-and-export.md", r"recipe dropdown\s+\((\d+)\s+recipes\)"
    ),
)

#: Mandatory baseline roster pairs that must never drop out of DOC_SITES discovery.
MANDATORY_SITE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (
        ("src/soup_cli/recipes/catalog.py", r"#\s*Recipe catalog\s*\((\d+)\s*recipes\)"),
        ("CONTRIBUTING.md", r"\((\d+)\s+recipes\)"),
        ("docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"),
        ("docs/serving-and-export.md", r"templates or\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"),
        ("docs/serving-and-export.md", r"recipe dropdown\s+\((\d+)\s+recipes\)"),
    )
)


def audit_doc_recipe_counts(
    root: pathlib.Path = ROOT,
    sites: Sequence[DocCountSite] = DOC_SITES,
    expected_count: int | None = None,
) -> SyncVerdict:
    """Pure auditor: inspect documentation sites without assertion side-effects.

    Returns an immutable `SyncVerdict` containing all matches, mismatches, and
    missing patterns.
    """
    if expected_count is None:
        expected_count = len(RECIPES)

    matches: list[DocCountMatch] = []
    mismatches: list[DocCountMatch] = []
    missing_patterns: list[DocCountSite] = []

    for site in sites:
        file_path = root / site.rel_path
        if not file_path.exists():
            missing_patterns.append(site)
            continue
        text = file_path.read_text(encoding="utf-8")
        found_matches = list(re.finditer(site.pattern, text, flags=re.IGNORECASE))
        if not found_matches:
            missing_patterns.append(site)
            continue
        for match in found_matches:
            lineno = text.count("\n", 0, match.start()) + 1
            count = int(match.group(1))
            entry = DocCountMatch(
                rel_path=site.rel_path,
                lineno=lineno,
                pattern=site.pattern,
                count=count,
            )
            matches.append(entry)
            if count != expected_count:
                mismatches.append(entry)

    return SyncVerdict(
        expected_count=expected_count,
        matches=tuple(matches),
        mismatches=tuple(mismatches),
        missing_patterns=tuple(missing_patterns),
    )


class TestRecipeCountIsSynchronised:
    """Every declared documentation site must match the true catalog size."""

    def test_every_documentation_site_matches_catalog_size(self) -> None:
        verdict = audit_doc_recipe_counts(ROOT, DOC_SITES, len(RECIPES))
        assert verdict.is_synced, (
            f"Recipe count in documentation is out of sync:\n{verdict.format_diagnostic()}\n"
            "Update the documentation sites to match the catalog count."
        )
        assert len(verdict.matches) >= len(DOC_SITES)

    def test_all_declared_sites_are_covered(self) -> None:
        """Control: ensure no declared site or pattern drops out of discovery."""
        current_pairs = {(s.rel_path, s.pattern) for s in DOC_SITES}
        missing_pairs = MANDATORY_SITE_PAIRS - current_pairs
        assert not missing_pairs, (
            f"Mandatory documentation pattern dropped from DOC_SITES: {missing_pairs}"
        )

        verdict = audit_doc_recipe_counts(ROOT, DOC_SITES, len(RECIPES))
        assert verdict.missing_patterns == ()
        assert len(verdict.matches) >= len(DOC_SITES)

    def test_catalog_size_matches_actual_recipes_dictionary(self) -> None:
        """Control: ensure the catalog dictionary itself is non-empty."""
        assert len(RECIPES) > 0


class TestTheGuardHasTeeth:
    """CONTROL. Prove the auditor catches stale counts, reworded lines, and mutations."""

    def test_stale_doc_count_produces_unsynced_verdict(
        self, tmp_path: pathlib.Path
    ) -> None:
        stale_file = tmp_path / "docs" / "commands.md"
        stale_file.parent.mkdir(parents=True, exist_ok=True)
        stale_file.write_text(
            "soup recipes list  List all 144 ready-made recipes\n",
            encoding="utf-8",
        )

        sites = (
            DocCountSite("docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"),
        )
        verdict = audit_doc_recipe_counts(tmp_path, sites, expected_count=200)
        assert not verdict.is_synced
        assert len(verdict.mismatches) == 1
        assert verdict.mismatches[0].count == 144
        assert verdict.mismatches[0].lineno == 1
        assert "docs/commands.md:1 declares 144" in verdict.format_diagnostic()

    def test_duplicate_correct_mentions_in_single_file_are_accepted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """CONTROL: Multiple valid count mentions in a single file must pass without error."""
        doc_file = tmp_path / "CONTRIBUTING.md"
        doc_file.write_text(
            "recipes/ - Ready-made models (200 recipes)\n"
            "See full list in catalog (200 recipes)\n",
            encoding="utf-8",
        )
        sites = (DocCountSite("CONTRIBUTING.md", r"\((\d+)\s+recipes\)"),)
        verdict = audit_doc_recipe_counts(tmp_path, sites, expected_count=200)
        assert verdict.is_synced
        assert len(verdict.matches) == 2
        assert verdict.mismatches == ()
        assert verdict.missing_patterns == ()

    def test_mixed_counts_in_single_file_flags_stale_line(
        self, tmp_path: pathlib.Path
    ) -> None:
        """CONTROL: If a file has one correct count and one stale count, flag the stale line."""
        doc_file = tmp_path / "CONTRIBUTING.md"
        doc_file.write_text(
            "Line 1: (200 recipes)\n"
            "Line 2: (199 recipes)\n",
            encoding="utf-8",
        )
        sites = (DocCountSite("CONTRIBUTING.md", r"\((\d+)\s+recipes\)"),)
        verdict = audit_doc_recipe_counts(tmp_path, sites, expected_count=200)
        assert not verdict.is_synced
        assert len(verdict.matches) == 2
        assert len(verdict.mismatches) == 1
        assert verdict.mismatches[0].lineno == 2
        assert verdict.mismatches[0].count == 199

    def test_dropped_site_from_doc_sites_fails_roster_check(self) -> None:
        """CONTROL: Dropping any mandatory site/pattern from DOC_SITES
        must fail the roster check."""
        mutated_sites = tuple(
            s for s in DOC_SITES if s.rel_path != "docs/commands.md"
        )
        current_pairs = {(s.rel_path, s.pattern) for s in mutated_sites}
        missing_pairs = MANDATORY_SITE_PAIRS - current_pairs
        assert missing_pairs, "Expected dropped site to be identified as missing."
        assert any(p[0] == "docs/commands.md" for p in missing_pairs)

    def test_reworded_or_missing_pattern_produces_missing_patterns_verdict(
        self, tmp_path: pathlib.Path
    ) -> None:
        reworded_file = tmp_path / "CONTRIBUTING.md"
        reworded_file.write_text(
            "recipes/ - Ready-made model configurations\n",
            encoding="utf-8",
        )

        sites = (DocCountSite("CONTRIBUTING.md", r"\((\d+)\s+recipes\)"),)
        verdict = audit_doc_recipe_counts(tmp_path, sites, expected_count=200)
        assert not verdict.is_synced
        assert len(verdict.missing_patterns) == 1
        assert verdict.missing_patterns[0].rel_path == "CONTRIBUTING.md"
        assert "Missing pattern matches" in verdict.format_diagnostic()

    def test_missing_file_produces_missing_patterns_verdict(
        self, tmp_path: pathlib.Path
    ) -> None:
        sites = (DocCountSite("non_existent.md", r"\((\d+)\s+recipes\)"),)
        verdict = audit_doc_recipe_counts(tmp_path, sites, expected_count=200)
        assert not verdict.is_synced
        assert len(verdict.missing_patterns) == 1
        assert verdict.missing_patterns[0].rel_path == "non_existent.md"

    def test_mutated_catalog_count_fails_all_sites(self) -> None:
        fake_count = len(RECIPES) + 1
        verdict = audit_doc_recipe_counts(ROOT, DOC_SITES, expected_count=fake_count)
        assert not verdict.is_synced
        assert len(verdict.mismatches) >= len(DOC_SITES)
