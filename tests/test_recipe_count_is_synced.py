"""Repo-wide ratchet: recipe count in documentation must match the catalog.

When new recipes land in the catalog, the recipe count is stated across multiple
documentation and module docstring sites. When two recipe PRs merge sequentially,
Git's 3-way merge conflicts loudly on Python test files, but silently auto-merges
Markdown documentation lines, leaving documentation counts quietly out of date.

This test derives the expected count dynamically from `len(RECIPES)` and scans every
declared documentation site in `DOC_SITES` to guarantee that documentation and code never drift.

The Python-dictionary milestone assertions in `test_recipes.py`, `test_v07124.py`,
`test_v07130.py` and `test_v07132.py` stay, because they pin `RECIPES` itself,
which the documentation audit does not. Since #1016 they compare against
`EXPECTED_RECIPE_COUNT` in `tests/recipe_count.py` instead of four copies of one
literal, and `MILESTONE_SITES` below keeps a literal from coming back. A recipe
count stated in any other tracked file is detected too (`TestNoUndeclaredCountSite`),
so a new doc line cannot sit outside `DOC_SITES` unnoticed.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import Sequence

import pytest

from soup_cli.recipes.catalog import RECIPES
from tests.recipe_count import EXPECTED_RECIPE_COUNT, recipe_count_hint

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


#: The milestone test files that pin ``len(RECIPES)``. Each must compare against
#: ``EXPECTED_RECIPE_COUNT`` and must not carry a literal of its own (#1016).
MILESTONE_SITES: tuple[str, ...] = (
    "tests/test_recipes.py",
    "tests/test_v07124.py",
    "tests/test_v07130.py",
    "tests/test_v07132.py",
)

_LITERAL_MILESTONE = re.compile(r"len\(RECIPES\)\s*==\s*\d+")

#: A recipe count stated in prose: ``176 recipes``, ``176 ready-made recipes``. A
#: preceding ``#`` is an issue or PR reference (``pre-#330 recipes``), not a count.
_STATED_COUNT = re.compile(
    r"(?<![#\w])(\d{2,4})\s+(?:ready[-\s]?made\s+)?recipes\b", re.IGNORECASE
)

_SCANNED_SUFFIXES = frozenset(
    {".md", ".py", ".txt", ".yaml", ".yml", ".toml", ".json", ".html", ".rst", ".cfg"}
)

#: Tracked paths whose recipe counts are history rather than a statement about
#: the current catalog, so they legitimately hold old numbers.
_HISTORICAL_PREFIXES: tuple[str, ...] = (
    "CHANGELOG.md",
    "CONTRIBUTORS.md",
    "changelog.d/",
    # Tests state counts in fixtures and in milestone narration; the milestone
    # assertions themselves are covered by MILESTONE_SITES.
    "tests/",
)

#: Individual lines that state a count on purpose and must not follow the
#: catalog, as ``(rel_path, text on that line)`` so an entry survives edits above it.
_DATED_MEASUREMENTS: frozenset[tuple[str, str]] = frozenset(
    {
        # The recipe-repo-ids sweep's measured cost on 2026-09-06: a dated benchmark.
        (".github/workflows/recipe-repo-ids.yml", "165 recipes / 330 surfaces"),
    }
)


def literal_milestone_sites(
    root: pathlib.Path, rel_paths: Sequence[str] = MILESTONE_SITES
) -> list[str]:
    """``file: literal`` for every milestone file that pins a count by literal."""
    found: list[str] = []
    for rel in rel_paths:
        text = (root / rel).read_text(encoding="utf-8")
        found.extend(f"{rel}: {literal}" for literal in _LITERAL_MILESTONE.findall(text))
    return found


def _tracked_files(root: pathlib.Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:  # pragma: no cover - not a git checkout (sdist)
        pytest.skip("not a git checkout; tracked files cannot be listed")
    return [path for path in result.stdout.split("\0") if path]


def undeclared_count_sites(
    root: pathlib.Path,
    rel_paths: Sequence[str],
    sites: Sequence[DocCountSite] = DOC_SITES,
) -> list[str]:
    """``file:line: text`` for every stated recipe count outside the declared tables."""
    declared = {
        (m.rel_path, m.lineno)
        for m in audit_doc_recipe_counts(root, sites, expected_count=len(RECIPES)).matches
    }
    found: list[str] = []
    for rel in rel_paths:
        if pathlib.PurePosixPath(rel).suffix.lower() not in _SCANNED_SUFFIXES:
            continue
        if rel.startswith(_HISTORICAL_PREFIXES):
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = text.splitlines()
        for match in _STATED_COUNT.finditer(text):
            lineno = text.count("\n", 0, match.start()) + 1
            if (rel, lineno) in declared:
                continue
            line = lines[lineno - 1]
            if any(rel == dated and needle in line for dated, needle in _DATED_MEASUREMENTS):
                continue
            found.append(f"{rel}:{lineno}: {line.strip()}")
    return found


class TestRecipeCountIsSynchronised:
    """Every declared documentation site must match the true catalog size."""

    def test_the_pinned_count_matches_the_catalog(self) -> None:
        assert len(RECIPES) == EXPECTED_RECIPE_COUNT, recipe_count_hint(len(RECIPES))

    def test_milestone_sites_use_the_pinned_count_not_a_literal(self) -> None:
        literals = literal_milestone_sites(ROOT)
        assert not literals, (
            "These milestone files pin the recipe count with a literal; compare against "
            "EXPECTED_RECIPE_COUNT from tests/recipe_count.py so adding a recipe is one "
            "edit, not one per milestone file (#1016):\n  " + "\n  ".join(literals)
        )
        for rel in MILESTONE_SITES:
            text = (ROOT / rel).read_text(encoding="utf-8")
            assert "EXPECTED_RECIPE_COUNT" in text, (
                f"{rel} is listed in MILESTONE_SITES but no longer pins the count; "
                "drop it from the table if that milestone was retired on purpose."
            )

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

    def test_a_literal_milestone_is_caught(self, tmp_path: pathlib.Path) -> None:
        """CONTROL: a milestone file that goes back to a literal is reported by
        name, and one comparing against the pinned count is not."""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_old.py").write_text(
            "def test_size():\n    assert len(RECIPES) == 175\n", encoding="utf-8"
        )
        (tmp_path / "tests" / "test_new.py").write_text(
            "def test_size():\n    assert len(RECIPES) == EXPECTED_RECIPE_COUNT\n",
            encoding="utf-8",
        )
        found = literal_milestone_sites(tmp_path, ("tests/test_old.py", "tests/test_new.py"))
        assert found == ["tests/test_old.py: len(RECIPES) == 175"]

    def test_mutated_catalog_count_fails_all_sites(self) -> None:
        fake_count = len(RECIPES) + 1
        verdict = audit_doc_recipe_counts(ROOT, DOC_SITES, expected_count=fake_count)
        assert not verdict.is_synced
        assert len(verdict.mismatches) >= len(DOC_SITES)


class TestNoUndeclaredCountSite:
    """A tracked file stating a recipe count outside DOC_SITES is detected (#1016)."""

    def test_no_tracked_file_states_a_count_outside_the_declared_sites(self) -> None:
        found = undeclared_count_sites(ROOT, _tracked_files(ROOT))
        assert not found, (
            "These lines state a recipe count that no table covers, so they would go "
            "stale silently when a recipe is added. Add each to DOC_SITES in "
            "tests/test_recipe_count_is_synced.py, or reword it without a number:\n  "
            + "\n  ".join(found)
        )

    def test_a_new_doc_line_with_a_count_is_caught(self, tmp_path: pathlib.Path) -> None:
        """CONTROL: the scanner fires on a count no table declares, and not on a
        declared site, an issue reference, a historical file or a dated measurement."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "new-guide.md").write_text(
            "Pick one of our 176 ready-made recipes.\n", encoding="utf-8"
        )
        (tmp_path / "docs" / "commands.md").write_text(
            "soup recipes list  List all 176 ready-made recipes\n", encoding="utf-8"
        )
        (tmp_path / "src.py").write_text("# matches pre-#330 recipes on disk\n", encoding="utf-8")
        (tmp_path / "CHANGELOG.md").write_text("Shipped 162 recipes.\n", encoding="utf-8")
        (tmp_path / ".github" / "workflows").mkdir(parents=True)
        (tmp_path / ".github" / "workflows" / "recipe-repo-ids.yml").write_text(
            "#   165 recipes / 330 surfaces / 118 unique ids\n", encoding="utf-8"
        )
        sites = (
            DocCountSite(
                "docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"
            ),
        )
        found = undeclared_count_sites(
            tmp_path,
            [
                "docs/new-guide.md",
                "docs/commands.md",
                "src.py",
                "CHANGELOG.md",
                ".github/workflows/recipe-repo-ids.yml",
            ],
            sites,
        )
        assert found == ["docs/new-guide.md:1: Pick one of our 176 ready-made recipes."]
