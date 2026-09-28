"""Rewrite every declared DOC_SITES count to match the catalog (#1016).

`tests/test_recipe_count_is_synced.py::DOC_SITES` is the single table that
tells `audit_doc_recipe_counts` what a synced repo looks like. This script
rewrites against that exact same table, imported rather than duplicated, so
the checker and the fixer can never disagree about which sites exist or what
their patterns are.

For each site, the number captured by `DocCountSite.pattern`'s group(1) is
replaced with `len(RECIPES)` in place, via a callable `re.sub` replacement
that rebuilds the matched text around the new number rather than substituting
the whole match outright. That preserves the surrounding wording ("List all
N ready-made recipes") exactly, and leaves an already-synced site
byte-identical.

`--check` reports what is out of sync and exits 1 without writing anything.
With no flag, it writes the fix and prints a per-file summary of how many
lines changed. Either way, it exits 0 when every declared site already
matches the catalog.

This script deliberately does NOT rewrite `EXPECTED_RECIPE_COUNT` in
`tests/recipe_count.py`. That literal is the milestone pin: a maintainer
deciding the catalog has reached a size worth recording, not a side effect of
running a script. Deriving it from `len(RECIPES)` would reduce the four
milestone assertions to `len(RECIPES) == len(RECIPES)`, which can never
fail. So adding a recipe is one number, edited by hand, plus one command.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from tests.test_recipe_count_is_synced import DocCountSite

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_root_on_path() -> None:
    """Make ``tests`` importable regardless of the caller's working directory.

    ``DOC_SITES`` lives in a test module and is imported here rather than
    duplicated, so the checker and the fixer can never drift apart. That only
    works if the repo root (not just ``src``) is on ``sys.path``.
    """
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass(frozen=True)
class SiteChange:
    """How many lines changed (or would change) at one DOC_SITES entry."""

    rel_path: str
    lines_changed: int


def _substitute_count(match: re.Match[str], new_count: int) -> str:
    """Rebuild a whole regex match with only its captured number replaced.

    ``DocCountSite.pattern`` always captures the number as group(1) inside a
    larger phrase (for example "List all N ready-made recipes"). Replacing
    ``match.group(0)`` outright would work today, but rebuilding the match
    from its own text and touching only group(1)'s span is what keeps this
    correct if a pattern is ever widened to capture more surrounding context,
    and it is what guarantees the untouched text on either side of the number
    survives byte-for-byte.
    """
    whole = match.group(0)
    relative_start = match.start(1) - match.start(0)
    relative_end = match.end(1) - match.start(0)
    return whole[:relative_start] + str(new_count) + whole[relative_end:]


def rewrite_text(text: str, pattern: str, new_count: int) -> tuple[str, int]:
    """Apply one site's pattern to ``text``. Returns ``(new_text, lines_changed)``.

    A pure function, independent of any file I/O, so the substitution logic
    is directly testable without touching the filesystem.
    """
    lines_changed = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal lines_changed
        if match.group(1) == str(new_count):
            return match.group(0)
        lines_changed += 1
        return _substitute_count(match, new_count)

    new_text = re.sub(pattern, _replace, text, flags=re.IGNORECASE)
    return new_text, lines_changed


def _resolve_sites(sites: Sequence[DocCountSite] | None) -> Sequence[DocCountSite]:
    """Default to the real ``DOC_SITES`` table, imported lazily (#1016)."""
    if sites is not None:
        return sites
    _ensure_repo_root_on_path()
    from tests.test_recipe_count_is_synced import DOC_SITES

    return DOC_SITES


def plan_changes(
    root: Path,
    new_count: int,
    sites: Sequence[DocCountSite] | None = None,
) -> tuple[SiteChange, ...]:
    """Compute, without writing anything, what would change at each site."""
    resolved_sites = _resolve_sites(sites)

    changes: list[SiteChange] = []
    for site in resolved_sites:
        file_path = root / site.rel_path
        if not file_path.exists():
            continue
        text = file_path.read_bytes().decode("utf-8")
        _, lines_changed = rewrite_text(text, site.pattern, new_count)
        if lines_changed:
            changes.append(SiteChange(site.rel_path, lines_changed))
    return tuple(changes)


def apply_changes(
    root: Path,
    new_count: int,
    sites: Sequence[DocCountSite] | None = None,
) -> tuple[SiteChange, ...]:
    """Write the rewritten text for every site whose count actually changes.

    Each file is read fresh from disk immediately before it is rewritten, so
    two sites that share a file (``docs/serving-and-export.md`` has two)
    apply on top of each other's edits instead of racing to overwrite one
    another.

    I/O goes through ``read_bytes``/``write_bytes`` rather than
    ``read_text``/``write_text`` on purpose. Text mode translates newlines
    twice: CRLF collapses to a bare newline on the way in, and every newline
    expands to ``os.linesep`` on the way out. On Windows with
    ``core.autocrlf=false`` that rewrites every line ending in the file, so a
    one-number edit arrives as a whole-file diff. Bytes translate nothing, so
    a run touches exactly the characters it means to.
    (``Path.read_text(newline=...)`` would say this more directly but only
    exists on 3.13+, and this project supports 3.10.)
    """
    resolved_sites = _resolve_sites(sites)

    changes: list[SiteChange] = []
    for site in resolved_sites:
        file_path = root / site.rel_path
        if not file_path.exists():
            continue
        text = file_path.read_bytes().decode("utf-8")
        new_text, lines_changed = rewrite_text(text, site.pattern, new_count)
        if lines_changed:
            file_path.write_bytes(new_text.encode("utf-8"))
            changes.append(SiteChange(site.rel_path, lines_changed))
    return tuple(changes)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: sync every declared DOC_SITES count to the catalog."""
    _ensure_repo_root_on_path()
    from soup_cli.recipes.catalog import RECIPES

    parser = argparse.ArgumentParser(
        description="Sync DOC_SITES recipe counts to the catalog (#1016)."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change and exit 1, without writing anything",
    )
    args = parser.parse_args(argv)

    new_count = len(RECIPES)

    if args.check:
        changes = plan_changes(_REPO_ROOT, new_count)
        if not changes:
            print(f"All DOC_SITES entries already match the catalog ({new_count} recipes).")
            return 0
        print(f"Out of sync with the catalog ({new_count} recipes):")
        for change in changes:
            print(f"  {change.rel_path}: {change.lines_changed} line(s) would change")
        return 1

    changes = apply_changes(_REPO_ROOT, new_count)
    if not changes:
        print(f"Nothing to do: all DOC_SITES entries already match {new_count} recipes.")
        return 0
    print(f"Synced DOC_SITES entries to {new_count} recipes:")
    for change in changes:
        print(f"  {change.rel_path}: {change.lines_changed} line(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
