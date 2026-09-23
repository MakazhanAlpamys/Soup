"""Tests for `scripts/sync_recipe_count.py` (#1016).

The rewriter is driven by the same `DOC_SITES` table the checker in
`tests/test_recipe_count_is_synced.py` reads, so these tests exercise the
substitution logic directly against synthetic `tmp_path` files rather than
the real repo tree: the real tree is already covered end-to-end by
`test_recipe_count_is_synced.py` itself.

Fixtures here state 200, never the live catalog count. A synthetic count that
happens to equal `len(RECIPES)` is flagged by `find_undeclared_count_sites`
the moment the file is tracked, and rightly so: it cannot tell a fixture from
a new documentation site. `TestTheGuardHasTeeth` uses 200/199/144 for the
same reason.
"""

from __future__ import annotations

import pathlib

import pytest


class TestRewriteText:
    """`rewrite_text` is the pure substitution kernel; everything else is I/O."""

    def test_stale_count_is_replaced_and_reported_as_changed(self) -> None:
        from scripts.sync_recipe_count import rewrite_text

        new_text, lines_changed = rewrite_text(
            "soup recipes list  List all 170 ready-made recipes\n",
            r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes",
            200,
        )
        assert new_text == "soup recipes list  List all 200 ready-made recipes\n"
        assert lines_changed == 1

    def test_already_synced_count_is_left_untouched(self) -> None:
        from scripts.sync_recipe_count import rewrite_text

        original = "soup recipes list  List all 200 ready-made recipes\n"
        new_text, lines_changed = rewrite_text(
            original,
            r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes",
            200,
        )
        assert new_text == original
        assert lines_changed == 0

    def test_replacement_preserves_surrounding_text_exactly(self) -> None:
        """The callable replacement must only touch the captured digits.

        A naive whole-match replacement would also have to reconstruct the
        surrounding wording; rebuilding around group(1)'s span means anything
        outside it, including odd spacing, is never at risk.
        """
        from scripts.sync_recipe_count import rewrite_text

        original = "Ship with     29    recipes   today, seriously.\n"
        new_text, lines_changed = rewrite_text(original, r"(\d+)\s+recipes", 200)
        assert new_text == "Ship with     200    recipes   today, seriously.\n"
        assert lines_changed == 1


class TestPlanAndApplyChanges:
    """`plan_changes` (read-only) and `apply_changes` (writes) against tmp_path."""

    def test_apply_changes_updates_a_stale_doc_site(self, tmp_path: pathlib.Path) -> None:
        from scripts.sync_recipe_count import SiteChange, apply_changes
        from tests.test_recipe_count_is_synced import DocCountSite

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc_file = docs_dir / "commands.md"
        doc_file.write_text(
            "soup recipes list  List all 170 ready-made recipes\n", encoding="utf-8"
        )

        sites = (
            DocCountSite(
                "docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"
            ),
        )
        changes = apply_changes(tmp_path, 200, sites=sites)

        assert changes == (SiteChange("docs/commands.md", 1),)
        assert "List all 200 ready-made recipes" in doc_file.read_text(encoding="utf-8")

    def test_check_mode_reports_without_writing(self, tmp_path: pathlib.Path) -> None:
        from scripts.sync_recipe_count import plan_changes
        from tests.test_recipe_count_is_synced import DocCountSite

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc_file = docs_dir / "commands.md"
        original = "soup recipes list  List all 170 ready-made recipes\n"
        doc_file.write_text(original, encoding="utf-8")

        sites = (
            DocCountSite(
                "docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"
            ),
        )
        changes = plan_changes(tmp_path, 200, sites=sites)

        assert len(changes) == 1
        assert changes[0].rel_path == "docs/commands.md"
        assert changes[0].lines_changed == 1
        # The whole point of --check: nothing on disk moves.
        assert doc_file.read_text(encoding="utf-8") == original

    def test_file_already_in_sync_is_left_byte_identical(self, tmp_path: pathlib.Path) -> None:
        from scripts.sync_recipe_count import apply_changes
        from tests.test_recipe_count_is_synced import DocCountSite

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc_file = docs_dir / "commands.md"
        original_bytes = b"soup recipes list  List all 200 ready-made recipes\n"
        doc_file.write_bytes(original_bytes)

        sites = (
            DocCountSite(
                "docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"
            ),
        )
        changes = apply_changes(tmp_path, 200, sites=sites)

        assert changes == ()
        assert doc_file.read_bytes() == original_bytes

    def test_lf_line_endings_survive_a_rewrite(self, tmp_path: pathlib.Path) -> None:
        """Only the changed number may differ; every other byte stays put.

        `read_text`/`write_text` translate newlines twice, so on Windows with
        `core.autocrlf=false` a rewrite converts the whole file to CRLF and
        puts every line in the diff. Byte I/O is what keeps a one-number edit
        a one-line diff.
        """
        from scripts.sync_recipe_count import apply_changes
        from tests.test_recipe_count_is_synced import DocCountSite

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc_file = docs_dir / "commands.md"
        before = b"intro\nsoup recipes list  List all 199 ready-made recipes\ntrailer\n"
        doc_file.write_bytes(before)

        sites = (
            DocCountSite(
                "docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"
            ),
        )
        changes = apply_changes(tmp_path, 200, sites=sites)

        assert len(changes) == 1
        assert doc_file.read_bytes() == before.replace(b"199", b"200")

    def test_crlf_line_endings_survive_a_rewrite(self, tmp_path: pathlib.Path) -> None:
        """The mirror of the LF case: a CRLF file must not be collapsed to LF."""
        from scripts.sync_recipe_count import apply_changes
        from tests.test_recipe_count_is_synced import DocCountSite

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        doc_file = docs_dir / "commands.md"
        before = b"intro\r\nsoup recipes list  List all 199 ready-made recipes\r\ntrailer\r\n"
        doc_file.write_bytes(before)

        sites = (
            DocCountSite(
                "docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"
            ),
        )
        changes = apply_changes(tmp_path, 200, sites=sites)

        assert len(changes) == 1
        assert doc_file.read_bytes() == before.replace(b"199", b"200")

    def test_missing_site_file_is_skipped_without_error(self, tmp_path: pathlib.Path) -> None:
        from scripts.sync_recipe_count import apply_changes
        from tests.test_recipe_count_is_synced import DocCountSite

        sites = (DocCountSite("docs/does_not_exist.md", r"(\d+)\s+recipes"),)
        changes = apply_changes(tmp_path, 200, sites=sites)
        assert changes == ()


class TestMainCli:
    """The CLI wiring: exit codes and the real DOC_SITES/RECIPES default path."""

    def test_check_exits_zero_when_the_real_repo_is_synced(self) -> None:
        """The real repo tree, on the branch under test, must already be synced.

        This is the control that keeps the CLI wired to the real DOC_SITES
        and the real catalog rather than only ever exercised against fixtures.
        """
        from scripts.sync_recipe_count import main

        assert main(["--check"]) == 0

    def test_no_flag_exits_zero_when_nothing_to_do(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercises the no-flag path end to end without ever touching the real repo.

        `main` resolves the catalog and `_REPO_ROOT` itself, so both are
        monkeypatched to a hermetic, already-synced `tmp_path` fixture. This
        deliberately does NOT call `main([])` against the real repository:
        doing that would write to it for real whenever it happened to be out
        of sync, which is exactly the state this file's own mutation-testing
        run puts it in.
        """
        import scripts.sync_recipe_count as sync_recipe_count
        import soup_cli.recipes.catalog as catalog
        from tests.test_recipe_count_is_synced import DocCountSite

        monkeypatch.setattr(catalog, "RECIPES", {"a": None, "b": None, "c": None})
        monkeypatch.setattr(sync_recipe_count, "_REPO_ROOT", tmp_path)

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "commands.md").write_text(
            "soup recipes list  List all 3 ready-made recipes\n", encoding="utf-8"
        )
        fixture_sites = (
            DocCountSite(
                "docs/commands.md", r"List all\s+(\d+)\s+(?:ready[-\s]?made\s+)?recipes"
            ),
        )
        monkeypatch.setattr(
            sync_recipe_count,
            "_resolve_sites",
            lambda sites: sites if sites is not None else fixture_sites,
        )

        assert sync_recipe_count.main([]) == 0
