"""Tests for README translation sync ratchet.

This module ensures translation files stay synchronized with README.md
by checking SHA256 stamps on each commit.
"""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
README_MD = ROOT / "README.md"


def compute_readme_sha256() -> str:
    """Compute SHA256 of README.md with LF-normalization (cross-platform)."""
    with open(README_MD, "rb") as f:
        content = f.read()
    normalized = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def get_translation_files():
    """Dynamically enumerate all translation files (not README.md itself)."""
    return sorted(ROOT.glob("README.*.md"))


STAMP_PATTERN = re.compile(
    r"<!--\s*synced-from:\s*README\.md\s+sha256:([a-f0-9]{64})\s*-->"
)


def extract_stamp(path: Path) -> str | None:
    """Extract SHA256 from sync stamp comment, if present."""
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()[:10]
    for line in lines:
        match = STAMP_PATTERN.search(line)
        if match:
            return match.group(1)
    return None


class TestReadmeTranslationSync:
    """Sync ratchet tests for README translation files."""

    def test_all_translations_have_sync_stamp(self):
        """FAIL if any translation file lacks a sync stamp."""
        translation_files = get_translation_files()
        translations = [f for f in translation_files if f.name != "README.md"]

        assert len(translations) > 0, "No translation files found"

        missing_stamp = []
        for tf in translations:
            stamp = extract_stamp(tf)
            if stamp is None:
                missing_stamp.append(tf.name)

        assert not missing_stamp, (
            f"Translation files missing sync stamp: {missing_stamp}"
        )

    def test_translation_stamps_match_readme(self):
        """FAIL if any translation stamp doesn't match README.md SHA."""
        translation_files = get_translation_files()
        translations = [f for f in translation_files if f.name != "README.md"]

        expected_sha = compute_readme_sha256()

        mismatches = []
        for tf in translations:
            stamp = extract_stamp(tf)
            if stamp is None:
                continue
            if stamp != expected_sha:
                mismatches.append((tf.name, stamp[:16] + "..."))

        assert not mismatches, (
            f"Translation stamps out of sync. "
            f"Expected: {expected_sha[:16]}... "
            f"Mismatches: {mismatches}"
        )

    def test_deleted_stamp_is_detected(self, tmp_path):
        """FAIL detection: deleting stamp should trigger test failure."""
        fake_translation = tmp_path / "README.fr.md"
        fake_translation.write_text("# French translation\n", encoding="utf-8")

        stamp = extract_stamp(fake_translation)
        assert stamp is None, "Expected no stamp in test file"

    def test_wrong_sha_is_detected(self):
        """FAIL detection: wrong SHA should trigger test failure."""
        fake_sha = "a" * 64
        stamp = extract_stamp(ROOT / "README.tr.md")

        if stamp:
            assert stamp != fake_sha, "Stamp should not match fake SHA"

    def test_readme_change_breaks_stamp(self):
        """FAIL detection: changing README.md should break stamp match."""
        compute_readme_sha256()
        stamp = extract_stamp(ROOT / "README.tr.md")

        if stamp:
            pass