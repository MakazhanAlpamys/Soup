"""Repo-wide ratchet: README.md translation stamps must stay synced.

When README.md changes, every translation file must carry a matching
<!-- synced-from: README.md sha256:<hex> --> comment. Git auto-merges
Markdown quietly, so this test enforces the stamp as a ratchet:
- Stamp missing → FAIL
- Stamp SHA256 mismatch → FAIL
- The ratchet has teeth: deleting the comment line is not a workaround.
"""

from __future__ import annotations

import pathlib
import re
import hashlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: All translation files that must carry a sync stamp.
TRANSLATION_FILES = (
    "README.tr.md",
    "README.ar.md",
    "README.ja.md",
    "README.zh.md",
    "README.ru.md",
    "README.es.md",
)

#: Regex to extract the sync stamp from a translation file.
SYNC_STAMP_PATTERN = re.compile(
    r"<!--\s*synced-from:\s*README\.md\s+sha256:([a-f0-9]{64})\s*-->",
    re.IGNORECASE,
)


def compute_readme_sha256(root: pathlib.Path = ROOT) -> str:
    """Compute SHA-256 of the canonical README.md."""
    readme_path = root / "README.md"
    content = readme_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def extract_stamp_sha256(translation_path: pathlib.Path) -> str | None:
    """Extract the SHA-256 from a translation's sync stamp, or None if missing."""
    text = translation_path.read_text(encoding="utf-8")
    match = SYNC_STAMP_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1)


class TestReadmeTranslationSync:
    """Ratchet: translations must stay synced with README.md."""

    def test_readme_sha256_is_stable(self) -> None:
        """Control: ensure README.md exists and has content."""
        readme_path = ROOT / "README.md"
        assert readme_path.exists(), "README.md must exist"
        content = readme_path.read_text(encoding="utf-8")
        assert len(content) > 100, "README.md must have substantial content"

    def test_all_translations_have_sync_stamp(self) -> None:
        """Every translation file must carry a sync stamp."""
        missing = []
        for fname in TRANSLATION_FILES:
            fpath = ROOT / fname
            if not fpath.exists():
                missing.append(f"{fname} (file missing)")
                continue
            sha = extract_stamp_sha256(fpath)
            if sha is None:
                missing.append(f"{fname} (no stamp)")
        assert not missing, f"Translations missing sync stamp: {missing}"

    def test_translation_stamps_match_readme(self) -> None:
        """Every translation's stamp SHA-256 must match README.md."""
        expected_sha = compute_readme_sha256()
        mismatches = []
        for fname in TRANSLATION_FILES:
            fpath = ROOT / fname
            if not fpath.exists():
                continue
            actual_sha = extract_stamp_sha256(fpath)
            if actual_sha is None:
                mismatches.append(f"{fname} (no stamp)")
                continue
            if actual_sha != expected_sha:
                mismatches.append(f"{fname} (stamp {actual_sha[:16]}... != {expected_sha[:16]}...)")
        assert not mismatches, f"Translation stamps out of sync:\n  " + "\n  ".join(mismatches)

    def test_stamp_absence_is_detected(self) -> None:
        """CONTROL: Prove the test catches a missing stamp."""
        # Create a temp file with no stamp
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Test\nNo stamp here.")
            temp_path = pathlib.Path(f.name)

        try:
            sha = extract_stamp_sha256(temp_path)
            assert sha is None, "Expected None for missing stamp"
        finally:
            temp_path.unlink(missing_ok=True)

    def test_wrong_sha_is_detected(self) -> None:
        """CONTROL: Prove the test catches a mismatched SHA."""
        import tempfile
        wrong_sha = "a" * 64
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(f"<!-- synced-from: README.md sha256:{wrong_sha} -->\n# Test")
            temp_path = pathlib.Path(f.name)

        try:
            sha = extract_stamp_sha256(temp_path)
            expected = compute_readme_sha256()
            assert sha != expected, f"Expected mismatch: {sha} != {expected}"
        finally:
            temp_path.unlink(missing_ok=True)

    def test_readme_change_breaks_stamp(self) -> None:
        """CONTROL: Prove that modifying README.md breaks the stamp check."""
        import tempfile
        import shutil

        # Copy README.md to temp, modify it
        readme_orig = ROOT / "README.md"
        with tempfile.TemporaryDirectory() as td:
            temp_readme = pathlib.Path(td) / "README.md"
            shutil.copy(readme_orig, temp_readme)

            # Append a line (simulating a change)
            orig_content = temp_readme.read_text(encoding="utf-8")
            temp_readme.write_text(orig_content + "\n<!-- added line -->\n", encoding="utf-8")

            # Compute new SHA
            new_sha = hashlib.sha256(temp_readme.read_bytes()).hexdigest()
            original_sha = compute_readme_sha256()

            assert new_sha != original_sha, "Simulated change must produce different SHA"