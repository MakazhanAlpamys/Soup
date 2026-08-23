"""Acceptance tests for conflict-free changelog fragments (#487)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.assemble_changelog import ChangelogError, assemble, check_empty, validate_fragments

CHANGELOG = """# Changelog

## [Unreleased]

### Added

- Existing addition.

### Fixed

- Existing fix.

## [0.73.3] - 2026-08-18

### Added

- Released addition.
"""


def _write_repo(root: Path, changelog: str = CHANGELOG) -> None:
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    (root / "changelog.d" / "0.73.3").mkdir(parents=True)


def _write_fragment(root: Path, name: str, content: str) -> Path:
    path = root / "changelog.d" / "0.73.3" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path


def test_assembly_preserves_long_markdown_verbatim(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    fragment = """- **A long change (#487 by @contributor).** First paragraph.

  A second paragraph keeps its indentation.

  | column | value |
  | --- | --- |
  | alpha | beta |

  ```python
  result = "spacing stays intact"
  ```"""
    path = _write_fragment(tmp_path, "487.changed.md", fragment)

    assembled = assemble(tmp_path)
    rendered = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")

    assert assembled == [path]
    assert fragment in rendered
    assert rendered.index("### Changed") < rendered.index(fragment)
    assert rendered.count(fragment) == 1
    assert not path.exists()
    assert not path.parent.exists()


def test_fragments_are_sorted_by_number_inside_their_section(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    _write_fragment(tmp_path, "902.fixed.md", "- Later identifier (#902).\n")
    _write_fragment(tmp_path, "101.fixed.md", "- Earlier identifier (#101).\n")

    assemble(tmp_path)
    rendered = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")

    assert rendered.index("Earlier identifier") < rendered.index("Later identifier")
    assert rendered.index("Existing fix") < rendered.index("Earlier identifier")


def test_recovery_after_interrupted_run_does_not_duplicate_entry(tmp_path: Path) -> None:
    entry = "- Already assembled exactly once (#487).\n"
    changelog = CHANGELOG.replace("- Existing fix.\n", f"- Existing fix.\n\n{entry}")
    (tmp_path / "CHANGELOG.md").write_bytes(changelog.replace("\n", "\r\n").encode())
    (tmp_path / "changelog.d" / "0.73.3").mkdir(parents=True)
    path = _write_fragment(tmp_path, "487.fixed.md", entry)

    assemble(tmp_path)

    rendered = (tmp_path / "CHANGELOG.md").read_bytes()
    normalised = rendered.decode().replace("\r\n", "\n")
    assert normalised.count(entry) == 1
    assert normalised.count("### Fixed") == 1
    assert b"- Existing fix.\r\n" in rendered
    assert not path.exists()


def test_crlf_assembly_does_not_add_an_extra_blank_line(tmp_path: Path) -> None:
    (tmp_path / "CHANGELOG.md").write_bytes(CHANGELOG.replace("\n", "\r\n").encode())
    (tmp_path / "changelog.d" / "0.73.3").mkdir(parents=True)
    entry = "- Newly assembled fix (#487).\n"
    _write_fragment(tmp_path, "487.fixed.md", entry)

    assemble(tmp_path)

    rendered = (tmp_path / "CHANGELOG.md").read_bytes()
    normalised = rendered.decode().replace("\r\n", "\n")
    assert f"- Existing fix.\n\n{entry}\n## [0.73.3]" in normalised
    assert f"- Existing fix.\n\n\n{entry}" not in normalised
    assert b"- Existing fix.\r\n\r\n" in rendered


def test_stale_fragment_fails_loudly_without_mutating_files(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    current_dir = tmp_path / "changelog.d" / "0.73.3"
    current_dir.rmdir()
    stale_dir = tmp_path / "changelog.d" / "0.73.2"
    stale_dir.mkdir()
    path = stale_dir / "487.fixed.md"
    path.write_text("- Stale entry (#487).\n", encoding="utf-8")
    before = (tmp_path / "CHANGELOG.md").read_bytes()

    with pytest.raises(ChangelogError, match="0.73.2.*latest release is 0.73.3"):
        assemble(tmp_path)

    assert (tmp_path / "CHANGELOG.md").read_bytes() == before
    assert path.exists()


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("0.73.3/not-a-fragment.txt", "invalid fragment filename"),
        ("loose.md", "must be inside a release-version directory"),
        ("0.73.3/nested/487.fixed.md", "must be exactly one directory"),
    ],
)
def test_unknown_fragment_layout_is_never_silently_ignored(
    tmp_path: Path, relative_path: str, message: str
) -> None:
    _write_repo(tmp_path)
    path = tmp_path / "changelog.d" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("- Entry (#487).\n", encoding="utf-8")

    with pytest.raises(ChangelogError, match=message):
        validate_fragments(tmp_path)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("487.fixed.md", "", "must not be empty"),
        ("487.fixed.md", "No list marker (#487).\n", "must start with a Markdown list item"),
        ("487.fixed.md", "- Wrong reference (#486).\n", "must reference #487"),
    ],
)
def test_malformed_fragment_is_rejected(
    tmp_path: Path, name: str, content: str, message: str
) -> None:
    _write_repo(tmp_path)
    _write_fragment(tmp_path, name, content)

    with pytest.raises(ChangelogError, match=message):
        validate_fragments(tmp_path)


def test_one_repository_number_cannot_claim_two_fragments(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    _write_fragment(tmp_path, "487.fixed.md", "- Fixed half (#487).\n")
    _write_fragment(tmp_path, "487.changed.md", "- Changed half (#487).\n")

    with pytest.raises(ChangelogError, match="#487 already has fragment"):
        validate_fragments(tmp_path)


def test_release_gate_refuses_every_remaining_fragment(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    _write_fragment(tmp_path, "487.fixed.md", "- Pending entry (#487).\n")

    with pytest.raises(ChangelogError, match="assemble them before tagging"):
        check_empty(tmp_path)

    (tmp_path / "changelog.d" / "0.73.3" / "487.fixed.md").unlink()
    (tmp_path / "changelog.d" / "0.73.3").rmdir()
    check_empty(tmp_path)


def test_cli_failure_is_nonzero_and_actionable(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    _write_fragment(tmp_path, "487.fixed.md", "- Pending entry (#487).\n")
    script = Path(__file__).parents[1] / "scripts" / "assemble_changelog.py"

    result = subprocess.run(
        [sys.executable, str(script), "--root", str(tmp_path), "--check-empty"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert "assemble them before tagging" in result.stderr


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        check=True,
        text=True,
    )


def test_two_branches_add_fragments_without_a_merge_conflict(tmp_path: Path) -> None:
    """The regression in #487: parallel changelog additions must merge cleanly."""
    _write_repo(tmp_path)
    (tmp_path / "changelog.d" / "0.73.3").rmdir()
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Soup test")
    _git(tmp_path, "config", "user.email", "soup-test@example.invalid")
    _git(tmp_path, "add", "CHANGELOG.md")
    _git(tmp_path, "commit", "-m", "base")

    _git(tmp_path, "checkout", "-b", "change-a")
    _write_fragment(tmp_path, "101.fixed.md", "- Branch A fix (#101).\n")
    _git(tmp_path, "add", "changelog.d/0.73.3/101.fixed.md")
    _git(tmp_path, "commit", "-m", "add branch A fragment")

    _git(tmp_path, "checkout", "main")
    _git(tmp_path, "checkout", "-b", "change-b")
    _write_fragment(tmp_path, "102.added.md", "- Branch B feature (#102).\n")
    _git(tmp_path, "add", "changelog.d/0.73.3/102.added.md")
    _git(tmp_path, "commit", "-m", "add branch B fragment")

    merged = _git(tmp_path, "merge", "--no-edit", "change-a")

    assert "CONFLICT" not in merged.stdout
    assert (tmp_path / "changelog.d" / "0.73.3" / "101.fixed.md").is_file()
    assert (tmp_path / "changelog.d" / "0.73.3" / "102.added.md").is_file()


def test_repository_wires_the_release_guard_and_contributor_guidance() -> None:
    root = Path(__file__).parents[1]
    workflow = (root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    contributing = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    template = (root / ".github" / "pull_request_template.md").read_text(encoding="utf-8")

    assert "python scripts/assemble_changelog.py --check-empty" in workflow
    assert "changelog.d/<latest-release>/<number>.<category>.md" in contributing
    assert "changelog fragment" in template.lower()


def test_repository_fragments_match_the_newest_release() -> None:
    """CI must fail after a release until every open fragment is rebased."""
    root = Path(__file__).parents[1]

    # The call is the assertion: stale or malformed repository fragments raise here.
    fragments = validate_fragments(root)

    assert all(fragment.path.is_file() for fragment in fragments)
