"""No tracked file may carry committed Git conflict markers (#1125).

A committed conflict marker previously escaped review into
``docs/performance-and-quantization.md`` on #1066 (literal ``<<<<<<< HEAD``,
``=======``, and ``>>>>>>> <sha>`` lines) with all 15 CI contexts green, because
CI never runs pre-commit and ``docs/`` is outside Ruff.

This test provides an automated repository-wide guard across every tracked text
file (including ``docs/``, ``.claude/``, fixtures, configuration, and source
trees).

**Handling the `=======` false-positive case:**
In Markdown and ReST, a line of ``=======`` is a legal setext-style heading
underline or divider. To avoid crying wolf on valid documentation, a line equal
to ``=======`` is flagged only when the file ALSO contains at least one conflict
boundary marker (``<<<<<<<`` or ``>>>>>>>`` or ``|||||||``). A file
containing only setext underlines without conflict boundaries produces zero hits.

**Deliberately no escape hatch:**
Consistent with ``.pre-commit-config.yaml``'s own ``check-merge-conflict`` hook,
there is deliberately no inline escape hatch comment. Genuine documentation
illustrating conflict resolution should indent markers or format them mid-line.

**Self-consistency:**
This test file builds marker strings dynamically via concatenation (e.g.
``"<" * 7``) and formats fixtures so literal markers never appear at column 0,
preventing the scanner from ever flagging its own source code.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Defined via concatenation to avoid matching this test file's own source lines:
MARKER_START = "<" * 7
MARKER_END = ">" * 7
MARKER_BASE = "|" * 7
MARKER_MID = "=" * 7


def _is_marker_line(line: str, prefix: str) -> bool:
    """Return True if line is exactly 7 chars or 7 chars followed by a space."""
    return line == prefix or line.startswith(prefix + " ")


def is_binary_file(path: Path) -> bool:
    """Check if a file is binary by inspecting for NULL bytes in the first 8KB."""
    try:
        with path.open("rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except OSError:
        return False


def find_conflict_markers(text: str) -> list[tuple[int, str, str]]:
    """Scan text for Git merge conflict markers.

    Returns a list of ``(lineno, marker_description, line_content)``.

    A line starting with ``<<<<<<<``, ``>>>>>>>``, or ``|||||||`` (either bare
    or followed by a space) is an unambiguous conflict boundary marker and is
    always flagged.

    A line equal to ``=======`` is only flagged if the file also contains at
    least one conflict boundary marker, preventing false positives on valid
    Markdown/ReST setext heading underlines (#1125).

    ``splitlines()`` natively strips LF and CRLF endings uniformly across OSes.
    """
    lines = text.splitlines()
    # A line of ======= is legal Markdown/ReST setext underline; only flag it
    # as a conflict divider when the file also contains a boundary marker (#1125).
    has_boundary = any(
        _is_marker_line(line, MARKER_START)
        or _is_marker_line(line, MARKER_END)
        or _is_marker_line(line, MARKER_BASE)
        for line in lines
    )
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(lines, start=1):
        if _is_marker_line(line, MARKER_START):
            hits.append((lineno, "start marker (`<<<<<<<`)", line))
        elif _is_marker_line(line, MARKER_END):
            hits.append((lineno, "end marker (`>>>>>>>`)", line))
        elif _is_marker_line(line, MARKER_BASE):
            hits.append((lineno, "base ancestor marker (`|||||||`)", line))
        elif has_boundary and line.rstrip(" ") == MARKER_MID:
            hits.append((lineno, "divider marker (`=======`)", line))
    return hits


def _tracked_files() -> list[Path]:
    """Return all tracked files in the repository via `git ls-files`."""
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("guard requires a git checkout")
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def scan_tracked_files(file_list: list[Path] | None = None) -> list[str]:
    """Scan tracked text files for merge conflict markers."""
    targets = file_list if file_list is not None else _tracked_files()
    offenders: list[str] = []
    for path in targets:
        if not path.is_file():
            continue
        if is_binary_file(path):
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")

        hits = find_conflict_markers(text)
        if hits:
            try:
                rel = path.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                rel = path.as_posix()
            for lineno, what, line in hits:
                offenders.append(f"{rel}:{lineno} carries git conflict {what}: {line.strip()}")
    return offenders


class TestNoGitConflictMarkersInTrackedFiles:
    """Repository-wide guard enforcing zero committed conflict markers."""

    def test_no_tracked_file_contains_git_conflict_markers(self):
        offenders = scan_tracked_files()
        assert not offenders, (
            "Committed Git conflict marker(s) detected in tracked files (#1125). "
            "Resolve the merge conflict and remove all conflict markers:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_scan_actually_covers_the_repository(self):
        """A scanner that silently stopped discovering files would pass vacuously."""
        tracked = _tracked_files()
        assert len(tracked) >= 1000, (
            f"only {len(tracked)} tracked files found; `git ls-files` or repository "
            "root resolution may be broken"
        )
        names = {p.as_posix() for p in tracked}
        # The exact documentation file that had committed markers in #1066:
        assert (REPO_ROOT / "docs" / "performance-and-quantization.md").as_posix() in names
        assert (REPO_ROOT / "pyproject.toml").as_posix() in names
        assert (REPO_ROOT / "src" / "soup_cli" / "__init__.py").as_posix() in names

    def test_this_very_file_passes_its_own_scan(self):
        """End-to-end self-test: the guard must not flag its own source code."""
        text = io.open(__file__, encoding="utf-8", errors="replace").read()
        assert find_conflict_markers(text) == [], (
            "test_issue1125_git_conflict_markers.py flagged itself; ensure marker "
            "strings in the test file are concatenated or escaped rather than written "
            "at column 0"
        )


class TestTheScannerCanActuallyFail:
    """Prove the scanner catches real markers and handles false positives (#1125).

    The repository is clean, so these tests prove the scanner detects real
    planted conflict markers (both directions) and avoids Markdown/ReST setext
    heading false positives.
    """

    def test_it_catches_standard_conflict_block(self):
        conflict_sample = "\n".join(
            [
                "def calculate_total(items):",
                MARKER_START + " HEAD",
                "    return sum(item.price for item in items)",
                MARKER_MID,
                "    return math.fsum(item.price for item in items)",
                MARKER_END + " feature-branch",
                "    pass",
            ]
        )
        hits = find_conflict_markers(conflict_sample)
        assert len(hits) == 3, hits
        assert hits[0][0] == 2
        assert "start marker" in hits[0][1]
        assert hits[1][0] == 4
        assert "divider marker" in hits[1][1]
        assert hits[2][0] == 6
        assert "end marker" in hits[2][1]

    def test_it_catches_diff3_base_ancestor_marker(self):
        diff3_sample = "\n".join(
            [
                MARKER_START + " HEAD",
                "foo = 1",
                MARKER_BASE + " ancestor",
                "foo = 0",
                MARKER_MID,
                "foo = 2",
                MARKER_END + " remote",
            ]
        )
        hits = find_conflict_markers(diff3_sample)
        assert len(hits) == 4, hits
        kinds = [h[1] for h in hits]
        assert any("base ancestor marker" in k for k in kinds)

    def test_it_catches_lone_start_marker(self):
        sample = f"line 1\n{MARKER_START} HEAD\nline 3\n"
        hits = find_conflict_markers(sample)
        assert len(hits) == 1
        assert hits[0][0] == 2
        assert "start marker" in hits[0][1]

    def test_it_catches_lone_end_marker(self):
        sample = f"line 1\nline 2\n{MARKER_END} main\n"
        hits = find_conflict_markers(sample)
        assert len(hits) == 1
        assert hits[0][0] == 3
        assert "end marker" in hits[0][1]

    def test_it_catches_bare_markers_without_labels(self):
        """Bare markers without branch labels or spaces must also be caught."""
        sample = f"{MARKER_START}\nconflict\n{MARKER_MID}\nresolution\n{MARKER_END}\n"
        hits = find_conflict_markers(sample)
        assert len(hits) == 3, hits
        assert "start marker" in hits[0][1]
        assert "divider marker" in hits[1][1]
        assert "end marker" in hits[2][1]

    def test_crlf_line_endings_handled_uniformly(self):
        """CRLF line endings (e.g. core.autocrlf on Windows) are handled identically."""
        crlf_sample = (
            f"line 1\r\n"
            f"{MARKER_START} HEAD\r\n"
            f"code\r\n"
            f"{MARKER_MID}\r\n"
            f"alt\r\n"
            f"{MARKER_END} branch\r\n"
        )
        hits = find_conflict_markers(crlf_sample)
        assert len(hits) == 3, hits
        assert hits[0][0] == 2
        assert hits[1][0] == 4
        assert hits[2][0] == 6

    def test_markdown_setext_heading_underline_is_not_flagged(self):
        """CONTROL: ======= without conflict boundaries is valid Markdown/ReST (#1125)."""
        markdown_text = "\n".join(
            [
                "# Title",
                "",
                "Some introductory text.",
                "",
                "Section Heading",
                "=======",
                "",
                "Body paragraph following the setext heading.",
            ]
        )
        assert find_conflict_markers(markdown_text) == []

    def test_rest_heading_underline_is_not_flagged(self):
        """CONTROL: ReST document with title underlines must not cry wolf."""
        rest_text = "\n".join(
            [
                "=======",
                "Heading",
                "=======",
                "",
                "Subheading",
                "----------",
            ]
        )
        assert find_conflict_markers(rest_text) == []

    def test_repeated_equals_of_varying_lengths_not_flagged_without_boundary(self):
        """CONTROL: Borders and lines like '===', '====', '===========' are ignored."""
        text = "\n".join(
            [
                "===",
                "====",
                "==========",
                "====================",
            ]
        )
        assert find_conflict_markers(text) == []

    def test_scan_tracked_files_detects_planted_file_with_line_number(self, tmp_path):
        """End-to-end failure proof: a planted conflict marker turns red with file and line."""
        planted = tmp_path / "docs" / "reproduction.md"
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted_content = (
            f"# Title\n\n{MARKER_START} HEAD\nold content\n"
            f"{MARKER_MID}\nnew\n{MARKER_END} 11f798e\n"
        )
        planted.write_text(planted_content, encoding="utf-8")

        offenders = scan_tracked_files([planted])
        assert len(offenders) == 3, offenders
        # Must name file and line number:
        assert any("reproduction.md:3" in o and "start marker" in o for o in offenders)
        assert any("reproduction.md:5" in o and "divider marker" in o for o in offenders)
        assert any("reproduction.md:7" in o and "end marker" in o for o in offenders)

    def test_non_git_checkout_skips(self, monkeypatch):
        """Outside a git checkout or release archive, the guard skips cleanly."""
        def mock_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, ["git", "ls-files"])

        monkeypatch.setattr(subprocess, "run", mock_run)
        with pytest.raises(pytest.skip.Exception, match="guard requires a git checkout"):
            _tracked_files()

