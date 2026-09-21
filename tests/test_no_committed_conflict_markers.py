"""#1125 — no tracked file may carry a committed Git conflict marker.

``docs/performance-and-quantization.md`` shipped literal ``<<<<<<< HEAD`` /
``=======`` / ``>>>>>>> <sha>`` lines in #1066 with all 15 CI contexts green.
Two reasons it stayed green:

* ``.pre-commit-config.yaml`` ships ``check-merge-conflict``, but **CI never runs
  pre-commit** — the ``lint`` job runs only ``ruff check`` and ``actionlint``.
* ``ruff`` does not read Markdown, so ``docs/`` is outside its paths anyway. A
  marker in a Markdown file is invisible twice over.

So this is a repo-wide scan over every tracked text file, run as a pytest test
rather than a workflow step: it lands on all nine matrix cells, is visible to
contributors running ``pytest`` locally, and cannot be skipped by a fork's
permissions. It deliberately does NOT add ``pre-commit run --all-files`` to CI —
that is the larger question the issue defers.

**The ``=======`` false positive.** A bare line of seven equals signs is legal
Markdown (a setext heading underline) and adef test_valid_utf8_returns_text(self, tmp_path: Path):ppears in ReST. Matching it on its
own would fire on legitimate documentation. So ``=======`` is reported only when
the *same file* also contains a ``<<<<<<<`` or ``>>>>>>>`` line — i.e. when it is
part of an actual conflict block, which is exactly how Git emits them. The two
angle-bracket markers are unambiguous and always reported.

Binary files are detected by attempting a strict UTF-8 decode; a file that fails
is treated as binary and skipped rather than crashing the scan.
"""

from __future__ import annotations

import io
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Conflict-marker forms Git writes, anchored at column 0. The angle-bracket
#: markers carry a trailing space then a branch/sha label; the divider is seven
#: equals signs to end of line.
_START = re.compile(r"^<{7}(?: .*)?$", re.M)
_END = re.compile(r"^>{7}(?: .*)?$", re.M)
_DIVIDER = re.compile(r"^={7}$", re.M)


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def _read_text(path: Path) -> str | None:
    """Return the decoded text, or ``None`` for a file we treat as binary."""
    try:
        raw = path.read_bytes()
    except OSError:  # pragma: no cover - unreadable tracked file
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def find_conflict_markers(text: str) -> list[tuple[int, str]]:
    """Return ``(lineno, marker_kind)`` for each conflict-marker line.

    ``marker_kind`` is one of ``"<<<<<<<"``, ``">>>>>>>"``, ``"======="``.
    A standalone ``=======`` with no companion angle-bracket marker in the same
    text is NOT returned — that is the setext-heading / ReST case the guard must
    tolerate.
    """
    starts = [m.start() for m in _START.finditer(text)]
    ends = [m.start() for m in _END.finditer(text)]
    has_block_context = bool(starts or ends)

    found: list[tuple[int, str]] = []
    for offset in starts:
        lineno = text.count("\n", 0, offset) + 1
        found.append((lineno, "<<<<<<<"))
    for offset in ends:
        lineno = text.count("\n", 0, offset) + 1
        found.append((lineno, ">>>>>>>"))
    if has_block_context:
        for m in _DIVIDER.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            found.append((lineno, "======="))
    found.sort()
    return found


def _scan_repo() -> list[str]:
    problems: list[str] = []
    for path in _tracked_files():
        text = _read_text(path)
        if text is None:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, kind in find_conflict_markers(text):
            problems.append(f"{rel}:{lineno}: stray conflict marker {kind}")
    return problems


class TestNoCommittedConflictMarkers:
    def test_no_tracked_file_carries_a_conflict_marker(self):
        problems = _scan_repo()
        assert not problems, (
            "These tracked files contain a committed Git conflict marker. "
            "Resolve the conflict and remove the marker lines before merging "
            "(see #1125):\n  " + "\n  ".join(problems)
        )

    def test_the_scan_actually_covers_the_suite(self):
        """A scan that silently stopped reading files would pass vacuously."""
        tracked = _tracked_files()
        scanned = [path for path in tracked if _read_text(path) is not None]
        assert tracked
        assert scanned


class TestTheScannerCanActuallyFail:
    """The repo is clean, so these are the only proof the scanner works."""

    def test_it_catches_a_start_marker(self):
        text = "# header\n<<<<<<< HEAD\nours\n"
        found = find_conflict_markers(text)
        assert (2, "<<<<<<<") in found, found

    def test_it_catches_an_end_marker(self):
        text = "# header\ntheir\n>>>>>>> abc1234\n"
        found = find_conflict_markers(text)
        assert (3, ">>>>>>>") in found, found

    def test_it_catches_a_complete_conflict_block(self):
        text = "before\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> topic\nafter\n"
        kinds = {kind for _, kind in find_conflict_markers(text)}
        assert kinds == {"<<<<<<<", "=======", ">>>>>>>"}

    def test_a_standalone_divider_is_not_flagged(self):
        """CONTROL. Seven equals signs alone is a setext underline / ReST rule."""
        text = "Title\n=======\nsome body prose\n"
        assert find_conflict_markers(text) == []

    def test_many_equals_signes_without_angle_markers_are_clean(self):
        text = "\n".join(["===="] * 5) + "\n=======\n"
        assert find_conflict_markers(text) == []

    def test_the_divider_only_counts_when_a_block_marker_shares_the_file(self):
        ours = "=======\n"
        theirs = "<<<<<<< HEAD\n=======\n"
        assert find_conflict_markers(ours) == []
        divider_lines = [ln for ln, k in find_conflict_markers(theirs) if k == "======="]
        assert divider_lines == [2], theirs

    def test_reported_line_numbers_are_one_indexed_and_correct(self):
        text = "line1\nline2\n<<<<<<< HEAD\nline4\n"
        found = find_conflict_markers(text)
        assert found == [(3, "<<<<<<<")], found


class TestReadTextTreatsUndecodableAsBinary:
    def test_undecodable_bytes_return_none(self, tmp_path: Path):
        target = tmp_path / "blob.bin"
        target.write_bytes(b"\xff\xfe\x00\x01\x80\x81")
        assert _read_text(target) is None

    def test_valid_utf8_returns_text(self, tmp_path: Path):
        target = tmp_path / "ok.txt"
        target.write_bytes("héllo ✓\n".encode("utf-8"))
        assert _read_text(target) == "héllo ✓\n"


class TestFailureNamesFileAndLine:
    def test_a_planted_marker_produces_a_diagnostic_with_path_and_line(self, tmp_path: Path):
        offender = tmp_path / "doc.md"
        offender.write_text("intro\n<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> x\n", encoding="utf-8")
        rel = "doc.md"
        problems = [
            f"{rel}:{lineno}: stray conflict marker {kind}"
            for lineno, kind in find_conflict_markers(offender.read_text(encoding="utf-8"))
        ]
        joined = "\n".join(problems)
        assert "doc.md:" in joined
        assert ":2:" in joined
        assert "<<<<<<<" in joined