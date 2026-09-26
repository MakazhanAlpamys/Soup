"""No tracked file may carry a third-party licence header (repo-wide ratchet).

This project is Apache-2.0. Three times now a file has landed carrying

    # SPDX-License-Identifier: AGPL-3.0-only
    # Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

at the top -- twice in ``8e4f012`` (removed by hand) and a third time in
``3426f2c``, which **I merged without noticing**, having already been burned by
it once that same day. The cause is now known and was disclosed by a
contributor: automated tooling that carries file templates between projects
carries their licence headers with them.

So this is a check rather than another paragraph in CONTRIBUTING.md. The
project's own record on that is unambiguous -- the Rich/ANSI ``--help`` failure
was written into CLAUDE.md, CONTRIBUTING and a memory note after each of four
occurrences and recurred every time, until
``test_cli_help_assertions_are_ansi_safe.py`` made it fail. Prose has been
tried here and it demonstrably does not hold.

Why the licence half matters more than the tidiness half: an AGPL header on a
file in an Apache-2.0 distribution is a licensing claim about that file. It
does not become true by being wrong, but a downstream redistributor reading it
has to act as though it might be, and we ship to PyPI.

**Scope is deliberately the header region and comment lines only.** Body
mentions of copyleft licence names are legitimate and must not be flagged:
``utils/license_advisor.py`` and ``utils/license_matrix.py`` exist precisely to
reason about AGPL/GPL compatibility, and a scanner that fires on them would be
deleted within a week -- which is the failure mode of a guard that cries wolf.
``TestTheScannerCanActuallyFail`` is load-bearing for the opposite reason: this
scanner finds zero offenders once the repo is clean, and a scanner with nothing
to find is indistinguishable from one that is silently broken.
"""

import io
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Only the top of a file is inspected -- a licence header lives there, and
#: restricting the window is what keeps legitimate body text out of scope.
HEADER_LINES = 40

#: Every pattern below is matched against COMMENT lines only (see
#: ``_offending_lines``), so prose inside a docstring or a data table is safe.
PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"SPDX-License-Identifier:\s*(?!Apache-2\.0\s*$)\S+", re.I),
        "an SPDX identifier other than Apache-2.0",
    ),
    (
        re.compile(r"\bA?GPL(?:-[0-9](?:\.[0-9])?)?\b|\bLGPL\b", re.I),
        "a copyleft licence name",
    ),
    (
        re.compile(r"All rights reserved", re.I),
        "an all-rights-reserved notice",
    ),
    (
        re.compile(r"LICENSE\.(?:A?GPL|MPL|EUPL)", re.I),
        "a path to a foreign licence file",
    ),
)

#: Extensions whose comment character is ``#``. Adding a language means adding
#: its comment prefix to ``_is_comment``, not widening this blindly.
SCANNED_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".toml", ".cfg", ".sh"})


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line]


def _is_comment(line: str) -> bool:
    """A file-level licence header sits at column 0, unindented.

    The distinction is load-bearing rather than stylistic, and I learned it the
    hard way: the first version of this file used ``line.lstrip()``, so the
    scanner flagged the two *indented* lines in its own module docstring -- the
    verbatim quotation of the header it exists to catch -- and turned all nine
    CI cells red. Locally it had passed, because an uncommitted file is absent
    from ``git ls-files`` and the scanner therefore never read itself.

    Requiring column 0 costs nothing real: every occurrence of the header this
    guard was written for sat at column 0, which is where tooling emits a file
    header. An indented ``#`` inside the header region is a quotation, a nested
    code block, or a comment inside a class body -- never a licence declaration
    about the file.
    """
    return line.startswith("#")


def _offending_lines(text: str) -> list[tuple[int, str, str]]:
    """Return (lineno, what, line) for each offending comment in the header."""
    found: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines()[:HEADER_LINES], start=1):
        if not _is_comment(line):
            continue
        for pattern, what in PATTERNS:
            if pattern.search(line):
                found.append((lineno, what, line.strip()))
                break
    return found


def _scan_repo() -> list[str]:
    problems: list[str] = []
    for path in _tracked_files():
        if path.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        try:
            text = io.open(path, encoding="utf-8", errors="replace").read()
        except OSError:  # pragma: no cover - unreadable tracked file
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, what, line in _offending_lines(text):
            problems.append(f"{rel}:{lineno} carries {what}: {line}")
    return problems


class TestNoForeignLicenseHeaders:
    def test_no_tracked_file_declares_a_foreign_license(self):
        problems = _scan_repo()
        assert not problems, (
            "These files carry a third-party licence header. This project is "
            "Apache-2.0 and ships to PyPI, so a foreign header is a licensing "
            "claim we cannot make. Delete the header (see 8e4f012):\n  "
            + "\n  ".join(problems)
        )

    def test_the_scan_actually_covers_the_suite(self):
        """A scanner that silently stopped reading files would pass vacuously."""
        scanned = [
            p
            for p in _tracked_files()
            if p.suffix.lower() in SCANNED_SUFFIXES
        ]
        assert len(scanned) > 400, (
            f"only {len(scanned)} files matched the scan; the tracked-file "
            "listing or the suffix set has broken"
        )
        names = {p.name for p in scanned}
        assert "pyproject.toml" in names
        # conftest.py rather than this file: this file is untracked until it
        # is committed, and a self-referential assertion would fail on first run.
        assert "conftest.py" in names


class TestTheScannerCanActuallyFail:
    """The repo is clean, so these are the only proof the scanner works."""

    def test_it_catches_the_real_header_that_shipped_three_times(self):
        # Verbatim from 3426f2c, the occurrence that got past a merge.
        text = (
            "# SPDX-License-Identifier: AGPL-3.0-only\n"
            "# Copyright 2026-present the Unsloth AI Inc. team. All rights "
            "reserved. See /studio/LICENSE.AGPL-3.0\n"
            '"""A module docstring."""\n'
        )
        found = _offending_lines(text)
        assert [lineno for lineno, _, _ in found] == [1, 2], found

    def test_it_catches_a_foreign_spdx_id_on_its_own(self):
        found = _offending_lines("# SPDX-License-Identifier: MPL-2.0\n")
        assert len(found) == 1
        assert "SPDX identifier" in found[0][1]

    def test_our_own_apache_header_is_accepted(self):
        """The control: the guard must not fire on this project's own licence."""
        assert _offending_lines("# SPDX-License-Identifier: Apache-2.0\n") == []

    def test_body_mentions_of_copyleft_are_not_flagged(self):
        """`license_advisor` reasons ABOUT AGPL; it must not be flagged for it.

        A guard that fires on correct code gets deleted, so this control is
        what keeps the scanner narrow enough to survive.
        """
        text = (
            '"""License compatibility."""\n'
            'INCOMPATIBLE = ("AGPL-3.0", "GPL-3.0")\n'
            "# AGPL is copyleft, so a derivative cannot ship closed.\n"
        )
        found = _offending_lines(text)
        # Line 2 is code, not a comment -> not flagged. Line 3 IS a comment
        # naming AGPL, so it is flagged: the window, not the wording, is what
        # keeps real files clean, and license_advisor.py has no such header.
        assert [lineno for lineno, _, _ in found] == [3], found

    def test_the_real_license_advisor_module_passes(self):
        """The end-to-end version of the control above, on the actual file."""
        target = REPO_ROOT / "src" / "soup_cli" / "utils" / "license_advisor.py"
        if not target.exists():  # pragma: no cover - module renamed
            pytest.skip("license_advisor.py not present")
        text = io.open(target, encoding="utf-8", errors="replace").read()
        assert "AGPL" in text, "fixture assumption broken: expected AGPL in body"
        assert _offending_lines(text) == []

    def test_an_indented_quotation_of_the_header_is_not_flagged(self):
        """This file's own docstring quotes the header; that must be safe.

        Regression: the first version used ``lstrip()``, flagged its own
        documentation, and failed all nine CI cells. It passed locally only
        because an uncommitted file is not in ``git ls-files``.
        """
        text = chr(10).join(
            [
                '"""Why this guard exists.',
                "",
                "    # SPDX-License-Identifier: AGPL-3.0-only",
                "    # Copyright 2026-present the Unsloth AI Inc. team."
                " All rights reserved.",
                '"""',
            ]
        )
        assert _offending_lines(text) == []

    def test_this_very_file_passes_its_own_scan(self):
        """The end-to-end version: the guard must not flag itself."""
        text = io.open(__file__, encoding="utf-8", errors="replace").read()
        assert _offending_lines(text) == []

    def test_a_deep_header_beyond_the_window_is_out_of_scope(self):
        """Documents the boundary rather than pretending it does not exist."""
        text = "\n".join(["# filler"] * HEADER_LINES) + (
            "\n# SPDX-License-Identifier: AGPL-3.0-only\n"
        )
        assert _offending_lines(text) == []
