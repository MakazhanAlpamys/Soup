r"""Repo-wide shape invariant: CONTRIBUTING.md must not carry hardcoded test counts.

When contributing to Soup, over 50% of pull requests add or modify unit tests.
Maintaining hardcoded test suite counts (e.g. `370 files, 18151 tests`) in
`CONTRIBUTING.md` creates a continuous drift treadmill, off-by-one guesswork,
and unnecessary merge conflicts across concurrent contributions.

Per Issue #465, the resolution is subtraction over redundant maintenance:
eliminating the exact count metrics makes drift structurally impossible, while
preserving developer ergonomics.

This test acts as a permanent anti-regression shape fence: it guarantees that
the `tests/` directory entry and the `### Test Files` section header remain
present in `CONTRIBUTING.md` (anti-rot), while asserting that neither line carries
hardcoded file or test counts, while safely admitting version tags (v0.69.0),
Python version ranges (Python 3.10-3.12), and tool requirements (pytest 8.x).
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRIBUTING_PATH = ROOT / "CONTRIBUTING.md"

# Matches count metrics: e.g. "370 files", "18151 tests", "18,151 tests", "(370)"
_COUNT_PATTERN = re.compile(
    r"\b\d+[\d,]*\+?\s*(?:files?|tests?)\b|\(\s*\d+\s*(?:files?)?\s*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContributingShapeVerdict:
    """The result of auditing CONTRIBUTING.md for hardcoded test counts."""

    is_valid: bool
    violations: tuple[tuple[int, str, str], ...]
    missing_sections: tuple[str, ...]

    def format_diagnostic(self) -> str:
        lines = []
        if self.missing_sections:
            lines.append("Missing required documentation sections (anti-rot violation):")
            for s in self.missing_sections:
                lines.append(f"  * Missing: {s}")
        if self.violations:
            lines.append("Hardcoded test or file counts found in CONTRIBUTING.md:")
            for lineno, line_text, section in self.violations:
                lines.append(f"  * Line {lineno} ({section}): {line_text!r}")
        return "\n".join(lines)


def audit_contributing_test_count_shape(
    file_path: pathlib.Path = CONTRIBUTING_PATH,
) -> ContributingShapeVerdict:
    """Inspect CONTRIBUTING.md test suite lines for hardcoded count metrics."""
    if not file_path.exists():
        return ContributingShapeVerdict(False, (), (f"File not found: {file_path}",))

    lines = file_path.read_text(encoding="utf-8").splitlines()
    violations = []
    missing_sections = []

    # 1. Inspect the `tests/` directory overview line
    test_dir_match = None
    for lineno, line in enumerate(lines, 1):
        if re.match(r"^\s*tests/\s+-\s+Test suite", line):
            test_dir_match = (lineno, line)
            break

    if test_dir_match is None:
        missing_sections.append("Directory overview line: 'tests/ - Test suite ...'")
    elif _COUNT_PATTERN.search(test_dir_match[1]):
        violations.append((test_dir_match[0], test_dir_match[1], "tests/ overview line"))

    # 2. Inspect the `### Test Files` section header
    test_files_match = None
    for lineno, line in enumerate(lines, 1):
        if re.match(r"^###\s+Test Files", line):
            test_files_match = (lineno, line)
            break

    if test_files_match is None:
        missing_sections.append("Section header: '### Test Files ...'")
    elif _COUNT_PATTERN.search(test_files_match[1]):
        violations.append((test_files_match[0], test_files_match[1], "### Test Files header"))

    return ContributingShapeVerdict(
        is_valid=(len(violations) == 0 and len(missing_sections) == 0),
        violations=tuple(violations),
        missing_sections=tuple(missing_sections),
    )


class TestContributingTestCountsAreNotHardcoded:
    """CONTRIBUTING.md must describe the test suite without drifting hardcoded numbers."""

    def test_contributing_test_suite_lines_contain_no_hardcoded_counts(self) -> None:
        verdict = audit_contributing_test_count_shape(CONTRIBUTING_PATH)
        assert verdict.is_valid, (
            f"CONTRIBUTING.md failed test count shape invariant:\n"
            f"{verdict.format_diagnostic()}\n"
            f"Remove hardcoded file/test count numbers to prevent drift (#465)."
        )

    def test_contributing_target_sections_are_present(self) -> None:
        """Control: ensure the target sections are not silently deleted."""
        verdict = audit_contributing_test_count_shape(CONTRIBUTING_PATH)
        assert verdict.missing_sections == (), (
            f"Required sections missing from CONTRIBUTING.md: {verdict.missing_sections}"
        )


class TestTheGuardHasTeeth:
    """CONTROL. Prove the auditor catches hardcoded counts while admitting version tags."""

    def test_hardcoded_counts_in_test_suite_line_fails(self, tmp_path: pathlib.Path) -> None:
        doc = tmp_path / "CONTRIBUTING.md"
        doc.write_text(
            "tests/ - Test suite (370 files, 18151 tests)\n### Test Files\n",
            encoding="utf-8",
        )
        verdict = audit_contributing_test_count_shape(doc)
        assert not verdict.is_valid
        assert len(verdict.violations) == 1
        assert verdict.violations[0][0] == 1

    def test_hardcoded_counts_in_test_files_header_fails(self, tmp_path: pathlib.Path) -> None:
        doc = tmp_path / "CONTRIBUTING.md"
        doc.write_text("tests/ - Test suite\n### Test Files (370 files)\n", encoding="utf-8")
        verdict = audit_contributing_test_count_shape(doc)
        assert not verdict.is_valid
        assert len(verdict.violations) == 1
        assert verdict.violations[0][0] == 2

    def test_bare_digits_in_test_files_header_fails(self, tmp_path: pathlib.Path) -> None:
        doc = tmp_path / "CONTRIBUTING.md"
        doc.write_text("tests/ - Test suite\n### Test Files (370)\n", encoding="utf-8")
        verdict = audit_contributing_test_count_shape(doc)
        assert not verdict.is_valid
        assert len(verdict.violations) == 1
        assert verdict.violations[0][0] == 2

    def test_version_tags_and_python_versions_are_accepted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """CONTROL: Legitimate version tags (v0.69.0), pytest, and Python ranges must pass."""
        doc = tmp_path / "CONTRIBUTING.md"
        doc.write_text(
            "tests/  - Test suite (v0.69.0, pytest 8.x, Python 3.10-3.12)\n"
            "### Test Files\n",
            encoding="utf-8",
        )
        verdict = audit_contributing_test_count_shape(doc)
        assert verdict.is_valid
        assert verdict.violations == ()

    def test_missing_test_suite_line_fails_anti_rot_check(self, tmp_path: pathlib.Path) -> None:
        doc = tmp_path / "CONTRIBUTING.md"
        doc.write_text("### Test Files\n", encoding="utf-8")
        verdict = audit_contributing_test_count_shape(doc)
        assert not verdict.is_valid
        assert len(verdict.missing_sections) == 1

    def test_missing_test_files_header_fails_anti_rot_check(self, tmp_path: pathlib.Path) -> None:
        doc = tmp_path / "CONTRIBUTING.md"
        doc.write_text("tests/ - Test suite\n", encoding="utf-8")
        verdict = audit_contributing_test_count_shape(doc)
        assert not verdict.is_valid
        assert len(verdict.missing_sections) == 1

    def test_nonexistent_file_fails(self, tmp_path: pathlib.Path) -> None:
        nonexistent = tmp_path / "DOES_NOT_EXIST.md"
        verdict = audit_contributing_test_count_shape(nonexistent)
        assert not verdict.is_valid

    def test_valid_unrelated_digits_in_contributing_are_accepted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """CONTROL: Numbers in other sections (recipe count, Python version) must pass."""
        doc = tmp_path / "CONTRIBUTING.md"
        doc.write_text(
            "Python 3.11 is required.\n"
            "recipes/ - Ready-made models (146 recipes)\n"
            "tests/ - Test suite covering all modules\n"
            "### Test Files\n",
            encoding="utf-8",
        )
        verdict = audit_contributing_test_count_shape(doc)
        assert verdict.is_valid
