"""The licence files are pinned in the package metadata and say the same thing (#1677).

Until #1677 the wheel and the sdist carried ``LICENSE`` and ``NOTICE`` only because
hatchling's default ``license-files`` globs happen to match them: ``pyproject.toml``
named neither, and nothing failed if the default changed or a file was renamed. The
``LICENSE`` itself had drifted from the published Apache-2.0 text (section 6 lost
"reasonable and customary use in", section 9 said "Support" where the published text
says "Additional Liability"), and four different holder spellings were in circulation.

Three locks:

1. ``[project] license-files`` names ``LICENSE`` and ``NOTICE``, and both exist.
2. ``LICENSE`` carries the two passages that had drifted, and not the drifted wording.
3. The holder line is one string in ``NOTICE``, the READMEs and the package authors.

``pyproject.toml`` is read with a regex, not ``tomllib``, which is 3.11+ while this
project supports 3.10 (same reason as ``test_requires_python_bound.py``).
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

#: The one holder spelling. Change it here and in every file below together.
HOLDER_NAME = "Makazhan Alpamys and the Soup contributors"
HOLDER = f"Copyright 2026 {HOLDER_NAME}"
README_NAMES = ["README.md", *sorted(path.name for path in ROOT.glob("README.*.md"))]

_LICENSE_FILES = re.compile(r"^license-files\s*=\s*\[([^\]]*)\]", re.M)


def _read(path: pathlib.Path) -> str:
    """UTF-8 text with LF endings: a Windows checkout with autocrlf has CRLF."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _project_table(text: str) -> str:
    project = re.search(r"^\[project\]\s*$(.*?)^\[", text, re.M | re.S)
    assert project, "pyproject.toml has no [project] table"
    return project.group(1)


def _license_files(text: str) -> list[str]:
    match = _LICENSE_FILES.search(_project_table(text))
    assert match, "no license-files key in the [project] table of pyproject.toml"
    return re.findall(r'"([^"]+)"', match.group(1))


class TestLicenseFilesArePinned:
    def test_project_table_lists_license_and_notice(self):
        assert sorted(_license_files(_read(PYPROJECT))) == ["LICENSE", "NOTICE"]

    @pytest.mark.parametrize("name", ["LICENSE", "NOTICE"])
    def test_the_listed_file_exists_at_the_repo_root_and_is_not_empty(self, name):
        path = ROOT / name
        assert path.is_file(), f"{name} is listed in license-files but is not at the repo root"
        assert _read(path).strip(), f"{name} is empty"

    def test_the_scanner_can_fail_on_a_missing_key(self):
        stripped = re.sub(r"^license-files", "licence-files", _read(PYPROJECT), flags=re.M)
        assert stripped != _read(PYPROJECT)  # CONTROL: the key was there to remove
        with pytest.raises(AssertionError, match="no license-files key"):
            _license_files(stripped)


class TestLicenseIsThePublishedApacheText:
    def test_section_9_says_additional_liability(self):
        text = _read(ROOT / "LICENSE")
        assert "9. Accepting Warranty or Additional Liability." in text
        assert "warranty or additional liability." in text
        assert "Warranty or Support" not in text

    def test_section_6_says_reasonable_and_customary_use(self):
        text = " ".join(_read(ROOT / "LICENSE").split())
        assert "reasonable and customary use in describing the origin of the Work" in text

    def test_the_appendix_is_the_unfilled_template(self):
        assert "Copyright [yyyy] [name of copyright owner]" in _read(ROOT / "LICENSE")


class TestHolderIsSpelledOnce:
    def test_notice_is_the_product_name_and_the_holder_line(self):
        assert _read(ROOT / "NOTICE").splitlines() == ["Soup", HOLDER]

    @pytest.mark.parametrize("name", README_NAMES)
    def test_every_readme_names_the_holder_in_english(self, name):
        assert HOLDER in _read(ROOT / name)

    def test_package_authors_use_the_holder_name(self):
        text = _read(PYPROJECT)
        authors = re.search(r"^authors\s*=\s*\[(.*?)^\]", _project_table(text), re.M | re.S)
        assert authors, "no authors list in the [project] table of pyproject.toml"
        assert HOLDER_NAME in authors.group(1)
        assert "Soup Team" not in authors.group(1)
