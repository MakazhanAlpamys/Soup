"""The version lives in two files and nothing asserted they agree.

`pyproject.toml` and `src/soup_cli/__init__.py` both carry the version, and the
release checklist says to bump both. Every other version test in this suite is a
FLOOR check (`>= 0.56.0`, `>= 0.53.11`, ...), which passes happily when one file
is bumped and the other is forgotten — the wheel then ships with metadata that
disagrees with `soup version`, and CI stays green. Until now that drift was
caught only by a human eye on step 6a of the checklist.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "soup_cli" / "__init__.py"

_PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)
_INIT_VERSION = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.M)


def _pyproject_version(text: str) -> str:
    """The version from `[project]`, not from some other table.

    A naive first-match search would also accept a `version = "..."` belonging to
    a tool section, so anchor on the `[project]` table and stop at the next one.
    """
    project = re.search(r"^\[project\]\s*$(.*?)^\[", text, re.M | re.S)
    assert project, "pyproject.toml has no [project] table"
    match = _PYPROJECT_VERSION.search(project.group(1))
    assert match, "no version line in the [project] table of pyproject.toml"
    return match.group(1)


def _init_version(text: str) -> str:
    match = _INIT_VERSION.search(text)
    assert match, "no __version__ line in src/soup_cli/__init__.py"
    return match.group(1)


class TestVersionSync:
    def test_the_two_files_agree(self):
        declared = _pyproject_version(PYPROJECT.read_text(encoding="utf-8"))
        package = _init_version(INIT.read_text(encoding="utf-8"))
        assert declared == package, (
            f"pyproject.toml says {declared!r} but src/soup_cli/__init__.py says "
            f"{package!r} — the release checklist requires BOTH to be bumped. "
            f"The wheel would ship metadata that disagrees with `soup version`."
        )

    def test_the_importable_package_agrees_too(self):
        """The parsed file and the imported module can differ if a stale build
        artifact shadows the source tree — the exact "stale shadow install"
        the checklist warns about at step 6a."""
        from soup_cli import __version__

        assert __version__ == _init_version(INIT.read_text(encoding="utf-8"))

    @pytest.mark.parametrize(
        "text,expected",
        [
            ('[project]\nname = "x"\nversion = "1.2.3"\n\n[tool.ruff]\n', "1.2.3"),
            # a version in another table must not be mistaken for the project's
            ('[project]\nname = "x"\nversion = "1.2.3"\n\n[tool.x]\nversion = "9.9.9"\n', "1.2.3"),
        ],
    )
    def test_parser_reads_the_project_table(self, text, expected):
        assert _pyproject_version(text) == expected

    def test_the_assertion_would_actually_catch_a_drift(self):
        """CONTROL. Without this, `test_the_two_files_agree` proves nothing about
        the comparison itself — it would pass just as happily against a helper
        that returned the same constant twice."""
        drifted = _pyproject_version('[project]\nversion = "0.99.0"\n\n[tool.x]\n')
        package = _init_version('__version__ = "0.72.4"\n')
        assert drifted != package

    def test_version_is_a_plausible_release_string(self):
        version = _init_version(INIT.read_text(encoding="utf-8"))
        assert re.fullmatch(r"\d+\.\d+\.\d+(?:[.-]?(?:a|b|rc|dev)\d*)?", version), version
