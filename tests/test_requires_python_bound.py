"""`requires-python` must carry an UPPER bound, and it must match the CI matrix.

`requires-python = ">=3.10"` with no ceiling tells pip that every future Python is
supported. It is not: CI tests 3.10, 3.11 and 3.12 only. On 3.13+ pip happily
resolves a torch wheel nobody here has ever run, and the failure is not a Soup
error message — it is a loader crash inside `c10.dll` / `libc10.so` before any
Soup code executes, so the user has nothing to act on.

The bound is derived from the CI matrix rather than hardcoded, because a floating
declaration and a fixed matrix drifting apart is exactly the failure this file
exists to catch. Widening the matrix and widening the bound must happen together.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

_REQUIRES_PYTHON = re.compile(r'^requires-python\s*=\s*"([^"]+)"', re.M)
_MATRIX_LINE = re.compile(r'^\s*python-version:\s*\[([^\]]+)\]', re.M)


def _requires_python(text: str) -> str:
    """Read `requires-python` from the `[project]` table specifically."""
    project = re.search(r"^\[project\]\s*$(.*?)^\[", text, re.M | re.S)
    assert project, "pyproject.toml has no [project] table"
    match = _REQUIRES_PYTHON.search(project.group(1))
    assert match, "no requires-python line in the [project] table of pyproject.toml"
    return match.group(1)


def _parse_bounds(spec: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ((floor_major, floor_minor), (ceiling_major, ceiling_minor))."""
    floor = re.search(r">=\s*(\d+)\.(\d+)", spec)
    ceiling = re.search(r"<\s*(\d+)\.(\d+)", spec)
    assert floor, f"no >= floor in requires-python {spec!r}"
    assert ceiling, (
        f"requires-python is {spec!r} and has NO upper bound. Without one, pip on an "
        f"untested Python resolves untested torch wheels that die in the native "
        f"extension before Soup runs. Set an upper bound matching the CI matrix."
    )
    return (int(floor.group(1)), int(floor.group(2))), (
        int(ceiling.group(1)),
        int(ceiling.group(2)),
    )


def _ci_python_versions() -> list[tuple[int, int]]:
    """The matrix `python-version` list from the test job in ci.yml."""
    match = _MATRIX_LINE.search(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert match, "no `python-version: [...]` matrix found in .github/workflows/ci.yml"
    versions = re.findall(r"(\d+)\.(\d+)", match.group(1))
    assert versions, "the python-version matrix parsed to an empty list"
    return sorted((int(major), int(minor)) for major, minor in versions)


class TestRequiresPythonHasAnUpperBound:
    def test_an_upper_bound_is_declared(self):
        _parse_bounds(_requires_python(PYPROJECT.read_text(encoding="utf-8")))

    def test_the_floor_is_the_oldest_tested_python(self):
        floor, _ = _parse_bounds(_requires_python(PYPROJECT.read_text(encoding="utf-8")))
        assert floor == _ci_python_versions()[0], (
            f"requires-python floor {floor} does not match the oldest CI python "
            f"{_ci_python_versions()[0]}"
        )

    def test_the_ceiling_excludes_the_first_untested_python(self):
        """`<3.13` when the newest tested is 3.12 — the exclusive bound is the
        next minor after the last one CI actually runs."""
        _, ceiling = _parse_bounds(_requires_python(PYPROJECT.read_text(encoding="utf-8")))
        newest_major, newest_minor = _ci_python_versions()[-1]
        assert ceiling == (newest_major, newest_minor + 1), (
            f"requires-python ceiling is <{ceiling[0]}.{ceiling[1]} but CI's newest "
            f"tested Python is {newest_major}.{newest_minor}. Either the matrix grew "
            f"and the bound did not, or the bound was widened without adding the cell."
        )

    def test_every_ci_python_satisfies_the_declared_range(self):
        floor, ceiling = _parse_bounds(_requires_python(PYPROJECT.read_text(encoding="utf-8")))
        for version in _ci_python_versions():
            assert floor <= version < ceiling, (
                f"CI runs Python {version[0]}.{version[1]}, which the declared "
                f"requires-python range excludes"
            )


class TestTheBoundCheckHasTeeth:
    """CONTROL. Without these, the tests above would pass just as happily against
    a parser that never rejects anything."""

    def test_a_missing_ceiling_is_rejected(self):
        with pytest.raises(AssertionError, match="NO upper bound"):
            _parse_bounds(">=3.10")

    @pytest.mark.parametrize("spec", [">=3.10,<3.13", ">= 3.10, < 3.13"])
    def test_a_declared_ceiling_is_accepted(self, spec):
        assert _parse_bounds(spec) == ((3, 10), (3, 13))

    def test_a_ceiling_that_lags_the_matrix_is_detectable(self):
        """A bound of `<3.12` against a matrix ending at 3.12 must not read as
        agreement — otherwise the ceiling test could never fail."""
        _, ceiling = _parse_bounds(">=3.10,<3.12")
        assert ceiling != (3, 13)
