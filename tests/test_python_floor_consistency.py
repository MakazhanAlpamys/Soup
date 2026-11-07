"""The supported Python window is declared once, in ``requires-python``; the rest must agree.

Moving the floor from 3.10 to 3.11 (#1805) touched five places: ``requires-python``, the
``core-floor`` job in ``ci.yml`` (it installs the core on the oldest supported Python), ruff's
``target-version``, mypy's ``python_version`` and the README badge.
``tests/test_requires_python_bound.py`` ties ``requires-python`` to the CI support matrix; this
file ties the other four to ``requires-python``, so a later move cannot leave one behind.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
README = ROOT / "README.md"
BADGE_RE = re.compile(
    r'<img src="https://img\.shields\.io/badge/python-(\d+)\.(\d+)--(\d+)\.(\d+)-blue"'
    r' alt="Python (\d+)\.(\d+)-(\d+)\.(\d+)">'
)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _window() -> tuple[tuple[int, int], tuple[int, int]]:
    """(floor, newest supported) from ``requires-python = ">=A.B,<C.D"``."""
    spec = _pyproject()["project"]["requires-python"]
    floor = re.search(r">=\s*(\d+)\.(\d+)", spec)
    ceiling = re.search(r"<\s*(\d+)\.(\d+)", spec)
    assert floor and ceiling, f"requires-python {spec!r} needs a >= floor and a < ceiling"
    ceiling_major, ceiling_minor = int(ceiling.group(1)), int(ceiling.group(2))
    assert ceiling_minor >= 1, f"cannot name the newest Python below {spec!r}"
    return (int(floor.group(1)), int(floor.group(2))), (ceiling_major, ceiling_minor - 1)


def _dotted(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def test_the_core_floor_job_runs_the_oldest_supported_python():
    floor, _ = _window()
    steps = yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]["core-floor"]["steps"]
    setups = [
        step for step in steps if str(step.get("uses", "")).startswith("actions/setup-python@")
    ]
    assert len(setups) == 1, setups
    assert str(setups[0]["with"]["python-version"]) == _dotted(floor), (
        "core-floor must install the core on the floor requires-python declares"
    )


def test_ruff_targets_the_floor():
    floor, _ = _window()
    assert _pyproject()["tool"]["ruff"]["target-version"] == f"py{floor[0]}{floor[1]}"


def test_mypy_checks_against_the_floor():
    floor, _ = _window()
    assert _pyproject()["tool"]["mypy"]["python_version"] == _dotted(floor)


def test_the_readme_badge_names_the_supported_window():
    floor, newest = _window()
    badges = BADGE_RE.findall(README.read_text(encoding="utf-8"))
    assert len(badges) == 1, f"expected one Python badge in README.md, found {badges}"
    numbers = tuple(int(part) for part in badges[0])
    expected = (*floor, *newest)
    assert numbers[:4] == expected, f"badge URL {numbers[:4]}, requires-python {expected}"
    assert numbers[4:] == expected, f"badge alt text {numbers[4:]}, requires-python {expected}"
