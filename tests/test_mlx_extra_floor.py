"""#2471 - the ``mlx`` extra declares the ``mlx`` floor that ``mlx-lm`` needs.

mlx-lm 0.31.3 requires ``mlx>=0.31.2`` on macOS, so the extra's ``mlx>=0.20.0``
named a lower bound no resolver could ever install for it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def _mlx_extra() -> dict[str, Requirement]:
    with (ROOT / "pyproject.toml").open("rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    requirements = [Requirement(line) for line in extras["mlx"]]
    return {requirement.name: requirement for requirement in requirements}


def _floor(requirement: Requirement) -> Version:
    floors = [Version(s.version) for s in requirement.specifier if s.operator == ">="]
    assert len(floors) == 1, str(requirement)
    return floors[0]


def test_the_mlx_floor_is_the_one_mlx_lm_requires():
    extra = _mlx_extra()
    assert _floor(extra["mlx-lm"]) == Version("0.31.3")
    assert _floor(extra["mlx"]) == Version("0.31.2")
