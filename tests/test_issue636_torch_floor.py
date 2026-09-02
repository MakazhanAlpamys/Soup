"""The declared torch floor must be the binding one, and doctor's copy pinned (#636).

``pyproject.toml [train]`` declared ``torch>=2.3.0`` while requiring
``transformers>=5.16.1``, whose own torch extra forces ``torch>=2.5`` — so the
declared floor could never bind, and ``soup doctor`` carried an unpinned second
copy of it. These tests pin both halves:

* the pyproject floor is at least what the *declared* transformers floor's own
  torch extra requires. That requirement is read from real metadata only where
  the installed transformers is exactly the declared floor — the
  ``transformers-floor`` CI job installs precisely that release under
  ``.github/constraints/transformers-floor.txt``. Anywhere else the check skips
  with a reason instead of passing vacuously against whatever transformers
  happens to be installed (an older one forces a lower floor, a newer one could
  turn the repo red without any repo change).
* the constraints file pins the declared transformers floor, so the metadata
  check above can never silently validate a different release than the one
  ``[train]`` declares.
* doctor's torch literal equals the pyproject declaration — hermetic, repo text
  against src literal, the pin that keeps the second copy honest.

pyproject is parsed with regex rather than ``tomllib`` because ``tomllib`` is
3.11+ and this repo supports 3.10 — the same convention as
``tests/test_requires_python_bound.py``.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from packaging.version import Version

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CONSTRAINTS = ROOT / ".github" / "constraints" / "transformers-floor.txt"


def _train_extra_block() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    table = re.search(
        r"^\[project\.optional-dependencies\]\s*$(.*?)^\[", text, re.M | re.S
    )
    assert table, "pyproject.toml has no [project.optional-dependencies] table"
    train = re.search(r"^train\s*=\s*\[(.*?)^\]", table.group(1), re.M | re.S)
    assert train, "no train = [...] array in [project.optional-dependencies]"
    return train.group(1)


def _declared_torch_floor() -> str:
    match = re.search(r'"torch\s*>=\s*([0-9][^"]*)"', _train_extra_block())
    assert match, "the [train] extra declares no torch>= floor"
    return match.group(1)


def _declared_transformers_floor() -> str:
    # No closing-quote anchor: the entry is "transformers>=5.16.1,<6.0.0".
    match = re.search(r'"transformers\s*>=\s*([0-9][^",]*)', _train_extra_block())
    assert match, "the [train] extra declares no transformers>= floor"
    return match.group(1)


def _constraints_pins() -> dict[str, str]:
    text = CONSTRAINTS.read_text(encoding="utf-8")
    return dict(re.findall(r"^([A-Za-z0-9_-]+)==([^\s#]+)\s*$", text, re.M))


class TestIssue636TorchFloor:
    def test_declared_floor_meets_the_declared_transformers_requirement(self) -> None:
        # Mutation check: on the transformers-floor CI job — where the installed
        # transformers IS the declared floor — setting pyproject back to
        # torch>=2.3.0 fails this test by name, because 5.16.1's own torch
        # extra requires >=2.5.
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import requires as _requires
        from importlib.metadata import version as _version

        from packaging.requirements import Requirement

        declared_tf = Version(_declared_transformers_floor())
        try:
            installed_tf = Version(_version("transformers"))
        except PackageNotFoundError:
            pytest.skip(
                "transformers is not installed; the guard runs against real "
                "metadata on the transformers-floor CI job"
            )
        if installed_tf != declared_tf:
            pytest.skip(
                f"installed transformers {installed_tf} is not the declared "
                f"floor {declared_tf}; comparing against a different release's "
                f"metadata would validate the wrong requirement — the "
                f"transformers-floor CI job installs exactly {declared_tf}"
            )

        floors = []
        for raw in _requires("transformers") or []:
            req = Requirement(raw)
            if req.name.lower() != "torch":
                continue
            if req.marker is not None and not req.marker.evaluate({"extra": "torch"}):
                continue
            floors.extend(
                spec.version for spec in req.specifier if spec.operator == ">="
            )
        assert floors, f"transformers {declared_tf} declares no torch>= floor"
        required = Version(max(floors, key=Version))

        declared = Version(_declared_torch_floor())
        assert declared >= required, (
            f"pyproject [train] declares torch>={declared} but its own declared "
            f"transformers floor {declared_tf} requires torch>={required}; the "
            f"declared floor cannot bind and invites trust it does not "
            f"deserve (#636)"
        )

    def test_constraints_file_pins_the_declared_floors(self) -> None:
        # Hermetic. The metadata guard above only runs where the installed
        # transformers equals the declared floor, and the transformers-floor
        # CI job creates that state from this constraints file — so the pin
        # must be the declared floor, or the guard validates another release.
        # The torch pin is the version that job actually installs beside it;
        # the declared floor must not exceed it or the job cannot resolve.
        pins = _constraints_pins()
        assert pins.get("transformers") == _declared_transformers_floor(), (
            "the transformers-floor CI job validates "
            f"transformers=={pins.get('transformers')} but [train] declares "
            f">={_declared_transformers_floor()} — move the pin with the "
            "declaration or the drift guard checks the wrong release (#636)"
        )
        assert Version(_declared_torch_floor()) <= Version(pins["torch"]), (
            f"[train] declares torch>={_declared_torch_floor()} but the floor "
            f"job pins torch=={pins['torch']}, which would not satisfy it (#636)"
        )

    def test_doctor_torch_floor_matches_the_pyproject_declaration(self) -> None:
        # Hermetic. doctor.py keeps a literal copy of the floor; this is the
        # pin that keeps it honest — mutating either side alone fails here by
        # name. Reading installed metadata instead was tried and rejected:
        # dist-info records the install's history, so an editable checkout
        # whose pyproject moved on reports a floor nobody declared, and an
        # uninstalled source tree reports "?" which _version_ok treats as OK.
        from soup_cli.commands.doctor import DEPS

        torch_rows = [row for row in DEPS if row[1] == "torch"]
        assert len(torch_rows) == 1, "expected exactly one torch row in DEPS"
        assert torch_rows[0][2] == _declared_torch_floor(), (
            f"doctor.py checks torch>={torch_rows[0][2]} but pyproject.toml "
            f"[train] declares torch>={_declared_torch_floor()} — doctor must "
            f"report the declared floor (#636)"
        )
