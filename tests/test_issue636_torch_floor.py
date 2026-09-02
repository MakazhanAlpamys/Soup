"""The declared torch floor must be the binding one, declared in one place (#636).

``pyproject.toml [train]`` declared ``torch>=2.3.0`` while requiring
``transformers>=5.16.1``, whose own torch extra forces ``torch>=2.5`` — so the
declared floor could never bind, and ``soup doctor`` carried a second hardcoded
copy of it. These tests pin both halves:

* the pyproject floor is at least what the *installed* transformers' torch
  extra requires (mutating pyproject back to 2.3.0 fails by name), and
* doctor's torch row reads the declared floor from installed metadata rather
  than a literal (re-hardcoding any literal in doctor.py fails by name).

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
DOCTOR_PY = ROOT / "src" / "soup_cli" / "commands" / "doctor.py"


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


def _transformers_torch_floor() -> str:
    """The torch floor the *installed* transformers declares in its torch extra."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import requires as _requires

    try:
        reqs = _requires("transformers")
    except PackageNotFoundError:
        pytest.skip("transformers is not installed in this environment")
    from packaging.requirements import Requirement

    floors = []
    for raw in reqs or []:
        req = Requirement(raw)
        if req.name.lower() != "torch":
            continue
        if req.marker is not None and not req.marker.evaluate({"extra": "torch"}):
            continue
        floors.extend(spec.version for spec in req.specifier if spec.operator == ">=")
    assert floors, "installed transformers declares no torch>= floor"
    return max(floors, key=Version)


class TestIssue636TorchFloor:
    def test_declared_floor_meets_transformers_torch_requirement(self) -> None:
        # Mutation check: setting pyproject back to torch>=2.3.0 fails this test
        # by name — 2.3.0 is below the floor transformers itself forces.
        declared = Version(_declared_torch_floor())
        required = Version(_transformers_torch_floor())
        assert declared >= required, (
            f"pyproject [train] declares torch>={declared} but the installed "
            f"transformers requires torch>={required}; the declared floor cannot "
            f"bind and invites trust it does not deserve (#636)"
        )

    def test_doctor_torch_row_reads_the_declared_floor(self) -> None:
        # Mutation check: re-hardcoding a literal floor in doctor.py's DEPS
        # fails this test by name — the row must equal the metadata-derived
        # floor, which is what _declared_train_floor returns at runtime.
        from soup_cli.commands.doctor import DEPS, _declared_train_floor

        torch_rows = [row for row in DEPS if row[1] == "torch"]
        assert len(torch_rows) == 1, "expected exactly one torch row in DEPS"
        floor = _declared_train_floor("torch")
        if floor == "?":
            pytest.skip("soup-cli metadata unreadable in this environment")
        assert torch_rows[0][2] == floor, (
            f"doctor.py's torch row declares {torch_rows[0][2]!r} but soup-cli's "
            f"installed [train] metadata declares {floor!r} — doctor must read the "
            f"declared floor, not carry its own copy (#636)"
        )

    def test_doctor_source_declares_no_torch_floor_literal(self) -> None:
        # "Exactly one place in the repo declares it": the DEPS torch row must
        # not contain a version literal again.
        source = DOCTOR_PY.read_text(encoding="utf-8")
        assert not re.search(r'\(\s*"torch"\s*,\s*"torch"\s*,\s*"[0-9]', source), (
            "doctor.py hardcodes a torch floor literal in DEPS again; read it via "
            "_declared_train_floor so pyproject stays the single declaration (#636)"
        )
