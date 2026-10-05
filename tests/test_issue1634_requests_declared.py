"""#1634: the ``[train]`` extra declares ``requests``.

trl 0.29 imports ``requests`` at module level in ``trl/generation/vllm_client.py``,
which ``trl/trainer/grpo_trainer.py`` imports, but lists it only under its ``vllm``
extra. Until datasets 5.1.0 the package arrived through datasets' own dependency;
when that was dropped, a fresh ``[train]`` install could not import the GRPO or
online-DPO trainers at all (``ModuleNotFoundError: No module named 'requests'``,
29 tests in CI). Soup declares the dependency itself until trl does.

pyproject is parsed with regex rather than ``tomllib`` because ``tomllib`` is 3.11+
and this repo supports 3.10 (the convention of ``tests/test_issue636_torch_floor.py``).
"""

from __future__ import annotations

import importlib
import pathlib
import re

import pytest
from packaging.version import Version

PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"

#: datasets 5.0.1 declared ``requests>=2.32.2``; the floor Soup inherits must not be lower
_INHERITED_FLOOR = Version("2.32.2")


def _train_extra_block() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    table = re.search(r"^\[project\.optional-dependencies\]\s*$(.*?)^\[", text, re.M | re.S)
    assert table, "pyproject.toml has no [project.optional-dependencies] table"
    train = re.search(r"^train\s*=\s*\[(.*?)^\]", table.group(1), re.M | re.S)
    assert train, "no train = [...] array in [project.optional-dependencies]"
    return train.group(1)


def test_the_train_extra_declares_requests() -> None:
    match = re.search(r'^\s*"requests\s*>=\s*([0-9][^",]*)', _train_extra_block(), re.M)
    assert match, "the [train] extra declares no requests>= floor (#1634)"
    assert Version(match.group(1)) >= _INHERITED_FLOOR


def test_requests_is_importable_wherever_trl_is() -> None:
    """The symptom itself: with trl installed, ``requests`` must be present, whichever
    package used to bring it in."""
    pytest.importorskip("trl")
    importlib.import_module("requests")
