"""Every file of the packed-LoRA package says what licence it is under, and the ported ones say
where they come from (the Soup Kernel Challenge submission of team Luck and the idea teams)."""

from pathlib import Path

import pytest

import soup_cli.kernels.packed_lora as package

ROOT = Path(package.__file__).parent
FILES = sorted(ROOT.rglob("*.py"))
# written for Soup, not ported
OWN = {"config.py", "loader.py", "selftest.py"}
SPDX = "# SPDX-License-Identifier: Apache-2.0"


def _id(path):
    return str(path.relative_to(ROOT))


def test_the_package_has_files():
    assert len(FILES) >= 20 and {f.name for f in FILES} >= {"nf4.py", "q8.py", "head.py"}


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_spdx_line_on_every_file(path):
    assert path.read_text(encoding="utf-8").splitlines()[0] == SPDX


@pytest.mark.parametrize("path", [f for f in FILES if f.name not in OWN], ids=_id)
def test_origin_on_every_ported_file(path):
    head = path.read_text(encoding="utf-8")[:900]
    assert "Soup Kernel Challenge (Decentrathon 6.0)" in head
    assert "BAITC-Hacks/hack-e79529fc-luck@b3f1225f" in head
    assert "Null Pointers, 21, DarkDaur and OrbiTeam" in head


@pytest.mark.parametrize("path", sorted((ROOT / "triton").glob("*.py")), ids=_id)
def test_kernel_files_say_what_was_modified_for_soup(path):
    text = path.read_text(encoding="utf-8")[:900]
    assert "Modified for Soup" in text or path.name == "__init__.py"


def test_qwen3_gate_is_closed_in_the_docstrings():
    from soup_cli.kernels.packed_lora import support

    assert "gate closed" in (support.__doc__ or "")
    assert "gate is closed" in (package.__doc__ or "")
    assert support.OPEN_ARCHITECTURES == ("Qwen2ForCausalLM",)
