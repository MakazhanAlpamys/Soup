"""Every file of the packed-LoRA package says what licence it is under, and the ported ones say
where they come from (the Soup Kernel Challenge submission of team Luck and the idea teams); the
ones written for Soup say that instead. A file is one or the other by its content: ``OWN`` lists
the ones with no ported line in them."""

from pathlib import Path

import pytest

import soup_cli.kernels.packed_lora as package

ROOT = Path(package.__file__).parent
FILES = sorted(ROOT.rglob("*.py"))
# written for Soup, not ported
OWN = {"config.py", "loader.py", "selftest.py"}
# the package above this one, and the test files adapted from the challenge team's
KERNELS_INIT = ROOT.parent / "__init__.py"
PORTED_TESTS = [Path(__file__).parent / "test_packed_lora_kernels_gpu.py"]
SPDX = "# SPDX-License-Identifier: Apache-2.0"


def _id(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def test_the_package_has_files():
    assert len(FILES) >= 20 and {f.name for f in FILES} >= {"nf4.py", "q8.py", "head.py"}


@pytest.mark.parametrize("path", [*FILES, KERNELS_INIT, *PORTED_TESTS], ids=_id)
def test_spdx_line_on_every_file(path):
    assert path.read_text(encoding="utf-8").splitlines()[0] == SPDX


@pytest.mark.parametrize(
    "path", [*(f for f in FILES if f.name not in OWN), *PORTED_TESTS], ids=_id)
def test_origin_on_every_ported_file(path):
    head = path.read_text(encoding="utf-8")[:900]
    assert "Soup Kernel Challenge (Decentrathon 6.0)" in head
    assert "BAITC-Hacks/hack-e79529fc-luck@b3f1225f" in head
    assert "Null Pointers, 21, DarkDaur and OrbiTeam" in head


@pytest.mark.parametrize("path", [f for f in FILES if f.name in OWN] + [KERNELS_INIT], ids=_id)
def test_files_written_for_soup_say_so_and_claim_no_origin(path):
    head = path.read_text(encoding="utf-8")[:900]
    assert "Written for Soup" in head
    assert "BAITC-Hacks" not in head


@pytest.mark.parametrize("path", sorted((ROOT / "triton").glob("*.py")), ids=_id)
def test_kernel_files_say_what_was_modified_for_soup(path):
    text = path.read_text(encoding="utf-8")[:900]
    assert "Modified for Soup" in text or path.name == "__init__.py"


def test_qwen3_gate_is_closed_in_the_docstrings():
    from soup_cli.kernels.packed_lora import support

    assert "gate closed" in (support.__doc__ or "")
    assert "gate is closed" in (package.__doc__ or "")
    assert support.OPEN_ARCHITECTURES == ("Qwen2ForCausalLM",)
