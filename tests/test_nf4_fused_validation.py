"""The fused NF4 candidate refuses shapes and quant states it cannot decode (#3455).

No GPU is needed: validation runs before any Triton launch. The supported case
is the layout bitsandbytes writes for nested NF4 at blocksize 64 with state2
blocksize 256; every altered field must be refused with a named reason.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from soup_cli.kernels.packed_lora.nf4_layout import validate_supported  # noqa: E402


def _state(n: int = 256, k: int = 256):
    absmax = torch.zeros(n * k // 64, dtype=torch.uint8)
    state2 = SimpleNamespace(
        absmax=torch.zeros(absmax.numel() // 256),
        blocksize=256,
        code=torch.arange(256, dtype=torch.uint8),
    )
    return SimpleNamespace(
        shape=(n, k),
        blocksize=64,
        quant_type="nf4",
        nested=True,
        state2=state2,
        absmax=absmax,
        offset=torch.zeros(1),
        code=torch.arange(16, dtype=torch.uint8),
    )


def _packed(n: int = 256, k: int = 256):
    return torch.zeros(n * k // 2, dtype=torch.uint8)


def test_the_supported_layout_is_accepted():
    validate_supported(_packed(), _state())


@pytest.mark.parametrize("shape", [(255, 256), (256, 255), (1, 256)])
def test_a_dimension_not_divisible_by_64_is_refused(shape):
    state = _state()
    state.shape = shape
    with pytest.raises(ValueError, match="divisible by 64"):
        validate_supported(_packed(*shape), state)


def test_a_wrong_blocksize_is_refused():
    state = _state()
    state.blocksize = 128
    with pytest.raises(ValueError, match="blocksize"):
        validate_supported(_packed(), state)


def test_a_wrong_quant_type_is_refused():
    state = _state()
    state.quant_type = "fp8"
    with pytest.raises(ValueError, match="nf4"):
        validate_supported(_packed(), state)


def test_a_non_nested_state_is_refused():
    state = _state()
    state.nested = False
    with pytest.raises(ValueError, match="nested"):
        validate_supported(_packed(), state)


def test_a_wrong_state2_blocksize_is_refused():
    state = _state()
    state.state2.blocksize = 128
    with pytest.raises(ValueError, match="state2 blocksize"):
        validate_supported(_packed(), state)


def test_a_short_packed_weight_is_refused():
    with pytest.raises(ValueError, match="packed weight"):
        validate_supported(_packed()[:-1], _state())


def test_a_wrong_absmax_count_is_refused():
    state = _state()
    state.absmax = state.absmax[:-1]
    with pytest.raises(ValueError, match="absmax"):
        validate_supported(_packed(), state)


def test_a_non_uint8_absmax_is_refused():
    state = _state()
    state.absmax = state.absmax.float()
    with pytest.raises(ValueError, match="uint8"):
        validate_supported(_packed(), state)


def test_a_wrong_state2_count_is_refused():
    state = _state()
    state.state2.absmax = state.state2.absmax[:-1]
    with pytest.raises(ValueError, match="state2 absmax"):
        validate_supported(_packed(), state)


def test_a_non_scalar_offset_is_refused():
    state = _state()
    state.offset = torch.zeros(2)
    with pytest.raises(ValueError, match="scalar offset"):
        validate_supported(_packed(), state)
