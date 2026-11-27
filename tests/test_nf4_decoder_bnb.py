"""The NF4 pair decoder must equal bitsandbytes 0.50.2 on ``uint16``, with one documented
exception: bf16-denormal products, where bitsandbytes flushes to signed zero.

For every output whose float32 product ``scale * value``
has magnitude at least the bf16 minimum normal (``2**-126``), the two implementations must
agree bit for bit. On the denormal band they may differ by design, and the extension is then
checked against the exact CPU reference instead (float32 multiply, round-to-nearest-even bf16),
which is what the CUDA kernel does with ``--ftz=false``.

Why the exception exists. bitsandbytes' CUDA dequantisation kernel runs with flush-to-zero:
a result in the bf16 subnormal range becomes signed zero. The native kernel is built with
``--ftz=false`` and keeps the subnormal. Neither is a corruption; only the subnormal band
differs, and only on scale values small enough to put products there (measured: the
``1e-38`` block, 60 of its 64 outputs; ``1e-45`` rounds to zero in bf16 on both sides).
NaN payloads are not part of the contract either: IEEE 754 leaves them unspecified, so a NaN
output is compared as a NaN, not by bits.

The 16 NF4 codes (bitsandbytes 0.50.2 ``get_4bit_type("nf4")``, float32 hex in the comment):

    0  0xBF800000  -1.0                   8  0x3DA2FAFF   0.07958029955625534
    1  0xBF3239B1  -0.6961928009986877    9  0x3E24CAE3   0.16093020141124725
    2  0xBF066B30  -0.5250730514526367   10  0x3E7C04DD   0.24611230194568634
    3  0xBECA32A0  -0.39491748809814453  11  0x3EAD033A   0.33791524171829224
    4  0xBE91A24D  -0.28444138169288635  12  0x3EE1A4B8   0.44070982933044434
    5  0xBE3D353F  -0.18477343022823334  13  0x3F1007AB   0.5626170039176941
    6  0xBDBA7871  -0.09105003625154495  14  0x3F3913B3   0.7229568362236023
    7  0x00000000   0.0                  15  0x3F800000   1.0

Run it on the card with ``pytest tests/test_nf4_decoder_bnb.py -m gpu --no-cov -v``. Any
differing bit outside the denormal band is a failure that names the code byte, scale and output
index; a benchmark never replaces this equality.
"""

from __future__ import annotations

from typing import Any, Tuple

import pytest

from tests.test_native_expert_reader_port import BUILD_COMMAND, extension_or_skip

pytestmark = pytest.mark.gpu

BLOCKSIZE = 64
BNB_VERSION = "0.50.2"
#: bf16's smallest normal; below it a product is a subnormal (or zero).
BF16_MIN_NORMAL = 2.0**-126
#: The 16 NF4 code values in code order; the source of truth the exact CPU reference uses.
NF4_CODE_TABLE = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def _decoder_or_skip() -> Any:
    extension = extension_or_skip()
    assert callable(extension.PackedStreamAsync)
    if not hasattr(extension, "decode_nf4_bf16_pair"):
        pytest.skip(
            f"The native reader has no CUDA NF4 decoder. Run {BUILD_COMMAND} --with-cuda-nf4."
        )
    return extension


def _bitsandbytes() -> Tuple[Any, Any]:
    bitsandbytes = pytest.importorskip("bitsandbytes")
    assert bitsandbytes.__version__ == BNB_VERSION, (
        f"this gate is defined against bitsandbytes {BNB_VERSION}, "
        f"got {bitsandbytes.__version__}"
    )
    from bitsandbytes import functional

    return bitsandbytes, functional


def _lookup(device: Any) -> Any:
    _, functional = _bitsandbytes()
    return functional.get_4bit_type("nf4", device=device)


def _reference(codes: Any, scales: Any) -> Any:
    """bitsandbytes' own BF16 dequantisation, as ``uint16`` bits."""
    import torch

    _, functional = _bitsandbytes()
    count = int(codes.numel()) * 2
    state = functional.QuantState(
        absmax=scales,
        code=_lookup(codes.device),
        shape=(count, 1),
        dtype=torch.bfloat16,
        blocksize=BLOCKSIZE,
        quant_type="nf4",
    )
    dequantised = functional.dequantize_4bit(codes.view(-1, 1), quant_state=state)
    return dequantised.view(torch.uint16).flatten()


def _decode_pair(
    extension: Any,
    gate_codes: Any,
    gate_scales: Any,
    down_codes: Any,
    down_scales: Any,
) -> Tuple[Any, Any]:
    """Run the extension on one gate block and one down block, or either alone."""
    import torch

    device = gate_codes.device if gate_codes is not None else down_codes.device
    gate_bits = (
        torch.empty(int(gate_codes.numel()) * 2, dtype=torch.uint16, device=device)
        if gate_codes is not None
        else torch.empty(0, dtype=torch.uint16, device=device)
    )
    down_bits = (
        torch.empty(int(down_codes.numel()) * 2, dtype=torch.uint16, device=device)
        if down_codes is not None
        else torch.empty(0, dtype=torch.uint16, device=device)
    )
    empty = torch.empty(0, dtype=torch.uint8, device=device)
    lookup = _lookup(device)
    extension.decode_nf4_bf16_pair(
        device.index,
        torch.cuda.current_stream(device).cuda_stream,
        lookup.data_ptr(),
        gate_codes.data_ptr() if gate_codes is not None else empty.data_ptr(),
        gate_scales.data_ptr() if gate_scales is not None else empty.data_ptr(),
        gate_bits.data_ptr(),
        int(gate_codes.numel()) if gate_codes is not None else 0,
        down_codes.data_ptr() if down_codes is not None else empty.data_ptr(),
        down_scales.data_ptr() if down_scales is not None else empty.data_ptr(),
        down_bits.data_ptr(),
        int(down_codes.numel()) if down_codes is not None else 0,
    )
    torch.cuda.synchronize(device)
    return gate_bits, down_bits


def _first_mismatch(expected: Any, actual: Any) -> str:
    import torch

    if torch.equal(expected, actual):
        return ""
    differing = (expected.view(torch.int16) != actual.view(torch.int16)).nonzero().flatten()
    index = int(differing[0])
    return (
        f"{differing.numel()} of {expected.numel()} bits differ; first at output index "
        f"{index} (code byte {index // 2}, nibble {'high' if index % 2 == 0 else 'low'}): "
        f"bnb {int(expected[index])} != extension {int(actual[index])}"
    )


def _codes_covering_every_code(count: int, device: Any) -> Any:
    """Random packed bytes with the first 16 bytes spelling every NF4 code in both nibbles."""
    import torch

    codes = torch.randint(0, 256, ((count + 1) // 2, 1), dtype=torch.uint8, device=device)
    codes[:16, 0] = torch.arange(16, dtype=torch.int64, device=device).to(torch.uint8) * 17
    return codes


def _random_scales(count: int, device: Any) -> Any:
    import torch

    return torch.rand(count // BLOCKSIZE, dtype=torch.float32, device=device) * 2.0


def _exact_reference(codes: Any, scales: Any) -> Tuple[Any, Any]:
    """``(bits, products)`` on the CPU: ``scale * code`` in float32, then bf16 RNE.

    No bitsandbytes is involved. ``products`` is what the denormal band is decided from,
    one entry per decoded output. The kernel multiplies in float32 and converts with
    ``__float2bfloat16_rn``; CPU float32 multiply plus torch's round-to-nearest-even cast is
    the same operation.
    """
    import torch

    table = torch.tensor(NF4_CODE_TABLE, dtype=torch.float32)
    flat = codes.flatten().to("cpu").to(torch.int64)
    values = torch.stack((flat >> 4, flat & 15), dim=1).flatten()
    products = table[values] * scales.cpu().repeat_interleave(BLOCKSIZE)
    return products.to(torch.bfloat16).view(torch.uint16), products


def _difference(
    expected: Any, actual: Any, products: Any, *, denormal_only: bool
) -> Any:
    """Boolean mask of outputs whose bits differ, NaN compared as a class, not a payload."""
    import torch

    expected_float = expected.view(torch.bfloat16).cpu()
    actual_float = actual.view(torch.bfloat16).cpu()
    both_nan = torch.isnan(expected_float) & torch.isnan(actual_float)
    differing = (expected_float.view(torch.int16) != actual_float.view(torch.int16)) & ~both_nan
    if denormal_only:
        differing = differing & ~(products.abs() < BF16_MIN_NORMAL)
    return differing


def _first_difference(expected: Any, actual: Any, products: Any, *, denormal_only: bool) -> str:
    differing = _difference(expected, actual, products, denormal_only=denormal_only)
    if not differing.any():
        return ""
    index = int(differing.nonzero().flatten()[0])
    return (
        f"{int(differing.sum())} of {expected.numel()} bits differ; first at output index "
        f"{index} (code byte {index // 2}, block {index // BLOCKSIZE}, product "
        f"{float(products[index])!r}): reference {int(expected[index])} != "
        f"extension {int(actual[index])}"
    )


def test_random_packets_cover_every_nf4_code() -> None:
    import torch

    extension = _decoder_or_skip()
    torch.manual_seed(20261123)
    count = BLOCKSIZE * 16
    codes = _codes_covering_every_code(count, "cuda")
    scales = _random_scales(count, "cuda")
    (gate_bits, _) = _decode_pair(extension, codes, scales, None, None)
    expected = _reference(codes, scales)
    assert _first_mismatch(expected, gate_bits) == ""


def test_boundary_scales_match_bitsandbytes_except_bf16_denormals() -> None:
    """Compare native output to bitsandbytes outside the bf16 subnormal band."""
    import torch

    extension = _decoder_or_skip()
    scales = torch.tensor(
        [0.0, -0.0, 1e-45, 1e-38, 1.0, -1.0, float("inf"), float("nan")],
        dtype=torch.float32,
        device="cuda",
    )
    count = int(scales.numel()) * BLOCKSIZE
    codes = _codes_covering_every_code(count, "cuda")
    (gate_bits, _) = _decode_pair(extension, codes, scales, None, None)
    expected = _reference(codes, scales)
    exact, products = _exact_reference(codes, scales)

    assert _first_difference(expected, gate_bits, products, denormal_only=True) == ""

    differing = _difference(expected, gate_bits, products, denormal_only=False)
    denormal = products.abs() < BF16_MIN_NORMAL
    assert differing.any(), "this fixture must exercise the denormal exception"
    assert not (differing & ~denormal).any(), (
        "bits differ outside the bf16 denormal band: " + _first_difference(
            expected, gate_bits, products, denormal_only=True
        )
    )
    # The measured exception: the 1e-38 block (index 3). bnb flushes its subnormal products.
    assert differing[3 * BLOCKSIZE : 4 * BLOCKSIZE].any()
    # The denormal band is answered by the exact CPU reference, not by bnb.
    assert _first_difference(exact, gate_bits, products, denormal_only=False) == ""


def test_the_denormal_block_matches_the_exact_cpu_reference() -> None:
    """With scale 1e-38 every nonzero product is a bf16 subnormal: the CPU reference wins."""
    import torch

    extension = _decoder_or_skip()
    assert torch.equal(
        torch.tensor(NF4_CODE_TABLE, dtype=torch.float32), _lookup("cpu")
    ), "the documented NF4 table no longer matches bitsandbytes"
    scales = torch.tensor([1e-38], dtype=torch.float32, device="cuda")
    codes = _codes_covering_every_code(BLOCKSIZE, "cuda")
    (gate_bits, _) = _decode_pair(extension, codes, scales, None, None)
    exact, products = _exact_reference(codes, scales)
    nonzero = products != 0
    assert nonzero.any()
    assert (products[nonzero].abs() < BF16_MIN_NORMAL).all()
    assert _first_difference(exact, gate_bits, products, denormal_only=False) == ""
    # The subnormals survive (no flush to signed zero), which is the whole exception.
    assert gate_bits.cpu().view(torch.int16)[nonzero].ne(0).all()
    # And bitsandbytes flushes the very same block to signed zeros.
    expected = _reference(codes, scales)
    assert (expected.view(torch.bfloat16) == 0).all()
    assert expected.view(torch.int16).ne(gate_bits.view(torch.int16)).any()


def _quantised_matrix(shape: Tuple[int, int], device: Any) -> Tuple[Any, Any]:
    import torch

    _, functional = _bitsandbytes()
    source = torch.randn(*shape, dtype=torch.float32, device=device)
    codes, state = functional.quantize_4bit(
        source, blocksize=BLOCKSIZE, compress_statistics=False, quant_type="nf4"
    )
    assert not state.nested, "plain absmax is the native decoder format"
    return codes.contiguous(), state.absmax.contiguous()


def test_real_gate_up_and_down_blocks_match_bitsandbytes() -> None:
    import torch

    extension = _decoder_or_skip()
    torch.manual_seed(77)
    hidden = 256
    gate_codes, gate_scales = _quantised_matrix((2 * hidden, hidden), "cuda")
    down_codes, down_scales = _quantised_matrix((hidden, 2 * hidden), "cuda")
    gate_bits, down_bits = _decode_pair(
        extension, gate_codes, gate_scales, down_codes, down_scales
    )
    mismatch = _first_mismatch(_reference(gate_codes, gate_scales), gate_bits)
    assert mismatch == "", f"gate_up: {mismatch}"
    mismatch = _first_mismatch(_reference(down_codes, down_scales), down_bits)
    assert mismatch == "", f"down: {mismatch}"
