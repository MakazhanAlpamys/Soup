"""#776 — document the two bitsandbytes NF4 forward paths.

The streaming repair in #857 deliberately uses ``dequantize_4bit`` +
``torch.nn.functional.linear``. A resident NF4 ``Linear4bit`` normally uses
bitsandbytes' ``matmul_4bit`` path instead. With bitsandbytes 0.50.2 those
kernels are observably different at small M, so the bit-exact streaming test
must compare matching computation paths rather than treating this backend
property as a streaming regression.
"""

import importlib.metadata

import pytest


torch = pytest.importorskip("torch")

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA to exercise bitsandbytes NF4 kernels",
)


@CUDA
@pytest.mark.parametrize("rows", [64, 2048])
def test_nf4_fused_and_dequant_paths_are_close(rows):
    """The divergence is a bitsandbytes 0.50.2 property, not a Soup invariant."""
    import bitsandbytes as bnb
    import torch.nn.functional as functional

    bnb_version = importlib.metadata.version("bitsandbytes")
    if bnb_version != "0.50.2":
        pytest.skip(
            "measured kernel-path behaviour is pinned to bitsandbytes 0.50.2"
        )

    torch.manual_seed(0)

    inputs = torch.randn(
        rows, 4096, device="cuda", dtype=torch.float16
    )
    weights = torch.randn(
        4096, 4096, device="cuda", dtype=torch.float32
    )

    packed, quant_state = bnb.functional.quantize_nf4(
        weights,
        blocksize=64,
    )

    fused = bnb.functional.matmul_4bit(
        inputs,
        packed,
        bias=None,
        quant_state=quant_state,
    )

    dequant = functional.linear(
        inputs,
        bnb.functional.dequantize_4bit(
            packed,
            quant_state,
        ),
    )

    max_error = (fused.float() - dequant.float()).abs().max().item()

    if rows == 64:
        assert max_error > 0.0
        assert max_error <= 1e-2
    else:
        assert max_error <= 1e-6
