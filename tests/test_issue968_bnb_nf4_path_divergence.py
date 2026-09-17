"""#968 — pin the bitsandbytes fused-vs-dequant NF4 divergence that #857 rests on.

#857 (merged ``d456a1d3``) moved the NF4 bit-exactness reference in
``tests/test_issue385_stream_dtype.py`` onto the streamed model's *computation
path*: the resident reference now gets ``install_dequant_forward`` applied too.
The justification is that production streams NF4 through ``dequantize_4bit`` +
``F.linear`` (``utils/layer_stream_runtime.py``, the #331 repair), while an
unpatched resident ``Linear4bit`` goes through bitsandbytes' own fused kernel —
so the old comparison was measuring two KERNELS against each other, not
streaming against itself.

That correction is right. The problem #968 records is that after the merge, the
property it rests on is asserted NOWHERE: it survives only in two docstrings.
A bitsandbytes release that fixed the fused path would make the correction
unnecessary and nothing would say so; a release that worsened it would change
the magnitude and nothing would notice.

So this file asserts the property directly, with **no streaming anywhere in the
experiment**. Two resident NF4 models are built from the same checkpoint, their
packed weights are asserted byte-identical, and the only difference between them
is whether ``install_dequant_forward`` was applied. On sm_120 with bitsandbytes
0.50.2 they disagree; with the patch applied to BOTH (or to NEITHER) they agree
exactly, which is what makes the divergence attributable to the path and not to
the harness.

WHAT THE "FUSED" PATH ACTUALLY IS, in bitsandbytes 0.50.2. ``#331``'s docstring
names ``MatMul4Bit``, which is where the *aliasing* bug lives, but numerically
both routes land on the same kernel: ``matmul_4bit`` calls
``torch.ops.bitsandbytes.gemm_4bit`` directly when no gradient is needed and
``MatMul4Bit.apply`` otherwise, and ``MatMul4Bit.forward`` calls that same op.
Measured: the divergence below is bit-identical under ``torch.no_grad()`` and
under ``torch.enable_grad()``, which is the evidence for that reading.

THE DIVERGENCE IS SHAPE-GATED, and this is the part that is easy to get wrong.
``gemm_4bit`` only runs its own fused dequantise-and-multiply for a small
leading dimension; above it, bitsandbytes dequantises and calls a normal GEMM,
which agrees with our path bit-for-bit. Measured on this card at
``M in {1..32}`` -> divergent, ``M >= 33`` -> exactly 0.0. That is why
``test_issue385``'s ``(1, 12)`` input sees the divergence at all, and it is the
trap the first attempt at this probe fell into (see the class docstring in
``TestTheDivergenceIsGatedOnTheLeadingDimension``).

This can only ever be a dev-box signal: GitHub runners have no GPU, so the whole
module skips in CI, exactly like its neighbours in ``test_issue385_stream_dtype``.
It is additionally pinned to the bitsandbytes version it was measured against, so
a version bump reports "unverified" rather than silently passing.

**#968 does not resolve #776.** Which of the two bitsandbytes paths is
numerically CORRECT on sm_120 is still unestablished. This file pins that they
differ, and by how much.

Environment the constants below were measured on: RTX 5070 Laptop GPU (sm_120),
driver 616.92 / CUDA 13.4, torch 2.14.0+cu130, bitsandbytes 0.50.2,
transformers 5.17.0, peft 0.20.0, Python 3.12.10, Windows 11.
"""

import pytest

# --------------------------------------------------------------------------
# The measurement, as data. Prose in a docstring is what #968 exists to replace.
# --------------------------------------------------------------------------
PINNED_BITSANDBYTES = "0.50.2"

#: maxabs logit difference between two resident NF4 models that differ ONLY in
#: whether ``install_dequant_forward`` was applied. Byte-identical packed
#: weights, identical inputs, no streaming.
MEASURED_LOGIT_DIVERGENCE = {
    "float16": 0.0006103515625,
    "bfloat16": 0.00439453125,
}

#: The assertion is about EXISTENCE and ROUGH MAGNITUDE, never equality. An
#: exact-float assertion would break on any driver-level change in GEMM kernel
#: selection and teach the next person to delete the test. One order of
#: magnitude either side is wide enough to absorb that (repeated runs on this
#: box are bit-stable, and re-seeding the model moves it by ~1.3x) and tight
#: enough that a bitsandbytes release which fixed the fused path (-> 0.0) or
#: worsened it (-> 10x) turns this red.
MAGNITUDE_BAND = 10.0

#: ``gemm_4bit`` runs its fused path only up to this leading dimension; above
#: it bitsandbytes dequantises and calls an ordinary GEMM, which agrees with
#: ``install_dequant_forward`` exactly. Measured by bisection on this card.
FUSED_KERNEL_MAX_ROWS = 32

_SEED = 7
_HIDDEN = 64
_SEQ_LEN = 12  # <= FUSED_KERNEL_MAX_ROWS, and the shape test_issue385 uses


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _bnb_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("bitsandbytes")
    except Exception:
        return ""


CUDA = pytest.mark.skipif(
    not _cuda_available(),
    reason="requires CUDA (the bitsandbytes 4-bit kernels are GPU-only)",
)

PINNED = pytest.mark.skipif(
    _bnb_version() != PINNED_BITSANDBYTES,
    reason=(
        f"UNVERIFIED: the fused-vs-dequant NF4 divergence was measured against "
        f"bitsandbytes {PINNED_BITSANDBYTES}, this env has "
        f"{_bnb_version() or 'no bitsandbytes'}. Re-measure and re-pin (#968) "
        f"rather than widening the band."
    ),
)


# --------------------------------------------------------------------------
# harness — deliberately a copy of the neighbour's, not an import of it.
# test_issue385_stream_dtype.py is the file whose green this one explains;
# reaching into it would make the explanation depend on the thing explained.
# --------------------------------------------------------------------------
def _tiny_llama_dir(tmp_path):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(_SEED)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _resident_nf4(weights: str, dtype: str):
    """A plain resident NF4 model. No shards, no pool, no streaming."""
    import torch
    from transformers import AutoModelForCausalLM

    from soup_cli.utils.layer_stream_runtime import build_nf4_config

    model = AutoModelForCausalLM.from_pretrained(
        weights,
        quantization_config=build_nf4_config(dtype),
        dtype=getattr(torch, dtype),
        device_map={"": "cuda"},
    )
    model.config.use_cache = False
    return model.eval()


def _packed_weights(model):
    import bitsandbytes as bnb

    return {
        name: child.weight.data
        for name, child in model.named_modules()
        if isinstance(child, bnb.nn.Linear4bit)
    }


def _logits(model, ids):
    import torch

    with torch.no_grad():
        return model(input_ids=ids).logits.float()


def _divergence(tmp_path, dtype: str, *, patch_reference: bool) -> float:
    """maxabs logit gap between two resident NF4 models built from one file.

    ``patch_reference=False`` is the real comparison: fused kernel vs
    ``install_dequant_forward``. ``patch_reference=True`` patches BOTH, which
    must be exactly 0.0 — that is the control, and it is also the lever the
    can-it-fail demonstration pulls.
    """
    import torch

    from soup_cli.utils.layer_stream_runtime import install_dequant_forward

    weights = _tiny_llama_dir(tmp_path / f"{dtype}-{patch_reference}")
    reference = _resident_nf4(weights, dtype)
    patched = _resident_nf4(weights, dtype)

    left, right = _packed_weights(reference), _packed_weights(patched)
    assert left and set(left) == set(right), "the two loads disagree on structure"
    assert all(torch.equal(left[k], right[k]) for k in left), (
        "the packed NF4 bytes differ between the two loads, so any divergence "
        "below would be quantiser nondeterminism rather than a path difference"
    )

    if patch_reference:
        assert install_dequant_forward(reference) > 0
    assert install_dequant_forward(patched) > 0

    generator = torch.Generator(device="cuda").manual_seed(3)
    ids = torch.randint(0, 64, (1, _SEQ_LEN), device="cuda", generator=generator)
    return (_logits(reference, ids) - _logits(patched, ids)).abs().max().item()


def _require_bf16():
    import torch

    if not torch.cuda.is_bf16_supported():
        pytest.skip("this card has no bf16")


@CUDA
@PINNED
class TestTheTwoBitsandbytesFourBitPathsDisagree:
    """The property #857's reference correction rests on, asserted.

    No streaming is involved. Two resident NF4 models, byte-identical packed
    weights, identical input, differing only in ``install_dequant_forward``.
    """

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_the_paths_diverge_by_the_measured_magnitude(self, tmp_path, dtype):
        if dtype == "bfloat16":
            _require_bf16()
        expected = MEASURED_LOGIT_DIVERGENCE[dtype]

        got = _divergence(tmp_path, dtype, patch_reference=False)

        assert got > 0.0, (
            f"the two bitsandbytes 4-bit paths now AGREE in {dtype}. If that is "
            f"real, #857's reference correction is no longer load-bearing and "
            f"#776 may be resolvable — re-measure before deleting anything."
        )
        assert expected / MAGNITUDE_BAND <= got <= expected * MAGNITUDE_BAND, (
            f"{dtype}: divergence {got!r} is outside one order of magnitude of "
            f"the recorded {expected!r} on bitsandbytes {PINNED_BITSANDBYTES}"
        )

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_the_same_path_twice_is_exact(self, tmp_path, dtype):
        """The control. Without it, the test above could be measuring load
        nondeterminism, a non-deterministic kernel, or a flaky harness, and
        would read the same. Patch both -> the only difference is gone -> 0.0."""
        if dtype == "bfloat16":
            _require_bf16()
        assert _divergence(tmp_path, dtype, patch_reference=True) == 0.0

    def test_bfloat16_diverges_further_than_float16(self):
        """Structural, and it survives a change of shape or seed: bf16 carries
        8 mantissa bits to fp16's 11, so the same relative kernel difference
        lands about an order of magnitude larger."""
        assert (
            MEASURED_LOGIT_DIVERGENCE["bfloat16"] > MEASURED_LOGIT_DIVERGENCE["float16"]
        )


@CUDA
@PINNED
class TestTheDivergenceIsGatedOnTheLeadingDimension:
    """The same property at the kernel, and the shape trap that hides it.

    Credit for the shape of this probe goes to @Samearth17, who wrote it in
    ``237057c2`` as ``tests/test_issue776_bnb_nf4_paths.py`` and removed it in
    ``b3678971`` as "invalid". #968 asks what was invalid about it; running the
    deleted file answers, and both answers are worth keeping:

    1. It called ``bnb.functional.matmul_4bit``, which does not exist in
       bitsandbytes 0.50.2 — the symbol is ``bnb.matmul_4bit``. Against the very
       version it pinned itself to, it raised ``AttributeError`` and could never
       have passed.
    2. Repairing that is not enough. It probed ``rows in (64, 2048)`` and
       asserted a NON-ZERO gap at 64 — but the fused kernel only runs up to
       ``M = 32``, so both of its shapes sit on the side where the two paths
       agree exactly and its positive assertion was unsatisfiable.

    The second one is the useful finding, and it is why the model-level probe
    above uses a 12-token input.
    """

    def _kernel_gap(self, rows: int, dtype_name: str) -> float:
        import bitsandbytes as bnb
        import torch
        import torch.nn.functional as functional

        dtype = getattr(torch, dtype_name)
        torch.manual_seed(0)
        weight = torch.randn(128, _HIDDEN, device="cuda", dtype=dtype)
        packed, quant_state = bnb.functional.quantize_nf4(weight, blocksize=64)
        dense = bnb.functional.dequantize_4bit(packed, quant_state)

        inputs = torch.randn(rows, _HIDDEN, device="cuda", dtype=dtype)
        fused = bnb.matmul_4bit(inputs, packed, quant_state=quant_state, bias=None)
        dequant = functional.linear(inputs, dense.to(inputs.dtype))
        return (fused.float() - dequant.float()).abs().max().item()

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_they_disagree_inside_the_fused_range(self, dtype):
        if dtype == "bfloat16":
            _require_bf16()
        assert self._kernel_gap(FUSED_KERNEL_MAX_ROWS, dtype) > 0.0

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_they_agree_once_bitsandbytes_falls_back(self, dtype):
        """Above the threshold bitsandbytes dequantises and calls an ordinary
        GEMM, which is what ``install_dequant_forward`` does — so the two agree
        bit-for-bit. Pinning both sides is what makes the boundary a fact rather
        than a guess, and is what the deleted probe needed."""
        if dtype == "bfloat16":
            _require_bf16()
        assert self._kernel_gap(FUSED_KERNEL_MAX_ROWS + 1, dtype) == 0.0
