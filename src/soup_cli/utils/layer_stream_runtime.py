"""training.stream_layers — streaming runtime (v0.72.0 BETA).

The torch half: pre-allocated VRAM buffer pool, the CPU-RAM weight source, the
prefetch scheduler, the layer wrapper, and the meta-device model build.

Data flow per step (plan 5.2)::

    FORWARD   layer i: wait(i) -> prefetch(i+1) -> checkpoint(body_i)
    BACKWARD  layer i: wait(i) -> prefetch(i-1) -> recompute + backward

Each layer is read TWICE per step and that cannot be optimised away:
``dL/dx = W^T . dL/dy``, so the backward pass needs W to reach lower layers and
their adapters. This is physics, not an implementation detail.

**No top-level torch / peft / transformers** — all lazy, so the light CLI keeps
importing without the training stack.
"""

import contextlib
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# Stdlib-only at import (it defers torch to its own constructor), so the
# default is safe to read at module scope. ``AsyncDiskSource`` itself is
# imported inside ``_build_source``, where torch is already unavoidable.
from soup_cli.utils.async_disk_source import DEFAULT_STREAM_READ_AHEAD

logger = logging.getLogger(__name__)

#: Storage dtypes a shard may hold. ``uint8`` is not a *base* dtype — it is what
#: NF4 packs its nibbles and (under double quant) its absmax into.
_DTYPE_NAMES = ("bfloat16", "float16", "float32", "uint8")

#: safetensors header dtype strings -> our names.
_SAFETENSORS_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "U8": "uint8",
}


# ==========================================================================
# model-graph navigation
# ==========================================================================
def decoder_owner(model: Any) -> Any:
    """Return the module that owns ``.layers`` (LlamaModel / Qwen2Model).

    This is deliberately NOT the CausalLM wrapper. PEFT's ``LoraModel.forward``
    calls ``self.model.forward(...)`` directly, bypassing ``__call__`` and
    therefore every forward hook registered on the wrapper. transformers always
    reaches the layer container through ``__call__``, so hooks land here.
    """
    node = model
    for _ in range(8):
        if hasattr(node, "layers"):
            return node
        for attr in ("base_model", "model", "transformer"):
            child = getattr(node, attr, None)
            if child is not None and child is not node:
                node = child
                break
        else:
            break
    raise ValueError(
        "could not locate the decoder-layer container (a module with .layers) — "
        "layer streaming supports Llama/Qwen-shaped models only"
    )


def _set_module_param(root: Any, full_name: str, tensor: Any) -> None:
    import torch.nn as nn

    parts = full_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    module._parameters[parts[-1]] = nn.Parameter(tensor, requires_grad=False)


def _torch_dtype(name: str):
    import torch

    if name not in _DTYPE_NAMES:
        raise ValueError(f"unsupported dtype {name!r}; supported: {_DTYPE_NAMES}")
    return getattr(torch, name)


# ==========================================================================
# NF4 — Params4bit views rebuilt over the pooled buffers (plan P3, v0.72.2)
# ==========================================================================
def rebuild_quant_state(key: str, buffers: Mapping[str, Any], spec: Any, codes: Mapping[str, Any]):
    """Reassemble one weight's ``QuantState`` from streamed tensors.

    ``Params4bit`` carries its ``quant_state`` alongside the packed bytes and
    quantises on transfer to CUDA, so NF4 weights cannot simply be ``copy_``d
    into a plain buffer (plan P3). The packed nibbles and the per-block absmax
    ARE streamed; the two code tables are constant across every weight and are
    held resident once.
    """
    import torch
    from bitsandbytes.functional import QuantState

    from soup_cli.utils.layer_shard import (
        ABSMAX_SUFFIX,
        NESTED_ABSMAX_SUFFIX,
        NESTED_OFFSET_SUFFIX,
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
    )

    code = codes.get(NF4_CODE_KEY)
    if code is None:
        raise ValueError(
            "the NF4 code table is missing from the shard extras — reshard the "
            "checkpoint (a stale v0.72.0 cache has no quantisation data)"
        )
    state2 = None
    offset = None
    if spec.nested:
        nested_code = codes.get(NF4_NESTED_CODE_KEY)
        if nested_code is None:
            raise ValueError(f"nested NF4 code table missing for {key}")
        state2 = QuantState(
            absmax=buffers[key + NESTED_ABSMAX_SUFFIX],
            code=nested_code,
            blocksize=spec.nested_blocksize,
            dtype=torch.float32,
        )
        offset = buffers[key + NESTED_OFFSET_SUFFIX]
    return QuantState(
        absmax=buffers[key + ABSMAX_SUFFIX],
        shape=torch.Size(spec.shape),
        dtype=_torch_dtype(spec.dtype),
        blocksize=spec.blocksize,
        code=code,
        quant_type=spec.quant_type,
        offset=offset,
        state2=state2,
    )


def rebuild_params4bit(key: str, buffers: Mapping[str, Any], spec: Any, codes: Mapping[str, Any]):
    """A ``Params4bit`` VIEW over the pooled buffer — no copy, no re-quantise.

    ``bnb_quantized=True`` is what stops ``Params4bit`` from trying to quantise
    the already-packed bytes again on its next ``.cuda()``.
    """
    import bitsandbytes as bnb

    return bnb.nn.Params4bit(
        data=buffers[key],
        requires_grad=False,
        quant_state=rebuild_quant_state(key, buffers, spec, codes),
        blocksize=spec.blocksize,
        compress_statistics=spec.nested,
        quant_type=spec.quant_type,
        bnb_quantized=True,
    )


class NF4ScaleCache:
    """Expanded fp32 block scales of streamed nested-NF4 weights (opt-in).

    bitsandbytes' dequantise + linear path expands a double-quantised weight's
    scales on every call: ``dequantize_blockwise`` of the 8-bit codes, then
    ``+= offset``. The base is frozen, so the result for a (layer, weight) never
    changes; this keeps it on the compute device after the first visit and hands
    later visits a non-nested ``QuantState`` over it. The expansion below is the
    one in ``bitsandbytes.functional.dequantize_4bit``, so the 4-bit dequantise
    that follows reads identical scales.

    Keyed by layer index and shard key, never by pool slot: several layers take
    turns in one slot. Only the dense path may use it; the fused custom GEMM
    expands the nested scales inside its kernel.
    """

    def __init__(self) -> None:
        self._states: Dict[Tuple[int, str], Any] = {}

    def __len__(self) -> int:
        return len(self._states)

    @property
    def nbytes(self) -> int:
        return sum(
            state.absmax.numel() * state.absmax.element_size() for state in self._states.values()
        )

    def dense_state(self, key: Tuple[int, str], quant_state: Any) -> Any:
        state = self._states.get(key)
        if state is None:
            state = self._expand(quant_state)
            self._states[key] = state
        elif state.shape != quant_state.shape or state.blocksize != quant_state.blocksize:
            raise RuntimeError(
                f"NF4 scale cache entry {key} was built for shape {tuple(state.shape)} "
                f"/ block {state.blocksize}, but the weight now has shape "
                f"{tuple(quant_state.shape)} / block {quant_state.blocksize}"
            )
        return state

    @staticmethod
    def _expand(quant_state: Any) -> Any:
        import torch
        from bitsandbytes import functional as bnb_functional

        absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax += quant_state.offset
        if absmax.dtype != torch.float32:
            absmax = absmax.float()
        return bnb_functional.QuantState(
            absmax=absmax,
            shape=quant_state.shape,
            code=quant_state.code,
            blocksize=quant_state.blocksize,
            quant_type=quant_state.quant_type,
            dtype=quant_state.dtype,
        )

    def clear(self) -> None:
        self._states.clear()


_CHECKPOINT_VISIBLE_NF4_FUNCTION = None


def _checkpoint_visible_nf4_function():
    """Lazy custom autograd Function whose saved state is visible to checkpoint (#842)."""
    global _CHECKPOINT_VISIBLE_NF4_FUNCTION
    if _CHECKPOINT_VISIBLE_NF4_FUNCTION is not None:
        return _CHECKPOINT_VISIBLE_NF4_FUNCTION

    import bitsandbytes.functional as bnb_functional
    import torch

    class CheckpointVisibleNF4Matmul(torch.autograd.Function):
        @staticmethod
        @torch.amp.custom_fwd(device_type="cuda")
        def forward(
            ctx,
            x,
            packed,
            absmax,
            state2_absmax,
            state2_code,
            offset,
            bias,
            shape,
            weight_dtype,
            blocksize,
            quant_type,
            nested,
            state2_blocksize,
        ):
            # #331/#842: every tensor the backward needs goes through
            # save_for_backward. Non-reentrant checkpoint can therefore discard
            # these references and recreate them after the streaming slot has
            # been refilled with the correct layer during recompute.
            ctx.save_for_backward(packed, absmax, state2_absmax, state2_code, offset)
            ctx.shape = tuple(int(dim) for dim in shape)
            ctx.weight_dtype = weight_dtype
            ctx.blocksize = int(blocksize)
            ctx.quant_type = str(quant_type)
            ctx.nested = bool(nested)
            ctx.state2_blocksize = int(state2_blocksize)
            ctx.bias_dtype = None if bias is None else bias.dtype
            ctx.owner_check = getattr(packed, "_soup_stream_owner_check", None)

            if ctx.nested:
                return torch.ops.bitsandbytes.gemm_4bit.default(
                    x,
                    packed,
                    ctx.shape,
                    state2_absmax,
                    ctx.blocksize,
                    ctx.quant_type,
                    bias=bias,
                    absmax_8bit=absmax,
                    absmax_code=state2_code,
                    absmax_offset=offset,
                )
            return torch.ops.bitsandbytes.gemm_4bit.default(
                x, packed, ctx.shape, absmax, ctx.blocksize, ctx.quant_type, bias=bias
            )

        @staticmethod
        @torch.amp.custom_bwd(device_type="cuda")
        def backward(ctx, grad_output):
            # Recompute must have reloaded this layer into its slot, and the
            # one-ahead prefetcher must not recycle that slot before its
            # backward finishes. A deeper lookahead would otherwise make the
            # aliases below silently describe another layer (#842).
            if ctx.owner_check is not None:
                ctx.owner_check()
            packed, absmax, state2_absmax, state2_code, offset = ctx.saved_tensors
            state2 = None
            q_offset = None
            if ctx.nested:
                state2 = bnb_functional.QuantState(
                    absmax=state2_absmax,
                    code=state2_code,
                    blocksize=ctx.state2_blocksize,
                    dtype=torch.float32,
                )
                q_offset = offset
            quant_state = bnb_functional.QuantState(
                absmax=absmax,
                shape=torch.Size(ctx.shape),
                dtype=ctx.weight_dtype,
                blocksize=ctx.blocksize,
                quant_type=ctx.quant_type,
                offset=q_offset,
                state2=state2,
            )
            # bitsandbytes 0.50.x exposes a fused forward GEMM but no transposed
            # 4-bit GEMM for dX. Keep this fallback explicit: it is correct and
            # checkpoint-visible, but #842 remains open until this dense
            # dequantisation is replaced by a real transposed kernel.
            grad_x = None
            if ctx.needs_input_grad[0]:
                weight = bnb_functional.dequantize_4bit(packed, quant_state).to(
                    grad_output.dtype
                )
                grad_x = torch.matmul(grad_output, weight)
            grad_bias = None
            if ctx.needs_input_grad[6] and ctx.bias_dtype is not None:
                grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(
                    dim=0, dtype=ctx.bias_dtype
                )
            return (
                grad_x,
                None,
                None,
                None,
                None,
                None,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    _CHECKPOINT_VISIBLE_NF4_FUNCTION = CheckpointVisibleNF4Matmul
    return CheckpointVisibleNF4Matmul


_PACKED_DEQUANT_NF4_FUNCTION = None


def _packed_dequant_nf4_function():
    """Lazy autograd Function: dense-dequant forward, PACKED-only saved state.

    For a streamed layer that runs WITHOUT checkpoint recompute. ``F.linear`` on
    a dequantised weight would save the dense bf16 weight for the backward (437 MB
    per Mistral-7B layer); this saves only the packed bytes and the quantisation
    tensors through ``save_for_backward`` and dequantises again for ``grad_x``.
    The forward arithmetic is the compatibility path's (dequantise + ``F.linear``)
    and the backward's ``grad_y @ W`` is the same GEMM ``F.linear``'s backward
    issues, so outputs and gradients stay bit-identical to the recompute path.
    """
    global _PACKED_DEQUANT_NF4_FUNCTION
    if _PACKED_DEQUANT_NF4_FUNCTION is not None:
        return _PACKED_DEQUANT_NF4_FUNCTION

    import bitsandbytes.functional as bnb_functional
    import torch
    import torch.nn.functional as functional

    def _state(ctx, absmax, state2_absmax, state2_code, offset):
        state2 = None
        q_offset = None
        if ctx.nested:
            state2 = bnb_functional.QuantState(
                absmax=state2_absmax,
                code=state2_code,
                blocksize=ctx.state2_blocksize,
                dtype=torch.float32,
            )
            q_offset = offset
        return bnb_functional.QuantState(
            absmax=absmax,
            shape=torch.Size(ctx.shape),
            code=ctx.code,
            dtype=ctx.weight_dtype,
            blocksize=ctx.blocksize,
            quant_type=ctx.quant_type,
            offset=q_offset,
            state2=state2,
        )

    class PackedDequantNF4Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, packed, absmax, state2_absmax, state2_code, offset, bias, meta):
            shape, weight_dtype, blocksize, quant_type, nested, state2_blocksize, code = meta
            ctx.shape = tuple(int(dim) for dim in shape)
            ctx.weight_dtype = weight_dtype
            ctx.blocksize = int(blocksize)
            ctx.quant_type = str(quant_type)
            ctx.nested = bool(nested)
            ctx.state2_blocksize = int(state2_blocksize)
            ctx.code = code
            ctx.bias_dtype = None if bias is None else bias.dtype
            ctx.owner_check = getattr(packed, "_soup_stream_owner_check", None)
            weight = bnb_functional.dequantize_4bit(
                packed, _state(ctx, absmax, state2_absmax, state2_code, offset)
            ).to(x.dtype)
            ctx.save_for_backward(packed, absmax, state2_absmax, state2_code, offset)
            return functional.linear(x, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            packed, absmax, state2_absmax, state2_code, offset = ctx.saved_tensors
            if ctx.owner_check is not None:
                ctx.owner_check()
            grad_x = None
            if ctx.needs_input_grad[0]:
                weight = bnb_functional.dequantize_4bit(
                    packed, _state(ctx, absmax, state2_absmax, state2_code, offset)
                ).to(grad_output.dtype)
                grad_x = torch.matmul(grad_output, weight)
            grad_bias = None
            if ctx.needs_input_grad[6] and ctx.bias_dtype is not None:
                grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(
                    dim=0, dtype=ctx.bias_dtype
                )
            return grad_x, None, None, None, None, None, grad_bias, None

    _PACKED_DEQUANT_NF4_FUNCTION = PackedDequantNF4Linear
    return PackedDequantNF4Linear


def packed_dequant_nf4_linear(x: Any, packed: Any, quant_state: Any, bias: Any = None) -> Any:
    """Dense-dequant NF4 linear that saves only packed state for its backward
    (training.stream_recompute 'off' / 'auto')."""
    empty = packed.new_empty((0,))
    state2_absmax = empty
    state2_code = empty
    offset = empty
    state2_blocksize = 0
    if quant_state.nested:
        state2_absmax = quant_state.state2.absmax
        state2_code = quant_state.state2.code
        offset = quant_state.offset
        state2_blocksize = int(quant_state.state2.blocksize)
    meta = (
        tuple(quant_state.shape),
        quant_state.dtype,
        int(quant_state.blocksize),
        str(quant_state.quant_type),
        bool(quant_state.nested),
        state2_blocksize,
        quant_state.code,
    )
    return _packed_dequant_nf4_function().apply(
        x, packed, quant_state.absmax, state2_absmax, state2_code, offset, bias, meta
    )


def checkpoint_visible_nf4_linear(
    x: Any, packed: Any, quant_state: Any, bias: Any = None
) -> Any:
    """Fused NF4 forward with checkpoint-visible packed state (#842)."""
    empty = packed.new_empty((0,))
    state2_absmax = empty
    state2_code = empty
    offset = empty
    state2_blocksize = 0
    if quant_state.nested:
        state2_absmax = quant_state.state2.absmax
        state2_code = quant_state.state2.code
        offset = quant_state.offset
        state2_blocksize = int(quant_state.state2.blocksize)
    return _checkpoint_visible_nf4_function().apply(
        x,
        packed,
        quant_state.absmax,
        state2_absmax,
        state2_code,
        offset,
        bias,
        tuple(quant_state.shape),
        quant_state.dtype,
        int(quant_state.blocksize),
        str(quant_state.quant_type),
        bool(quant_state.nested),
        state2_blocksize,
    )


def _can_use_checkpoint_visible_nf4_gemm(x: Any, quant_state: Any) -> bool:
    """Match bitsandbytes' own decision to use its custom 4-bit CUDA GEMM.

    The private dispatch names are intentionally read from bitsandbytes rather
    than copying a card-specific M window. If those private names move, prefer
    the checkpoint-visible Function: it remains exact, though potentially
    slower, and the compatibility test makes the upstream change visible.
    """
    if getattr(getattr(x, "device", None), "type", None) != "cuda":
        return False
    if bool(getattr(quant_state, "nested", False)):
        state2 = getattr(quant_state, "state2", None)
        if state2 is None or int(getattr(state2, "blocksize", 0)) != 256:
            return False
    try:
        import torch

        torch.ops.bitsandbytes.gemm_4bit.default
    except (ImportError, AttributeError, RuntimeError):
        return False
    try:
        from bitsandbytes.backends.cuda import ops as bnb_cuda_ops

        custom_max_m = bnb_cuda_ops._gemm_4bit_custom_max_m
        use_custom_fn = bnb_cuda_ops._gemm_4bit_use_custom_fn
    except (ImportError, AttributeError, RuntimeError):
        return True

    k = int(x.shape[-1])
    m = int(x.numel() // k)
    n = int(quant_state.shape[0])
    blocksize = int(quant_state.blocksize)
    if m > int(custom_max_m) or k % blocksize != 0:
        return False
    return bool(use_custom_fn(x.device.index, x.dtype, m, n, k))


def install_dequant_forward(module: Any) -> int:
    """#331 — keep a STREAMED NF4 weight out of ``bitsandbytes``' ``MatMul4Bit``.

    ``MatMul4Bit.forward`` stashes the packed weight and the ``quant_state`` on
    ``ctx`` as plain attributes rather than through ``save_for_backward``::

        ctx.state = quant_state
        ctx.tensors = (None, B)

    ``torch.utils.checkpoint`` discards and recomputes *saved tensors*. These are
    not saved tensors, so it cannot see them: the reference taken in the forward
    survives, it ALIASES the buffer pool, and the backward reads it after that slot
    has been refilled with a different layer. Measured on 8xH100 against a resident
    NF4 reference, that is a bit-exact forward, a healthy-looking loss curve, and
    gradients wrong on every layer but the last ``stream_buffers``.

    De-aliasing was measured and rejected: bnb holds the reference across the whole
    forward-to-backward span, so any copy keeps one layer alive for that span and
    costs O(model). On real 32B, peak VRAM 4 220 -> 19 720 MiB.

    The streamed weight still never enters bitsandbytes' ``MatMul4Bit`` Function.
    When bitsandbytes' own CUDA dispatch selects its custom 4-bit GEMM, #842 routes
    that same operation through Soup's autograd Function: packed bytes and every
    quantisation tensor go through ``save_for_backward``, so non-reentrant
    checkpointing can discard and recompute them after the pool refills the correct
    layer. Whenever bitsandbytes selects its dequantise + linear fallback, Soup
    keeps the v0.73.0 path rather than paying a third dequantisation in backward.
    Missing private dispatch symbols fail toward the Function: exact but possibly
    slower, never toward an alias-unsafe ``MatMul4Bit``.

    The #842 Function is deliberately only the first half of the intended kernel.
    bitsandbytes exposes a fused forward GEMM but no transposed 4-bit GEMM for
    ``grad_x = grad_y @ W``; its backward also dequantises. Soup therefore keeps
    that dense-dequant backward explicitly until a transposed kernel exists. Do
    not claim the full #842 throughput ceiling from the fused-forward path alone.

    Returns the number of modules patched, so a caller can assert it patched
    something. Zero would mean the model carries no 4-bit linears at all.
    """
    import types

    import bitsandbytes as bnb
    import torch
    import torch.nn.functional as functional
    from bitsandbytes.functional import dequantize_4bit

    def _dequant_forward(self, x):
        if getattr(self, "quant_state", None) is not None and self.weight.quant_state is None:
            self.weight.quant_state = self.quant_state
        quant_state = self.weight.quant_state

        if not getattr(self, "compute_type_is_set", True):
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if getattr(self, "compute_dtype", None) is not None:
            x = x.to(self.compute_dtype)

        bias = self.bias
        if bias is not None:
            bias = bias.to(x.dtype)

        # Match bnb's own private dispatch. Only the custom-GEMM arm needs
        # Soup's Function; on bnb's dequant+linear arm the old path below has
        # identical arithmetic and avoids an extra backward dequantisation.
        if _can_use_checkpoint_visible_nf4_gemm(x, quant_state):
            return checkpoint_visible_nf4_linear(
                x, self.weight, quant_state, bias
            ).to(inp_dtype)

        # Compatibility path: dequantise inside the checkpointed region and let
        # F.linear save the dense weight properly. The fused arm above keeps the
        # nested state: its kernel expands the scales itself.
        scale_cache = getattr(self.weight, "_soup_scale_cache", None)
        if scale_cache is not None:
            quant_state = scale_cache.dense_state(self.weight._soup_scale_cache_key, quant_state)
        if getattr(self, "_soup_packed_backward", False) and torch.is_grad_enabled():
            # training.stream_recompute — no checkpoint recompute around this
            # layer: never let F.linear save the dense weight; save the packed
            # bytes instead.
            return packed_dequant_nf4_linear(x, self.weight, quant_state, bias).to(inp_dtype)
        weight = dequantize_4bit(self.weight, quant_state).to(x.dtype)
        return functional.linear(x, weight, bias).to(inp_dtype)

    patched = 0
    for child in module.modules():
        if isinstance(child, bnb.nn.Linear4bit):
            child.forward = types.MethodType(_dequant_forward, child)
            patched += 1
    return patched


def validate_quant_shape(key: str, spec: Any, shard_spec: Mapping[str, Any]) -> None:
    """The index's claimed shape must be backed by the bytes actually on disk.

    ``dequantize_4bit`` allocates ``prod(spec.shape)`` outputs and reads that
    many values out of the packed buffer and the absmax with no bounds check, so
    an index that OVERSTATES a tensor turns into an out-of-bounds read in native
    code. Checks are one-sided — under-claiming is fine, since bitsandbytes pads
    a non-block-aligned tensor up to the next block.
    """
    from soup_cli.utils.layer_shard import ABSMAX_SUFFIX, NESTED_ABSMAX_SUFFIX

    elements = math.prod(spec.shape)
    packed_shape, _dtype = shard_spec[key]
    packed = math.prod(packed_shape) if packed_shape else 0
    if elements > packed * 2:
        raise ValueError(
            f"shard is inconsistent with its index at {key}: the index claims "
            f"{elements} elements but only {packed} packed bytes are stored "
            f"(NF4 holds two values per byte). Reshard the checkpoint."
        )
    absmax_shape, _ = shard_spec[key + ABSMAX_SUFFIX]
    absmax = math.prod(absmax_shape) if absmax_shape else 0
    blocks = -(-elements // spec.blocksize)
    if blocks > absmax:
        raise ValueError(
            f"shard is inconsistent with its index at {key}: the index implies "
            f"{blocks} absmax blocks but only {absmax} are stored. Reshard the "
            f"checkpoint."
        )
    if spec.nested:
        nested_shape, _ = shard_spec[key + NESTED_ABSMAX_SUFFIX]
        nested = math.prod(nested_shape) if nested_shape else 0
        nested_blocks = -(-absmax // spec.nested_blocksize)
        if nested_blocks > nested:
            raise ValueError(
                f"shard is inconsistent with its index at {key}: the index "
                f"implies {nested_blocks} nested absmax blocks but only "
                f"{nested} are stored. Reshard the checkpoint."
            )


def quant_sidecar_keys(key: str, spec: Any) -> Tuple[str, ...]:
    """Every shard key that must be streamed for ``key`` to be rebuildable."""
    from soup_cli.utils.layer_shard import (
        ABSMAX_SUFFIX,
        NESTED_ABSMAX_SUFFIX,
        NESTED_OFFSET_SUFFIX,
    )

    keys = [key, key + ABSMAX_SUFFIX]
    if spec.nested:
        keys.append(key + NESTED_ABSMAX_SUFFIX)
        keys.append(key + NESTED_OFFSET_SUFFIX)
    return tuple(keys)


# ==========================================================================
# Tier 1 — the whole frozen base in CPU RAM (plan 5.5)
# ==========================================================================
#: torch's caching host allocator hands out page-locked memory in blocks rounded
#: UP to the next power of two, so pinning the store one tensor at a time costs
#: far more than its byte count — measured on the dev box (#901): 1.73x on the
#: Qwen2.5-14B NF4 store (6.82 GB requested, 11.83 GB of private commit), 1.90x
#: on a run of 35 MB tensors, and a single 9 GB request refused outright because
#: it rounds to 16 GiB. That rounding, against the box's RAM, is what "could not
#: page-lock the base" meant on a 32 GB machine. The pinned store is therefore
#: packed into arenas whose sizes ARE powers of two, every tensor a view.
PINNED_ARENA_BYTES = 2**28
#: The largest arena the packer will open on its own. A tensor bigger than this
#: still gets an arena that fits it. Measured on the dev box: an 8 GB pinned
#: request was granted, 9 GB refused (it rounds to 16 GiB), 2 GB chunks fine.
PINNED_ARENA_MAX_BYTES = 2**31
#: Every view starts on this boundary: enough for any dtype and for the DMA
#: engine, and a rounding error of at most 255 bytes per tensor.
PINNED_ARENA_ALIGN = 256


@dataclass(frozen=True)
class ArenaPlan:
    """Where each tensor of a pinned store lands, and what the arenas cost."""

    #: Bytes per arena; each is a power of two.
    arena_sizes: Tuple[int, ...]
    #: ``(arena, offset)`` per tensor, in the order the sizes were given.
    placements: Tuple[Tuple[int, int], ...]
    #: The tensors' own bytes, before any rounding.
    requested_bytes: int

    @property
    def pinned_bytes(self) -> int:
        return sum(self.arena_sizes)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def plan_pinned_arenas(
    sizes: Sequence[int],
    *,
    arena_bytes: int = PINNED_ARENA_BYTES,
    align: int = PINNED_ARENA_ALIGN,
) -> ArenaPlan:
    """Pack ``sizes`` (bytes, allocation order) into power-of-two arenas.

    Two layouts are planned. The in-order walk puts each tensor into the arena
    opened last, else opens the next. The largest-first walk (stable first-fit
    decreasing) is returned ONLY when it page-locks strictly fewer bytes; on a
    tie the in-order plan comes back, placement for placement. Either way
    ``placements`` is indexed in the order the sizes were given, and a tensor
    never straddles two arenas, because it has to be one contiguous view.

    The in-order walk alone strands the space in front of a large tensor that
    arrives late (#1702). A Qwen3-8B NF4 store is 3.58 GB of decoder tensors
    followed by two 1.24 GB vocabulary matrices; neither fits behind the
    decoder, so each opened a 2 GiB arena of its own: 8 GiB page-locked for a
    6.07 GB store. Largest-first puts the two matrices down first and the
    decoder fills in around them: 6 GiB. The test is bytes, not arena count —
    after the trim a sorted layout can hold as many arenas and MORE bytes.

    The arena capacity is the power of two above FOUR times the store's largest
    tensor, floored at ``arena_bytes`` and capped at
    :data:`PINNED_ARENA_MAX_BYTES`; a tensor beyond the cap still gets an arena
    that fits it. Two review findings and one failed alternative shaped that.
    A fixed capacity packs badly the moment tensors approach it — at 256 MiB a
    bf16 Qwen2.5-14B store, whose 141.6 MB projections cannot share an arena
    two at a time, cost 1.46x. Sizing each arena from the tensor that opened it
    was tried and rejected: in that same store an arena opened by a 52 MB
    projection stayed at 256 MiB and packed the 141.6 MB tensors that followed
    one per arena again. Four times the largest tensor bounds every arena's
    tail waste to a quarter and measures ~5% on that store. The cost of a
    global capacity is that one outlier raises every arena's REQUEST size —
    which the cap bounds at 2 GiB, a request this box granted where a 9 GB one
    (rounded to 16 GiB) was refused; the total stays near the store's bytes
    either way. Each arena is finally trimmed to the power of two above what
    it holds — the last one is usually part-filled, and a tiny model must not
    page-lock a whole default arena for a 5 MB store. Pure arithmetic, no torch.
    """
    if arena_bytes <= 0 or arena_bytes & (arena_bytes - 1):
        raise ValueError(f"arena_bytes must be a power of two; got {arena_bytes}")
    if align <= 0 or align & (align - 1):
        raise ValueError(f"align must be a power of two; got {align}")
    if align > arena_bytes:
        raise ValueError(f"align ({align}) cannot exceed arena_bytes ({arena_bytes})")
    sizes = [int(size) for size in sizes]
    if any(size < 0 for size in sizes):
        raise ValueError(f"a negative tensor size was planned: {min(sizes)}")
    if not sizes:
        return ArenaPlan(arena_sizes=(), placements=(), requested_bytes=0)
    ceiling = max(arena_bytes, PINNED_ARENA_MAX_BYTES)
    capacity = min(max(arena_bytes, _next_power_of_two(4 * max(sizes))), ceiling)
    in_order = _plan_in_order(sizes, capacity, align)
    largest_first = _plan_largest_first(sizes, capacity, align)
    return largest_first if largest_first.pinned_bytes < in_order.pinned_bytes else in_order


def _plan_in_order(sizes: Sequence[int], capacity: int, align: int) -> ArenaPlan:
    """Allocation order: each tensor into the arena opened last, else a new one."""
    fills: List[int] = []
    capacities: List[int] = []
    placements: List[Tuple[int, int]] = []
    for size in sizes:
        if fills:
            start = -(-fills[-1] // align) * align
            if start + size <= capacities[-1]:
                placements.append((len(fills) - 1, start))
                fills[-1] = start + size
                continue
        fills.append(size)
        # Only a tensor beyond the ceiling opens an arena wider than the rest.
        capacities.append(max(capacity, _next_power_of_two(size)))
        placements.append((len(fills) - 1, 0))
    return ArenaPlan(
        arena_sizes=tuple(max(align, _next_power_of_two(fill)) for fill in fills),
        placements=tuple(placements),
        requested_bytes=sum(sizes),
    )


def _plan_largest_first(sizes: Sequence[int], capacity: int, align: int) -> ArenaPlan:
    """Largest tensor first, each into the FIRST arena with room, else a new one.

    The sort is stable, so equal sizes keep their allocation order. Each
    placement is written back at the tensor's ORIGINAL index: both sources read
    the plan in allocation order, whatever order the arenas were filled in.
    """
    fills: List[int] = []
    capacities: List[int] = []
    placements: List[Tuple[int, int]] = [(0, 0)] * len(sizes)
    for original, size in sorted(enumerate(sizes), key=lambda item: -item[1]):
        for arena, fill in enumerate(fills):
            start = -(-fill // align) * align
            if start + size <= capacities[arena]:
                placements[original] = (arena, start)
                fills[arena] = start + size
                break
        else:
            placements[original] = (len(fills), 0)
            fills.append(size)
            capacities.append(max(capacity, _next_power_of_two(size)))
    return ArenaPlan(
        arena_sizes=tuple(max(align, _next_power_of_two(fill)) for fill in fills),
        placements=tuple(placements),
        requested_bytes=sum(sizes),
    )


class RamSource:
    """The base held in CPU RAM, allocated ONCE and filled by ``copy_``.

    The obvious ``load_file -> .to(dtype) -> .pin_memory()`` costs three
    transient copies of every layer. Measured on the dev box, that transient —
    not the store — is what pushed a 5.55 GB base past the 7.12 GB page-locked
    ceiling and made a 3B run impossible. So the store is pre-allocated at its
    final dtype and each source tensor is streamed into it one at a time.

    Pinned, the store is a handful of power-of-two arenas with every tensor a
    view (:func:`plan_pinned_arenas`, #901): ``nbytes`` is still the tensors'
    own bytes, ``pinned_bytes`` is what is actually page-locked. Pageable, each
    tensor is its own allocation, as before — the CPU allocator does not round.
    """

    def __init__(
        self,
        shard_dir: str,
        n_layers: int,
        spec: Union[
            Mapping[str, Tuple[Tuple[int, ...], str]],
            Sequence[Mapping[str, Tuple[Tuple[int, ...], str]]],
        ],
        *,
        pin: bool = True,
        shard_paths: Optional[Sequence[str]] = None,
        arena_bytes: int = PINNED_ARENA_BYTES,
    ):
        import torch
        from safetensors import safe_open

        layer_specs = self._normalize_layer_specs(spec, n_layers)
        paths = self._normalize_shard_paths(shard_dir, n_layers, shard_paths)
        self.store: list = []
        self.nbytes = 0
        self.pinned = bool(pin)
        self.arena_sizes: Tuple[int, ...] = ()
        self.pinned_bytes = 0
        plan: Optional[ArenaPlan] = None
        planned_sizes: List[int] = []
        arenas: List[Any] = []
        if self.pinned:
            # ONE walk defines the order the plan is indexed by; the fill loop
            # below re-derives each length and refuses to proceed if it ever
            # disagrees, so the two cannot drift apart silently.
            planned_sizes = [
                math.prod(shape) * _dtype_size(dtype)
                for idx in range(n_layers)
                for _name, (shape, dtype) in layer_specs[idx].items()
            ]
            plan = plan_pinned_arenas(planned_sizes, arena_bytes=arena_bytes)
            for size in plan.arena_sizes:
                # This is the host store; never inherit a process-wide MPS/CUDA default.
                arena = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
                self._check_host_tensor(arena)
                arenas.append(arena)
            self.arena_sizes = plan.arena_sizes
            self.pinned_bytes = plan.pinned_bytes
        self._arenas = arenas
        position = 0
        for idx in range(n_layers):
            held: Dict[str, Any] = {}
            with safe_open(paths[idx], framework="pt") as handle:
                for name, (shape, dtype) in layer_specs[idx].items():
                    torch_dtype = _torch_dtype(dtype)
                    if plan is not None:
                        arena_index, offset = plan.placements[position]
                        length = math.prod(shape) * _dtype_size(dtype)
                        if planned_sizes[position] != length:
                            raise RuntimeError(
                                f"layer streaming's pinned-arena plan is out of step with "
                                f"the store at layer {idx}, tensor {name!r} (planned "
                                f"{planned_sizes[position]} bytes, filling {length}); this "
                                f"is a bug, please report it"
                            )
                        position += 1
                        dst = (
                            arenas[arena_index][offset : offset + length]
                            .view(torch_dtype)
                            .view(tuple(shape))
                        )
                    else:
                        dst = torch.empty(
                            tuple(shape), dtype=torch_dtype, device="cpu", pin_memory=False
                        )
                        self._check_host_tensor(dst)
                    src = handle.get_tensor(name)
                    dst.copy_(src)
                    del src
                    held[name] = dst
                    self.nbytes += dst.numel() * dst.element_size()
            self.store.append(held)

    def _check_host_tensor(self, tensor: Any) -> None:
        # PyTorch 2.7+ on Apple Silicon may return an MPS tensor for
        # ``device="cpu", pin_memory=True``.  Accepting that would put
        # the entire supposed host store in the accelerator allocator
        # while reporting it as pinned CPU RAM (#434).
        if tensor.device.type != "cpu":
            raise RuntimeError(
                "layer streaming's RAM source requested a CPU tensor, "
                f"but torch returned {tensor.device}. Pinned host memory is "
                "CUDA-only here; retry with pin=False."
            )
        if self.pinned and not tensor.is_pinned():
            raise RuntimeError(
                "layer streaming requested pinned CPU RAM, but torch "
                "returned pageable memory; retry with pin=False."
            )

    @staticmethod
    def spec_from_path(path: str) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        """Shape AND dtype for the decoder-layer shard at ``path``, read from its header.

        The dtype is read per tensor rather than taken from ``index.dtype``: an NF4 shard is
        deliberately mixed — packed nibbles and (under double quant) absmax are ``uint8``
        while the nested absmax, the offset and the layernorms are floats. Allocating one
        dtype across the pool would reinterpret packed bytes as floats.
        """
        from safetensors import safe_open

        spec: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():
                sliced = handle.get_slice(name)
                shape = tuple(int(d) for d in sliced.get_shape())
                raw = sliced.get_dtype()
                if raw not in _SAFETENSORS_DTYPES:
                    raise ValueError(
                        f"shard tensor {name} has unsupported dtype {raw!r}; "
                        f"supported: {', '.join(sorted(_SAFETENSORS_DTYPES))}"
                    )
                spec[name] = (shape, _SAFETENSORS_DTYPES[raw])
        return spec

    @staticmethod
    def spec_from_shard(shard_dir: str, idx: int = 0) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        """``spec_from_path`` for layer ``idx`` of a ONE-root cache."""
        from soup_cli.utils.layer_shard import layer_shard_path

        return RamSource.spec_from_path(layer_shard_path(shard_dir, idx))

    @classmethod
    def layer_specs_from_paths(
        cls, paths: Sequence[str]
    ) -> list[Dict[str, Tuple[Tuple[int, ...], str]]]:
        """Every decoder layer's spec, one shard path per layer, wherever each file lives.

        The form the setup and ``install_streaming`` use, with ``layer_paths(shard_dir,
        index)``: on a striped cache (R4) the layers sit on several roots.
        """
        return [cls.spec_from_path(path) for path in paths]

    @classmethod
    def layer_specs_from_shards(
        cls, shard_dir: str, n_layers: int
    ) -> list[Dict[str, Tuple[Tuple[int, ...], str]]]:
        """The ONE-root form: every layer under ``shard_dir``. Wrong for a striped cache.

        Kept for one-root callers (tests, the #974 harnesses). The setup goes through
        :meth:`layer_specs_from_paths` with ``layer_paths(shard_dir, index)``, which knows
        which root holds each layer.
        """
        return [cls.spec_from_shard(shard_dir, idx) for idx in range(n_layers)]

    @staticmethod
    def merge_layer_specs(
        layer_specs: Sequence[Mapping[str, Tuple[Tuple[int, ...], str]]],
    ) -> Dict[str, Tuple[Tuple[int, ...], str]]:
        merged: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        for idx, spec in enumerate(layer_specs):
            for name, value in spec.items():
                prior = merged.get(name)
                if prior is not None and prior != value:
                    raise ValueError(
                        f"decoder weight {name!r} has different stored shapes or dtypes "
                        f"across layers ({prior} vs {value}, first seen before layer "
                        f"{idx}) — layer streaming can vary which weights exist per "
                        f"layer, but a shared key must keep one storage layout"
                    )
                merged.setdefault(name, value)
        return merged

    @staticmethod
    def _normalize_layer_specs(
        spec: Union[
            Mapping[str, Tuple[Tuple[int, ...], str]],
            Sequence[Mapping[str, Tuple[Tuple[int, ...], str]]],
        ],
        n_layers: int,
    ) -> list[Mapping[str, Tuple[Tuple[int, ...], str]]]:
        if isinstance(spec, Mapping):
            return [spec] * n_layers
        layer_specs = list(spec)
        if len(layer_specs) != n_layers:
            raise ValueError(f"expected {n_layers} layer specs, but got {len(layer_specs)}")
        return layer_specs

    @staticmethod
    def _normalize_shard_paths(
        shard_dir: str, n_layers: int, shard_paths: Optional[Sequence[str]]
    ) -> list[str]:
        from soup_cli.utils.layer_shard import layer_shard_path

        if shard_paths is None:
            return [layer_shard_path(shard_dir, idx) for idx in range(n_layers)]
        paths = [str(path) for path in shard_paths]
        if len(paths) != n_layers:
            raise ValueError(f"expected {n_layers} shard paths, but got {len(paths)}")
        return paths

    def get(self, idx: int, name: str):
        return self.store[idx][name]


class DiskSource:
    """Tier 2 — the base stays on disk and each layer is read on demand.

    Same interface as :class:`RamSource` (``get(idx, name)`` + ``nbytes``), so
    the buffer pool, the prefetcher and the layer wrapper are untouched: the
    only thing that changes is where the bytes come from. That is what lets the
    disk tier inherit v0.72.0's correctness gates rather than needing new ones
    for the scheduler.

    Handles are opened ONCE and held. safetensors memory-maps them, so the
    operating system's page cache — not this class — decides what stays
    resident. Two honest consequences:

    * On a machine with spare RAM the "disk" tier is partly a RAM tier, because
      the pages survive between steps. That makes a like-for-like RAM-vs-disk
      comparison hard to construct, and it is why this release publishes the
      code path but no gap measurement (see the v0.72.3 gate notes).
    * ``nbytes`` is 0: nothing is deliberately held resident. The on-disk size
      is reported separately so the pre-flight can still show the operator what
      the model costs.

    NVMe only — ``choose_tier`` refuses spinning disks, where each step costs
    two seeks per layer (plan P11) and the run thrashes rather than merely
    running slower.
    """

    def __init__(
        self,
        shard_dir: str,
        n_layers: int,
        spec: Union[
            Mapping[str, Tuple[Tuple[int, ...], str]],
            Sequence[Mapping[str, Tuple[Tuple[int, ...], str]]],
        ],
        *,
        shard_paths: Optional[Sequence[str]] = None,
    ):
        import contextlib

        from safetensors import safe_open

        layer_specs = RamSource._normalize_layer_specs(spec, n_layers)
        paths = RamSource._normalize_shard_paths(shard_dir, n_layers, shard_paths)
        self._stack = contextlib.ExitStack()
        # ExitStack, not a comprehension of __enter__(): if shard N fails to
        # open, everything opened before it must still be closed.
        try:
            self._handles = [
                self._stack.enter_context(safe_open(paths[idx], framework="pt"))
                for idx in range(n_layers)
            ]
        except BaseException:
            self._stack.close()
            raise
        self.nbytes = 0  # nothing is held resident by design
        self.disk_bytes = sum(
            math.prod(shape) * _dtype_size(dtype)
            for per_layer in layer_specs
            for shape, dtype in per_layer.values()
        )
        self.pinned = False

    def get(self, idx: int, name: str):
        return self._handles[idx].get_tensor(name)

    def close(self) -> None:
        """Release every shard handle. Idempotent — ``ExitStack.close`` is."""
        self._stack.close()
        self._handles = []

    def __del__(self) -> None:
        # Backstop only. The trainer closes the runtime explicitly; this exists
        # so a caller that drops the source without doing so still releases one
        # file handle per decoder layer (80+ on a large model) rather than
        # holding them until the process exits.
        try:
            self.close()
        except Exception:  # noqa: BLE001 — never raise from a finaliser
            pass


def _dtype_size(name: str) -> int:
    from soup_cli.utils.layer_stream import dtype_bytes

    return dtype_bytes(name)


def extras_resident_bytes(shard_dir: str) -> int:
    """Bytes the non-layer weights occupy on the GPU (final norm / buffers).

    Vocabulary-sized embeddings and an untied head live in dedicated large
    shards as of shard format v2 and are deliberately absent from this number.
    Read from the extras header — no tensor is materialised.

    The shared NF4 code tables live in the same file but are 272 floats of
    machinery, not model weights, so they are excluded.
    """
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import (
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        extras_shard_path,
    )
    from soup_cli.utils.layer_stream import dtype_bytes

    skip = {NF4_CODE_KEY, NF4_NESTED_CODE_KEY}
    total = 0
    with safe_open(extras_shard_path(shard_dir), framework="pt") as handle:
        for name in handle.keys():
            if name in skip:
                continue
            sliced = handle.get_slice(name)
            raw = sliced.get_dtype()
            if raw not in _SAFETENSORS_DTYPES:
                # Raise, don't skip: this number feeds the RAM-tier decision, so
                # silently under-reporting it would let a base that does not fit
                # be accepted. Mirrors spec_from_shard.
                raise ValueError(
                    f"extras tensor {name} has unsupported dtype {raw!r}; "
                    f"supported: {', '.join(sorted(_SAFETENSORS_DTYPES))}"
                )
            total += math.prod(int(dim) for dim in sliced.get_shape()) * dtype_bytes(
                _SAFETENSORS_DTYPES[raw]
            )
    return total


def large_layer_specs(shard_dir: str, index: Any) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Shape and dtype for the streamed embedding/head shards."""
    from safetensors import safe_open

    from soup_cli.utils.layer_shard import large_shard_path, large_weight_role

    specs: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    roles = set()
    for key in tuple(getattr(index, "large_keys", ()) or ()):
        role = large_weight_role(key)
        if role is None or role in roles:
            raise ValueError(
                f"shard index has an invalid or duplicate large-layer key {key!r}; "
                "reshard the checkpoint"
            )
        roles.add(role)
        with safe_open(large_shard_path(shard_dir, key), framework="pt") as handle:
            stored = tuple(handle.keys())
            if stored != (key,):
                raise ValueError(
                    f"large-layer shard for {key!r} contains {stored!r}; expected "
                    "that key alone — reshard the checkpoint"
                )
            sliced = handle.get_slice(key)
            raw = sliced.get_dtype()
            if raw not in _SAFETENSORS_DTYPES:
                raise ValueError(
                    f"large-layer tensor {key} has unsupported dtype {raw!r}; "
                    f"supported: {', '.join(sorted(_SAFETENSORS_DTYPES))}"
                )
            specs[key] = (
                tuple(int(dim) for dim in sliced.get_shape()),
                _SAFETENSORS_DTYPES[raw],
            )
    return specs


def large_layer_store_bytes(shard_dir: str, index: Any) -> int:
    """Host/disk bytes occupied by every large-layer shard."""
    return sum(
        math.prod(shape) * _dtype_size(dtype)
        for shape, dtype in large_layer_specs(shard_dir, index).values()
    )


def large_layer_buffer_bytes(shard_dir: str, index: Any) -> int:
    """Bytes in the single reusable large-layer device slot."""
    sizes = [
        math.prod(shape) * _dtype_size(dtype)
        for shape, dtype in large_layer_specs(shard_dir, index).values()
    ]
    return max(sizes, default=0)


# ==========================================================================
# Tier 0 — pre-allocated VRAM buffers (plan 5.4)
# ==========================================================================
def _release_source(source: Any, idx: int, event: Any) -> None:
    """Tell a staging source the copy out of layer ``idx`` has been enqueued.

    ``RamSource`` holds every layer for the whole run and ``DiskSource`` returns
    a freshly allocated tensor per call, so neither can have a buffer recycled
    underneath an in-flight copy and neither defines ``release``. A source that
    stages into a small pool of reusable HOST buffers can: out of PINNED memory
    ``dst.copy_(..., non_blocking=True)`` is still draining when ``load_async``
    returns, and ``wait()`` is a GPU-side ``wait_event`` that never blocks the
    Python thread — so the reader is free to run ahead and overwrite the bytes
    the copy is reading. Measured through this pool against ``AsyncDiskSource``
    before this call existed: 7 of 8 layers reached the device holding another
    layer's weights at read_ahead=1, 6 of 8 at the default 2, 4 of 8 at 4.

    Duck-typed rather than isinstance-gated so the two shipped sources stay
    untouched and a future source opts in by defining the method.
    """
    release = getattr(source, "release", None)
    if callable(release):
        release(idx, event)


class LayerBufferPool:
    """N pre-allocated per-layer buffers. Never allocates inside the loop —
    that is what keeps the allocator from fragmenting (plan P7)."""

    def __init__(
        self,
        layer_spec: Mapping[str, Tuple[Tuple[int, ...], str]],
        n_buffers: int = 2,
        device: str = "cuda",
        active_keys_by_layer: Optional[Sequence[Sequence[str]]] = None,
    ):
        import torch

        self.device = device
        self.is_cuda = str(device).startswith("cuda")
        self.n = int(n_buffers)
        self.buffers = [
            {
                name: torch.empty(tuple(shape), dtype=_torch_dtype(dtype), device=device)
                for name, (shape, dtype) in layer_spec.items()
            }
            for _ in range(self.n)
        ]
        self.events = [torch.cuda.Event() for _ in range(self.n)] if self.is_cuda else []
        self.owner: list = [None] * self.n
        # Bumped on every load into a slot, so a reader can tell a slot that was
        # refilled (even with the same layer) from one that was not.
        self.generation: list = [0] * self.n
        if active_keys_by_layer is None:
            shared = tuple(layer_spec)
            self.active_keys_by_layer = [shared]
        else:
            self.active_keys_by_layer = [tuple(keys) for keys in active_keys_by_layer]
        self.loads = 0
        self.nbytes = (
            sum(buf.numel() * buf.element_size() for buf in self.buffers[0].values()) * self.n
        )

    def slot_for(self, idx: int) -> int:
        return idx % self.n

    def load_async(self, idx: int, source: RamSource, stream: Any = None) -> int:
        import torch

        slot = self.slot_for(idx)
        keys = (
            self.active_keys_by_layer[0]
            if len(self.active_keys_by_layer) == 1
            else self.active_keys_by_layer[idx]
        )
        if self.is_cuda and stream is not None:
            # The slot's previous owner may still be in flight on the compute
            # stream; the prefetch must not clobber it (plan P1).
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for name in keys:
                    dst = self.buffers[slot][name]
                    dst.copy_(source.get(idx, name), non_blocking=True)
                self.events[slot].record(stream)
            _release_source(source, idx, self.events[slot])
        else:
            for name in keys:
                dst = self.buffers[slot][name]
                dst.copy_(source.get(idx, name))
            # No event: this branch's copy has already finished when `copy_`
            # returns, so the source's buffer is free NOW rather than when a
            # stream drains. Saying so is not decoration — a source that stages
            # into a small reusable pool refills sooner for it.
            _release_source(source, idx, None)
        self.owner[slot] = idx
        self.generation[slot] += 1
        self.loads += 1
        return slot

    def wait(self, idx: int) -> Dict[str, Any]:
        """Block the compute stream until layer ``idx`` is resident.

        The ownership check is the plan-P1 tripwire: a buffer recycled while an
        autograd node still references it produces silently WRONG gradients, not
        a crash. Failing loudly here is the whole point.
        """
        import torch

        slot = self.slot_for(idx)
        if self.owner[slot] != idx:
            raise RuntimeError(
                f"layer-stream scheduler bug: buffer slot {slot} holds layer "
                f"{self.owner[slot]}, but layer {idx} was requested. Raise "
                f"training.stream_buffers (currently {self.n}) or report this."
            )
        if self.is_cuda:
            torch.cuda.current_stream().wait_event(self.events[slot])
        return self.buffers[slot]


class LargeLayerBufferPool:
    """One reusable device slot for ``embed_tokens`` and an untied ``lm_head``.

    Both weights are stored at the same stream dtype.  A flat allocation sized
    to the larger matrix can therefore expose a correctly shaped view for
    either boundary without holding both vocabulary-sized tensors resident.
    """

    def __init__(
        self,
        specs: Mapping[str, Tuple[Tuple[int, ...], str]],
        source_indices: Mapping[str, int],
        *,
        device: str = "cuda",
    ):
        import torch

        self.device = str(device)
        self.is_cuda = self.device.startswith("cuda")
        self.specs = dict(specs)
        self.source_indices = dict(source_indices)
        dtypes = {dtype for _shape, dtype in self.specs.values()}
        if len(dtypes) > 1:
            raise ValueError(
                "streamed large-layer weights must share one dtype; reshard the checkpoint"
            )
        self.dtype = next(iter(dtypes), "float32")
        largest = max((math.prod(shape) for shape, _dtype in self.specs.values()), default=0)
        self.buffer = torch.empty(largest, dtype=_torch_dtype(self.dtype), device=device)
        self.event = torch.cuda.Event() if self.is_cuda and largest else None
        self.owner: Optional[str] = None
        self.loads = 0
        self.nbytes = self.buffer.numel() * self.buffer.element_size()

    def _view(self, key: str) -> Any:
        shape, dtype = self.specs[key]
        if dtype != self.dtype:
            raise RuntimeError(f"large-layer dtype changed for {key!r}")
        return self.buffer[: math.prod(shape)].view(tuple(shape))

    def load_async(self, key: str, source: Any, stream: Any = None) -> None:
        import torch

        if self.owner == key:
            return
        if key not in self.specs or key not in self.source_indices:
            raise ValueError(f"large-layer source is missing {key!r}")
        dst = self._view(key)
        source_idx = self.source_indices[key]
        if self.is_cuda and stream is not None:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                dst.copy_(source.get(source_idx, key), non_blocking=True)
                self.event.record(stream)
            _release_source(source, source_idx, self.event)
        else:
            dst.copy_(source.get(source_idx, key))
            _release_source(source, source_idx, None)
        self.owner = key
        self.loads += 1

    def wait(self, key: str) -> Any:
        import torch

        if self.owner != key:
            raise RuntimeError(
                f"large-layer scheduler bug: slot holds {self.owner!r}, but {key!r} was requested"
            )
        if self.is_cuda and self.event is not None:
            torch.cuda.current_stream().wait_event(self.event)
        return self._view(key)


class StreamPrefetcher:
    """Drives the prefetch. Forward walks 0..L-1; backward recompute walks
    L-1..0, so the direction is inferred from the call order."""

    def __init__(
        self,
        pool: Any,
        source: Any,
        n_layers: int,
        stream: Any = None,
        tail_prefetch: Any = None,
        backward_tail_prefetch: Any = None,
        head_prefetch_layer: Optional[int] = None,
    ):
        self.pool = pool
        self.source = source
        self.n_layers = int(n_layers)
        self.stream = stream
        self.prev: Optional[int] = None
        self.direction = 1
        self.primes = 0
        self.tail_prefetch = tail_prefetch
        self.tail_prefetched = False
        # #1174 part 2 — which decoder layer's forward issues `tail_prefetch`, the
        # head's load into the shared slot. `None` is the original timing, the
        # last layer, where the copy has only that one layer's compute to hide
        # behind. A smaller index issues it earlier in the forward: the
        # embedding's bytes in the slot are dead once the lookup has run, and
        # the lookup precedes layer 0's `advance`, so layer 0 is the earliest
        # point. It is a property, so a harness can flip it between steps and
        # run both timings in one process, and an assignment is checked the same
        # way the constructor argument is.
        self.head_prefetch_layer = head_prefetch_layer
        # #975 — the embedding's `_prime`-time load pays a full head-sized H2D
        # copy with nothing to overlap it against, because it fires right as
        # the next step's forward starts and is needed almost immediately.
        # This callback lets the caller issue that SAME load earlier, once
        # layer 0's backward recompute confirms this step's decoder walk is
        # done, so it can overlap with whatever backward work is still ahead
        # (the embedding's own backward, optimiser bookkeeping) instead of
        # blocking the next step's first op.
        self.backward_tail_prefetch = backward_tail_prefetch
        self.backward_tail_prefetched = False

    @property
    def head_prefetch_layer(self) -> Optional[int]:
        return self._head_prefetch_layer

    @head_prefetch_layer.setter
    def head_prefetch_layer(self, value: Optional[int]) -> None:
        # `bool` is an `int`, so `True` would otherwise pass as layer 1. A bad
        # value assigned later would only surface at the head's forward as
        # "large-layer scheduler bug: slot holds ...", which names the wrong cause.
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.n_layers
        ):
            raise ValueError(
                f"head_prefetch_layer must be None or in [0, {self.n_layers}), got {value!r}"
            )
        self._head_prefetch_layer = value

    def prime(self) -> None:
        """Start of a forward pass: layer 0, walking upward."""
        self.prev = None
        self.direction = 1
        self.primes += 1
        self.backward_tail_prefetched = False
        self.tail_prefetched = False
        if self.pool.owner[self.pool.slot_for(0)] != 0:
            self.pool.load_async(0, self.source, self.stream)

    def advance(self, idx: int) -> None:
        # Direction is explicit state, not re-derived per call. It only ever
        # flips downward, at the forward/backward turnaround, and is reset by
        # prime() at the start of the next step. Inferring it fresh each call
        # happens to work today only because the turnaround index is the last
        # layer; a future deeper lookahead would break that assumption
        # silently.
        if self.prev is not None and idx < self.prev:
            self.direction = -1
        self.prev = idx
        nxt = idx + self.direction
        if 0 <= nxt < self.n_layers and self.pool.owner[self.pool.slot_for(nxt)] != nxt:
            self.pool.load_async(nxt, self.source, self.stream)
        head_layer = self.head_prefetch_layer
        if head_layer is None:
            head_layer = self.n_layers - 1
        if (
            self.direction == 1
            and idx == head_layer
            and not self.tail_prefetched
            and self.tail_prefetch is not None
        ):
            # The refill of the shared slot with the head. It is ordered after
            # the embedding lookup (queued on the compute stream, reading the
            # slot) by the `wait_stream` in `LargeLayerBufferPool.load_async`,
            # the same guard the backward-tail refill below relies on.
            self.tail_prefetch()
            self.tail_prefetched = True
        # #975 — layer 0 reached going backward is the last decoder recompute
        # of this step's backward pass, so this refills the shared slot with
        # the embedding while the backward is still running. Two separate
        # things make that safe, and only the second protects the GPU:
        #
        # * CPU side: `lm_head`'s backward Node sits between the loss and every
        #   decoder layer, so the autograd engine has already dispatched it, and
        #   the saved-tensor version check happens at dispatch. The embedding's
        #   own backward reads indices, never the weight values, so it never
        #   checks this slot's version at all.
        # * GPU side: when the CPU reaches layer 0, the head's backward GEMM can
        #   still be queued on the compute stream, reading this slot. What
        #   orders the refill after it is `stream.wait_stream(torch.cuda.
        #   current_stream())` in `LargeLayerBufferPool.load_async`, which makes
        #   the copy stream wait for everything already enqueued on the compute
        #   stream. Without that wait the refill can overwrite bytes the head's
        #   backward has not read yet, and the gradients come out silently
        #   wrong. Keep it if `load_async` is touched.
        #
        # `_prime()` still issues this same load unconditionally at the next
        # step's start — this only makes that call a same-owner no-op on the
        # hot path, by getting there first with time to overlap.
        if (
            self.direction == -1
            and idx == 0
            and not self.backward_tail_prefetched
            and self.backward_tail_prefetch is not None
        ):
            self.backward_tail_prefetch()
            self.backward_tail_prefetched = True


# ==========================================================================
# the streamed layer (plan 5.6)
# ==========================================================================
def _substituted_names_are_tied(module: Any, names: Any) -> bool:
    """True when a name in ``names`` shares its tensor with another registered name.

    ``functional_call(tie_weights=True)`` rewrites the mapping only for a GIVEN
    name whose tensor is registered under more than one name
    (``torch.nn.utils.stateless._untie_named_tensors_map``); for every other
    mapping it returns a plain copy. Deciding this once lets the hot path pass
    ``tie_weights=False`` with an identical result.
    """
    wanted = set(names)
    names_by_tensor: dict = {}
    for name, tensor in (
        *module.named_parameters(remove_duplicate=False),
        *module.named_buffers(remove_duplicate=False),
    ):
        names_by_tensor.setdefault(id(tensor), set()).add(name)
    return any(len(group) > 1 and group & wanted for group in names_by_tensor.values())


def _build_streamed_layer_class():
    import torch
    import torch.nn as nn
    from torch.func import functional_call
    from torch.utils.checkpoint import checkpoint, noop_context_fn

    class StreamedDecoderLayer(nn.Module):
        def __init__(
            self,
            inner,
            idx,
            pool,
            prefetcher,
            name_map=None,
            use_checkpoint=True,
            quant_specs=None,
            codes=None,
            scale_cache=None,
            checkpoint_context_fn=None,
        ):
            super().__init__()
            self.inner = inner
            self.idx = int(idx)
            self.pool = pool
            self.prefetcher = prefetcher
            self.name_map = dict(name_map or {})
            self.use_checkpoint = bool(use_checkpoint)
            # training.stream_recompute — False: no checkpoint recompute.
            # Activations stay saved; pool-slot tensors are saved as tokens and
            # the layer is re-visited (wait + prefetch) at its first backward
            # unpack instead of at the recompute. See _body_retained /
            # set_stream_recompute.
            self.recompute = True
            self._bwd_visited = True
            # The pool slot's load generation at the first backward visit; a
            # later unpack refuses a slot that was reloaded since, even with the
            # same owner.
            self._bwd_generation = None
            self._slot_ptrs: Dict[int, frozenset] = {}
            # training.stream_recompute='auto' — a RetainedLayerProbe armed on
            # this layer measures its next no-recompute forward, then disarms.
            self._retained_probe = None
            self.quant_specs = dict(quant_specs or {})
            self.codes = dict(codes or {})
            # training.stream_nf4_scale_cache — an NF4ScaleCache owned by the
            # runtime, or None.
            self.scale_cache = scale_cache
            # Two-read experts (``expert_stream.two_read_checkpoint_contexts``): the
            # recompute replays the routers' decisions and reads no expert.
            self.checkpoint_context_fn = checkpoint_context_fn or noop_context_fn
            # #841 — each wrapper returns to the same pool slot on every visit.
            # Cache the tensor/Params4bit substitution views for that slot rather
            # than reconstructing Python wrappers and QuantState objects on every
            # forward and checkpoint recompute. The views share storage with the
            # pool tensors, so later layer loads update their contents in place.
            self._cached_substitution_buffers = None
            self._cached_substitution_weights = None
            # functional_call's default tie_weights=True walks every parameter
            # and buffer of the layer on each call, forward and recompute alike
            # (~104 us on a PEFT Mistral layer, ~6.7 ms per 32-layer step). The
            # walk changes nothing unless a substituted name is tied, and
            # get_peft_model() has already run, so the graph is final here.
            self._tie_weights = _substituted_names_are_tied(self.inner, self.name_map)
            # v0.72.5 (#331) — a streamed NF4 weight must not reach MatMul4Bit,
            # which captures it outside save_for_backward and so aliases the pool
            # across the checkpoint boundary. See install_dequant_forward.
            if self.quant_specs:
                self.n_dequant_forward = install_dequant_forward(self.inner)
            else:
                self.n_dequant_forward = 0
            # v0.72.3 — the mirror of the state_dict() override below.
            self._register_load_state_dict_pre_hook(self._redirect_canonical_keys)

        def _redirect_canonical_keys(self, state_dict, prefix, *_args: Any, **_kwargs: Any) -> None:
            # v0.72.1 made SAVING canonical; this makes LOADING accept the same
            # keys, which is what `--resume` needs.
            #
            # ``nn.Module.load_state_dict`` narrows the dict by CHILD NAME as it
            # descends: at this wrapper it keeps only keys under `<prefix>`, then
            # recurses into each child with `<prefix><child>.`. Our sole child is
            # `inner`, so a canonical `...layers.0.self_attn...` key matches no
            # child prefix and is silently dropped. Measured before this hook:
            # `load_adapter` landed **0 of 12** tensors, PEFT emitted only a
            # UserWarning, and the resumed loss curve was byte-identical to a
            # from-scratch one.
            #
            # This pre-hook runs at the START of our own `_load_from_state_dict`
            # and may mutate `state_dict` in place; the child loop reads that
            # same object afterwards, so the redirected copies are visible when
            # torch descends into `inner`. Nothing is deleted (the originals
            # match no child and are inert) and nothing is double-added.
            #
            # Deliberately load-side only. It redirects keys rather than
            # re-parenting the module tree, so the forward path is untouched and
            # v0.72.0's bit-exactness gates remain valid without being re-run.
            #
            # The original key is MOVED, not copied. Leaving it behind makes
            # this wrapper's own strict scan flag it `unexpected` — `self_attn`
            # is not one of our children, only `inner` is — so a caller passing
            # ``strict=True`` would get "Unexpected key(s)" on a load that in
            # fact succeeded. (PEFT always loads with ``strict=False`` and never
            # inspects the list, so this was inert in practice; it is still a
            # landmine for any direct caller.) Moving is safe: torch rebuilds a
            # fresh filtered dict at every recursion level, so the dict mutated
            # here is an intermediate one, never the caller's own.
            inner_prefix = prefix + "inner."
            for key in [
                k for k in state_dict if k.startswith(prefix) and not k.startswith(inner_prefix)
            ]:
                redirected = inner_prefix + key[len(prefix) :]
                value = state_dict.pop(key)
                if redirected in state_dict:
                    # A checkpoint carrying BOTH spellings of one weight is
                    # malformed, and silently keeping one would load a tensor
                    # the file does not unambiguously specify.
                    raise ValueError(
                        f"checkpoint contains both {key!r} and {redirected!r} "
                        f"for the same weight — it is malformed; re-save the "
                        f"adapter"
                    )
                state_dict[redirected] = value

        def _apply(self, fn: Any, recurse: bool = True) -> Any:
            # `.to(device)` / `.to(dtype)` walk the module tree via _apply. The
            # wrapped layer's weights are META PLACEHOLDERS, substituted per
            # call from the buffer pool, and moving a meta tensor raises
            # NotImplementedError — which transformers' Trainer.__init__ AND
            # accelerate's prepare_model both trigger. Pass meta tensors
            # through untouched; everything real (the LoRA adapters, which live
            # inside this same subtree) still moves and casts normally.
            def _skip_meta(tensor: Any) -> Any:
                if getattr(tensor, "is_meta", False):
                    return tensor
                return fn(tensor)

            # #1005 — `named_children()` below no longer lists `inner`, and
            # torch's `_apply` recurses over `children()`, so the inner layer's
            # OWN parameters and buffers (none on a decoder layer today, but
            # not a contract) would be skipped. Reach it explicitly.
            if recurse:
                self.inner._apply(_skip_meta, recurse=True)
            return super()._apply(_skip_meta, recurse=False)

        # ------------------------------------------------------------------
        # #1005 — the wrapper's NAMES are canonical, not only its KEYS.
        #
        # v0.72.1 made `state_dict()` write `...layers.0.self_attn...` (below)
        # while `named_modules()` / `named_parameters()` still said
        # `...layers.0.inner.self_attn...` — two spellings of one tensor. That
        # was harmless while every consumer joined keys to keys. peft 0.21
        # selects the adapter entries STRUCTURALLY: it collects key prefixes
        # from `named_modules()` (every tuner layer's attribute names) and keeps
        # the `state_dict()` entries under them — prefixes built from one
        # spelling, applied to the other, so a streamed adapter saved as ZERO
        # tensors with no exception (9 CI cells, every PR, the day 0.21.0 was
        # published). transformers' `get_parameter_names` — the weight-decay
        # grouping — walks `named_children()` and joins the result with
        # `named_parameters()`: self-consistent before this fix (both spelled
        # `.inner.`) and broken the moment only `named_modules()` changed, so
        # the two move together.
        #
        # So the inner layer's subtree is reported at THIS wrapper's prefix,
        # and the wrapper itself is NOT enumerated — `named_modules()` yields
        # the inner layer under the wrapper's own name and then its children;
        # `named_children()` yields the inner layer's children. The wrapper
        # stays reachable where the tree is walked structurally (`children()`
        # of the container, `get_submodule`, attribute access), which is what
        # `train()` / `apply()` / `_apply()` go through. Enumerating the
        # wrapper TOO would be worse than either: it is attribute-transparent
        # (`__getattr__` below), so transformers' `_set_gradient_checkpointing`
        # — `hasattr(module, "gradient_checkpointing")` then `setattr` over
        # `modules()` — would answer on the wrapper and land the write on it,
        # and a name-keyed patcher (`utils/gradient_ckpt.py`) would patch two
        # objects under one name. `named_parameters()` / `named_buffers()` /
        # `parameters()` derive from `named_modules()` and become canonical
        # with it. Whatever reads `_modules` directly needs no help and gets
        # none — `__repr__`, `get_submodule`, `load_state_dict`'s recursion
        # (for which the load-side pre-hook above keeps redirecting canonical
        # keys into `.inner.`); whatever goes through `children()` —
        # `train()`, `apply()`, `_apply()` — is overridden below to reach
        # `inner` itself. `apply()` deliberately still visits the wrapper too,
        # AFTER the inner tree: it serves caller-supplied initialisers and
        # hook installers that should see every object, and the
        # hasattr/setattr hazard above is specific to the `modules()`- and
        # name-driven walks; a test pins the opt-back-in. Naming only: nothing
        # on the streamed forward path changes, so the bit-exactness gates
        # stay valid un-re-run.
        # ------------------------------------------------------------------
        def named_children(self) -> Any:
            yield from self.inner.named_children()

        def named_modules(
            self, memo: Any = None, prefix: str = "", remove_duplicate: bool = True
        ) -> Any:
            if memo is None:
                memo = set()
            if remove_duplicate:
                if self in memo:
                    return
                memo.add(self)
            yield from self.inner.named_modules(memo, prefix, remove_duplicate)

        def train(self, mode: bool = True) -> Any:
            if not isinstance(mode, bool):
                raise ValueError("training mode is expected to be boolean")
            self.training = mode
            self.inner.train(mode)
            return self

        def apply(self, fn: Any) -> Any:
            self.inner.apply(fn)
            fn(self)
            return self

        def state_dict(
            self,
            *args: Any,
            destination: Any = None,
            prefix: str = "",
            keep_vars: bool = False,
        ) -> Any:
            # v0.72.1 — serialise as though this wrapper were not in the tree.
            #
            # The wrapper holds the real layer as a child named `inner`, so
            # every adapter parameter would otherwise be written as
            # `...layers.0.inner.self_attn.q_proj.lora_A.weight`. That file
            # loads as ZERO tensors into any normal model — PEFT reports the
            # keys as missing and returns the untuned base, with no exception.
            # Every adapter artifact (the final `trainer.save_model()`, each
            # `save_steps` checkpoint, and therefore everything downstream:
            # `soup merge` / `serve` / `chat` / `adapters *` / the Registry)
            # reaches disk through this method, so delegating at OUR prefix is
            # what makes a streamed adapter indistinguishable from a normal
            # LoRA run.
            #
            # Serialisation-only, deliberately: the forward path is untouched,
            # so v0.72.0's bit-exactness gates remain valid. Until #1005 the
            # cost was that `named_parameters()` still showed `.inner.` (which
            # is why `canonical_named_parameters()` below existed); since #1005
            # the naming methods above agree with these keys, and that helper
            # is an identity kept for its callers. `--resume` / `--hf-resume`
            # load INTO a streamed model fine (v0.72.3, `train.py`): a separate
            # load-side pre-hook redirects canonical keys at load time,
            # mirroring this save-side delegation.
            #
            # The wrapper owns no parameters or buffers of its own — they all
            # live on `inner` — so nothing is lost by not serialising it. It
            # also means bypassing nn.Module.state_dict skips only hooks
            # registered on the WRAPPER itself, of which there are none (the
            # prefetch hook lives on the decoder container, not here).
            if args:
                # torch's legacy positional form: (destination, prefix, keep_vars)
                if destination is None:
                    destination = args[0]
                if len(args) > 1 and prefix == "":
                    prefix = args[1]
                if len(args) > 2 and keep_vars is False:
                    keep_vars = args[2]
            return self.inner.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )

        def __getattr__(self, name: str) -> Any:
            # transformers reads contract attributes straight off the layer
            # object (this version reads `decoder_layer.attention_type`). The
            # wrapper must be attribute-transparent or the model breaks at
            # forward time — and a wrapper returning a DEFAULT instead would
            # silently pick the wrong attention path.
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name == "inner":
                    raise
                inner = self._modules.get("inner")
                if inner is None:
                    raise
                return getattr(inner, name)

        def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            if self.use_checkpoint and torch.is_grad_enabled():
                if not self.recompute:
                    return self._body_retained(hidden_states, *args, **kwargs)
                return checkpoint(
                    self._body, hidden_states, *args, use_reentrant=False,
                    context_fn=self.checkpoint_context_fn, **kwargs,
                )
            return self._body(hidden_states, *args, **kwargs)

        def _pool_ptrs(self) -> frozenset:
            slot = self.pool.slot_for(self.idx)
            ptrs = self._slot_ptrs.get(slot)
            if ptrs is None:
                ptrs = frozenset(
                    buf.untyped_storage().data_ptr() for buf in self.pool.buffers[slot].values()
                )
                self._slot_ptrs[slot] = ptrs
            return ptrs

        def _backward_visit(self) -> None:
            # The first unpack of this layer's pool tensors in the backward is
            # where the recompute used to run: same wait + prefetch, same order.
            slot = self.pool.slot_for(self.idx)
            if not self._bwd_visited:
                self._bwd_visited = True
                self.pool.wait(self.idx)
                # Recorded after the wait (the slot holds this layer now) and
                # before the prefetch, which loads a DIFFERENT slot.
                self._bwd_generation = self.pool.generation[slot]
                self.prefetcher.advance(self.idx)
                return
            if self.pool.owner[slot] != self.idx:
                raise RuntimeError(
                    f"streamed backward (no recompute) found slot {slot} holding layer "
                    f"{self.pool.owner[slot]} while layer {self.idx} still unpacks from it"
                )
            if self.pool.generation[slot] != self._bwd_generation:
                raise RuntimeError(
                    f"streamed backward (no recompute) found slot {slot} reloaded "
                    f"(load generation {self._bwd_generation} -> "
                    f"{self.pool.generation[slot]}) while layer {self.idx} still unpacks "
                    "from it"
                )

        def _body_retained(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            ptrs = self._pool_ptrs()
            token = _POOL_TOKEN

            def pack(tensor: Any) -> Any:
                try:
                    ptr = tensor.untyped_storage().data_ptr()
                except (AttributeError, RuntimeError):
                    return tensor
                if ptr in ptrs:
                    return (token, tensor)
                return tensor

            def unpack(obj: Any) -> Any:
                if type(obj) is tuple and len(obj) == 2 and obj[0] is token:
                    self._backward_visit()
                    return obj[1]
                return obj

            self._bwd_visited = False
            probe = self._retained_probe
            if probe is not None:
                # One forward only: the probe restores the recompute after it.
                self._retained_probe = None
                probe.start(self)
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                out = self._body(hidden_states, *args, **kwargs)
            if probe is not None:
                probe.finish(self, out)
            return out

        def _substituted_weights(self, buffers: Any) -> Any:
            # Weights arrive with requires_grad=False, so autograd allocates no
            # grad buffers for them — but W STAYS IN THE GRAPH for W^T . dL/dy,
            # which is how the lower adapters receive gradient at all.
            #
            # #841 — a wrapper revisits the same pool-slot mapping every forward
            # and checkpoint recompute. Only the tensors' CONTENTS change when a
            # layer is loaded. Reuse the mapping and Params4bit/QuantState views
            # while the mapping object is identical; rebuilding them is pure
            # Python/object churn and does not create fresher storage.
            if (
                self._cached_substitution_buffers is buffers
                and self._cached_substitution_weights is not None
            ):
                return self._cached_substitution_weights

            weights = {}
            for meta, ckpt in self.name_map.items():
                spec = self.quant_specs.get(ckpt)
                if spec is None:
                    weights[meta] = buffers[ckpt]
                else:
                    weight = rebuild_params4bit(ckpt, buffers, spec, self.codes)
                    weight._soup_stream_owner_check = self._assert_nf4_slot_owner
                    if self.scale_cache is not None:
                        weight._soup_scale_cache = self.scale_cache
                        weight._soup_scale_cache_key = (self.idx, ckpt)
                    weights[meta] = weight
            self._cached_substitution_buffers = buffers
            self._cached_substitution_weights = weights
            return weights

        def _assert_nf4_slot_owner(self) -> None:
            slot = self.pool.slot_for(self.idx)
            owner = self.pool.owner[slot]
            if owner != self.idx:
                raise RuntimeError(
                    "streamed NF4 backward found a recycled weight slot: "
                    f"slot {slot} holds layer {owner}, expected layer {self.idx}. "
                    "The prefetch lookahead exceeded the checkpoint-visible "
                    "alias lifetime."
                )

        def _body(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            buffers = self.pool.wait(self.idx)
            self.prefetcher.advance(self.idx)
            return functional_call(
                self.inner,
                self._substituted_weights(buffers),
                (hidden_states, *args),
                kwargs,
                tie_weights=self._tie_weights,
            )

    return StreamedDecoderLayer


_STREAMED_LAYER_CLASS = None
_POOL_TOKEN = object()


def set_stream_recompute(model: Any, enabled: bool, *, retain: Optional[int] = None) -> int:
    """training.stream_recompute — switch every streamed decoder layer between the
    checkpoint recompute (True, the default) and retained activations (False).
    Returns the number of streamed layers found.

    ``retain`` (only with ``enabled=False``): how many layers skip the recompute
    (the LAST ``retain`` layers); the rest keep the checkpoint recompute.
    ``None`` = all layers. Peak VRAM grows by one layer's activations per
    retained layer, whichever layers they are. A retained layer's NF4 linears
    save only the packed bytes for the backward (``packed_dequant_nf4_linear``)."""
    if retain is not None and (
        isinstance(retain, bool) or not isinstance(retain, int) or retain < 0
    ):
        raise ValueError(f"retain must be None or an int >= 0; got {retain!r}")
    streamed = streamed_decoder_layers(model)
    order = [module.idx for module in streamed]
    kept = set(order if retain is None else (order[-retain:] if retain > 0 else []))
    for module in streamed:
        skip = not enabled and module.idx in kept
        _set_layer_recompute(module, skip=skip)
        module._bwd_visited = True
    return len(streamed)


def streamed_decoder_layers(model: Any) -> list:
    """Every ``StreamedDecoderLayer`` under ``model``, sorted by layer index.

    The wrapper is transparent to ``named_modules()`` (it yields only its inner
    layer's modules), so it is reached through its parent's ``_modules``.
    """
    cls = _streamed_layer_class()
    streamed = []
    seen = set()
    for parent in model.modules():
        for child in parent._modules.values():
            if isinstance(child, cls) and id(child) not in seen:
                seen.add(id(child))
                streamed.append(child)
    return sorted(streamed, key=lambda module: module.idx)


def _linear4bit_class() -> Any:
    """``bitsandbytes.nn.Linear4bit``. Only reached for a layer whose NF4
    linears were patched, which already required bitsandbytes."""
    import bitsandbytes as bnb

    return bnb.nn.Linear4bit


def _set_layer_recompute(module: Any, *, skip: bool) -> None:
    """One streamed layer: ``skip`` = no checkpoint recompute. Its NF4 linears
    then save the packed bytes, not the dense weight, for the backward. A layer
    with no patched NF4 linear (``n_dequant_forward == 0``) has none to flag, and
    bitsandbytes is not imported for it."""
    module.recompute = not skip
    if not getattr(module, "n_dequant_forward", 0):
        return
    linear4bit = _linear4bit_class()
    for child in module.inner.modules():
        if isinstance(child, linear4bit):
            child._soup_packed_backward = skip


def _output_nbytes(output: Any) -> int:
    """Bytes of every tensor in a layer's output (a tensor, or nested containers)."""
    if hasattr(output, "untyped_storage") and hasattr(output, "element_size"):
        return int(output.numel() * output.element_size())
    if isinstance(output, Mapping):
        return sum(_output_nbytes(value) for value in output.values())
    if isinstance(output, (tuple, list)):
        return sum(_output_nbytes(value) for value in output)
    return 0


class RetainedLayerProbe:
    """training.stream_recompute='auto' — what ONE streamed layer keeps allocated
    when it skips the checkpoint recompute.

    ``arm(model)`` switches the LAST streamed layer to no-recompute for its next
    grad-enabled forward only, and that forward reads the allocator before and
    after the layer's body: ``after - before - output bytes`` is what the layer
    holds for its backward. The input was allocated before and is saved by the
    recompute path too, so it is not counted; pool-slot tensors are saved as
    tokens and cost nothing. Bytes the NF4 scale cache fills during that forward
    are persistent, not retained, and are subtracted. Allocated memory is read,
    never a difference of peaks: peaks are not additive, and a peak difference
    under-estimated a Mistral-7B layer 3x at 1024 tokens (benchmarks/exp-1c-
    streaming-without-recompute.md, section 6). The layer's mode is restored
    right after the measured forward; its backward still runs the no-recompute
    path, bit-identical to the recompute.

    ``allocated`` reads the allocator; None = ``torch.cuda.memory_allocated`` on
    the layer's device, which needs a CUDA pool. Every failure is recorded in
    ``error`` instead of being raised into the training step.
    """

    def __init__(self, allocated: Optional[Callable[[], int]] = None):
        self._allocated = allocated
        self.per_layer_bytes: Optional[int] = None
        self.layer_idx: Optional[int] = None
        self.error: Optional[str] = None
        self._before: Optional[int] = None
        self._cache_before = 0

    def arm(self, model: Any) -> Optional[int]:
        """Arm the last streamed layer; returns its index, or None (``error`` says why)."""
        layers = streamed_decoder_layers(model)
        if not layers:
            self.error = "the model has no streamed decoder layers"
            return None
        last = layers[-1]
        if self._allocated is None:
            pool = getattr(last, "pool", None)
            if not getattr(pool, "is_cuda", False):
                self.error = "no CUDA device to measure the retained bytes on"
                return None
            import torch

            device = pool.device
            self._allocated = lambda: int(torch.cuda.memory_allocated(device))
        _set_layer_recompute(last, skip=True)
        last._retained_probe = self
        self.error = None
        return last.idx

    def disarm(self, model: Any) -> None:
        """Drop an armed probe whose forward never ran, restoring that layer."""
        for layer in streamed_decoder_layers(model):
            if layer._retained_probe is self:
                layer._retained_probe = None
                _set_layer_recompute(layer, skip=False)

    @staticmethod
    def _cache_bytes(layer: Any) -> int:
        return int(getattr(getattr(layer, "scale_cache", None), "nbytes", 0) or 0)

    def start(self, layer: Any) -> None:
        try:
            self._cache_before = self._cache_bytes(layer)
            self._before = int(self._allocated())
        except Exception as exc:  # noqa: BLE001 — a measurement must not stop training
            self._before = None
            self.error = f"the measurement raised {type(exc).__name__}: {exc}"

    def finish(self, layer: Any, output: Any) -> None:
        try:
            if self._before is not None:
                after = int(self._allocated())
                filled = self._cache_bytes(layer) - self._cache_before
                self.per_layer_bytes = after - self._before - filled - _output_nbytes(output)
                self.layer_idx = layer.idx
        except Exception as exc:  # noqa: BLE001 — a measurement must not stop training
            self.per_layer_bytes = None
            self.error = f"the measurement raised {type(exc).__name__}: {exc}"
        finally:
            # The flags are read in the forward, so restoring them now leaves this
            # step's backward on the no-recompute path it was recorded for.
            _set_layer_recompute(layer, skip=False)


def _fmt_bytes(count: int) -> str:
    if abs(count) >= 10**9:
        return f"{count / 1e9:.2f} GB"
    if abs(count) >= 10**6:
        return f"{count / 1e6:.1f} MB"
    return f"{count} bytes"


class CudaMemoryReader:
    """The allocator figures ``stream_recompute='auto'`` reads, on one CUDA device."""

    def __init__(self, device: Any):
        self.device = device

    def reset_peak(self) -> None:  # pragma: no cover - CUDA only
        import torch

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def synchronize(self) -> None:  # pragma: no cover - CUDA only
        import torch

        torch.cuda.synchronize(self.device)

    def allocated(self) -> int:  # pragma: no cover - CUDA only
        import torch

        return int(torch.cuda.memory_allocated(self.device))

    def peak_allocated(self) -> int:  # pragma: no cover - CUDA only
        import torch

        return int(torch.cuda.max_memory_allocated(self.device))

    def peak_reserved(self) -> int:  # pragma: no cover - CUDA only
        import torch

        return int(torch.cuda.max_memory_reserved(self.device))

    def reserved(self) -> int:  # pragma: no cover - CUDA only
        import torch

        return int(torch.cuda.memory_reserved(self.device))

    def free(self) -> int:  # pragma: no cover - CUDA only
        import torch

        return int(torch.cuda.mem_get_info(self.device)[0])

    def total(self) -> int:  # pragma: no cover - CUDA only
        import torch

        return int(torch.cuda.get_device_properties(self.device).total_memory)

    def release(self) -> None:  # pragma: no cover - CUDA only
        import torch

        torch.cuda.empty_cache()


def _synthetic_step(model: Any, *, rows: int, seq_len: int, vocab_size: int, device: Any) -> None:
    """One forward+backward on random ids at the configured shape, gradients dropped.

    Synthetic for the reason :func:`measure_step_peak_bytes` gives: the CONFIGURED
    shape is the worst case any real batch can pad up to, a real batch measures
    whatever length it happened to have.
    """
    import torch

    ids = torch.randint(0, vocab_size, (rows, seq_len), device=device)
    out = None
    try:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
    finally:
        del ids, out
        _zero_probe_grads(model)


def plan_stream_recompute_auto(
    model: Any,
    *,
    rows: int,
    seq_len: int,
    vocab_size: int,
    notify: Callable[[str], Any],
    override_bytes: Optional[int] = None,
    headroom: Optional[float] = None,
    memory: Any = None,
    step: Optional[Callable[..., None]] = None,
) -> int:
    """training.stream_recompute='auto' — decide, before training, how many of the
    LAST streamed layers keep their activations, and switch them. Returns ``K``.

    One synthetic forward+backward at the configured shape (``rows`` x
    ``seq_len``, the shape the VRAM probe measures) runs with the checkpoint
    recompute on every layer but the last, which a :class:`RetainedLayerProbe`
    measures: its allocated bytes, charged at the step's reserved/allocated ratio
    when that is above 1. The step's RESERVED peak (which therefore already holds
    the measured layer, so ``K`` errs low by up to one) is taken from
    :func:`~soup_cli.utils.layer_stream.stream_recompute_budget`'s cap, and ``K =
    budget // charge`` clamped to the layer count. Anything that cannot be
    measured keeps the checkpoint recompute everywhere and says why; that
    includes an out-of-memory in the synthetic step. Any other exception of the
    step propagates, as the VRAM probe treats one (:func:`_classify_probe_exception`).

    ``memory`` reads the allocator (:class:`CudaMemoryReader` on the streamed
    layers' device when None, which needs a CUDA pool); ``step`` runs the
    synthetic step (:func:`_synthetic_step` when None).
    """
    from soup_cli.utils.layer_stream import (
        RECOMPUTE_CAP_CARD,
        RECOMPUTE_CAP_FREE,
        STREAM_RECOMPUTE_HEADROOM,
        plan_retained_layers,
        retained_layer_charge,
        stream_recompute_budget,
    )
    from soup_cli.utils.terminal import for_terminal

    headroom = STREAM_RECOMPUTE_HEADROOM if headroom is None else headroom
    layers = streamed_decoder_layers(model)
    n_layers = len(layers)

    def keep(why: str) -> int:
        set_stream_recompute(model, True)
        notify(
            f"[yellow]training.stream_recompute='auto' keeps the recompute on all "
            f"{n_layers} streamed layers:[/] {for_terminal(why)}"
        )
        return 0

    if not layers:
        return keep("the model has no streamed decoder layers")
    if min(rows, seq_len, vocab_size) < 1:
        return keep(
            f"no step shape to measure (batch {rows} x seq {seq_len}, vocab_size {vocab_size})"
        )
    device = layers[-1].pool.device
    if memory is None:
        if not getattr(layers[-1].pool, "is_cuda", False):
            return keep("no CUDA device to measure VRAM on")
        memory = CudaMemoryReader(device)
    run = _synthetic_step if step is None else step
    shape = f"batch {rows} x seq {seq_len}"
    probe = RetainedLayerProbe(allocated=memory.allocated)
    probe.arm(model)
    try:
        memory.reset_peak()
        run(model, rows=rows, seq_len=seq_len, vocab_size=vocab_size, device=device)
        memory.synchronize()
        peak_reserved = memory.peak_reserved()
        peak_allocated = memory.peak_allocated()
        free = memory.free()
        reserved = memory.reserved()
        total = memory.total()
    except Exception as exc:
        # An out-of-memory is a result, as it is for the VRAM probe: this shape
        # does not fit even with one layer retained. Anything else (an illegal
        # access, a device assert, a bug) may leave the context unusable, so it
        # propagates and the setup closes the runtime and refuses the run.
        if not _is_out_of_memory(exc):
            raise
        return keep(
            f"the synthetic step at {shape} ran out of memory, so the configured shape "
            "does not fit with one layer retained; set training.stream_vram_probe: true "
            "to measure whether it fits at all"
        )
    finally:
        probe.disarm(model)
        try:
            memory.release()
        except Exception:  # noqa: BLE001 — a cache flush must not mask the result
            logger.warning("stream_recompute='auto': releasing the allocator cache failed")
    per_layer = probe.per_layer_bytes
    if per_layer is None:
        return keep(probe.error or "the measured forward did not run")
    if per_layer <= 0:
        return keep(f"one layer measured at {_fmt_bytes(per_layer)} retained (<= 0)")
    charge = retained_layer_charge(per_layer, peak_reserved, peak_allocated)
    budget = stream_recompute_budget(
        total_bytes=total,
        free_bytes=free,
        reserved_bytes=reserved,
        reserved_peak_bytes=peak_reserved,
        override_bytes=override_bytes,
        headroom=headroom,
    )
    if budget.bound == RECOMPUTE_CAP_CARD:
        cap = f"{headroom:.0%} of the card's {_fmt_bytes(total)}"
    elif budget.bound == RECOMPUTE_CAP_FREE:
        cap = f"free + reserved VRAM {_fmt_bytes(free + reserved)}"
    else:
        cap = f"training.stream_vram_override {_fmt_bytes(budget.cap_bytes)}"
    if budget.budget_bytes <= 0:
        return keep(
            f"no VRAM headroom: {cap} minus the reserved peak of the {shape} step, "
            f"{_fmt_bytes(peak_reserved)}, is {_fmt_bytes(budget.budget_bytes)}"
        )
    k = plan_retained_layers(budget.budget_bytes, charge, n_layers)
    if k == 0:
        return keep(
            f"one layer ({_fmt_bytes(charge)} charged) does not fit the budget of "
            f"{_fmt_bytes(budget.budget_bytes)}"
        )
    set_stream_recompute(model, False, retain=k)
    ratio = peak_reserved / peak_allocated if peak_allocated > 0 else 1.0
    notify(
        f"[cyan]training.stream_recompute='auto':[/] the last {k} of {n_layers} streamed "
        f"layers keep their activations, the rest recompute (measured at {shape}: one "
        f"layer {_fmt_bytes(per_layer)} retained, charged {_fmt_bytes(charge)} at "
        f"reserved/allocated {ratio:.2f}; budget {_fmt_bytes(budget.budget_bytes)} = "
        f"{cap} minus the reserved peak {_fmt_bytes(peak_reserved)})"
    )
    return k


def _streamed_layer_class():
    global _STREAMED_LAYER_CLASS
    if _STREAMED_LAYER_CLASS is None:
        _STREAMED_LAYER_CLASS = _build_streamed_layer_class()
    return _STREAMED_LAYER_CLASS


class _StreamedDecoderLayerProxy:
    """Callable shim so ``StreamedDecoderLayer(...)`` works as a name."""

    def __call__(self, *args, **kwargs):
        return _streamed_layer_class()(*args, **kwargs)

    def __instancecheck__(self, instance):
        return isinstance(instance, _streamed_layer_class())


StreamedDecoderLayer = _StreamedDecoderLayerProxy()


def _large_layer_weight_param_name(inner: Any) -> str:
    """The meta parameter name ``StreamedLargeLayer.forward`` must substitute.

    Normally ``"weight"``. get_peft_model() runs before install_streaming(),
    so a LoRA target on the input/output embedding wraps ``inner`` in a peft
    tuner layer first (#1012): the real, still-meta base weight moves to
    ``inner.base_layer.weight`` (``inner`` is the leaf embedding/head module
    itself here, so the prefix carries no leading dot, unlike a nested
    decoder-layer projection), and peft's own ``weight`` exposes a READ-ONLY
    property delegating to it, which a plain ``{"weight": ...}`` substitution
    cannot setattr onto. ``_layer_name_map`` resolves the same indirection one
    level down, for decoder-layer projections (q_proj/v_proj); this mirrors
    it for the large-layer boundary modules.
    """
    for pname, param in inner.named_parameters(recurse=True):
        if param.is_meta and pname in ("weight", "base_layer.weight"):
            return pname
    return "weight"


def _unwrap_tuner_base(module: Any) -> Any:
    """The real module under any peft tuner wrapper (#1012 wraps a LoRA target)."""
    seen = 0
    while hasattr(module, "base_layer") and seen < 8:
        module = module.base_layer
        seen += 1
    return module


def _refuse_trainable_streamed_boundary(model: Any, index: Any, lora_config: Any) -> None:
    """An externally supplied modules_to_save must not unfreeze a streamed base.

    Soup exposes LoRA targets, not modules_to_save. Guard the lower-level build
    API before PEFT wraps/copies these meta modules, where it would fail late.
    """
    targets = getattr(lora_config, "modules_to_save", None)
    if not targets or not tuple(getattr(index, "large_keys", ()) or ()):
        return
    if isinstance(targets, str):
        targets = (targets,)
    boundaries = (model.get_input_embeddings(), model.get_output_embeddings())
    for name, module in model.named_modules():
        if any(module is boundary for boundary in boundaries) and any(
            name == target or name.endswith("." + target) for target in targets
        ):
            raise ValueError(
                f"modules_to_save targets the frozen streamed boundary {name!r}; "
                "use a LoRA target for this module or train without stream_layers"
            )


def _build_streamed_large_layer_class():
    import torch
    import torch.nn as nn
    from torch.func import functional_call

    class StreamedLargeLayer(nn.Module):
        """Embedding or output projection backed by the shared large slot."""

        def __init__(
            self,
            inner: Any,
            key: str,
            pool: Any,
            refill_before_backward: bool = False,
            weight_path: Optional[str] = None,
        ) -> None:
            super().__init__()
            self.inner = inner
            self.key = str(key)
            self.pool = pool
            self.weight_path = weight_path
            # #1049 -- set by the trainers whose loss runs a second forward before
            # the step's backward (see `_needs_a_private_weight`).
            self.refill_before_backward = bool(refill_before_backward)
            self._register_load_state_dict_pre_hook(self._redirect_canonical_weight)
            self._weight_param_name = _large_layer_weight_param_name(inner)

        def _redirect_canonical_weight(
            self, state_dict: Any, prefix: str, *_args: Any, **_kwargs: Any
        ) -> None:
            # Mirrors StreamedDecoderLayer._redirect_canonical_keys: redirect
            # every key under this prefix, not only the literal "weight" one
            # (#1048: a peft-wrapped inner saves lora_A/lora_B keys too).
            inner_prefix = prefix + "inner."
            for key in [
                k for k in state_dict if k.startswith(prefix) and not k.startswith(inner_prefix)
            ]:
                redirected = inner_prefix + key[len(prefix) :]
                value = state_dict.pop(key)
                if redirected in state_dict:
                    raise ValueError(
                        f"checkpoint contains both {key!r} and {redirected!r} "
                        f"for the same streamed large-layer weight; it is "
                        f"malformed, re-save the adapter"
                    )
                state_dict[redirected] = value
            # PEFT's embedding export may include the frozen base alongside
            # the adapter. It must agree with the shard feeding forward: a
            # torch copy into a meta parameter would silently discard a
            # different base. Keep the placeholder, even with assign=True.
            weight_key = inner_prefix + self._weight_param_name
            value = state_dict.get(weight_key)
            if self.weight_path is not None and value is not None and not value.is_meta:
                if not torch.equal(value.detach().cpu(), self._frozen_weight_snapshot()):
                    raise ValueError(
                        f"checkpoint frozen base for {self.key!r} differs from the "
                        "streaming shard; resume with the original base checkpoint "
                        "and its matching shard cache"
                    )
                state_dict[weight_key] = self.inner.get_parameter(self._weight_param_name).detach()

        def _frozen_weight_snapshot(self) -> Any:
            from safetensors import safe_open

            # Never read pool.wait(): input and output share storage and the
            # slot may currently hold the other boundary (or an async copy).
            # Reading the shard also works independently of a disk-tier
            # source's live staging buffers. Clone before closing its mapping.
            with safe_open(self.weight_path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(self.key).clone()

        def _apply(self, fn: Any, recurse: bool = True) -> Any:
            def _skip_meta(tensor: Any) -> Any:
                return tensor if getattr(tensor, "is_meta", False) else fn(tensor)

            # #1005 — `named_children()` no longer lists `inner`; reach it
            # explicitly (its `weight` is the meta placeholder `_skip_meta`
            # passes through).
            if recurse:
                self.inner._apply(_skip_meta, recurse=True)
            return super()._apply(_skip_meta, recurse=False)

        # #1005 — canonical NAMES, mirroring StreamedDecoderLayer: the inner
        # module (an Embedding or a Linear, which owns `weight` directly) is
        # enumerated under this wrapper's own prefix in place of the wrapper,
        # so `named_parameters()` says `model.embed_tokens.weight` exactly as
        # `state_dict()` does.
        #
        # ONE documented exception (found in #1010's review): the inner module
        # has no children, so `named_children()` yields nothing and this
        # wrapper owns no `_parameters` — transformers' `get_parameter_names`
        # (the weight-decay grouping, which walks `named_children()` and reads
        # each visited module's own `_parameters`) therefore never lists the
        # streamed large weight, where a resident model lists it. That is
        # safe, and only because of WHAT the weight is: a frozen meta
        # placeholder (`requires_grad=False`; the large slot serves the real
        # bytes), and `Trainer.create_optimizer` filters every group on
        # `requires_grad`, so the optimizer groups equal a resident model's
        # exactly — pinned, with the exact name difference, by
        # tests/test_issue1005_peft021_review_round.py. The alternative,
        # registering `inner.weight` on this wrapper as well, is WRONG rather
        # than untidy: `_load_from_state_dict` would then report the canonical
        # key missing, because the pre-hook above has already renamed it into
        # `.inner.` by the time the wrapper's own parameters are checked.
        def named_children(self) -> Any:
            yield from self.inner.named_children()

        def named_modules(
            self, memo: Any = None, prefix: str = "", remove_duplicate: bool = True
        ) -> Any:
            if memo is None:
                memo = set()
            if remove_duplicate:
                if self in memo:
                    return
                memo.add(self)
            yield from self.inner.named_modules(memo, prefix, remove_duplicate)

        def train(self, mode: bool = True) -> Any:
            if not isinstance(mode, bool):
                raise ValueError("training mode is expected to be boolean")
            self.training = mode
            self.inner.train(mode)
            return self

        def apply(self, fn: Any) -> Any:
            self.inner.apply(fn)
            fn(self)
            return self

        def state_dict(
            self,
            *args: Any,
            destination: Any = None,
            prefix: str = "",
            keep_vars: bool = False,
        ) -> Any:
            if args:
                if destination is None:
                    destination = args[0]
                if len(args) > 1 and prefix == "":
                    prefix = args[1]
                if len(args) > 2 and keep_vars is False:
                    keep_vars = args[2]
            state = self.inner.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
            if self.weight_path is not None and hasattr(self.inner, "base_layer"):
                # PEFT obtains state_dict BEFORE deciding auto/True/False for
                # save_embedding_layers. Only tuner-wrapped boundaries need
                # independent CPU bytes; leave untargeted weights on meta and
                # avoid reading their vocabulary-sized shards on every save.
                state[prefix + self._weight_param_name] = self._frozen_weight_snapshot()
            return state

        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name == "inner":
                    raise
                inner = self._modules.get("inner")
                if inner is None:
                    raise
                return getattr(inner, name)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            weight = self.pool.wait(self.key)
            if torch.is_grad_enabled() and self._needs_a_private_weight():
                # #1049 -- embed_tokens and an untied lm_head SHARE one slot, and
                # `wait` returns a view of it, so the next `load_async` refills the
                # bytes autograd is holding. The projection saves its weight to build
                # `grad wrt x`, so a second forward in the same step -- the reference
                # pass of a preference loss -- bumped that tensor's version and the
                # policy backward died with "modified by an inplace operation".
                # Hand autograd a private copy: one head-sized allocation, alive only
                # while the graph is, and only when a second forward can refill the
                # slot before the backward. The VRAM pre-flight charges it.
                weight = weight.clone()
            return functional_call(
                self.inner,
                {self._weight_param_name: weight},
                args,
                kwargs,
            )

        def _needs_a_private_weight(self) -> bool:
            """True when this layer's weight can be refilled while autograd holds it.

            Three things have to be true. The loss must run a second forward
            before the step's backward -- the reference pass of a preference
            loss. SFT, ORPO and SimPO run one forward per step, and nothing
            refills the slot between it and its backward, so they never need the
            copy (``refill_before_backward``, set by the trainer). The slot must
            be SHARED: a tied checkpoint streams no large key, so its buffer is never
            refilled within a step. And the weight must be one autograd can save,
            which an embedding's is not -- ``embedding_backward`` works from the
            indices and the vocabulary size, never from the weight values, so
            nothing holds those bytes past the lookup. That leaves the
            projection, i.e. the untied ``lm_head``, and keeps the cost at one
            head-sized copy rather than two.
            """
            if not self.refill_before_backward:
                return False
            specs = getattr(self.pool, "specs", None)
            if specs is None or len(specs) <= 1:
                return False
            return not isinstance(_unwrap_tuner_base(self.inner), nn.Embedding)

    return StreamedLargeLayer


_STREAMED_LARGE_LAYER_CLASS = None


def _streamed_large_layer_class():
    global _STREAMED_LARGE_LAYER_CLASS
    if _STREAMED_LARGE_LAYER_CLASS is None:
        _STREAMED_LARGE_LAYER_CLASS = _build_streamed_large_layer_class()
    return _STREAMED_LARGE_LAYER_CLASS


def _replace_module_references(root: Any, target: Any, replacement: Any) -> int:
    """Replace every child reference to ``target`` without relying on its path."""
    replaced = 0
    # Walk the graph STRUCTURALLY (`_modules`, by identity) rather than through
    # `root.modules()`: since #1005 a streamed wrapper enumerates its inner
    # layer in its own place, so `modules()` never visits a wrapper and would
    # skip its `_modules` — inert for today's callers (embed_tokens / lm_head
    # hang off top-level containers, and every call is guarded by
    # `if not _replace_module_references(...): raise`), but the docstring's
    # promise should not depend on that. Snapshot before mutating: visiting
    # ``replacement`` after the first assignment would replace its own
    # ``inner`` reference and make the wrapper its own child.
    seen: set = set()
    stack = [root]
    snapshot = []
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        snapshot.append(module)
        stack.extend(child for child in module._modules.values() if child is not None)
    for module in snapshot:
        # ``named_children()`` removes duplicate modules, but tied architectures
        # may expose the same boundary module through more than one attribute.
        for name, child in tuple(module._modules.items()):
            if child is target:
                setattr(module, name, replacement)
                replaced += 1
    return replaced


def _attach_streamed_boundary_adapter_bridge(model: Any) -> None:
    """Let PEFT inject/load adapters without changing streaming placement.

    PEFT resolves a target through its parent attribute, which is our streamed
    wrapper rather than the canonical tuner yielded by named_modules(). TRL
    DPO uses add_adapter to make its reference. Expose the tuner only during
    injection and restore streaming even if PEFT raises partway through it.
    PEFT's load_adapter otherwise re-plans CPU/disk models across available
    accelerators, which cannot move this runtime's independently owned pools.
    This is scoped to this model, never a patch of PEFT's global classes.
    """
    if not hasattr(model, "peft_config"):
        return
    from types import MethodType

    add_adapter = type(model).add_adapter
    load_adapter = type(model).load_adapter

    def _load_adapter(
        self: Any, model_id: Any, adapter_name: str, is_trainable: bool = False,
        torch_device: Optional[str] = None, *args: Any, **kwargs: Any,
    ) -> Any:
        runtime_map = getattr(self, "hf_device_map", None)
        if isinstance(runtime_map, dict) and "" in runtime_map:
            if torch_device is None:
                device = runtime_map[""]
                torch_device = f"cuda:{device}" if isinstance(device, int) else str(device)
            kwargs.setdefault("device_map", dict(runtime_map))
        return load_adapter(
            self, model_id, adapter_name, is_trainable, torch_device, *args, **kwargs
        )

    def _add_adapter(self: Any, *args: Any, **kwargs: Any) -> Any:
        boundaries = {}
        for getter in (self.get_input_embeddings, self.get_output_embeddings):
            wrapper = getter()
            if isinstance(wrapper, _streamed_large_layer_class()):
                boundaries[id(wrapper)] = (wrapper, [])
        seen = set()
        stack = [self]
        while stack:
            parent = stack.pop()
            if id(parent) in seen:
                continue
            seen.add(id(parent))
            for name, child in parent._modules.items():
                if id(child) in boundaries:
                    boundaries[id(child)][1].append((parent, name))
                elif child is not None:
                    stack.append(child)
        for wrapper, references in boundaries.values():
            for parent, name in references:
                parent._modules[name] = wrapper.inner
        try:
            return add_adapter(self, *args, **kwargs)
        finally:
            for wrapper, references in boundaries.values():
                if not references:
                    continue
                parent, name = references[0]
                inner = parent._modules[name]
                wrapper.inner = inner
                wrapper._weight_param_name = _large_layer_weight_param_name(inner)
                for parent, name in references:
                    parent._modules[name] = wrapper

    model.add_adapter = MethodType(_add_adapter, model)
    model.load_adapter = MethodType(_load_adapter, model)


def canonical_named_parameters(model: Any) -> Iterator[Tuple[str, Any]]:
    """Yield ``(name, param)`` with any wrapper ``.inner.`` segment stripped,
    matching the spelling ``StreamedDecoderLayer.state_dict()`` uses (v0.72.1).

    Since #1005 a streamed model's ``named_parameters()`` is canonical on its
    own (the wrapper reports the inner layer's tree at its own prefix), so this
    is an identity for it; it is kept for its callers and for any tree that
    still nests a wrapped module under a child named ``inner``.
    """
    for name, param in model.named_parameters():
        yield name.replace(".inner.", "."), param


def assert_canonical_parameters_intersect(model_a: Any, model_b: Any) -> FrozenSet[str]:
    """Raise if two models (either may be streamed) share no canonical
    parameter name. A per-parameter comparison (gradients, weights, any other
    property) can only walk names both sides have; if the intersection is
    empty, a loop over it is vacuously satisfied and reports success on
    nothing compared. Returns the intersecting name set on success so a
    caller need not recompute it.
    """
    names_a = frozenset(name for name, _ in canonical_named_parameters(model_a))
    names_b = frozenset(name for name, _ in canonical_named_parameters(model_b))
    shared = names_a & names_b
    if not shared:
        raise ValueError(
            f"no canonical parameter name is shared between the two models "
            f"({len(names_a)} vs {len(names_b)} parameters); a comparison "
            f"over an empty intersection is not a comparison"
        )
    return shared


# ==========================================================================
# allocator
# ==========================================================================
@dataclass(frozen=True)
class GemmCeiling:
    """A dense GEMM rate measured on THIS card, in THIS session."""

    tflops: float
    sm_clock_mhz: Optional[int]
    size: int
    dtype: str
    samples: tuple[float, ...] = ()


def sm_clock_mhz() -> Optional[int]:
    """Current SM clock, or None when it cannot be read.

    Quoted next to every throughput number: a fraction-of-ceiling without a
    stated clock is meaningless, and this box's boost clock moved 442..952 MHz
    inside one measurement run.
    """
    import subprocess

    from soup_cli.utils.layer_stream import _resolve_tool

    # Run by the absolute path found in an absolute PATH directory, never by the
    # bare name: on Windows a bare name is looked up in the current directory
    # before PATH, so the nvidia-smi run would depend on where soup was started.
    tool = _resolve_tool("nvidia-smi")
    if tool is None:
        return None
    try:
        out = subprocess.run(
            [tool, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


#: 4096^3 per matmul (~137 GFLOP, ~20 ms here) and best-of-3. Both were measured,
#: not chosen for tidiness: at 2048 the same probe spread **38%** across five
#: repeats and ramped monotonically upward (3.19 -> 4.41 TFLOPS) because the
#: sample is too short for the boost clock to engage; at 4096 the spread fell to
#: 9%. Taking the BEST repeat is not cherry-picking — a ceiling's noise is
#: one-sided, since contention, a cold clock and thermal throttling can only ever
#: make an achievable rate look slower than it is.
_GEMM_SIZE = 4096
_GEMM_REPS = 3
_GEMM_ITERS = 8


def measure_gemm_tflops(
    device: str = "cuda",
    *,
    size: int = _GEMM_SIZE,
    iters: int = _GEMM_ITERS,
    reps: int = _GEMM_REPS,
) -> Optional[GemmCeiling]:
    """Benchmark a dense matmul using this card's resolved stream dtype.

    Returns None off CUDA rather than inventing a number — the forecast rests
    entirely on a measurement, so there is nothing honest to return when no
    measurement is possible. A per-card constant compiled into the source would
    be a fabrication: this box alone produced 3.5 and 6.75 TFLOPS in two sessions
    at the same *reported* clock, which is precisely why the number is taken now
    and quoted as a bound rather than a promise.

    Costs ~0.7 s and ~100 MB, once, before the model is built.
    """
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a [train] extra
        return None
    from soup_cli.utils.layer_stream import resolve_stream_dtype

    if not torch.cuda.is_available():
        return None
    dtype_name = resolve_stream_dtype(device)
    dtype = getattr(torch, dtype_name)
    samples: list[float] = []
    try:
        left = torch.randn(size, size, device=device, dtype=dtype)
        right = torch.randn(size, size, device=device, dtype=dtype)
        for _ in range(max(1, reps)):
            for _ in range(3):  # warm up: the first matmul pays kernel selection
                left @ right
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                left @ right
            end.record()
            torch.cuda.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
            if seconds > 0:
                samples.append(2.0 * (size**3) * iters / seconds / 1e12)
        del left, right
        torch.cuda.empty_cache()
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return None
    if not samples:
        return None
    best = max(samples)
    if best <= 0:
        return None
    return GemmCeiling(
        tflops=best,
        sm_clock_mhz=sm_clock_mhz(),
        size=size,
        dtype=dtype_name,
        samples=tuple(samples),
    )


@dataclass(frozen=True)
class StepPeak:
    """Peak VRAM of ONE real forward+backward, measured on THIS model and shape.

    Three outcomes, deliberately not two. ``oom`` is a measurement RESULT ("this
    shape does not fit"). ``failed`` means the probe ran a real CUDA op and that
    op raised — the fit is unknown AND the context may no longer be usable, so
    proceeding is not safe merely because the formula was happy. A ``None``
    return from :func:`measure_step_peak_bytes` is the third: the probe was never
    attempted (no CUDA, no torch), nothing was touched, and falling back to the
    prediction is correct. Collapsing ``failed`` into ``None`` would let a run
    continue past a broken CUDA context on the strength of arithmetic.
    """

    peak_bytes: int
    reserved_bytes: int
    seconds: float
    rows: int
    seq_len: int
    oom: bool = False
    failed: bool = False
    error: Optional[str] = None


def _classify_probe_exception(
    exc: BaseException,
    *,
    rows: int,
    seq_len: int,
    seconds: float = 0.0,
) -> StepPeak:
    """Map a probe-step failure to its StepPeak outcome without requiring CUDA.

    An out-of-memory exception (allocator OOM or an asynchronous CUDA OOM
    surfacing via AcceleratorError/RuntimeError) is a result (oom=True), not an
    instrument failure. Any other exception is an instrument failure that may
    leave the device context poisoned (failed=True).
    """
    if _is_out_of_memory(exc):
        # A result, not a failure: the shape provably does not fit. That is
        # the allocator's own `torch.OutOfMemoryError` on Linux, and — #649's
        # shape, seen again in #901 — under WDDM, where the allocator has
        # often already spilled, an `AcceleratorError("CUDA error: out of
        # memory")` surfacing later at a synchronise. One spelling for both:
        # the verdict is what tells the operator to lower batch or
        # max_length, where "instrument failure" tells them nothing they can
        # act on. (A WDDM run that spills WITHOUT raising reaches the success
        # path with a peak above free VRAM, which the caller refuses too.)
        return StepPeak(
            peak_bytes=0,
            reserved_bytes=0,
            seconds=seconds,
            rows=rows,
            seq_len=seq_len,
            oom=True,
        )
    # NOT `return None`. None means "never attempted"; this op ran and broke,
    # which can leave the CUDA context poisoned (an illegal access or device
    # assert surfaces exactly here). Reporting that as "cannot tell" would
    # let the caller proceed on the prediction alone, into a context that may
    # no longer work.
    logger.warning("stream VRAM probe raised during the step: %r", exc)
    return StepPeak(
        peak_bytes=0,
        reserved_bytes=0,
        seconds=seconds,
        rows=rows,
        seq_len=seq_len,
        failed=True,
        error=type(exc).__name__,
    )


def measure_step_peak_bytes(
    model: Any,
    *,
    rows: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda",
) -> Optional[StepPeak]:
    """Run one forward+backward at the configured shape and read the real peak.

    This is the #349 instrument. The pre-flight's formula is fitted and, measured
    against the real training run, under-predicts past seq 4352 (0.934x the real
    peak at seq 5120, 0.787x at 6144); against this probe the same formula reads
    0.992x at seq 4096 and 0.830x at 5120, because the probe runs 12.5-14.3%
    above the real training step — conservatively high, the direction that makes
    it safe as a gate. A formula cannot model a term nobody has identified, so
    past that point only a measurement is honest. Synthetic
    ``input_ids`` are used rather than a real batch because the quantity being
    bounded is the CONFIGURED shape, which is the worst case any real batch can
    pad up to; a real batch would measure whatever length it happened to have.
    Validated against a full ``soup train`` run of the same config: probe
    4.3117 GB reserved vs the real run's 4.3159 GB, 0.1% apart.

    Costs one step. Measured on an RTX 3050 Laptop: 1.02-1.15 s for
    SmolLM2-135M at 1x1024, 5.09 s at 2x2048, and 5.33 s for Llama-3.1-8B NF4 at
    1x512 — against a training run of minutes to hours.

    Returns ``None`` off CUDA or when the probe itself breaks, rather than
    raising: a pre-flight that dies because its own instrument failed is worse
    than one that falls back to the shipped prediction. That is the same contract
    :func:`~soup_cli.utils.layer_stream.measure_logits_loss_bytes_per_element`
    already keeps.
    """
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a [train] extra
        return None
    if not torch.cuda.is_available():
        return None
    if rows < 1 or seq_len < 1 or vocab_size < 1:
        raise ValueError(
            f"rows/seq_len/vocab_size must all be >= 1; got {rows}/{seq_len}/{vocab_size}"
        )
    ids = None
    out = None
    # Bound before the try: `synchronize()` and `reset_peak_memory_stats()` can
    # both surface an earlier async CUDA error as an OOM, and the OOM handler
    # reads `started`. Assigning it inside the try would turn that into an
    # UnboundLocalError escaping setup() instead of the clean refusal below.
    started = time.perf_counter()
    try:  # pragma: no cover - exercised only where torch + CUDA exist
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        ids = torch.randint(0, vocab_size, (rows, seq_len), device=device)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    except Exception as exc:  # pragma: no cover - a real CUDA op raised
        return _classify_probe_exception(
            exc,
            rows=rows,
            seq_len=seq_len,
            seconds=time.perf_counter() - started,
        )
    finally:  # pragma: no cover - exercised only where torch + CUDA exist
        del ids, out
        _zero_probe_grads(model)
        torch.cuda.empty_cache()
    return StepPeak(
        peak_bytes=peak,
        reserved_bytes=reserved,
        seconds=elapsed,
        rows=rows,
        seq_len=seq_len,
    )


def _zero_probe_grads(model: Any) -> None:
    """Drop the gradients the probe's backward left on the adapter.

    Without this the first real optimizer step would apply a gradient computed
    from random token ids — training on noise for one step, silently, and only
    when the probe is enabled.

    A failure here is never re-raised (it would mask the measurement the caller
    came for) but it is never silent either: swallowing it would leave exactly
    the corruption this function exists to prevent, with nothing to find it by.
    """
    try:
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
    except Exception as exc:  # pragma: no cover - never let cleanup mask the result
        logger.warning(
            "could not clear the VRAM probe's gradients (%r); the first "
            "optimizer step may include a gradient computed from random tokens",
            exc,
        )


def expandable_segments_status() -> tuple[bool, str]:
    """``(enabled, why_not)`` for the ``expandable_segments:True`` hint.

    Split out from :func:`probe_expandable_segments` because the caller used to
    print "unavailable on this platform (silently ignored on Windows)" for every
    False — and on a Colab T4, i.e. Linux, that sentence is simply untrue. There
    the hint fails for a different reason: the allocator reads
    ``PYTORCH_CUDA_ALLOC_CONF`` when the CUDA context is created, so once
    anything has touched CUDA it is too late to set it. Reporting the wrong
    cause is worse than reporting none, and this panel is the most-read output
    the feature has.
    """
    if sys.platform.startswith("win"):
        return (False, "Windows ignores it — torch warns and carries on")
    try:
        import torch
    except ImportError:
        return (False, "torch is not importable")
    if not torch.cuda.is_available():
        return (False, "no CUDA device")
    if torch.cuda.is_initialized():
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" in conf:
            return (True, "")
        if conf:
            # ``soup train`` sets the hint before CUDA starts (#2149), with setdefault,
            # so a value the user exported is what kept it out, not the ordering.
            return (False, "PYTORCH_CUDA_ALLOC_CONF is set to something else")
        return (
            False,
            "CUDA was already initialised before Soup could set it — the "
            "allocator reads PYTORCH_CUDA_ALLOC_CONF once, at context creation",
        )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        return (True, "")
    return (False, "PYTORCH_CUDA_ALLOC_CONF is set to something else")


def probe_expandable_segments() -> bool:
    """Attempt ``expandable_segments:True`` (plan P7) and report whether it took.

    Never claim it is active when it is not; :func:`expandable_segments_status`
    carries the reason.
    """
    return expandable_segments_status()[0]


# ==========================================================================
# model construction — the resident load must NEVER happen (plan P14)
# ==========================================================================
def build_nf4_config(dtype: str, *, double_quant: bool = True):
    """The BitsAndBytesConfig the streamed skeleton and the sharder share.

    ``double_quant`` defaults to True to match
    ``quant_menu.build_quantization_config_for_loader``, whose own default is
    True for every resident 4-bit load in this repo. That agreement is
    load-bearing, not cosmetic: the release's central claim is that a streamed
    NF4 run is bit-exact against a resident one, and it holds only while both
    sides quantise with the same settings. Since #321,
    ``training.bnb_4bit_use_double_quant`` IS threaded through both paths from
    the SAME config field (``stream_setup`` reads it once and passes it to the
    sharder and this skeleton together), so honouring it can no longer make the
    two sides drift apart.
    """
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_torch_dtype(dtype),
        bnb_4bit_use_double_quant=bool(double_quant),
    )


def _stamp_4bit_markers(model: Any, quant_config: Any) -> None:
    """Stamp what ``from_pretrained`` stamps — PEFT and Trainer both read these.

    ``is_loaded_in_4bit`` is THE most important finding of the NF4 gate. PEFT
    checks it to choose ``lora.bnb.Linear4bit``; a ``meta`` skeleton carries no
    such marker, so PEFT silently falls back to the generic
    ``lora.layer.Linear``. That still *runs* against a ``Linear4bit`` base — it
    just casts and accumulates differently, measured as a **9.375e-01** logit
    divergence versus resident NF4 with byte-identical weights AND adapters. No
    crash, no warning, and a loss curve that looks perfectly healthy.

    ``hf_quantizer`` is not optional either, and its absence fails LOUDLY rather
    than silently: ``transformers.Trainer.__init__`` treats
    ``is_quantized and not hf_quantizer.is_trainable`` as "this quantization
    method cannot be fine-tuned" and then formats the error message from
    ``model.hf_quantizer.quantization_config.quant_method`` — so every streaming
    run would die at trainer construction with an AttributeError about an
    attribute the user has never heard of.
    """
    from transformers.quantizers import AutoHfQuantizer
    from transformers.utils.quantization_config import QuantizationMethod

    model.is_loaded_in_4bit = True
    model.is_quantized = True
    model.quantization_method = QuantizationMethod.BITS_AND_BYTES
    # pre_quantized=True is the truth here: the shards were quantised offline by
    # the sharder, not on load. Both settings report is_trainable=True, which is
    # the property Trainer actually gates on.
    model.hf_quantizer = AutoHfQuantizer.from_config(quant_config, pre_quantized=True)
    if getattr(model, "config", None) is not None:
        model.config.quantization_config = quant_config


def _drop_fp8_source_quantization(config: Any) -> None:
    """Build the streamed skeleton at the stream dtype even over an FP8 block source.

    An FP8 block checkpoint's ``config.json`` declares ``quantization_config`` with
    ``quant_method: "fp8"``. The sharder dequantises those weights (``utils/fp8_block``),
    so the shards hold the stream dtype (or NF4) and the modules must be plain
    ``nn.Linear`` (or ``Linear4bit``): transformers' FP8 quantizer, applied from this
    config, would turn them into ``FP8Linear`` modules expecting FP8 bytes and scales.
    ``from_config`` does not apply it today (transformers 5.17), but the config would
    still claim FP8 on a model holding none, for any reader of ``model.config``.

    Only this skeleton's own config object is changed: ``config.json`` on disk and every
    resident load (``from_pretrained``, which dequantises or runs FP8 itself) are not.
    """
    from soup_cli.utils.fp8_block import is_fp8_quantization_config

    if is_fp8_quantization_config(getattr(config, "quantization_config", None)):
        del config.quantization_config


#: A CPU tensor-factory request of this many bytes or more is made on ``meta`` while the
#: skeleton is built (#2154). Smaller requests stay real, so the small buffers
#: (rotary ``inv_freq``, ``e_score_correction_bias``) keep their ``__init__`` values.
SKELETON_META_REDIRECT_BYTES = 1 << 20

_SKELETON_FACTORIES = ("empty", "zeros", "ones", "full", "empty_strided")


@contextlib.contextmanager
def _large_cpu_requests_on_meta(threshold: int) -> Iterator[None]:
    """Make every CPU factory request of ``threshold`` bytes or more on ``meta``.

    ``init_empty_weights()`` moves a parameter to ``meta`` only after it is registered,
    and a fused-expert MoE builds its parameter with ``torch.empty(num_experts,
    2 * intermediate, hidden)`` first: 22.5 GB of commit per layer for Kimi K2, never
    touched, but charged (#2154). Redirecting the request itself is what avoids it.

    A ``TorchFunctionMode``, not a patch of the ``torch`` module: torch caches the
    factories that ``with torch.device(...)`` applies to the first time a device context
    is used, and a patched module seen then (``replace_with_bnb_linear`` opens one under
    ``nf4``) would stay in that cache for the rest of the process. The mode also applies
    to this thread only, and it ends with the build however the build ends.
    """
    import torch
    from torch.overrides import TorchFunctionMode

    class _Redirect(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if getattr(func, "__name__", None) in _SKELETON_FACTORIES and "out" not in kwargs:
                device = kwargs.get("device")
                target = torch.device(device if device is not None else torch.get_default_device())
                if target.type == "cpu":
                    probe = func(*args, **{**kwargs, "device": "meta"})
                    if probe.numel() * probe.element_size() >= threshold:
                        return probe
            return func(*args, **kwargs)

    with _Redirect():
        yield


def _skeleton_from_config(
    config: Any, torch_dtype: Any, quant_config: Any, trust_remote_code: bool
) -> Any:
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    with init_empty_weights():
        # Keep the model-construction spelling aligned with the load-site
        # compatibility guard in tests/test_transformers_floor_compat.py.
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
        )
        if quant_config is not None:
            from transformers.integrations.bitsandbytes import replace_with_bnb_linear

            # lm_head stays at the base dtype: it lives in extras (and is tied
            # to the embeddings on most small Llamas), and quantising the output
            # projection costs accuracy where it is felt most.
            model = replace_with_bnb_linear(
                model,
                modules_to_not_convert=["lm_head"],
                quantization_config=quant_config,
            )
    return model


def build_meta_skeleton(
    model_id: str,
    *,
    dtype: str,
    quant: str = "none",
    double_quant: bool = True,
    trust_remote_code: bool = False,
):
    """Build the model structure on ``meta``: no weight storage is allocated.

    Under ``quant='nf4'`` the decoder linears are replaced with
    ``bnb.nn.Linear4bit`` (still on ``meta``) so the streamed ``Params4bit``
    views land in modules that know how to consume them.

    Large CPU requests are made on ``meta`` during the build
    (:func:`_large_cpu_requests_on_meta`). A buffer has to stay real, so if one
    that large comes out on ``meta`` the model is built again without the
    redirect, as before #2154, and the buffer is named.
    """
    from soup_cli.utils.kimi_k2 import load_stream_model_config
    from soup_cli.utils.layer_shard import QUANT_NF4, SUPPORTED_STREAM_QUANTS
    from soup_cli.utils.run_provenance import base_revision_kwargs

    if quant not in SUPPORTED_STREAM_QUANTS:
        raise ValueError(
            f"unsupported quant {quant!r} for layer streaming; supported: "
            f"{', '.join(SUPPORTED_STREAM_QUANTS)}"
        )
    torch_dtype = _torch_dtype(dtype)
    # AutoConfig's config, or for Kimi K2 the guarded native one (utils/kimi_k2.py); both
    # reads pinned to the run's bound base commit (#2094).
    config = load_stream_model_config(
        model_id, trust_remote_code=trust_remote_code, **base_revision_kwargs(model_id),
    )
    config.use_cache = False
    _drop_fp8_source_quantization(config)
    quant_config = None
    if quant == QUANT_NF4:
        quant_config = build_nf4_config(dtype, double_quant=double_quant)
    with _large_cpu_requests_on_meta(SKELETON_META_REDIRECT_BYTES):
        model = _skeleton_from_config(config, torch_dtype, quant_config, trust_remote_code)
    meta_buffers = [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if meta_buffers:
        logger.warning(
            "%s registers a buffer of %d bytes or more (%s), which must hold real "
            "values; building its skeleton without the meta redirect, so its large "
            "parameters are allocated on the CPU before they move to meta",
            type(model).__name__,
            SKELETON_META_REDIRECT_BYTES,
            ", ".join(meta_buffers),
        )
        model = _skeleton_from_config(config, torch_dtype, quant_config, trust_remote_code)
    # ``from_config`` does not reliably preserve the source reference that
    # ``from_pretrained`` stamps on both the model and its config.  PEFT reads
    # ``model.__dict__["name_or_path"]`` while wrapping LoRA and copies it to
    # ``base_model_name_or_path`` in adapter_config.json.  Without this stamp a
    # healthy streamed adapter cannot be auto-loaded by chat/merge/serve (#531).
    model.name_or_path = model_id
    model.config.name_or_path = model_id
    if quant_config is not None:
        _stamp_4bit_markers(model, quant_config)
    return model


def quantised_layer_suffixes(model: Any) -> FrozenSet[str]:
    """Per-layer short keys that ``replace_with_bnb_linear`` actually converted.

    Authoritative by construction — derived from the skeleton rather than from a
    hard-coded name list that would drift the moment an architecture names its
    projections differently.
    """
    import bitsandbytes as bnb

    try:
        layers = decoder_owner(model).layers
    except ValueError:
        return frozenset()
    if not len(layers):
        return frozenset()
    found = set()
    for layer in layers:
        found.update(
            name.replace(".base_layer.", ".")
            for name, param in layer.named_parameters()
            if isinstance(param, bnb.nn.Params4bit)
        )
    return frozenset(found)


@dataclass(frozen=True)
class ExtrasLoad:
    """What ``materialize_extras`` produced."""

    placed: int
    #: The shared NF4 code tables, moved to the training device. Empty for bf16.
    codes: Mapping[str, Any]


def materialize_extras(
    model: Any, shard_dir: str, index: Any, *, device: str, dtype: str
) -> ExtrasLoad:
    """Give real storage to resident weights that are not streamed.

    Also lifts out the two shared NF4 code tables: they are constant across
    every weight (the sharder asserts it), so one resident copy serves the whole
    model instead of streaming 16 + 256 floats per layer. For an untied pair,
    the embedding and output head intentionally remain on ``meta`` until the
    large-layer wrappers are installed. A tied embedding stays resident.
    """
    from safetensors.torch import load_file

    from soup_cli.utils.layer_shard import (
        NF4_CODE_KEY,
        NF4_NESTED_CODE_KEY,
        extras_shard_path,
    )

    extras = load_file(extras_shard_path(shard_dir))
    codes = {
        key: extras.pop(key).to(device)
        for key in (NF4_CODE_KEY, NF4_NESTED_CODE_KEY)
        if key in extras
    }
    torch_dtype = _torch_dtype(dtype)
    large_keys = set(tuple(getattr(index, "large_keys", ()) or ()))
    placed = 0
    pending_tied = []
    for name, param in list(model.named_parameters()):
        if not param.is_meta or ".layers." in name:
            continue
        if name in extras:
            _set_module_param(model, name, extras[name].to(device=device, dtype=torch_dtype))
            placed += 1
        elif name in large_keys:
            continue
        else:
            pending_tied.append(name)
    if pending_tied:
        # tie_word_embeddings=True -> lm_head.weight is absent from the
        # checkpoint by design and is restored from the input embeddings.
        model.tie_weights()
        allowed_large = {
            id(module.weight)
            for module in (model.get_input_embeddings(), model.get_output_embeddings())
            if module is not None and hasattr(module, "weight")
        }
        still_meta = [
            name
            for name, param in model.named_parameters()
            if param.is_meta
            and ".layers." not in name
            and name not in large_keys
            and id(param) not in allowed_large
        ]
        if still_meta:
            raise RuntimeError(
                f"non-layer weights left unmaterialised after tying: {still_meta[:4]}"
            )
    for name, buf in list(model.named_buffers()):
        if buf is not None and str(buf.device) != str(device):
            parts = name.split(".")
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], buf.to(device))
    return ExtrasLoad(placed=placed, codes=codes)


def materialize_meta_adapters(model: Any, *, seed: int = 0, device: str = "cuda") -> int:
    """Give real storage to LoRA adapters PEFT initialised on ``meta``.

    PEFT creates adapter weights on the base layer's device. In a streaming
    build that device is ``meta``, so the adapters have no storage: the
    optimizer happily accepts them and the run trains nothing. Re-initialise
    with PEFT's own scheme (A ~ kaiming_uniform, B = 0).

    The return value is the number of parameters materialised, not a success
    signal.  In particular, zero is healthy when PEFT created real adapter
    tensors itself.  Call :func:`assert_trainable_adapters_materialized` after
    this function to enforce the actual postcondition.
    """
    import torch
    import torch.nn as nn

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    count = 0
    for module_name, module in model.named_modules():
        for pname, param in list(module.named_parameters(recurse=False)):
            full = f"{module_name}.{pname}" if module_name else pname
            if not param.is_meta or "lora_" not in full:
                continue
            data = torch.empty(param.shape, dtype=torch.float32)
            if "lora_embedding_A" in full or "lora_B" in full:
                data.zero_()
            elif "lora_embedding_B" in full:
                nn.init.normal_(data, generator=generator)
            else:
                nn.init.kaiming_uniform_(data, a=5**0.5, generator=generator)
            module._parameters[pname] = nn.Parameter(data.to(device), requires_grad=True)
            count += 1
    return count


def assert_trainable_adapters_materialized(model: Any) -> None:
    """Refuse a streamed build whose trainable LoRA tensors have no storage.

    Decoder weights deliberately remain on ``meta`` under layer streaming, so
    the invariant is restricted to trainable ``lora_*`` parameters.  It is a
    postcondition rather than an interpretation of
    :func:`materialize_meta_adapters`' return count: PEFT versions differ on
    whether adapters need materialising, but both must leave real tensors.
    """
    stranded = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" in name and param.is_meta
    ]
    if not stranded:
        return

    preview = ", ".join(stranded[:4])
    remainder = len(stranded) - 4
    if remainder > 0:
        preview += f", ... (+{remainder} more)"
    raise RuntimeError(
        "trainable LoRA parameters remain on the meta device after adapter "
        f"materialization: {preview}. Refusing to train adapters without storage."
    )


#: The file every PEFT adapter save writes, final and ``checkpoint-*`` alike.
_ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"

#: The child segment of ``StreamedDecoderLayer``. A saved key carrying it is the
#: v0.72.0 spelling of an adapter that reloads as zero tensors.
_WRAPPER_KEY_SEGMENT = ".inner."


def assert_streamed_adapter_saved(model: Any, output_dir: str) -> None:
    """Refuse a streamed save that wrote an adapter which reloads as nothing.

    #1011 — twice a streamed LoRA run has reported success and written an
    ``adapter_model.safetensors`` that held nothing loadable: v0.72.0's
    ``.inner.`` keys, and #1005's 40-byte, 0-tensor file under peft 0.21. Both
    times the only signal was a ``UserWarning`` at some later load, after the
    GPU hours were spent. :func:`assert_trainable_adapters_materialized` asks
    "will this train?" before training; this asks "did anything get written?"
    after each save, by reading back the file's header (keys only, no tensor
    data).

    The reference is the model's own trainable ``lora_*`` parameters, one saved
    key each, not a bare ``> 0``. There is no legitimately empty streamed save:
    streaming always trains a LoRA adapter, and its trainable tensors exist from
    construction, so a save before the first optimizer step still writes all of
    them. Non-``lora_`` keys (``modules_to_save``) are not counted on either
    side.
    """
    path = os.path.join(output_dir, _ADAPTER_WEIGHTS_FILE)
    expected = sum(
        1 for name, param in model.named_parameters() if param.requires_grad and "lora_" in name
    )
    if not os.path.isfile(path):
        raise RuntimeError(
            f"the streamed run saved no {_ADAPTER_WEIGHTS_FILE} in {output_dir}; "
            f"expected {expected} LoRA tensors. Refusing to report a run whose "
            "adapter was not written."
        )

    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        keys = list(handle.keys())
    wrapped = [key for key in keys if _WRAPPER_KEY_SEGMENT in key]
    if wrapped:
        raise RuntimeError(
            f"{path} carries the streaming wrapper's '{_WRAPPER_KEY_SEGMENT}' segment "
            f"in {len(wrapped)} keys (first: {wrapped[0]}). Those keys match no "
            "module of a normal model, so the adapter would reload as zero tensors."
        )
    saved = sum(1 for key in keys if "lora_" in key)
    if saved == 0 or saved != expected:
        raise RuntimeError(
            f"{path} holds {saved} LoRA tensors, but the streamed model trained "
            f"{expected}. The adapter on disk does not hold the adapter that was "
            "trained, and would reload as "
            + ("zero tensors." if saved == 0 else "a partial adapter.")
        )


def build_streamed_save_guard_callback() -> Any:
    """Return a callback that runs :func:`assert_streamed_adapter_saved` on
    every ``checkpoint-*`` the Trainer writes.

    The HF Trainer writes periodic checkpoints through the same ``save_model``
    as the final save, so #1005's empty file appeared in every ``save_steps``
    checkpoint too. The final save dispatches no ``on_save``, which is why the
    trainers also call the check after it. Built here rather than at module
    scope so importing this module stays free of transformers.
    """
    from transformers import TrainerCallback

    class StreamedSaveGuardCallback(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kwargs) -> None:
            # the same rank condition save_model gates its write on
            if not getattr(args, "should_save", True) or model is None:
                return
            step = int(getattr(state, "global_step", 0) or 0)
            output_dir = getattr(args, "output_dir", None)
            if step <= 0 or not output_dir:
                return
            assert_streamed_adapter_saved(model, os.path.join(output_dir, f"checkpoint-{step}"))

    return StreamedSaveGuardCallback()


def materialize_meta_adapter_copy(
    model: Any, *, source_adapter: str = "default", target_adapter: str = "ref"
) -> int:
    """Materialize a frozen adapter copy that PEFT created on ``meta``.

    TRL 0.29 creates a ``ref`` adapter when DPO receives an existing PEFT
    model. On a layer-streamed model the decoder skeleton lives on ``meta``, so
    PEFT also creates the new adapter there and TRL's subsequent ``copy_`` is a
    no-op. Copy each matching source parameter into real storage explicitly;
    the reference remains frozen and still costs only adapter-sized memory.
    """
    source_marker = f".{source_adapter}."
    target_marker = f".{target_adapter}."
    parameters = dict(model.named_parameters())
    copied = 0

    for target_name, target_param in list(parameters.items()):
        # Linear adapters have .ref.weight; embedding adapters are parameters
        # in a ParameterDict and end with .ref instead.
        has_target = target_marker in target_name or target_name.endswith("." + target_adapter)
        if not has_target or not target_param.is_meta:
            continue
        source_name = (
            target_name.replace(target_marker, source_marker, 1)
            if target_marker in target_name
            else target_name[: -len(target_adapter)] + source_adapter
        )
        source_param = parameters.get(source_name)
        if source_param is None:
            raise RuntimeError(
                f"cannot materialize adapter {target_adapter!r}: "
                f"no source parameter {source_name!r}"
            )
        if source_param.is_meta:
            raise RuntimeError(
                f"cannot materialize adapter {target_adapter!r}: "
                f"source parameter {source_name!r} is still on meta"
            )
        _set_module_param(model, target_name, source_param.detach().clone())
        copied += 1

    stranded = [
        name
        for name, param in model.named_parameters()
        if (target_marker in name or name.endswith("." + target_adapter)) and param.is_meta
    ]
    if stranded:
        raise RuntimeError(
            f"adapter {target_adapter!r} remains on meta after materialization: "
            + ", ".join(stranded[:4])
        )
    return copied


# ==========================================================================
# installation
# ==========================================================================
@dataclass
class StreamRuntime:
    """Live streaming state for one model."""

    pool: Any
    source: Any
    prefetcher: Any
    n_layers: int
    pinned: bool
    device: str
    large_pool: Any = None
    hook: Any = None
    #: "ram" or "disk" — which weight source is feeding the buffer pool.
    tier: str = "ram"
    #: True source parameter count, from the shard index. PEFT's own total
    #: is ~6.5x too high for a streamed NF4 model (see trainer/sft.py).
    total_params: int = 0
    #: Read-only Qwen4 PLE mappings, if the N-gram table is SSD-backed.
    external_sources: Tuple[Any, ...] = ()
    #: training.stream_nf4_scale_cache — the layers' shared NF4ScaleCache, or None.
    scale_cache: Any = None
    #: Per MoE layer, the reader of its experts file (expert_layout "per_expert").
    expert_readers: Tuple[Any, ...] = ()

    def close(self) -> None:
        """Release the weight source and detach the prefetch hook.

        On the disk tier this stops the reader THREAD and frees its host
        staging buffers, which on a pinned run are page-locked and so cost the
        box until they are released. It holds no shard handles to leak:
        ``AsyncDiskSource`` opens each shard per read precisely because a
        mapping charges Windows commit for the file's whole size (#926).
        A no-op for the RAM tier, which owns neither a thread nor handles.
        """
        try:
            source_close = getattr(self.source, "close", None)
            if callable(source_close):
                source_close()
            for external in self.external_sources:
                external_close = getattr(external, "close", None)
                if callable(external_close):
                    external_close()
        finally:
            # One open handle per MoE layer's experts file (expert_layout "per_expert").
            for reader in self.expert_readers:
                reader.close()
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        if self.scale_cache is not None:
            # The layers' substituted weights still reference the cache object;
            # emptying it is what returns the device memory.
            self.scale_cache.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "n_layers": self.n_layers,
            "buffers": self.pool.n,
            "buffer_bytes": self.pool.nbytes + getattr(self.large_pool, "nbytes", 0),
            "large_buffer_bytes": getattr(self.large_pool, "nbytes", 0),
            "store_bytes": self.source.nbytes,
            "pinned": self.pinned,
            # What is actually page-locked for the store (#901): the arenas'
            # power-of-two sizes on a pinned RamSource, 0 on a pageable one, and
            # None for a source that does not account for it (the disk tier's
            # staging still pins per tensor).
            "pinned_bytes": getattr(self.source, "pinned_bytes", None) if self.pinned else 0,
            "tier": self.tier,
            # The reader's depth, where there is a reader. None on the RAM
            # tier — RamSource holds every layer and reads nothing ahead, so a
            # number there would be an invented one.
            "read_ahead": getattr(self.source, "read_ahead", None),
            "disk_bytes": getattr(self.source, "disk_bytes", 0),
            "drive_count": len(getattr(self.source, "direct_io_roots", {})),
            "direct_io_drives": sum(
                getattr(self.source, "direct_io_roots", {}).values()
            ),
            "direct_io_roots": dict(getattr(self.source, "direct_io_roots", {})),
            "layer_loads": self.pool.loads,
            "large_loads": getattr(self.large_pool, "loads", 0),
            "device": self.device,
            "total_params": self.total_params,
            "nf4_scale_cache_bytes": getattr(self.scale_cache, "nbytes", 0),
            "expert_fetches": sum(reader.fetches for reader in self.expert_readers),
            "expert_bytes_read": sum(reader.bytes_read for reader in self.expert_readers),
        }


def _device_map_value(device: Any) -> Union[str, int]:
    """The ``hf_device_map`` value ``from_pretrained(device_map=...)`` records.

    It must carry a device INDEX. For a 4-bit model ``accelerate.prepare_model``
    reads ``set(model.hf_device_map.values())`` and does
    ``torch.device(value).index`` — a bare ``"cuda"`` has index ``None``, and
    the very next line calls ``torch.device(None)``, which raises a TypeError
    naming nothing the user could act on. A resident load stores ``0``, so this
    stores ``0``.
    """
    text = str(device)
    if not text.startswith("cuda"):
        return text
    import torch

    if ":" in text:
        return int(text.split(":", 1)[1])
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


def _layer_name_map(layer: Any) -> Dict[str, str]:
    """meta parameter name inside one layer -> its shard key."""
    return {
        pname: pname.replace(".base_layer.", ".")
        for pname, param in layer.named_parameters()
        if param.is_meta
    }


def _load_layer_buffers(
    layers: Any, paths: Sequence[str], specs: Sequence[Any], keys: Sequence[str]
) -> int:
    """Copy the named persistent buffers of each decoder layer from its shard.

    ``keys`` (``layer_shard.layer_buffer_keys(arch)``) are the only buffers
    loaded. The skeleton is built under ``init_empty_weights()``, which leaves
    buffers as real tensors holding their ``__init__`` value: DeepSeek-V3's
    router ``e_score_correction_bias`` would otherwise run at zero, which routes
    tokens to different experts with no error. Buffers are not trained, so they
    are loaded once and stay resident, at the dtype the shard stores them in. A
    listed buffer absent from a layer's shard keeps its init value, as
    ``from_pretrained`` would. Returns the count loaded.
    """
    from safetensors import safe_open

    wanted_keys = frozenset(keys)
    if not wanted_keys:
        return 0
    loaded = 0
    for idx, (layer, path) in enumerate(zip(layers, paths)):
        wanted = []
        for name, buf in layer.named_buffers():
            owner_name, _, leaf = name.rpartition(".")
            owner = layer.get_submodule(owner_name) if owner_name else layer
            if buf is None or leaf in owner._non_persistent_buffers_set:
                continue
            key = name.replace(".base_layer.", ".")
            if key in wanted_keys and key in specs[idx]:
                wanted.append((owner, leaf, key, buf))
        if not wanted:
            continue
        with safe_open(path, framework="pt") as handle:
            for owner, leaf, key, buf in wanted:
                value = handle.get_tensor(key)
                if tuple(value.shape) != tuple(buf.shape):
                    raise ValueError(
                        f"decoder layer {idx} buffer {key!r} has shape "
                        f"{tuple(value.shape)} in the shard but {tuple(buf.shape)} "
                        f"in the model — reshard the checkpoint"
                    )
                # copy=True: on a CPU target `.to` would return the mapped view,
                # keeping the whole layer file mapped for the model's lifetime.
                setattr(owner, leaf, value.to(device=buf.device, copy=True))
                loaded += 1
    return loaded


def _install_expert_streaming(
    layers: Any,
    shard_dir: str,
    index: Any,
    *,
    device: str,
    codes: Optional[Mapping[str, Any]],
    two_reads: bool = False,
) -> Tuple[Any, ...]:
    """Swap each MoE layer's experts module for a ``StreamedExperts``.

    Only for a cache sharded with ``expert_layout="per_expert"``; returns the
    experts-file readers (closed by ``StreamRuntime.close``). ``two_reads`` on
    any other cache is refused: accepted, it would read three times while the
    caller believes two.
    """
    from soup_cli.utils.expert_stream import (
        EXPERT_LAYOUT_PER_EXPERT,
        ExpertFetcher,
        ExpertShardReader,
        install_expert_streaming,
    )
    from soup_cli.utils.layer_shard import expert_paths

    if getattr(index, "expert_layout", "") != EXPERT_LAYOUT_PER_EXPERT:
        if two_reads:
            raise ValueError(
                "two_read_experts needs a cache sharded with expert_layout='per_expert'; "
                f"this one is {getattr(index, 'expert_layout', '') or 'fused'!r}"
            )
        return ()
    readers: Dict[int, Any] = {}
    try:
        for layer, path in expert_paths(shard_dir, index).items():
            readers[layer] = ExpertShardReader(
                path,
                count=index.expert_count,
                shapes=index.expert_shapes,
                quant_specs=index.expert_quant_specs,
                dtype=index.dtype,
            )
        fetcher = ExpertFetcher(device=device, dtype=index.dtype, codes=codes)
        install_expert_streaming(layers, readers, fetcher, two_reads=two_reads)
    except BaseException:
        for reader in readers.values():
            reader.close()
        raise
    return tuple(readers.values())


def install_streaming(
    model: Any,
    *,
    shard_dir: str,
    index: Any,
    buffers: int = 2,
    pin: bool = True,
    require_pin: bool = False,
    device: str = "cuda",
    console: Any = None,
    codes: Optional[Mapping[str, Any]] = None,
    tier: str = "ram",
    read_ahead: int = DEFAULT_STREAM_READ_AHEAD,
    refill_before_backward: bool = False,
    read_ahead_decision: Any = None,
    nf4_scale_cache: bool = False,
    two_read_experts: bool = False,
) -> StreamRuntime:
    """Wrap every decoder layer and wire the buffer pool + prefetch scheduler.

    ``refill_before_backward`` says the loss runs a second forward before each
    step's backward; an untied output head then hands autograd a private copy
    of its weight (#1049).

    ``nf4_scale_cache`` gives the layers one shared :class:`NF4ScaleCache`
    (``training.stream_nf4_scale_cache``); it needs double-quantised NF4 shards.

    ``read_ahead_decision`` (a ``stripe_roots.ReadAheadDecision``, R4) says how
    ``read_ahead`` came about, so the page-lock messages advise a depth that is
    not raised straight back up on a striped cache.

    ``two_read_experts`` (``expert_layout="per_expert"`` caches only): each MoE
    layer's checkpoint recompute replays its routers' top-k decisions and reads no
    expert, so a step reads every routed-to expert twice instead of three times
    (``expert_stream.two_read_checkpoint_contexts``).
    """
    import torch

    from soup_cli.utils.layer_shard import (
        QUANT_NF4,
        large_shard_path,
        large_weight_role,
        layer_buffer_keys,
        layer_paths,
    )

    # PyTorch 2.7+ on Apple Silicon can turn
    # ``device="cpu", pin_memory=True`` into an MPS allocation, placing the
    # whole base in the accelerator allocator and defeating the host-store
    # invariant (#434). Keep this MPS-specific guard at the runtime boundary as
    # well as in stream_setup so direct callers are safe. CPU retains its
    # existing try/fallback semantics, including the require_pin wiring tests.
    if str(device).startswith("mps"):
        if require_pin:
            message = (
                "layer streaming was called with require_pin=True, but the target "
                "is MPS. Pinned CPU host memory is CUDA-only here; proceeding "
                "with a pageable CPU source."
            )
            if console is not None:
                console.print(f"[yellow]{message}[/]")
            else:
                logger.warning(message)
        pin = False
        require_pin = False

    # Validate the index BEFORE touching the model: an index whose `quant` and
    # `quant_specs` disagree passes the cache key (which reads `quant`) and
    # would then reconstruct NF4 against a skeleton never converted to
    # Linear4bit.
    quant_specs = dict(getattr(index, "quant_specs", None) or {})
    if bool(quant_specs) != (getattr(index, "quant", None) == QUANT_NF4):
        raise ValueError(
            f"shard index is inconsistent: quant={getattr(index, 'quant', None)!r} "
            f"but {len(quant_specs)} quant_specs — reshard the checkpoint"
        )
    scale_cache = None
    if nf4_scale_cache:
        if not any(spec.nested for spec in quant_specs.values()):
            raise ValueError(
                "nf4_scale_cache=True needs double-quantised NF4 shards, but this "
                "index has no nested NF4 weights; there are no scales to cache"
            )
        scale_cache = NF4ScaleCache()

    owner = decoder_owner(model)
    layers = owner.layers
    n_layers = len(layers)
    if n_layers != index.n_layers:
        raise ValueError(
            f"model has {n_layers} decoder layers but the shard index has "
            f"{index.n_layers} — reshard the checkpoint"
        )

    _require_stripe_folders(shard_dir, index)
    # Before the name maps: the fused expert parameters leave the layer here,
    # so the layer pool never allocates them.
    expert_readers = _install_expert_streaming(
        layers, shard_dir, index, device=str(device), codes=codes, two_reads=two_read_experts
    )
    context_fns = [None] * n_layers
    if two_read_experts:
        from soup_cli.utils.expert_stream import checkpoint_context_fn

        context_fns = [checkpoint_context_fn(layer) for layer in layers]
    layer_name_maps = [_layer_name_map(layer) for layer in layers]
    if not any(layer_name_maps):
        raise RuntimeError(
            "no meta decoder weights found — the base was materialised, which "
            "defeats layer streaming entirely"
        )
    decoder_paths = layer_paths(shard_dir, index)
    layer_specs = RamSource.layer_specs_from_paths(decoder_paths)
    large_specs = large_layer_specs(shard_dir, index)
    large_keys = tuple(large_specs)
    role_keys = {large_weight_role(key): key for key in large_keys}
    embed_key = role_keys.get("embed_tokens")
    explicit_head_key = role_keys.get("lm_head")
    if large_keys and embed_key is None:
        raise ValueError(
            "shard index has large-layer weights but no model.embed_tokens.weight; "
            "reshard the checkpoint"
        )

    # NF4 streams the packed nibbles AND the statistics needed to rebuild the
    # QuantState; the two code tables are shared and stay resident.
    needed_specs_by_layer = []
    active_keys_by_layer = []
    for idx, name_map in enumerate(layer_name_maps):
        if not name_map:
            raise RuntimeError(
                f"decoder layer {idx} exposes no meta weights — the base was "
                f"materialised, which defeats layer streaming entirely"
            )
        shard_spec = layer_specs[idx]
        missing = sorted(set(name_map.values()) - set(shard_spec))
        if missing:
            raise ValueError(f"shard is missing decoder weights: {missing[:4]}")
        needed: Dict[str, Tuple[Tuple[int, ...], str]] = {}
        for ckpt in name_map.values():
            spec_q = quant_specs.get(ckpt)
            wanted = quant_sidecar_keys(ckpt, spec_q) if spec_q is not None else (ckpt,)
            for key in wanted:
                if key not in shard_spec:
                    raise ValueError(
                        f"shard is missing the NF4 sidecar {key!r} — reshard the checkpoint"
                    )
                needed[key] = shard_spec[key]
            if spec_q is not None:
                validate_quant_shape(ckpt, spec_q, shard_spec)
        needed_specs_by_layer.append(needed)
        active_keys_by_layer.append(tuple(sorted(needed)))
    spec = RamSource.merge_layer_specs(needed_specs_by_layer)
    # Before the source exists: a refusal here leaves no reader thread or
    # page-locked staging behind.
    _load_layer_buffers(
        layers, decoder_paths, layer_specs, layer_buffer_keys(getattr(index, "arch", ""))
    )

    large_source_indices = {key: n_layers + offset for offset, key in enumerate(large_keys)}
    source_specs = needed_specs_by_layer + [{key: large_specs[key]} for key in large_keys]
    source_paths = list(decoder_paths) + [large_shard_path(shard_dir, key) for key in large_keys]
    placement = tuple(getattr(index, "layer_roots", ()) or ())
    source_roots = placement + (0,) * len(large_keys) if placement else None
    source, pinned = _build_source(
        shard_dir,
        len(source_specs),
        source_specs,
        pin,
        console,
        tier,
        require_pin=require_pin,
        shard_paths=source_paths,
        read_ahead=read_ahead,
        layer_roots=source_roots,
        read_ahead_decision=read_ahead_decision,
    )
    pool = LayerBufferPool(
        spec,
        n_buffers=buffers,
        device=device,
        active_keys_by_layer=active_keys_by_layer,
    )
    stream = torch.cuda.Stream() if str(device).startswith("cuda") else None
    large_pool = (
        LargeLayerBufferPool(large_specs, large_source_indices, device=device)
        if large_specs
        else None
    )
    output_key = explicit_head_key or embed_key

    def _prefetch_output() -> None:
        if large_pool is not None and output_key is not None:
            large_pool.load_async(output_key, source, stream)

    def _prefetch_embed() -> None:
        if large_pool is not None and embed_key is not None:
            large_pool.load_async(embed_key, source, stream)

    prefetcher = StreamPrefetcher(
        pool,
        source,
        n_layers,
        stream,
        tail_prefetch=_prefetch_output if large_pool is not None else None,
        backward_tail_prefetch=_prefetch_embed if large_pool is not None else None,
        # #1174 part 2: the head's load right after the embedding lookup. Set
        # `prefetcher.head_prefetch_layer = None` to restore the last-layer timing.
        head_prefetch_layer=0 if large_pool is not None else None,
    )

    layer_cls = _streamed_layer_class()
    for idx in range(n_layers):
        layers[idx] = layer_cls(
            layers[idx],
            idx,
            pool,
            prefetcher,
            layer_name_maps[idx],
            quant_specs=quant_specs,
            codes=codes,
            scale_cache=scale_cache,
            checkpoint_context_fn=context_fns[idx],
        )

    if large_pool is not None:
        input_module = model.get_input_embeddings()
        output_module = model.get_output_embeddings()
        if input_module is None or output_module is None:
            raise RuntimeError("layer streaming needs both input and output embedding modules")
        for role, module, key in (
            ("input embedding", input_module, embed_key),
            ("output head", output_module, output_key),
        ):
            weight = getattr(module, "weight", None)
            if key is None or weight is None or not getattr(weight, "is_meta", False):
                raise RuntimeError(f"streamed {role} is not an unmaterialised meta weight")
            expected_shape = tuple(large_specs[key][0])
            if tuple(weight.shape) != expected_shape:
                raise ValueError(
                    f"streamed {role} shape {tuple(weight.shape)} does not match "
                    f"the shard {expected_shape} — reshard the checkpoint"
                )

        large_cls = _streamed_large_layer_class()
        if input_module is output_module:
            # UNEXERCISED by the test suite, deliberately (#1010 review): no
            # supported architecture returns ONE module object for both
            # boundaries -- a tied Llama shares the Parameter, not the module,
            # and takes the resident path above -- so this branch has no fixture
            # and its `named_modules()` memo de-duplication (one wrapper reached
            # under two attribute names yields once, as torch's own walk would)
            # is read, not run. Kept for the architecture that does share it.
            if embed_key != output_key:
                raise ValueError("one module cannot represent two untied large-layer weights")
            shared = large_cls(
                input_module,
                embed_key,
                large_pool,
                refill_before_backward=refill_before_backward,
                weight_path=large_shard_path(shard_dir, embed_key),
            )
            if not _replace_module_references(model, input_module, shared):
                raise RuntimeError("could not install the streamed tied embedding module")
        else:
            streamed_input = large_cls(
                input_module,
                embed_key,
                large_pool,
                refill_before_backward=refill_before_backward,
                weight_path=large_shard_path(shard_dir, embed_key),
            )
            streamed_output = large_cls(
                output_module,
                output_key,
                large_pool,
                refill_before_backward=refill_before_backward,
                weight_path=large_shard_path(shard_dir, output_key),
            )
            if not _replace_module_references(model, input_module, streamed_input):
                raise RuntimeError("could not install the streamed input embedding")
            if not _replace_module_references(model, output_module, streamed_output):
                raise RuntimeError("could not install the streamed output head")
        _attach_streamed_boundary_adapter_bridge(model)

    def _prime(*_args: Any, **_kwargs: Any) -> None:
        # #975 — the backward-tail prefetch above already loads this for every
        # step but the first, so `load_async`'s own-owner check makes this a
        # no-op on the hot path. Kept unconditional: it is the only load for
        # step 0, and for any forward that follows a backward that never
        # reached layer 0.
        if large_pool is not None and embed_key is not None:
            large_pool.load_async(embed_key, source, stream)
        prefetcher.prime()

    handle = owner.register_forward_pre_hook(_prime)

    # transformers' Trainer.__init__ calls _move_model_to_device -> model.to(),
    # and .to() on a module holding meta parameters raises NotImplementedError.
    # The decoder weights stay on meta BY DESIGN, so declare that this model
    # manages its own placement — exactly the marker a device_map-sharded model
    # carries, and the one _move_model_to_device short-circuits on.
    model.hf_device_map = {"": _device_map_value(device)}

    return StreamRuntime(
        pool=pool,
        source=source,
        prefetcher=prefetcher,
        n_layers=n_layers,
        pinned=pinned,
        device=str(device),
        large_pool=large_pool,
        hook=handle,
        total_params=int(getattr(index, "total_params", 0) or 0),
        expert_readers=expert_readers,
        tier=tier,
        scale_cache=scale_cache,
    )


def _is_out_of_memory(exc: BaseException) -> bool:
    """THE one spelling of "the device ran out of memory", shared with the batch
    probe: the allocator's ``torch.OutOfMemoryError``, or a ``RuntimeError``
    (``AcceleratorError`` is one) whose text says so."""
    import torch

    from soup_cli.utils.batch_probe import _is_cuda_oom

    return _is_cuda_oom(exc, torch)


def drain_stale_cuda_error(
    device: str = "cuda", *, launch: Optional[Callable[[], None]] = None
) -> bool:
    """#901 — consume the stale error a failed page-lock leaves on the CUDA runtime.

    ``cuMemHostAlloc`` refusing a ``pin_memory=True`` allocation raises
    ``AcceleratorError("CUDA error: out of memory")`` and leaves the runtime's
    per-thread last error set. Measured on the dev box after such a failure:
    ``cudaMalloc``, ``synchronize`` and a host-to-device copy all succeed, and the
    FIRST kernel launch raises that same "out of memory" with 7.3 GB of VRAM
    free, because its launch check reads the stale value — and clears it, so
    the second launch works. In the report that first launch was the adapter
    cast inside ``SFTTrainer.__init__``; with the probe on it was the forward.

    torch exposes no ``cudaGetLastError``, so this launches one trivial kernel
    to let its check consume the error, then a second to prove the context is
    healthy. Returns True when a stale error was drained. A launch that fails
    for any other reason, or twice, propagates: that is a context which is
    genuinely broken or genuinely out of memory, and hiding it would recreate
    the silent failure this exists to end. ``launch`` is injectable for tests;
    the default is a no-op without a CUDA device, where nothing was pinned.
    """
    if launch is None:
        import torch

        if not torch.cuda.is_available():
            return False

        def launch() -> None:
            torch.ones(1, device=device)
            torch.cuda.synchronize(device)

    try:
        launch()
    except RuntimeError as exc:
        if not _is_out_of_memory(exc):
            raise
    else:
        return False
    launch()
    return True


def release_cached_pinned_memory() -> int:
    """Return the page-locked blocks torch's caching host allocator kept.

    A pinned tensor that is freed goes back to that cache, not to the OS, so
    the blocks a partially pinned store allocated before its page-lock failed
    stay page-locked for the rest of the process — gigabytes on a 14B store,
    beside the pageable store that replaced it. Returns the bytes released, or
    0 when the stats or the cache call are unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        return 0
    empty = getattr(torch._C, "_host_emptyCache", None)
    if empty is None:  # pragma: no cover - torch without the host-cache call
        return 0
    stats = getattr(torch.cuda, "host_memory_stats", None)
    try:
        before = int(stats().get("allocated_bytes.current", 0)) if stats else 0
        empty()
        after = int(stats().get("allocated_bytes.current", 0)) if stats else 0
    except Exception as exc:  # noqa: BLE001 - a private API with no contract
        # Reported, never raised: this runs inside the page-lock fallback, and
        # an error here would replace "could not page-lock the base" with one
        # about the recovery. The cache staying full costs RAM, not correctness.
        logger.warning("could not release torch's cached pinned memory: %r", exc)
        return 0
    return max(0, before - after)


def recover_from_failed_page_lock(*, device: str = "cuda", console: Any = None) -> bool:
    """What a pageable fallback owes the run before it builds anything (#901).

    Returns True when a stale CUDA error was drained — the half that decides
    whether the run's next kernel launch lives. The cache release is reported
    in the message (bytes, when any were returned) but not in the return value:
    a full cache costs RAM, an undrained error costs the run.
    """
    drained = drain_stale_cuda_error(device)
    released = release_cached_pinned_memory()
    if not drained and not released:
        return False
    parts = []
    if drained:
        parts.append("cleared the stale CUDA out-of-memory error it left on the runtime")
    if released:
        parts.append(f"released {released / 1e9:.2f} GB of page-locked memory it left cached")
    message = f"after the failed page-lock: {' and '.join(parts)}."
    if console is not None:
        console.print(f"[dim]{message}[/]")
    else:
        logger.info(message)
    return drained


def _recover_before_refusing(console: Any) -> None:
    """The refusal branches of ``_build_source`` raise instead of falling back,
    but the stale error a failed page-lock leaves is per-thread and outlives
    the exception: a caller that catches the refusal (a test, a wrapper that
    retries with other settings) would inherit it. Best effort, because the
    refusal is the message that matters and nothing here may replace it."""
    try:
        recover_from_failed_page_lock(console=console)
    except Exception as exc:  # noqa: BLE001 - never let cleanup mask the refusal
        logger.warning("page-lock recovery failed before the refusal: %r", exc)


def _require_stripe_folders(shard_dir: str, index: Any) -> None:
    """Every stripe folder a striped cache reads from is there and is a real folder.

    Refused by name before any header is read: a missing root (an unmounted drive), a missing
    per-model folder under a present root, and a link or junction at the folder, which would
    send every read somewhere the cache check never looked.
    """
    from soup_cli.utils.layer_shard import stripe_dirs
    from soup_cli.utils.stripe_roots import STRIPE_DIRS_ENV, stripe_folder_problem

    placement = tuple(getattr(index, "layer_roots", ()) or ())
    roots = tuple(getattr(index, "stripe_roots", ()) or ())
    folders = stripe_dirs(shard_dir, roots)[1:]
    for position, (root, folder) in enumerate(zip(roots, folders), start=1):
        layers = [idx for idx, owner in enumerate(placement) if owner == position]
        shown = f"{layers[:8]}{' ...' if len(layers) > 8 else ''}"
        if not os.path.isdir(root):
            raise RuntimeError(
                f"layer streaming's cache keeps decoder layers {shown} on the stripe root "
                f"{root} (from {STRIPE_DIRS_ENV}), and that folder is not there — a drive "
                f"that is not mounted, or a drive letter that changed. Reconnect it, or unset "
                f"{STRIPE_DIRS_ENV} and let Soup re-shard to one root."
            )
        problem = stripe_folder_problem(folder, root)
        if problem is not None:
            raise RuntimeError(f"layer streaming's cache: {problem}.")
        if not os.path.isdir(folder):
            raise RuntimeError(
                f"layer streaming's cache keeps decoder layers {shown} in {folder} (on a "
                f"{STRIPE_DIRS_ENV} root), and that folder is gone although its root is there. "
                f"Let Soup re-shard (the cache check will rebuild it), or unset "
                f"{STRIPE_DIRS_ENV} to re-shard to one root."
            )


def _say_unaligned_staging(source: Any, console: Any) -> None:
    """Put the disk tier's page-cache fallback where a `soup train` user reads it (#1531).

    The source logs it; the logger is not what the Rich output shows.
    """
    if console is not None and not getattr(source, "staging_aligned", True):
        from soup_cli.utils.async_disk_source import UNALIGNED_STAGING_MESSAGE

        console.print(f"[yellow]{UNALIGNED_STAGING_MESSAGE}[/]")


def _say_page_cache_roots(source: Any, console: Any) -> None:
    """Name each drive whose operating system refused direct I/O (#1615)."""
    if console is None:
        return
    from soup_cli.utils.terminal import for_terminal

    for root, reason in getattr(source, "page_cache_reasons", {}).items():
        console.print(
            "[yellow]Layer streaming reads "
            f"{for_terminal(root)} through the page cache: direct I/O is unavailable there "
            f"({for_terminal(reason)}).[/]"
        )


def _say_disk_source_fallbacks(source: Any, console: Any) -> None:
    _say_unaligned_staging(source, console)
    _say_page_cache_roots(source, console)


def _build_source(
    shard_dir,
    n_layers,
    spec,
    pin,
    console,
    tier="ram",
    require_pin=False,
    shard_paths=None,
    read_ahead=DEFAULT_STREAM_READ_AHEAD,
    layer_roots=None,
    read_ahead_decision=None,
):
    """Build the weight source for the chosen tier.

    On the RAM tier, page-locking is bounded by the box rather than by free RAM
    (the dev box topped out at 7.12 GB with 9.1 GB "available"). Falling back to
    a pageable store is correct; hiding the ~97% -> ~79% GPU-utilisation cost is
    not, so the fallback says so out loud.

    ``require_pin`` (from ``training.stream_pin=true``) turns that fallback into
    a hard refusal on the RAM tier: the user asked for the pinned store
    explicitly, so silently degrading to a pageable one — spending the whole
    throughput margin pinning exists to provide — is exactly the outcome the flag
    exists to prevent.

    The DISK tier now behaves the same way, for the same reason. It used to
    *announce* that pinning was inapplicable — true while the base streamed from
    NVMe into a freshly allocated tensor per call, with nothing to page-lock.
    ``AsyncDiskSource`` reads ahead into reusable HOST STAGING (#971), and that
    staging is exactly the kind of memory pinning exists for: out of pageable
    memory the host-to-device copy is synchronous and the reader cannot overlap
    with compute. So an explicit ``training.stream_pin`` is honoured or refused
    here, never explained away.

    ``read_ahead`` (``training.stream_read_ahead``) is the reader's depth and
    therefore the multiplier on how much host memory is page-locked, which makes
    lowering it a remedy the RAM tier cannot offer. ``read_ahead_decision`` (R4)
    says how that depth came about; without it the depth is taken as set, over
    the drives ``layer_roots`` names. Either way the advice names a depth that
    the N + 1 rule does not raise straight back up.

    The second element of the returned tuple means the same thing on both tiers:
    the host-side source memory is page-locked.
    """
    source_kwargs = {} if shard_paths is None else {"shard_paths": shard_paths}
    if tier == "disk":
        from soup_cli.utils.async_disk_source import AsyncDiskSource
        from soup_cli.utils.stripe_roots import ReadAheadDecision

        decision = read_ahead_decision or ReadAheadDecision(
            depth=read_ahead,
            configured=read_ahead,
            n_roots=(max(layer_roots) + 1) if layer_roots else 1,
        )
        advice = decision.lowering_advice()
        open_kwargs = dict(read_ahead=read_ahead, **source_kwargs)
        if layer_roots is not None:
            # R4: which drive each source index lives on. Only the async reader uses it; the
            # RAM tier reads every file once whatever drive it is on.
            open_kwargs["layer_roots"] = layer_roots
        if pin:
            try:
                source = AsyncDiskSource(shard_dir, n_layers, spec, pin=True, **open_kwargs)
            except (RuntimeError, MemoryError) as exc:
                # Staging is allocated before the reader thread starts, so a
                # constructor that raised here owns no thread and no buffers —
                # there is nothing to close before retrying.
                if require_pin:
                    _recover_before_refusing(console)
                    raise RuntimeError(
                        "training.stream_pin=true but this box could not "
                        "page-lock the disk tier's host staging "
                        f"({type(exc).__name__}). The staging is "
                        f"training.stream_read_ahead={read_ahead} layers deep "
                        "per distinct layer shape, so that depth is what decides "
                        "how much gets page-locked. Refusing rather than silently "
                        "degrading to pageable staging, which makes host-to-device "
                        "copies synchronous and costs the ~97% -> ~79% "
                        "GPU-utilisation overlap pinning buys. "
                        + (f"{advice}, free RAM, " if advice else "Free RAM, ")
                        + "or unset training.stream_pin to allow the pageable fallback."
                    ) from exc
                message = (
                    "layer streaming could not page-lock the disk tier's host "
                    f"staging ({type(exc).__name__}); falling back to PAGEABLE "
                    "staging. Host-to-device copies become synchronous, which "
                    "costs overlap — measured GPU utilisation drops from ~97% to "
                    "~79%. The read-ahead depth decides how much is page-locked. "
                    + (f"{advice}, or free RAM" if advice else "Free RAM")
                    + " to keep the pinned staging."
                )
                if console is not None:
                    console.print(f"[yellow]{message}[/]")
                else:
                    logger.warning(message)
                # #901 — BEFORE the pageable retry: the failed page-lock leaves
                # a stale CUDA error that the run's first kernel launch would
                # otherwise report as an out-of-memory it never had.
                recover_from_failed_page_lock(console=console)
            else:
                _say_disk_source_fallbacks(source, console)
                return source, source.pinned
        source = AsyncDiskSource(shard_dir, n_layers, spec, pin=False, **open_kwargs)
        _say_disk_source_fallbacks(source, console)
        return source, source.pinned
    # `source.pinned` on both branches rather than the literal, so the tuple's
    # second element has ONE meaning to read off: what the source says about
    # itself. Value-identical today — `RamSource.__init__` raises rather than
    # returning a pageable store under pin=True — but a literal is a claim
    # about the constructor made at the call site, which is where the disk
    # branch's two spellings used to disagree.
    if not pin:
        source = RamSource(shard_dir, n_layers, spec, pin=False, **source_kwargs)
        return source, source.pinned
    try:
        source = RamSource(shard_dir, n_layers, spec, pin=True, **source_kwargs)
        return source, source.pinned
    except (RuntimeError, MemoryError) as exc:
        store_gb = _spec_bytes(spec, n_layers=n_layers) / 1e9
        if require_pin:
            _recover_before_refusing(console)
            raise RuntimeError(
                "training.stream_pin=true but this box could not page-lock the "
                f"{store_gb:.2f} GB RAM store ({type(exc).__name__}). Refusing "
                "rather than silently degrading to a pageable store, which makes "
                "host-to-device copies synchronous and costs the ~97% -> ~79% "
                "GPU-utilisation overlap pinning buys. Free RAM, use a smaller "
                "base, or unset training.stream_pin to allow the pageable "
                "fallback."
            ) from exc
        message = (
            "layer streaming could not page-lock the base "
            f"({type(exc).__name__}); falling back to a PAGEABLE RAM store. "
            "Host-to-device copies become synchronous, which costs overlap — "
            "measured GPU utilisation drops from ~97% to ~79%. Free RAM or use "
            "a smaller base to keep the pinned store."
        )
        if console is not None:
            console.print(f"[yellow]{message}[/]")
        else:
            logger.warning(message)
        # #901 — the fallback was dead on arrival without this: the failed
        # page-lock leaves a stale CUDA error, and the first kernel launch of
        # the run (in the report, `param.data.to(bfloat16)` in
        # SFTTrainer.__init__) reported it as an out-of-memory with gigabytes
        # free. Drain it, and give back the page-locked blocks the abandoned
        # attempt left in torch's host cache, before building the store.
        recover_from_failed_page_lock(console=console)
        source = RamSource(shard_dir, n_layers, spec, pin=False, **source_kwargs)
        return source, source.pinned


def _spec_bytes(
    spec: Union[
        Mapping[str, Tuple[Tuple[int, ...], str]],
        Sequence[Mapping[str, Tuple[Tuple[int, ...], str]]],
    ],
    *,
    n_layers: int = 1,
) -> int:
    """Exact RAM-store bytes represented by one shared or per-layer spec."""
    if isinstance(spec, Mapping):
        layer_specs = [spec] * n_layers
    else:
        layer_specs = list(spec)
        if len(layer_specs) != n_layers:
            raise ValueError(f"expected {n_layers} layer specs, but got {len(layer_specs)}")
    return sum(
        math.prod(shape) * _dtype_size(dtype)
        for layer_spec in layer_specs
        for shape, dtype in layer_spec.values()
    )


def build_streamed_model(
    *,
    model_id: str,
    shard_dir: str,
    index: Any,
    lora_config: Any,
    device: str = "cuda",
    dtype: str = "bfloat16",
    buffers: int = 2,
    pin: bool = True,
    require_pin: bool = False,
    seed: int = 0,
    trust_remote_code: bool = False,
    console: Any = None,
    quant: str = "none",
    double_quant: bool = True,
    tier: str = "ram",
    read_ahead: int = DEFAULT_STREAM_READ_AHEAD,
    weights_dir: Optional[str] = None,
    ngram_source: str = "disk",
    refill_before_backward: bool = False,
    read_ahead_decision: Any = None,
    nf4_scale_cache: bool = False,
    two_read_experts: bool = False,
) -> Tuple[Any, StreamRuntime]:
    """Meta skeleton -> extras -> LoRA -> streaming. No resident base load.

    ``read_ahead_decision`` (R4) is the setup's ``ReadAheadDecision`` for ``read_ahead``,
    handed to ``install_streaming`` for its page-lock advice. ``two_read_experts`` is
    ``install_streaming``'s: an API switch, no config field.
    """
    from peft import get_peft_model

    model = build_meta_skeleton(
        model_id,
        dtype=dtype,
        quant=quant,
        double_quant=double_quant,
        trust_remote_code=trust_remote_code,
    )
    external_tensors = dict(getattr(index, "external_tensors", None) or {})
    external_sources: Tuple[Any, ...] = ()
    if external_tensors:
        if weights_dir is None:
            raise ValueError("a Qwen4 shard index with external PLE tensors requires weights_dir")
        from soup_cli.utils.qwen4_ple import install_qwen4_ple_embeddings

        external_sources = install_qwen4_ple_embeddings(
            model,
            weights_dir=weights_dir,
            external_tensors=external_tensors,
            source=ngram_source,
        )
    try:
        extras = materialize_extras(model, shard_dir, index, device=device, dtype=dtype)
        for param in model.parameters():
            param.requires_grad = False
        from soup_cli.utils.peft_wiring import apply_pre_lora_patches

        apply_pre_lora_patches(model, model_id)
        _refuse_trainable_streamed_boundary(model, index, lora_config)
        model = get_peft_model(model, lora_config)
        materialize_meta_adapters(model, seed=seed, device=device)
        assert_trainable_adapters_materialized(model)
        runtime = install_streaming(
            model,
            shard_dir=shard_dir,
            index=index,
            buffers=buffers,
            pin=pin,
            require_pin=require_pin,
            device=device,
            console=console,
            codes=extras.codes,
            tier=tier,
            read_ahead=read_ahead,
            refill_before_backward=refill_before_backward,
            read_ahead_decision=read_ahead_decision,
            nf4_scale_cache=nf4_scale_cache,
            two_read_experts=two_read_experts,
        )
    except BaseException:
        for external in external_sources:
            external.close()
        raise
    runtime.external_sources = external_sources
    return model, runtime
