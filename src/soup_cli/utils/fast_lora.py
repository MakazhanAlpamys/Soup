"""Fast-LoRA single-projection autograd (correctness half of tracker #792).

Scope: the single-projection path from #839, which is ``o_proj`` plus any
adapted ``nn.Linear`` / ``bnb.nn.Linear4bit`` the MLP (#837) and QKV (#838)
paths do not cover. The kernel computes the same math as peft's generic
autograd with fewer intermediates; **no speedup is claimed here** -- the
tracker measures before any number is published.

What this module establishes for the sibling paths:

- instance-level forward patching (``types.MethodType``, as the streamed-layer
  dequant patch already does in ``layer_stream_runtime``);
- ``save_for_backward`` for every tensor the backward reads (#331: streamed
  pools rewrite their slots, so a saved reference is only safe when it went
  through the pool-aware save);
- NF4: the packed weight and its quant-state tensors are saved and the weight
  is re-dequantised in the backward; the dequantised matrix is never saved;
- delegation back to the unpatched peft forward whenever ``disable_adapters``,
  ``merged`` or ``adapter_names`` is set, the adapter is a fused variant
  (DoRA/VeRA and friends produce tuple results the hand-written backward does
  not model), dropout is non-zero, or the call shape is not a plain ``(x)``.

The patch covers only the single-projection path. When the QKV and MLP
patchers land they have to run first (or mark their modules) so those shapes
do not fall into this path.

The patched forward reads the base weight at call time on purpose: streamed
layers substitute weights via ``functional_call`` only for the duration of a
call, and a reference captured at patch time under streaming is a meta
placeholder.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_soup_fast_lora_single_projection"
_ORIGINAL_FORWARD_MARKER = "_soup_fast_lora_original_forward"

__all__ = [
    "patch_fast_lora_single_projection",
    "unpatch_fast_lora_single_projection",
]

_FUNCTION: Any = None


def _as_dtype(tensor: Any, dtype: Any) -> Any:
    """Cast only when the dtype differs, so the uniform case stays op-for-op."""
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _flatten(tensor: Any) -> Any:
    """Collapse the leading dimensions into one, for weight-shaped gradients.

    ``dB`` and ``dA`` sum over every leading dimension regardless of the input
    rank (``[N, in]`` for the unit tests, ``[B, S, in]`` from transformers).
    ``reshape`` is a view for the contiguous activation layouts that reach a
    projection; a non-contiguous grad pays a copy, which is the correct trade
    against a silent shape assumption.
    """
    return tensor.reshape(-1, tensor.shape[-1])


def _quant_state_parts(quant_state: Any) -> tuple[list[Any], dict[str, Any]]:
    """Split a ``QuantState`` into ``save_for_backward`` tensors and metadata.

    Roles are recorded in save order so the backward can rebuild the state
    without guessing which optional tensor is which (``offset`` and the nested
    ``state2`` are only present for some checkpoints).
    """
    roles = ["absmax", "code"]
    tensors = [quant_state.absmax, quant_state.code]
    meta: dict[str, Any] = {
        "shape": tuple(quant_state.shape),
        "dtype": quant_state.dtype,
        "blocksize": quant_state.blocksize,
        "quant_type": quant_state.quant_type,
    }
    if quant_state.offset is not None:
        roles.append("offset")
        tensors.append(quant_state.offset)
    if quant_state.state2 is not None:
        roles += ["state2_absmax", "state2_code"]
        tensors += [quant_state.state2.absmax, quant_state.state2.code]
        meta["state2_blocksize"] = quant_state.state2.blocksize
        if quant_state.state2.offset is not None:
            roles.append("state2_offset")
            tensors.append(quant_state.state2.offset)
    meta["roles"] = roles
    return tensors, meta


def _rebuild_quant_state(meta: dict[str, Any], tensors: list[Any]) -> Any:
    """Reassemble a ``QuantState`` from saved tensors (``layer_stream_runtime``
    precedent, including the nested ``state2`` dtype)."""
    import torch
    from bitsandbytes.functional import QuantState

    parts = dict(zip(meta["roles"], tensors))
    state2 = None
    if "state2_absmax" in parts:
        state2 = QuantState(
            absmax=parts["state2_absmax"],
            code=parts["state2_code"],
            blocksize=meta["state2_blocksize"],
            dtype=torch.float32,
            offset=parts.get("state2_offset"),
        )
    return QuantState(
        absmax=parts["absmax"],
        shape=torch.Size(meta["shape"]),
        dtype=meta["dtype"],
        blocksize=meta["blocksize"],
        code=parts["code"],
        quant_type=meta["quant_type"],
        offset=parts.get("offset"),
        state2=state2,
    )


def _single_projection_function() -> Any:
    """Build (once per process) the ``autograd.Function`` class.

    The class has to be built inside a function because its base class is
    ``torch.autograd.Function`` and torch must stay out of the light CLI import
    path (tests/test_cli_startup_is_light.py).
    """
    global _FUNCTION
    if _FUNCTION is not None:
        return _FUNCTION

    import torch

    class _FastLoraSingleProjection(torch.autograd.Function):
        """``Y = X @ W^T + b + s * (X @ A^T) @ B^T``, hand-written backward.

        ``W`` and ``b`` are frozen during LoRA training, so the backward returns
        ``None`` for both, and the input gradient still carries the base term
        ``dY @ W``. ``H`` (``[N, r]``) is saved rather than recomputed: the
        backward needs it twice (``dA`` and ``dB``) and it is rank-sized; the
        saved-bytes test reports the actual delta against peft's graph.

        NF4: ``weight`` is the packed tensor and ``quant_state`` carries the
        per-block tensors. The forward dequantises for the base matmul only;
        the backward rebuilds the ``QuantState`` from the saved tensors and
        dequantises again for ``dX``.
        """

        @staticmethod
        def forward(ctx, x, weight, bias, lora_a, lora_b, scaling, quant_state):
            import torch.nn.functional as functional

            h = functional.linear(_as_dtype(x, lora_a.dtype), lora_a)  # [N, r]
            if quant_state is None:
                out = functional.linear(x, weight, bias)
                ctx.save_for_backward(x, weight, lora_a, lora_b, h)
                ctx.qmeta = None
            else:
                from bitsandbytes.functional import dequantize_4bit

                # Transient on purpose: under streaming this dense matrix would
                # alias a pool slot, so it must not outlive the call.
                dense = dequantize_4bit(weight, quant_state).to(x.dtype)
                out = functional.linear(
                    x, dense, _as_dtype(bias, x.dtype) if bias is not None else None
                )
                del dense
                qs_tensors, qs_meta = _quant_state_parts(quant_state)
                ctx.save_for_backward(x, weight, lora_a, lora_b, h, *qs_tensors)
                ctx.qmeta = qs_meta

            ctx.scaling = float(scaling)
            # ``torch.addmm(out, h, lora_b.t(), alpha=s)`` is the 2-D form of
            # this sum. matmul + add keeps one code path that also covers the
            # ``[B, S, H]`` inputs transformers hands to real projections, and
            # the add's ``alpha`` still folds the scaling into the term.
            lora_term = torch.matmul(h, lora_b.t())
            if out.dtype == h.dtype:
                out = torch.add(out, lora_term, alpha=ctx.scaling)
            else:
                # Matches peft's promote-then-cast sum for mixed dtypes, at the
                # cost of one temporary; the uniform case never takes this path.
                promoted = torch.promote_types(out.dtype, h.dtype)
                out = torch.add(out.to(promoted), lora_term.to(promoted), alpha=ctx.scaling).to(
                    out.dtype
                )
            return out

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_out):
            scaling = ctx.scaling
            x, weight, lora_a, lora_b, h = ctx.saved_tensors[:5]

            # dB sums over every leading dimension, so flatten it; dB and dA
            # are weight-shaped regardless of the input's rank.
            grad_b = _as_dtype(_flatten(grad_out), h.dtype).t() @ _flatten(h) * scaling
            # dH keeps the input's leading dimensions; dA = dH^T @ X.
            grad_h = torch.matmul(_as_dtype(grad_out, lora_b.dtype), lora_b) * scaling
            grad_a = _flatten(grad_h).t() @ _as_dtype(_flatten(x), grad_h.dtype)

            grad_x = None
            if ctx.needs_input_grad[0]:
                if ctx.qmeta is None:
                    dense = weight
                else:
                    from bitsandbytes.functional import dequantize_4bit

                    quant_state = _rebuild_quant_state(ctx.qmeta, list(ctx.saved_tensors[5:]))
                    dense = dequantize_4bit(weight, quant_state).to(x.dtype)

                # dX = dY @ W, plus the LoRA term accumulated into the same
                # result: one add instead of separate dH @ A and sum
                # allocations. Skipped entirely when X needs no grad, which
                # recovers the 2x the unskipped dX GEMM costs on a layer-0
                # projection.
                grad_x = torch.matmul(grad_out, _as_dtype(dense, grad_out.dtype))
                grad_x = torch.add(
                    grad_x,
                    torch.matmul(
                        _as_dtype(grad_h, grad_x.dtype), _as_dtype(lora_a, grad_x.dtype)
                    ),
                )
            return grad_x, None, None, grad_a, grad_b, None, None

    _FUNCTION = _FastLoraSingleProjection
    return _FUNCTION



def _make_patched_forward(original_forward: Any) -> Any:
    """Wrap a bound peft forward; delegate whenever peft's own path must run."""
    fast = _single_projection_function()

    def _fast_lora_single_forward(self, x, *args, **kwargs):
        if args or kwargs:
            # Mixed-batch ``adapter_names`` and variant kwargs are peft's.
            return original_forward(x, *args, **kwargs)
        if getattr(self, "disable_adapters", False) or getattr(self, "merged", False):
            return original_forward(x)
        active = getattr(self, "active_adapters", None) or []
        if len(active) != 1:
            return original_forward(x)
        adapter = active[0]
        if adapter not in self.lora_A:
            return original_forward(x)
        if adapter in getattr(self, "lora_variant", {}):
            return original_forward(x)
        dropout = self.lora_dropout[adapter]
        if float(getattr(dropout, "p", 0.0)) != 0.0:
            # The trainer will refuse dropout alongside this flag once the
            # follow-up in #839 lands; until then a direct caller gets peft's
            # own path rather than silently unregularised math.
            return original_forward(x)

        # ``fan_in_fan_out`` means the base weight is stored transposed (GPT-2's
        # ``Conv1D``), so ``dY @ W`` and the LoRA term are both computed against
        # the wrong layout: on a square ``attn.c_proj`` that is silently wrong
        # and on a non-square ``mlp.c_proj`` it raises. #839 scopes Conv1D out,
        # so hand it back to peft.
        #
        # This is the only structural exclusion the kernel needs. peft decides
        # the flag from the base type and overrides a caller who sets it the
        # other way (its own warning names the module), so on a matched target
        # ``fan_in_fan_out`` is True exactly for a ``Conv1D`` base. A separate
        # "is the base nn.Linear" check would be unreachable.
        if getattr(self, "fan_in_fan_out", False):
            return original_forward(x)
        # ``lora_bias=True`` adds a bias on ``lora_B`` that the kernel does not
        # carry; dropping it would silently change the output.
        if getattr(self.lora_B[adapter], "bias", None) is not None:
            return original_forward(x)

        lora_a = self.lora_A[adapter].weight
        lora_b = self.lora_B[adapter].weight
        scaling = float(self.scaling[adapter])
        base = self.get_base_layer()

        weight = base.weight
        quant_state = getattr(weight, "quant_state", None)
        if quant_state is None and getattr(base, "quant_state", None) is not None:
            # Streamed layers rebuild ``Params4bit`` views; the state can live
            # on the module until the next ``.cuda()`` move (dequant precedent).
            weight.quant_state = base.quant_state
            quant_state = weight.quant_state

        bias = getattr(base, "bias", None)
        if quant_state is None:
            return fast.apply(x, weight, bias, lora_a, lora_b, scaling, None)

        if not getattr(base, "compute_type_is_set", True):
            base.set_compute_type(x)
            base.compute_type_is_set = True
        inp_dtype = x.dtype
        if getattr(base, "compute_dtype", None) is not None:
            x = x.to(base.compute_dtype)
        out = fast.apply(x, weight, bias, lora_a, lora_b, scaling, quant_state)
        return out.to(inp_dtype)

    return _fast_lora_single_forward


def patch_fast_lora_single_projection(model: Any) -> int:
    """Replace the forward of every single-projection LoRA linear in ``model``.

    Returns the number of patched modules so a caller can assert it patched
    something (``install_dequant_forward`` precedent). Already-patched modules
    are skipped, which makes the call idempotent.
    """
    import types

    from peft.tuners.lora import Linear as LoraLinear

    types_to_match: tuple[type, ...] = (LoraLinear,)
    try:
        from peft.tuners.lora import bnb as lora_bnb

        types_to_match = (LoraLinear, lora_bnb.Linear4bit)
    except Exception:  # noqa: BLE001 - bitsandbytes absence is a normal install
        pass

    targets = []
    for child in model.modules():
        if getattr(child, _PATCH_MARKER, False):
            continue
        if isinstance(child, types_to_match):
            targets.append(child)

    for child in targets:
        setattr(child, _ORIGINAL_FORWARD_MARKER, child.forward)
        setattr(child, _PATCH_MARKER, True)
        child.forward = types.MethodType(_make_patched_forward(child.forward), child)
    return len(targets)


def unpatch_fast_lora_single_projection(model: Any) -> int:
    """Restore the original peft forwards. Returns the number restored."""
    restored = 0
    for child in model.modules():
        original = getattr(child, _ORIGINAL_FORWARD_MARKER, None)
        if original is None or not getattr(child, _PATCH_MARKER, False):
            continue
        child.forward = original
        delattr(child, _ORIGINAL_FORWARD_MARKER)
        delattr(child, _PATCH_MARKER)
        restored += 1
    return restored
