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
  not model), dropout is non-zero without the explicit opt-in, or the call
  shape is not a plain ``(x)``. Dropout is modelled only when a caller asks
  for it (``patch_fast_lora_single_projection(..., allow_dropout=True)``): the
  keep-mask is drawn and saved (one byte per element) and ``X_d`` is rebuilt
  in the backward.

The patch covers only the single-projection path. Which kernel a projection
gets when the QKV or MLP patcher also wants it is written once, in the comment
above ``_GROUP_PATCH_OWNER_MARKER`` below.

The patched forward reads the base weight at call time on purpose: streamed
layers substitute weights via ``functional_call`` only for the duration of a
call, and a reference captured at patch time under streaming is a meta
placeholder.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_soup_fast_lora_single_projection"
_ORIGINAL_FORWARD_MARKER = "_soup_fast_lora_original_forward"
_HAD_INSTANCE_FORWARD_MARKER = "_soup_fast_lora_had_instance_forward"
# When two patchers want one projection, the group kernel takes it, whichever
# was patched first: shared-X Q/K/V owns q/k/v and the fused SwiGLU MLP owns
# gate/up/down. A group patcher releases this kernel from the projections it
# takes and marks them with _GROUP_PATCH_OWNER_MARKER, which this one skips.
_GROUP_PATCH_OWNER_MARKER = "_soup_fast_lora_group_owner"
_FORWARD_OWNER_MARKER = "_soup_fast_lora_forward_owner"
_OWNER = "single"

__all__ = [
    "patch_fast_lora_single_projection",
    "unpatch_fast_lora_single_projection",
]

_FUNCTION: Any = None


def _as_dtype(tensor: Any, dtype: Any) -> Any:
    """Cast only when the dtype differs, so the uniform case stays op-for-op."""
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _scaled_lora_add(out: Any, hidden: Any, lora_b: Any, scaling: float) -> Any:
    """Add a LoRA update with PEFT's promote, add, then cast semantics."""
    import torch

    lora_term = torch.matmul(hidden, lora_b.t())
    promoted = torch.promote_types(out.dtype, lora_term.dtype)
    return torch.add(
        out.to(promoted), lora_term.to(promoted), alpha=float(scaling)
    ).to(out.dtype)


def _flatten(tensor: Any) -> Any:
    """Collapse the leading dimensions into one, for weight-shaped gradients.

    ``dB`` and ``dA`` sum over every leading dimension regardless of the input
    rank (``[N, in]`` for the unit tests, ``[B, S, in]`` from transformers).
    ``reshape`` is a view for the contiguous activation layouts that reach a
    projection; a non-contiguous grad pays a copy, which is the correct trade
    against a silent shape assumption.
    """
    return tensor.reshape(-1, tensor.shape[-1])


def _dropout_mask(x: Any, p: float) -> tuple[Any, Any]:
    """Draw LoRA dropout's keep-mask and apply it to the adapter input.

    Returns ``(keep, x_d)`` where ``keep`` is boolean, one byte per element.
    The mask is what the backward saves: ``X_d`` is rebuilt from it, so the
    full-precision dropped copy peft's ``lora_A`` retains does not have to be
    kept (#839). The draw uses the global torch RNG, which is what lets a
    checkpoint recompute reproduce the same mask when ``preserve_rng_state``
    is on.
    """
    import torch

    keep = torch.empty_like(x, dtype=torch.bool).bernoulli_(1.0 - p)
    return keep, x * keep / (1.0 - p)


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


class _ProjectionState(NamedTuple):
    weight: Any
    bias: Any
    lora_a: Any
    lora_b: Any
    scaling: float
    qmeta: dict[str, Any] | None
    qparts: list[Any]
    compute_dtype: Any
    dropout_p: float
    dropout_training: bool


def _projection_state(
    proj: Any, x: Any, *, allow_dropout: bool = False, allow_unadapted: bool = False
) -> _ProjectionState | None:
    """Return one PEFT/base projection state or None when PEFT must run.

    This is shared by the single, MLP and QKV kernels so the seven delegation
    guards, streamed QuantState repair and bitsandbytes compute-dtype handling
    cannot drift between patchers.

    ``allow_dropout`` is the single-projection kernel's explicit opt-in: with
    it, a non-zero ``lora_dropout`` returns a state carrying ``dropout_p`` and
    ``dropout_training`` instead of delegating; ``p >= 1`` still delegates,
    since the mask rescale is undefined at 1. The MLP and QKV kernels never
    pass it, so for them the shared check keeps refusing exactly as before.
    """
    lora_a_map = getattr(proj, "lora_A", None)
    if lora_a_map is not None and not _is_supported_lora_projection(proj):
        return None
    base = proj.get_base_layer() if hasattr(proj, "get_base_layer") else proj
    if not hasattr(base, "weight"):
        return None

    adapter = None
    dropout_p = 0.0
    dropout_training = False
    if lora_a_map is None:
        if not allow_unadapted:
            return None
    else:
        if getattr(proj, "disable_adapters", False) or getattr(proj, "merged", False):
            return None
        active = getattr(proj, "active_adapters", None) or []
        if len(active) != 1:
            return None
        adapter = active[0]
        if adapter not in proj.lora_A:
            if not allow_unadapted:
                return None
            adapter = None
        else:
            if adapter in getattr(proj, "lora_variant", {}):
                return None
            # Dropout changes the adapter computation and mask lifetime. The
            # single-projection kernel models the mask when it is asked to
            # (``allow_dropout``); every other caller keeps handing dropout
            # calls back to peft.
            dropout = proj.lora_dropout[adapter]
            dropout_p = float(getattr(dropout, "p", 0.0))
            dropout_training = bool(getattr(dropout, "training", False))
            if dropout_p != 0.0 and not allow_dropout:
                return None
            if allow_dropout and dropout_p >= 1.0:
                # 1 / (1 - p) is undefined at p = 1; peft's native dropout
                # handles that edge, so the opt-in delegates it too.
                return None
            # Conv1D-style/transposed storage needs the PEFT fan-in/fan-out
            # path rather than the Linear weight orientation used below.
            if getattr(proj, "fan_in_fan_out", False):
                return None
            # ``lora_bias`` adds another trained term that these Functions do
            # not accept or differentiate.
            if getattr(proj.lora_B[adapter], "bias", None) is not None:
                return None

    weight = base.weight
    qstate = getattr(weight, "quant_state", None)
    if qstate is None and getattr(base, "quant_state", None) is not None:
        # Streamed Params4bit views can temporarily carry the state on the
        # module rather than on the view itself. Repair the same way as the
        # single-projection path before any sibling reads it.
        try:
            weight.quant_state = base.quant_state
        except AttributeError:
            pass
        qstate = getattr(weight, "quant_state", None) or base.quant_state

    # An unadapted sibling may still be an 8-bit/custom projection. Without a
    # 4-bit QuantState it is not a dense floating weight and cannot use F.linear.
    if qstate is None and not weight.is_floating_point():
        return None

    compute_dtype = None
    qmeta = None
    qparts: list[Any] = []
    if qstate is not None:
        if not getattr(base, "compute_type_is_set", True) and hasattr(base, "set_compute_type"):
            base.set_compute_type(x)
            base.compute_type_is_set = True
        compute_dtype = getattr(base, "compute_dtype", None)
        qparts, qmeta = _quant_state_parts(qstate)
        qmeta = dict(qmeta)
        qmeta["_count"] = len(qparts)

    if adapter is None:
        lora_a = x.new_empty(0)
        lora_b = x.new_empty(0)
        scaling = 0.0
    else:
        lora_a = proj.lora_A[adapter].weight
        lora_b = proj.lora_B[adapter].weight
        scaling = float(proj.scaling[adapter])

    return _ProjectionState(
        weight,
        getattr(base, "bias", None),
        lora_a,
        lora_b,
        scaling,
        qmeta,
        qparts,
        compute_dtype,
        dropout_p,
        dropout_training,
    )


def _supported_lora_projection_types() -> tuple[type, ...]:
    """Return PEFT projection types whose weight contracts the kernels model."""
    from peft.tuners.lora import Linear as LoraLinear

    supported: tuple[type, ...] = (LoraLinear,)
    try:
        from peft.tuners.lora import bnb as lora_bnb

        supported = (LoraLinear, lora_bnb.Linear4bit)
    except Exception:  # noqa: BLE001 - bitsandbytes absence is a normal install
        pass
    return supported


def _is_supported_lora_projection(proj: Any) -> bool:
    """Exclude PEFT 8-bit/custom layers whose storage math differs."""
    return isinstance(proj, _supported_lora_projection_types())


def _dense_weight(
    weight: Any,
    qmeta: dict[str, Any] | None,
    qparts: list[Any],
    dtype: Any,
) -> Any:
    """Return a dense view for one base projection without retaining it."""
    if qmeta is None:
        return _as_dtype(weight, dtype)
    from bitsandbytes.functional import dequantize_4bit

    state = _rebuild_quant_state(qmeta, qparts)
    return dequantize_4bit(weight, state).to(dtype)


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
        """``Y = X @ W^T + b + s * (X_d @ A^T) @ B^T``, hand-written backward.

        ``X_d`` is ``X`` unless dropout is modelled (the opt-in path); there it
        is ``X * keep / (1 - p)`` with ``keep`` the saved boolean mask, and the
        base term above still reads the undropped ``X``.

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
        def forward(
            ctx, x, weight, bias, lora_a, lora_b, scaling, quant_state, dropout_p, dropout_training
        ):
            import torch.nn.functional as functional

            keep = None
            x_lora = x
            if dropout_p > 0.0 and dropout_training:
                keep, x_lora = _dropout_mask(x, dropout_p)
            h = functional.linear(_as_dtype(x_lora, lora_a.dtype), lora_a)  # [N, r]
            if quant_state is None:
                out = functional.linear(x, weight, bias)
                ctx.save_for_backward(
                    x, weight, lora_a, lora_b, h, *((keep,) if keep is not None else ())
                )
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
                ctx.save_for_backward(
                    x,
                    weight,
                    lora_a,
                    lora_b,
                    h,
                    *((keep,) if keep is not None else ()),
                    *qs_tensors,
                )
                ctx.qmeta = qs_meta

            ctx.scaling = float(scaling)
            ctx.dropout_p = float(dropout_p)
            ctx.dropout_active = keep is not None
            ctx.qstart = 5 + (1 if keep is not None else 0)
            # ``torch.addmm(out, h, lora_b.t(), alpha=s)`` is the 2-D form of
            # this sum. matmul + add keeps one code path that also covers the
            # ``[B, S, H]`` inputs transformers hands to real projections, and
            # the add's ``alpha`` still folds the scaling into the term.
            out = _scaled_lora_add(out, h, lora_b, ctx.scaling)
            return out

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_out):
            scaling = ctx.scaling
            # One access only: under a non-reentrant checkpoint the saved
            # tensors go through an unpack hook that refuses a second read.
            saved = ctx.saved_tensors
            x, weight, lora_a, lora_b, h = saved[:5]
            keep = saved[5] if ctx.dropout_active else None
            # dA contracts against the DROPPED input; it is rebuilt from the
            # saved mask rather than retaining peft's full-precision copy.
            x_lora = x if keep is None else x * keep / (1.0 - ctx.dropout_p)

            # dB sums over every leading dimension, so flatten it; dB and dA
            # are weight-shaped regardless of the input's rank.
            grad_b = _as_dtype(_flatten(grad_out), h.dtype).t() @ _flatten(h) * scaling
            # dH keeps the input's leading dimensions; dA = dH^T @ X_d.
            grad_h = torch.matmul(_as_dtype(grad_out, lora_b.dtype), lora_b) * scaling
            grad_a = _flatten(grad_h).t() @ _as_dtype(_flatten(x_lora), grad_h.dtype)

            grad_x = None
            if ctx.needs_input_grad[0]:
                if ctx.qmeta is None:
                    dense = weight
                else:
                    from bitsandbytes.functional import dequantize_4bit

                    quant_state = _rebuild_quant_state(ctx.qmeta, list(saved[ctx.qstart :]))
                    dense = dequantize_4bit(weight, quant_state).to(x.dtype)

                # dX = dY @ W, plus the LoRA term accumulated into the same
                # result: one add instead of separate dH @ A and sum
                # allocations. Skipped entirely when X needs no grad, which
                # recovers the 2x the unskipped dX GEMM costs on a layer-0
                # projection.
                grad_x = torch.matmul(grad_out, _as_dtype(dense, grad_out.dtype))
                grad_lora = torch.matmul(
                    _as_dtype(grad_h, grad_x.dtype), _as_dtype(lora_a, grad_x.dtype)
                )
                # The LoRA branch read the dropped input, so its gradient takes
                # the mask and the 1/(1-p) rescale on the way back to X.
                if keep is not None:
                    grad_lora = grad_lora * keep / (1.0 - ctx.dropout_p)
                grad_x = torch.add(grad_x, grad_lora)
            return grad_x, None, None, grad_a, grad_b, None, None, None, None

    _FUNCTION = _FastLoraSingleProjection
    return _FUNCTION


def _make_patched_forward(original_forward: Any, *, allow_dropout: bool = False) -> Any:
    """Wrap a bound peft forward; delegate whenever peft's own path must run.

    ``allow_dropout`` is the explicit opt-in for this kernel: with it, a
    non-zero LoRA dropout is modelled (the mask is drawn, saved and applied in
    the backward) instead of delegating to peft.
    """
    fast = _single_projection_function()

    def _fast_lora_single_forward(self, x, *args, **kwargs):
        if args or kwargs:
            return original_forward(x, *args, **kwargs)
        state = _projection_state(self, x, allow_dropout=allow_dropout)
        if state is None:
            return original_forward(x)

        input_dtype = x.dtype
        work_x = x
        if state.compute_dtype is not None and work_x.dtype != state.compute_dtype:
            work_x = work_x.to(state.compute_dtype)

        out = fast.apply(
            work_x,
            state.weight,
            state.bias,
            state.lora_a,
            state.lora_b,
            state.scaling,
            None if state.qmeta is None else _rebuild_quant_state(state.qmeta, state.qparts),
            state.dropout_p,
            state.dropout_training,
        )
        return out if work_x is x else out.to(input_dtype)

    setattr(_fast_lora_single_forward, _FORWARD_OWNER_MARKER, _OWNER)
    return _fast_lora_single_forward


def patch_fast_lora_single_projection(model: Any, *, allow_dropout: bool = False) -> int:
    """Replace the forward of every single-projection LoRA linear in ``model``.

    Returns the number of patched modules so a caller can assert it patched
    something (``install_dequant_forward`` precedent). Already-patched modules
    are skipped, which makes the call idempotent.

    ``allow_dropout`` opts this kernel into modelling non-zero LoRA dropout;
    without it, dropout calls keep delegating to peft, which is the default
    the MLP and QKV kernels and the tracker's validators assume.
    """
    import types

    types_to_match = _supported_lora_projection_types()

    targets = []
    for child in model.modules():
        if getattr(child, _PATCH_MARKER, False):
            continue
        if getattr(child, _GROUP_PATCH_OWNER_MARKER, None) is not None:
            continue
        if isinstance(child, types_to_match):
            targets.append(child)

    for child in targets:
        setattr(child, _ORIGINAL_FORWARD_MARKER, child.forward)
        setattr(child, _HAD_INSTANCE_FORWARD_MARKER, "forward" in vars(child))
        setattr(child, _PATCH_MARKER, True)
        child.forward = types.MethodType(
            _make_patched_forward(child.forward, allow_dropout=allow_dropout), child
        )
    return len(targets)


def eligible_single_projections(model: Any) -> int:
    """Patched single projections the kernel can run (#792).

    A ``fan_in_fan_out`` LoRA (GPT-2's Conv1D layout) is patched like any other
    but delegates to peft on every call, so it is not counted.
    """
    return sum(
        1
        for child in model.modules()
        if getattr(child, _PATCH_MARKER, False) and not getattr(child, "fan_in_fan_out", False)
    )


def _release_single_projection(child: Any) -> bool:
    """Give ``child`` back the forward it had before this kernel; ``True`` if it had it.

    A forward someone else installed over this kernel's is left in place.
    """
    if not getattr(child, _PATCH_MARKER, False):
        return False
    if getattr(child.forward, _FORWARD_OWNER_MARKER, None) == _OWNER:
        if getattr(child, _HAD_INSTANCE_FORWARD_MARKER, False):
            child.forward = getattr(child, _ORIGINAL_FORWARD_MARKER)
        else:
            del child.forward
    for attr in (_ORIGINAL_FORWARD_MARKER, _HAD_INSTANCE_FORWARD_MARKER, _PATCH_MARKER):
        if attr in vars(child):
            delattr(child, attr)
    return True


def unpatch_fast_lora_single_projection(model: Any) -> int:
    """Restore the original peft forwards. Returns the number restored."""
    return sum(_release_single_projection(child) for child in list(model.modules()))
