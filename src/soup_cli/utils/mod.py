"""Mixture-of-Depths (MoD) selective-token routing — v0.41.0 schema / v0.71.12 #84 live.

MoD (`https://arxiv.org/abs/2404.02258`) adds a per-layer router that selects a
top-k subset of tokens (``k = floor(seq_len * capacity_factor)``) to receive the
transformer block's residual update; the remaining tokens bypass the block via
the residual connection. This trades a small accuracy delta for reduced
effective compute on the unselected tokens.

v0.41.0 Part C shipped the ``training.use_mod`` schema flag with a deferred
``NotImplementedError`` stub (the live patch lived in ``expand_model_blocks``'s
sibling slot). v0.71.12 #84 lifts the stub:

1. ``apply_mod_patch(model, capacity_factor=0.125)`` attaches a small router
   (``nn.Linear(hidden, 1)``) to each decoder layer and wraps the layer
   ``forward`` so only the top-k tokens receive the block's residual update,
   gated by the router weight (so the router learns which tokens to route).
2. ``apply_mod_if_configured`` is the shared SFT / Pretrain wiring helper
   (mirrors ``block_expansion.apply_block_expansion_if_configured``).

Architecture allowlist: Llama-family + Qwen + Mistral (confirmed in upstream
MoD implementations). Unknown / null-byte / empty model names are rejected via
the same word-boundary detectors used by LongLoRA (v0.39.0 ``is_gemma4_model``
policy). Applied AFTER ``get_peft_model`` so the freshly-added routers are
trainable (PEFT froze the base earlier), and the layer-forward wrapper calls the
LoRA-injected projections inside the block.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

MIN_CAPACITY_FACTOR: float = 0.0  # exclusive
MAX_CAPACITY_FACTOR: float = 1.0  # inclusive
DEFAULT_CAPACITY_FACTOR: float = 0.125


def validate_capacity_factor(value: object) -> float:
    """Validate the MoD ``capacity_factor`` — finite float in ``(0, 1]``.

    Rejects bool (subclass of int — project policy), non-numeric, NaN / ±inf,
    and out-of-range values.
    """
    if isinstance(value, bool):
        raise TypeError(f"capacity_factor must not be bool, got {value!r}")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"capacity_factor must be numeric, got {type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"capacity_factor must be finite, got {value!r}")
    if fval <= MIN_CAPACITY_FACTOR:
        raise ValueError(
            f"capacity_factor must be > {MIN_CAPACITY_FACTOR}, got {fval}"
        )
    if fval > MAX_CAPACITY_FACTOR:
        raise ValueError(
            f"capacity_factor must be <= {MAX_CAPACITY_FACTOR}, got {fval}"
        )
    return fval


def mod_capacity(seq_len: int, capacity_factor: object) -> int:
    """Number of tokens routed through a block: ``floor(seq_len * factor)``.

    Clamped to ``[1, seq_len]`` so at least one token is always routed and the
    capacity never exceeds the sequence length. Rejects bool / non-int
    ``seq_len`` and out-of-range ``capacity_factor`` (delegates to
    :func:`validate_capacity_factor`).
    """
    if isinstance(seq_len, bool) or not isinstance(seq_len, int):
        raise TypeError(f"seq_len must be int, got {type(seq_len).__name__}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    cf = validate_capacity_factor(capacity_factor)
    cap = int(math.floor(seq_len * cf))
    return max(1, min(seq_len, cap))


def is_mod_supported_arch(model_name: object) -> bool:
    """Return True if ``model_name`` is in the MoD allowlist (Llama / Qwen / Mistral).

    Defensive surface — returns ``False`` (never raises) on non-string / null-byte
    / oversize input (matches ``is_supported_longlora_arch`` policy).
    """
    if not isinstance(model_name, str):
        return False
    from soup_cli.utils.longlora import (
        is_llama_model,
        is_mistral_model,
        is_qwen_model,
    )

    try:
        return (
            is_llama_model(model_name)
            or is_qwen_model(model_name)
            or is_mistral_model(model_name)
        )
    except (TypeError, ValueError):
        return False


def _find_decoder_layers(model: Any) -> Any:
    """Return the decoder ``layers`` ModuleList, handling PEFT-wrapped models."""
    candidates = [model]
    if hasattr(model, "get_base_model"):
        try:
            candidates.append(model.get_base_model())
        except Exception:  # noqa: BLE001 — best-effort PEFT unwrap
            pass
    for cand in candidates:
        if cand is None:
            continue
        inner = getattr(cand, "model", None)
        if inner is None:
            inner = cand
        layers = getattr(inner, "layers", None)
        if layers is None and hasattr(inner, "model"):
            layers = getattr(inner.model, "layers", None)
        if layers is None and hasattr(inner, "decoder"):
            layers = getattr(inner.decoder, "layers", None)
        if layers is not None:
            return layers
    return None


def _resolve_hidden_size(model: Any, layers: Any) -> int | None:
    """Best-effort hidden size for the router input dim."""
    for obj in (model, getattr(model, "get_base_model", lambda: None)()):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            hs = getattr(cfg, "hidden_size", None)
            if isinstance(hs, int) and not isinstance(hs, bool) and hs > 0:
                return hs
    # Fallback — infer from a decoder block's float param last-dim.
    try:
        for layer in layers:
            for p in layer.parameters():
                if p.dtype.is_floating_point and p.dim() >= 1:
                    return int(p.shape[-1])
    except Exception:  # noqa: BLE001
        pass
    return None


def _gather_dim1(tensor, idx):
    """Gather ``tensor`` along its sequence dim (1) by ``idx`` ``[B, cap]``.

    Handles a leading batch of 1 (broadcast RoPE cos/sin) by expanding first,
    and trailing feature dims of any rank. ``[B, T]`` -> ``[B, cap]``;
    ``[B, T, D]`` -> ``[B, cap, D]``; ``[1, T, D]`` -> ``[B, cap, D]``.
    """
    import torch

    bsz, cap = idx.shape
    lead = tensor.shape[0]
    extra = tuple(tensor.shape[2:])
    if lead == 1 and bsz > 1:
        tensor = tensor.expand(bsz, *tensor.shape[1:])
    view = idx.reshape(bsz, cap, *([1] * len(extra))).expand(bsz, cap, *extra)
    return torch.gather(tensor, 1, view)


def _gather_block_inputs(hidden_states, topk, args, kwargs):
    """Gather the decoder-block inputs for the selected sub-sequence.

    Returns ``(sub_hidden, sub_kwargs)`` when the positional inputs can be
    gathered SAFELY, else ``None`` (the caller then falls back to running the
    full block + gate-blend). Conservative by design: any positional forward
    args, a KV cache, ``use_cache``, or a non-4D-causal attention mask abort the
    savings path so a mis-gathered mask can never silently corrupt attention.
    """
    import torch

    if args:  # older positional signature — don't risk mis-identifying args
        return None
    if kwargs.get("use_cache"):
        return None
    if (
        kwargs.get("past_key_value") is not None
        or kwargs.get("past_key_values") is not None
    ):
        return None

    bsz, cap = topk.shape
    seq_len = hidden_states.shape[1]
    hidden = hidden_states.shape[-1]
    new_kwargs = dict(kwargs)

    sub_hidden = _gather_dim1(hidden_states, topk)  # [B, cap, H]
    if sub_hidden.shape != (bsz, cap, hidden):
        return None

    am = kwargs.get("attention_mask")
    if am is not None:
        # Only a square 4D causal mask [B, h, T, T] can be safely narrowed to
        # [B, h, cap, cap] (gather query dim then key dim). Anything else aborts.
        if am.dim() != 4 or am.shape[-1] != seq_len or am.shape[-2] != seq_len:
            return None
        heads = am.shape[1]
        qi = topk.reshape(bsz, 1, cap, 1).expand(bsz, heads, cap, seq_len)
        am_q = torch.gather(am, 2, qi)  # [B, h, cap, T]
        ki = topk.reshape(bsz, 1, 1, cap).expand(bsz, heads, cap, cap)
        new_kwargs["attention_mask"] = torch.gather(am_q, 3, ki)  # [B, h, cap, cap]

    pid = kwargs.get("position_ids")
    if pid is not None:
        if pid.dim() != 2 or pid.shape[-1] != seq_len:
            return None
        new_kwargs["position_ids"] = _gather_dim1(pid, topk)

    cache_pos = kwargs.get("cache_position")
    if cache_pos is not None:
        if cache_pos.dim() != 1 or cache_pos.shape[0] != seq_len:
            return None
        # [T] -> gather with the first row of indices (cache_position is shared)
        new_kwargs["cache_position"] = cache_pos.index_select(0, topk[0])

    pe = kwargs.get("position_embeddings")
    if pe is not None:
        if not (isinstance(pe, (tuple, list)) and len(pe) == 2):
            return None
        gathered = []
        for part in pe:
            if (
                not hasattr(part, "dim")
                or part.dim() != 3
                or part.shape[1] != seq_len
            ):
                return None
            gathered.append(_gather_dim1(part, topk))
        new_kwargs["position_embeddings"] = tuple(gathered)

    return sub_hidden, new_kwargs


def _make_mod_forward(original, router, capacity_factor: float):
    """Wrap a decoder-layer forward to route only the top-k tokens.

    The router scores every token; the top ``floor(T * capacity_factor)`` tokens
    are GATHERED into a shorter sub-sequence, the block runs on ONLY those
    tokens (real compute saving), and the gated result is scattered back —
    unselected tokens keep their input hidden state unchanged. When the block's
    positional inputs can't be safely gathered, it falls back to running the
    full block once and gate-blending (correct, but no saving). Never crashes
    training (returns the original output on any shape mismatch).
    """

    def _fallback(hidden_states, args, kwargs, router_logits, topk):
        import torch

        out = original(hidden_states, *args, **kwargs)
        new_hidden = out[0] if isinstance(out, tuple) else out
        if (
            not hasattr(new_hidden, "shape")
            or new_hidden.shape != hidden_states.shape
        ):
            return out
        mask = torch.zeros_like(router_logits)
        mask.scatter_(1, topk, 1.0)
        weights = (torch.sigmoid(router_logits) * mask).unsqueeze(-1)
        blended = hidden_states + weights * (new_hidden - hidden_states)
        if isinstance(out, tuple):
            return (blended,) + tuple(out[1:])
        return blended

    def mod_forward(hidden_states, *args, **kwargs):
        import torch

        if not hasattr(hidden_states, "shape") or len(hidden_states.shape) != 3:
            return original(hidden_states, *args, **kwargs)
        seq_len = hidden_states.shape[1]
        try:
            cap = mod_capacity(int(seq_len), capacity_factor)
        except (TypeError, ValueError):
            return original(hidden_states, *args, **kwargs)
        if cap >= seq_len:
            return original(hidden_states, *args, **kwargs)  # no routing benefit

        router_logits = router(hidden_states).squeeze(-1)  # [B, T]
        # Sort the selected indices ascending so the gathered sub-sequence keeps
        # causal order for its self-attention + RoPE positions.
        topk = torch.topk(router_logits, k=cap, dim=-1).indices  # [B, cap]
        topk, _ = torch.sort(topk, dim=-1)

        gathered = _gather_block_inputs(hidden_states, topk, args, kwargs)
        if gathered is None:
            return _fallback(hidden_states, args, kwargs, router_logits, topk)

        sub_hidden, sub_kwargs = gathered
        sub_out = original(sub_hidden, **sub_kwargs)  # block on cap tokens only
        sub_new = sub_out[0] if isinstance(sub_out, tuple) else sub_out
        if not hasattr(sub_new, "shape") or sub_new.shape != sub_hidden.shape:
            # Unexpected sub-block output — bail to the safe full path.
            return _fallback(hidden_states, args, kwargs, router_logits, topk)

        # Learnable depth gate on the selected tokens.
        sel_logits = torch.gather(router_logits, 1, topk)  # [B, cap]
        gate = torch.sigmoid(sel_logits).unsqueeze(-1)  # [B, cap, 1]
        sub_updated = sub_hidden + gate * (sub_new - sub_hidden)

        idx = topk.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        result = hidden_states.scatter(1, idx, sub_updated)
        if isinstance(sub_out, tuple):
            return (result,) + tuple(sub_out[1:])
        return result

    mod_forward.__name__ = "mod_forward"
    return mod_forward


def apply_mod_patch(model: Any, *, capacity_factor: object = DEFAULT_CAPACITY_FACTOR) -> int:
    """Attach a MoD router to each decoder layer and wrap its forward.

    Returns the number of layers patched. Idempotent per layer (a layer
    already carrying ``_soup_mod_patched`` is skipped). Best-effort: returns 0
    if the decoder layers / hidden size cannot be resolved (no crash).
    """
    from torch import nn

    cf = validate_capacity_factor(capacity_factor)
    layers = _find_decoder_layers(model)
    if layers is None or not hasattr(layers, "__len__") or len(layers) == 0:
        return 0
    hidden = _resolve_hidden_size(model, layers)
    if hidden is None:
        return 0

    patched = 0
    for layer in layers:
        if getattr(layer, "_soup_mod_patched", False):
            continue
        router = nn.Linear(hidden, 1, bias=False)
        # Zero-init the router so the initial routing does not perturb the
        # pretrained model — all logits 0, sigmoid 0.5; the top-k selection is
        # arbitrary at step 0 but the gate magnitude is small + learnable.
        nn.init.zeros_(router.weight)
        router._soup_is_mod_router = True  # type: ignore[attr-defined]
        # Match the layer's device + a floating dtype (4-bit bases keep their
        # quantised weights in uint8; the router must use the compute dtype).
        device = None
        dtype = None
        try:
            for p in layer.parameters():
                if device is None:
                    device = p.device
                if p.dtype.is_floating_point:
                    dtype = p.dtype
                    break
        except Exception:  # noqa: BLE001
            pass
        if device is not None or dtype is not None:
            router = router.to(device=device, dtype=dtype)
        layer.add_module("_soup_mod_router", router)
        original_forward = layer.forward
        layer.forward = _make_mod_forward(original_forward, router, cf)
        layer._soup_mod_patched = True  # type: ignore[attr-defined]
        patched += 1
    return patched


def apply_mod_if_configured(
    model: Any,
    tcfg: Any,
    base: str,
    console: Any | None = None,
) -> int:
    """Shared SFT / Pretrain wiring helper for Mixture-of-Depths.

    Returns the number of layers patched (0 when ``use_mod`` is off or the
    architecture is unsupported). Designed to be called AFTER
    ``get_peft_model`` so the routers (added fresh) are trainable. Mirrors
    ``block_expansion.apply_block_expansion_if_configured``.
    """
    if not getattr(tcfg, "use_mod", False):
        return 0
    if not is_mod_supported_arch(base):
        # Best-effort skip (matches the surgical-patch policy) — MoD is only
        # validated on Llama / Qwen / Mistral; warn so the no-op is visible.
        import warnings

        warnings.warn(
            f"use_mod=True but base {base!r} is not in the MoD architecture "
            "allowlist (Llama / Qwen / Mistral) — skipping the MoD patch. "
            "Open a feature request to add another architecture.",
            stacklevel=2,
        )
        return 0
    cf = getattr(tcfg, "mod_capacity_factor", DEFAULT_CAPACITY_FACTOR)
    cf = float(cf or DEFAULT_CAPACITY_FACTOR)
    patched = apply_mod_patch(model, capacity_factor=cf)
    if console is not None and patched:
        try:
            console.print(
                f"[green]Mixture-of-Depths:[/] routed {patched} layers "
                f"(capacity_factor={cf})"
            )
        except Exception:  # noqa: BLE001
            pass
    return patched
