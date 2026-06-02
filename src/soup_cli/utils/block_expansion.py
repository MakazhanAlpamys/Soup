"""LLaMA Pro block expansion — schema in v0.41.0 Part C, live wiring v0.53.4 #83.

LLaMA Pro (`https://arxiv.org/abs/2401.02415`) adds ``N`` zero-initialised
transformer blocks to a base model and freezes the original blocks, training
only the new blocks. The result is a copy-and-extend that preserves base
behaviour while adding capacity in the new domain.

v0.41.0 Part C shipped the schema fields + a ``NotImplementedError`` stub.
v0.53.4 #83 lifts the stub with a real implementation:

1. Clone the last ``num_new_blocks`` decoder blocks via ``copy.deepcopy``.
2. Zero-init the MLP output projection ("down_proj") of each clone so the
   block initially acts as an identity (per the LLaMA Pro paper §3.1).
3. Append the clones to ``model.model.layers`` (HF causal-LM convention).
4. Update ``model.config.num_hidden_layers`` so cached counts stay coherent.

The new-block-only freeze policy is owned by the caller via
``freeze_trainable_layers`` (positive = train top-N, negative = train
bottom-N); ``apply_llama_pro_freeze`` is provided for the canonical
"train only the appended blocks" case.

References:
- LlamaFactory ``freeze_trainable_layers`` (positive = train top-N, negative
  = train bottom-N) + ``expand_layers`` (block count).
"""

from __future__ import annotations

import copy
from typing import Any

_MAX_EXPAND_LAYERS = 64
_MIN_EXPAND_LAYERS = 1


def validate_expand_layers(value: object) -> int:
    """Validate ``training.expand_layers`` (LLaMA Pro)."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"expand_layers must be int, got {type(value).__name__}"
        )
    if value < _MIN_EXPAND_LAYERS or value > _MAX_EXPAND_LAYERS:
        raise ValueError(
            f"expand_layers must be in [{_MIN_EXPAND_LAYERS}, "
            f"{_MAX_EXPAND_LAYERS}], got {value}"
        )
    return int(value)


def validate_freeze_trainable_layers(value: object) -> int:
    """Signed int — positive = train top-N, negative = train bottom-N."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"freeze_trainable_layers must be int, got {type(value).__name__}"
        )
    if abs(value) > 1000:
        raise ValueError(
            f"freeze_trainable_layers magnitude must be <= 1000, got {value}"
        )
    return int(value)


def expand_model_blocks(model: Any, num_new_blocks: int) -> int:
    """Insert ``num_new_blocks`` zero-init transformer layers at the end.

    v0.53.4 #83 — live implementation. Clones the last ``num_new_blocks``
    decoder blocks, zero-inits each clone's MLP output projection
    (``down_proj``) so the block initially acts as an identity, then
    appends the clones to ``model.model.layers``. ``config.num_hidden_layers``
    is updated to match.

    Returns the total number of layers after expansion.

    Passing ``num_new_blocks=0`` (or ``None``) is the no-op path — returns
    the current layer count without mutating the model.
    """
    if num_new_blocks is None or num_new_blocks == 0:
        return _count_layers(model)
    n = validate_expand_layers(num_new_blocks)

    layers = _get_layers_module(model)
    if layers is None:
        raise ValueError(
            "expand_model_blocks: could not find decoder layers list on "
            "the model (expected model.model.layers or model.decoder.layers)."
        )
    original_count = len(layers)
    if original_count == 0:
        raise ValueError(
            "expand_model_blocks: base model has zero decoder layers; "
            "cannot clone."
        )
    # Clone the last ``min(n, original_count)`` blocks — guards against
    # over-expansion on tiny test models. The LLaMA Pro paper interleaves
    # blocks, but appending-then-zero-init is mathematically equivalent
    # when zero-init is applied to the residual path (down_proj output).
    clone_count = min(n, original_count)
    any_zeroed = False
    for offset in range(clone_count):
        source = layers[original_count - clone_count + offset]
        clone = copy.deepcopy(source)
        if _zero_init_block_residual(clone):
            any_zeroed = True
        layers.append(clone)
    if not any_zeroed:
        # v0.53.4 security review LOW — surface that the residual projection
        # path missed every cloned block (non-Llama-shaped arch). The blocks
        # still get appended + trained, but they are NOT identity-initialised,
        # so the user should know their starting point degrades slightly.
        import warnings

        warnings.warn(
            "expand_model_blocks: could not locate standard residual "
            "projections (mlp.down_proj / self_attn.o_proj) on the cloned "
            "blocks — appended blocks are NOT zero-initialised. This is "
            "expected for non-Llama-shaped architectures; training will "
            "proceed but the identity-init guarantee from the LLaMA Pro "
            "paper is lost.",
            stacklevel=2,
        )

    # Keep config in sync so downstream code (e.g. HF generation) sees the
    # new layer count.
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        cfg.num_hidden_layers = len(layers)

    return len(layers)


def apply_llama_pro_freeze(model: Any, num_new_blocks: int) -> int:
    """Freeze every parameter EXCEPT the appended new blocks.

    Returns the count of trainable parameters after freezing. Intended to
    be called immediately after :func:`expand_model_blocks` when the caller
    wants the canonical LLaMA Pro behaviour (train only new blocks).
    """
    if num_new_blocks is None or num_new_blocks == 0:
        return 0
    n = validate_expand_layers(num_new_blocks)
    layers = _get_layers_module(model)
    if layers is None:
        return 0
    total = len(layers)
    new_start = max(0, total - n)
    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze new blocks.
    trainable = 0
    for idx in range(new_start, total):
        for param in layers[idx].parameters():
            param.requires_grad = True
            trainable += param.numel()
    return trainable


def _zero_init_block_residual(block: Any) -> bool:
    """Zero the output projections so the new block is initially identity.

    Targets the standard HF causal-LM layer shape:
      * ``block.mlp.down_proj.weight`` (and bias, if any)
      * ``block.self_attn.o_proj.weight`` (and bias, if any)

    Returns ``True`` if at least one residual projection was zeroed, else
    ``False`` (caller can emit a warning so a silent-degradation surface is
    visible — addresses the v0.53.4 security-review LOW finding on non-Llama
    arches).
    """
    zeroed_any = False
    for path in (("mlp", "down_proj"), ("self_attn", "o_proj")):
        mod = block
        for name in path:
            mod = getattr(mod, name, None)
            if mod is None:
                break
        if mod is None:
            continue
        weight = getattr(mod, "weight", None)
        if weight is not None and hasattr(weight, "data"):
            weight.data.zero_()
            zeroed_any = True
        bias = getattr(mod, "bias", None)
        if bias is not None and hasattr(bias, "data"):
            bias.data.zero_()
    return zeroed_any


def _get_layers_module(model: Any) -> Any:
    """Return the mutable ``layers`` ModuleList, or ``None`` if not found."""
    # Explicit ``is None`` check (code-review HIGH fix) — avoids the falsy
    # shortcut that would silently fall back to ``model`` if ``model.model``
    # overrides ``__bool__`` to return False (some nn.Module subclasses do).
    inner = getattr(model, "model", None)
    if inner is None:
        inner = model
    layers = getattr(inner, "layers", None)
    if layers is None and hasattr(inner, "decoder"):
        layers = getattr(inner.decoder, "layers", None)
    return layers


def apply_block_expansion_if_configured(
    model: Any,
    tcfg: Any,
    console: Any | None = None,
) -> int:
    """Shared SFT/Pretrain helper — run LLaMA Pro expansion + optional freeze.

    Returns the total layer count AFTER (possible) expansion. Callers can
    print(the resulting count themselves. Designed to be called BEFORE)
    ``get_peft_model`` so PEFT's matcher sees the new blocks.

    Mirrors the v0.40.6 ``peft_wiring`` centralisation policy (one source of
    truth for the wiring step across every trainer).
    """
    n = getattr(tcfg, "expand_layers", None)
    if not n:
        return _count_layers(model)
    new_total = expand_model_blocks(model, n)
    if console is not None:
        console.print(
            f"[green]LLaMA Pro:[/] expanded to {int(new_total)} layers "
            f"(+{int(n)} zero-init blocks)"
        )
    freeze = getattr(tcfg, "freeze_trainable_layers", None)
    # Project policy ``is None`` over falsy — but a value of 0 means "no
    # positive freeze direction", so the canonical "train only new blocks"
    # path runs iff the user opted in with a positive ``freeze_trainable_layers``.
    if freeze is not None and freeze > 0:
        trainable = apply_llama_pro_freeze(model, n)
        if console is not None:
            console.print(
                f"[green]LLaMA Pro freeze:[/] {trainable:,} parameters "
                f"trainable (only the {int(n)} new blocks)"
            )
    return new_total


def _count_layers(model: Any) -> int:
    """Best-effort count of decoder layers on an HF causal-LM."""
    layers = _get_layers_module(model)
    if layers is None or not hasattr(layers, "__len__"):
        return 0
    return len(layers)
