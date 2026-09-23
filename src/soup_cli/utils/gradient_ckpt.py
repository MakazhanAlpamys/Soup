"""Gradient checkpointing tiers — selective / medium / full / auto.

Gradient checkpointing trades compute for memory by re-computing activations
during the backward pass instead of storing them. Tiers control *how much*
re-computation happens:

- ``False`` / ``None``      — disabled (no memory savings)
- ``True`` / ``"full"``     — every transformer block (~30% slow, biggest save)
- ``"medium"``              — every other block (balance)
- ``"selective"``           — attention only (~10% slow, modest save)
- ``"auto"``                — pick based on detected VRAM headroom

``resolve_gradient_checkpointing`` returns a kwargs-dict suitable for
``TrainingArguments(**kwargs)``. ``medium`` uses Transformers' native
``every_n_layers`` support. ``selective`` is installed on the model through
``plan_gradient_checkpointing`` and deliberately leaves HuggingFace
checkpointing off, so the same activations are not recomputed twice.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Union

TierLike = Union[bool, str, None]


# Heuristic VRAM thresholds (GB) for tier=auto. Tuned for a 7-8B LoRA run at
# bf16 + 4bit quant + max_length≈4k.
# Below 24 GB: full checkpoint (largest memory savings).
# 24-80 GB:    medium (every other block) — balance.
# >80 GB:      selective (attention only) — minimize slowdown.
AUTO_FULL_THRESHOLD_GB = 24.0
AUTO_SELECTIVE_THRESHOLD_GB = 80.0


def resolve_granularity(
    tier: TierLike, gpu_memory_gb: float | None = None,
) -> str | None:
    """Return the granularity string the wrapper should install hooks for.

    One of: ``"full"`` | ``"medium"`` | ``"selective"`` | ``None`` (disabled).
    ``"auto"`` is resolved to full/medium/selective based on ``gpu_memory_gb``.
    """
    if not tier:
        return None
    if tier is True or tier == "full":
        return "full"
    if tier in ("medium", "selective"):
        return tier  # type: ignore[return-value]
    if tier == "auto":
        if gpu_memory_gb is None:
            return "full"
        if gpu_memory_gb < AUTO_FULL_THRESHOLD_GB:
            return "full"
        if gpu_memory_gb <= AUTO_SELECTIVE_THRESHOLD_GB:
            return "medium"
        return "selective"
    return None


def resolve_gradient_checkpointing(
    tier: TierLike, gpu_memory_gb: float | None = None,
) -> dict[str, Any]:
    """Resolve a gradient_checkpointing setting into TrainingArguments kwargs.

    Only returns keys that HuggingFace's ``TrainingArguments`` actually accepts.
    ``selective`` returns an explicit ``gradient_checkpointing=False`` because
    its attention-only hooks are installed separately by
    :func:`plan_gradient_checkpointing`.

    Args:
        tier: TrainingConfig.gradient_checkpointing value (bool or tier string).
        gpu_memory_gb: GPU memory (GB) used by ``"auto"`` tier. If None, falls
            back to full checkpointing on auto.

    Returns:
        Dict of kwargs suitable for ``TrainingArguments(**kwargs)``:
        - ``gradient_checkpointing`` (bool)
        - ``gradient_checkpointing_kwargs`` (dict)
    """
    granularity = resolve_granularity(tier, gpu_memory_gb=gpu_memory_gb)
    if granularity is None:
        return {}

    if granularity == "selective":
        return {"gradient_checkpointing": False}

    checkpoint_kwargs: dict[str, Any] = {"use_reentrant": False}
    if granularity == "medium":
        # Transformers forwards this separately to
        # model.gradient_checkpointing_enable(every_n_layers=2). It is not a
        # torch.utils.checkpoint keyword and must stay at this dictionary level.
        checkpoint_kwargs["every_n_layers"] = 2

    return {
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": checkpoint_kwargs,
    }


def install_selective_hooks(model, granularity: str) -> int:
    """Install selective / medium gradient-checkpoint hooks on transformer
    blocks (#44, v0.33.0).

    Iterates the model's named modules looking for transformer-block-shaped
    children, then wraps their ``forward`` with
    ``torch.utils.checkpoint.checkpoint`` based on the granularity:

    - ``"selective"``: only each block's direct attention child (for example
      ``self_attn``), never that child's q/k/v/o descendants
    - ``"medium"``: every second transformer block
    - ``"full"``: every transformer block (kept for symmetry; HF's native
      ``gradient_checkpointing`` already handles full — this path is a
      manual fallback for backends that don't expose that toggle)

    Args:
        model: a torch ``nn.Module`` (typically a HuggingFace model).
        granularity: one of ``"selective"`` / ``"medium"`` / ``"full"``.

    Returns:
        Number of modules that use a hook. Zero is a meaningful signal
        — caller should fall back to HF's native ``gradient_checkpointing``.

    Raises:
        ValueError: when ``granularity`` is not recognised.

    Notes:
        - Pure best-effort; the function never raises on a missing torch
          dependency at call site (it imports inside).
        - Existing Soup checkpoint wrappers are reused, so repeated planning
          is idempotent and never nests one checkpoint inside another.
    """
    if granularity not in {"selective", "medium", "full"}:
        raise ValueError(
            f"granularity must be one of selective/medium/full, "
            f"got {granularity!r}"
        )

    try:
        import torch.utils.checkpoint as ckpt_mod
    except ImportError:
        return 0

    layer_index = 0
    hooked = 0

    def _wrap(module):
        if getattr(module.forward, "__name__", "") == "_checkpointed_forward":
            return
        original_forward = module.forward

        def _checkpointed_forward(*args, **kwargs):
            return ckpt_mod.checkpoint(
                original_forward, *args, use_reentrant=False, **kwargs,
            )

        module.forward = _checkpointed_forward

    for name, module in model.named_modules():
        # Heuristic: HF transformer blocks are named like
        # `model.layers.<i>` (LLaMA), `transformer.h.<i>` (GPT-2),
        # `model.decoder.layers.<i>` (T5/Bart). We match on a numeric suffix.
        parts = name.rsplit(".", 1)
        if len(parts) != 2 or not parts[-1].isdigit():
            continue

        if granularity == "full":
            _wrap(module)
            hooked += 1
        elif granularity == "medium":
            if layer_index % 2 == 0:
                _wrap(module)
                hooked += 1
            layer_index += 1
        elif granularity == "selective":
            # Only direct children. Walking ``named_modules`` here used to wrap
            # self_attn AND q_proj/k_proj/v_proj/o_proj (plus names such as
            # post_attention_layernorm), producing nested checkpoints.
            for child_name, child in module.named_children():
                lc = child_name.lower()
                if (
                    lc in {"attn", "attention", "self_attention"}
                    or lc.endswith("_attn")
                    or lc.endswith("_attention")
                ):
                    _wrap(child)
                    hooked += 1

    return hooked


class GradientCheckpointingPlan(NamedTuple):
    """The model hooks and TrainingArguments kwargs actually selected."""

    kwargs: dict[str, Any]
    granularity: str | None
    hooked_modules: int
    description: str


def plan_gradient_checkpointing(
    model: Any,
    tier: TierLike,
    gpu_memory_gb: float | None = None,
) -> GradientCheckpointingPlan:
    """Apply model-side selective hooks and return truthful HF kwargs.

    Full and medium are native Transformers modes. Selective needs model-side
    hooks; when an architecture exposes no direct attention child, fall back to
    full checkpointing instead of printing a selective mode that did nothing.
    """
    granularity = resolve_granularity(tier, gpu_memory_gb=gpu_memory_gb)
    if granularity is None:
        return GradientCheckpointingPlan({}, None, 0, "off")

    requested_description = describe_tier(tier, gpu_memory_gb)
    if granularity != "selective":
        return GradientCheckpointingPlan(
            resolve_gradient_checkpointing(tier, gpu_memory_gb),
            granularity,
            0,
            requested_description,
        )

    hooked = install_selective_hooks(model, "selective")
    if hooked:
        # PEFT's ``prepare_model_for_kbit_training`` enables native HF
        # checkpointing before this plan runs. ``TrainingArguments(False)``
        # does not undo that model-side state, which would otherwise make a
        # selective run checkpoint both every decoder layer and its attention
        # child. The selective hooks are now the sole checkpointing mechanism.
        if getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
        return GradientCheckpointingPlan(
            resolve_gradient_checkpointing("selective"),
            "selective",
            hooked,
            f"{requested_description}; {hooked} attention module(s)",
        )

    return GradientCheckpointingPlan(
        resolve_gradient_checkpointing("full"),
        "full",
        0,
        f"{requested_description} unavailable on this architecture; full fallback",
    )


def describe_tier(tier: TierLike, gpu_memory_gb: float | None = None) -> str:
    """Return a short human-readable description of the selected tier."""
    if not tier:
        return "off"
    if tier is True or tier == "full":
        return "full (every block)"
    if tier == "medium":
        return "medium (every other block)"
    if tier == "selective":
        return "selective (attention only)"
    if tier == "auto":
        if gpu_memory_gb is None:
            return "auto → full (unknown VRAM)"
        if gpu_memory_gb < AUTO_FULL_THRESHOLD_GB:
            return f"auto → full (VRAM {gpu_memory_gb:.0f}GB)"
        if gpu_memory_gb <= AUTO_SELECTIVE_THRESHOLD_GB:
            return f"auto → medium (VRAM {gpu_memory_gb:.0f}GB)"
        return f"auto → selective (VRAM {gpu_memory_gb:.0f}GB)"
    return str(tier)
