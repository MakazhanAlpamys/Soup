"""Shared PEFT wiring helpers (v0.40.6 #67) — multi-trainer ReLoRA + surgical patches.

Centralises the v0.39.0 Part B (ReLoRA callback) and Part D (surgical PEFT
patches) wiring previously inlined only in the SFT trainer. Every
transformer-backend trainer (DPO / GRPO / KTO / ORPO / SimPO / IPO / PPO /
RewardModel / Pretrain / Embedding / BCO) calls these helpers from its
``_setup_transformers`` and ``train`` paths.

Helpers swallow per-patch exceptions at DEBUG level — best-effort by design;
training never crashes because a Gemma4 swap or 3-D dropout strip failed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_pre_lora_patches(model: Any, base: str) -> None:
    """Run pre-LoRA surgical patches (v0.39.0 Part D, multi-trainer in v0.40.6).

    Currently: Gemma4 ``ClippableLinear`` -> ``nn.Linear`` swap. Gated by
    ``is_gemma4_model(base)`` so the swap never runs on non-Gemma4 models.
    """
    from soup_cli.utils.peft_patches import apply_gemma4_clippable_patch, is_gemma4_model

    if not is_gemma4_model(base):
        return
    try:
        apply_gemma4_clippable_patch(model)
    except Exception as exc:  # noqa: BLE001 — best-effort patch, log + continue
        logger.debug("apply_gemma4_clippable_patch skipped: %s", exc)


def apply_post_lora_patches(model: Any) -> None:
    """Run post-LoRA surgical patches (v0.39.0 Part D, multi-trainer in v0.40.6).

    Currently: 3-D fused-MoE expert dropout strip. Architecture-detected via
    ``weight.ndim == 3`` inside the helper; safe to call unconditionally.
    """
    from soup_cli.utils.peft_patches import strip_lora_dropout_for_3d_experts

    try:
        strip_lora_dropout_for_3d_experts(model)
    except Exception as exc:  # noqa: BLE001 — best-effort patch, log + continue
        logger.debug("strip_lora_dropout_for_3d_experts skipped: %s", exc)


def attach_relora_callback(trainer: Any, tcfg: Any) -> bool:
    """Attach :class:`ReLoRACallback` when ``training.relora_steps`` is set.

    Returns ``True`` when a callback was attached, ``False`` otherwise.
    The schema-level cross-validator (``_validate_relora_supported_tasks``)
    already enforces the transformer-backend requirement, so this helper
    trusts the caller's task/backend.
    """
    relora_steps = getattr(tcfg, "relora_steps", None)
    # Use `is None` (not `not relora_steps`) so a schema-bypassing caller that
    # passes `relora_steps=0` surfaces as a loud `ReLoRAPolicy` ValueError
    # rather than a silent skip. Matches project policy (v0.34.0 / v0.39.0).
    if relora_steps is None:
        return False
    # Pydantic schema guarantees these fields exist on `TrainingConfig`. Read
    # them directly so a misnamed attr fails loudly with `AttributeError`.
    from soup_cli.utils.relora import ReLoRACallback, ReLoRAPolicy

    policy = ReLoRAPolicy(
        steps=int(relora_steps),
        warmup_ratio=float(tcfg.relora_warmup_ratio),
        reset_optimizer=bool(tcfg.relora_reset_optimizer),
        prune_ratio=float(tcfg.relora_prune_ratio),
    )
    trainer.add_callback(ReLoRACallback(policy=policy))
    return True
