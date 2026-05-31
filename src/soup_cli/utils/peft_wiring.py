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


def attach_curriculum_callback(
    trainer: Any,
    tcfg: Any,
    output_dir: str,
    console: Any = None,
) -> bool:
    """Attach :class:`DynamicCurriculumCallback` when ``curriculum_dynamic=true``.

    Returns ``True`` when attached, ``False`` otherwise. The schema-level
    cross-validator (``_validate_curriculum_dynamic_supported``) gates by
    backend / task, so this helper trusts the caller's config.

    Args:
        trainer: HF Trainer (or duck-typed equivalent with ``add_callback``).
        tcfg: ``SoupConfig.training`` model.
        output_dir: Directory under cwd to write
            ``curriculum_history.jsonl`` (the BETA history record).
        console: Optional Rich Console for the BETA advisory.
    """
    if not getattr(tcfg, "curriculum_dynamic", False):
        return False
    # Lazy import — the callback module touches transformers + torch.
    from soup_cli.monitoring.curriculum_callback import (
        DynamicCurriculumCallback,
    )
    from soup_cli.utils.curriculum_dynamic import DynamicCurriculumPolicy

    policy = DynamicCurriculumPolicy(
        num_buckets=int(tcfg.curriculum_buckets),
        recompute_every_n_steps=int(
            getattr(tcfg, "curriculum_dynamic_recompute_steps", 50) or 50
        ),
        floor=float(getattr(tcfg, "curriculum_dynamic_floor", 0.05) or 0.05),
        temperature=float(
            getattr(tcfg, "curriculum_dynamic_temperature", 1.0) or 1.0
        ),
    )
    try:
        callback = DynamicCurriculumCallback(policy=policy, output_dir=output_dir)
    except (TypeError, ValueError) as exc:
        logger.debug("attach_curriculum_callback rejected: %s", exc)
        return False
    trainer.add_callback(callback)
    if console is not None:
        try:
            console.print(
                "[yellow]BETA:[/yellow] dynamic curriculum callback attached "
                f"(buckets={policy.num_buckets}, recompute_every="
                f"{policy.recompute_every_n_steps})"
            )
        except Exception:  # noqa: BLE001 — never crash on console issues.
            pass
    return True


def attach_grpo_stability_callback(trainer: Any, tcfg: Any) -> bool:
    """Attach :class:`GRPOStabilityCallback` when any v0.50.0 Part D knob is set.

    Returns ``True`` when a callback was attached, ``False`` otherwise.
    Mirrors the v0.40.6 / v0.53.5 / v0.53.6 callback-attach pattern.

    The schema-level cross-validator already gates these fields to
    ``task='grpo'`` on non-mlx backends, so this helper trusts the caller.
    """
    stability_fields = (
        "ref_model_ema_alpha",
        "replay_buffer_size",
        "async_grpo_prefetch",
        "tis_threshold",
        "mask_truncated_completions",
        "defer_rerolling",
        "skip_zero_advantage",
        "off_policy_mask_threshold",
    )
    # `is None` policy (matches v0.40.6 review-fix policy on `attach_relora_callback`)
    has_any = False
    for field_name in stability_fields:
        val = getattr(tcfg, field_name, None)
        # bools count as set when True; numeric fields count when not None.
        if isinstance(val, bool):
            if val:
                has_any = True
                break
        elif val is not None:
            has_any = True
            break
    if not has_any:
        return False
    from soup_cli.monitoring.grpo_stability_callback import GRPOStabilityCallback

    try:
        callback = GRPOStabilityCallback(
            ref_model_ema_alpha=tcfg.ref_model_ema_alpha,
            replay_buffer_size=tcfg.replay_buffer_size,
            async_grpo_prefetch=bool(tcfg.async_grpo_prefetch),
            tis_threshold=tcfg.tis_threshold,
            mask_truncated_completions=bool(tcfg.mask_truncated_completions),
            defer_rerolling=bool(tcfg.defer_rerolling),
            skip_zero_advantage=bool(tcfg.skip_zero_advantage),
            off_policy_mask_threshold=tcfg.off_policy_mask_threshold,
        )
    except (TypeError, ValueError) as exc:
        logger.debug("attach_grpo_stability_callback rejected: %s", exc)
        return False
    trainer.add_callback(callback)
    return True


def attach_plugin_callback(trainer: Any, console: Any = None) -> bool:
    """Attach :class:`SoupPluginCallback` when any enabled plugin implements a hook.

    Returns ``True`` when a callback was attached, ``False`` otherwise
    (no plugins enabled OR none implement any hook — the build helper
    short-circuits to ``None`` in that case so the trainer pays zero
    overhead).

    Failures inside individual plugin hooks are swallowed at WARNING
    inside the callback itself; this helper only handles the
    construction failure path (transformers not importable / plugin
    registry corrupted).
    """
    try:
        from soup_cli.monitoring.plugin_callback import build_plugin_callback

        callback = build_plugin_callback()
    except Exception as exc:  # noqa: BLE001 — plugin infra must not crash training
        logger.debug("attach_plugin_callback skipped: %s", exc)
        return False
    if callback is None:
        return False
    try:
        trainer.add_callback(callback)
    except Exception as exc:  # noqa: BLE001
        logger.debug("attach_plugin_callback add_callback failed: %s", exc)
        return False
    if console is not None:
        try:
            # Number of plugins is the count of distinct (plugin_name, hooks)
            # pairs the callback snapshot will fan out to.
            from soup_cli.plugins import list_plugins

            n_enabled = sum(1 for s in list_plugins().values() if s.enabled)
            console.print(
                f"[dim]Plugin callback attached ({n_enabled} enabled plugin(s)).[/]"
            )
        except Exception:  # noqa: BLE001
            pass
    return True
