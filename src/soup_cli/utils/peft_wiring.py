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


def attach_lisa_callback(trainer: Any, tcfg: Any) -> bool:
    """Attach :class:`LisaCallback` when ``training.lisa_enabled`` is set.

    Returns ``True`` when a callback was attached, ``False`` otherwise. The
    schema-level cross-validator (``_validate_lisa_compat``) already enforces
    the sft / transformers / text / quantization=none gate and mutual
    exclusion, so this helper trusts the caller's task/backend (v0.71.34 #267).
    """
    if not getattr(tcfg, "lisa_enabled", False):
        return False
    from soup_cli.utils.lisa import LisaCallback, LisaPolicy

    # Read schema fields directly (they are guaranteed to exist on
    # TrainingConfig) so a misnamed attr fails loudly — mirrors
    # attach_relora_callback. seed is a fixed 0 (LISA reproducibility does not
    # need a user knob today; add a schema field if that changes).
    policy = LisaPolicy(
        num_layers=int(tcfg.lisa_num_layers),
        interval_steps=int(tcfg.lisa_interval_steps),
        reset_optimizer=bool(tcfg.lisa_reset_optimizer),
        seed=0,
    )
    trainer.add_callback(LisaCallback(policy=policy))
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
    # v0.71.5 #149 — thread curriculum_metric so the callback can bucket by
    # loss / perplexity percentile (round-robin fallback for `length`). Any
    # value that is not one of the three valid metrics (e.g. a missing field
    # or a test MagicMock) falls back to `length` so the callback always
    # constructs.
    curriculum_metric = getattr(tcfg, "curriculum_metric", "length")
    if curriculum_metric not in ("length", "perplexity", "loss"):
        curriculum_metric = "length"
    try:
        callback = DynamicCurriculumCallback(
            policy=policy,
            output_dir=output_dir,
            curriculum_metric=curriculum_metric,
        )
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


def rl_callbacks_need_buffer(tcfg: Any) -> bool:
    """True when a reward-fn capture buffer is needed (v0.71.11 #235/#240).

    The reward-hack + echo-trap callbacks observe the GRPO step's rewards
    + completions through the shared
    :class:`~soup_cli.utils.rl_signal_buffer.RLSignalBuffer`. The
    RL-checkpoint callback does not.
    """
    return (
        getattr(tcfg, "reward_hack_detector", None) is not None
        or bool(getattr(tcfg, "echo_trap_enabled", False))
        or getattr(tcfg, "reward_hack_mitigation", "off") != "off"
    )


def _attach_reward_hack(
    trainer: Any,
    tcfg: Any,
    *,
    buffer: Any,
    tokenizer: Any,
    output_dir: str,
    task: str,
    rl_checkpoint_cb: Any = None,
) -> int:
    """Attach the reward-hack callback: mitigation controller (v0.71.26) when a
    ``reward_hack_mitigation`` mode is set, else the plain v0.70.0 detector.

    Returns 1 when a callback was attached, 0 otherwise. The mitigation
    controller SUBSUMES the plain detector (they share the same signal), so
    exactly one of the two is ever attached. ``rl_checkpoint_cb`` is the
    (already-built) RL-checkpoint callback the pid_lagrangian rollback ladder
    restores from.
    """
    import os

    detector = getattr(tcfg, "reward_hack_detector", None)
    mitigation = getattr(tcfg, "reward_hack_mitigation", "off")
    if mitigation != "off" and detector is not None:
        from soup_cli.utils.reward_hack_control import (
            BangBangPolicy,
            MitigationLogWriter,
            PIDLagrangianPolicy,
            RewardHackMitigationCallback,
        )

        try:
            writer = MitigationLogWriter(
                os.path.join(output_dir, "mitigation_log.jsonl")
            )
            signals = tuple(
                getattr(tcfg, "reward_hack_signals", None) or ("info_rm",)
            )
            bang_bang = None
            pid = None
            if mitigation == "kl_control":
                bang_bang = BangBangPolicy(
                    beta_floor=tcfg.reward_hack_beta_floor,
                    beta_ceil=tcfg.reward_hack_beta_ceil,
                    trip_band=tcfg.reward_hack_trip_band,
                    release_band=tcfg.reward_hack_release_band,
                    dwell_steps=tcfg.reward_hack_dwell_steps,
                    release_patience=tcfg.reward_hack_release_patience,
                    kl_gain=tcfg.reward_hack_kl_gain,
                )
            elif mitigation == "pid_lagrangian":
                pid = PIDLagrangianPolicy(
                    kp=tcfg.reward_hack_pid_kp,
                    ki=tcfg.reward_hack_pid_ki,
                    kd=tcfg.reward_hack_pid_kd,
                    signal_target=tcfg.reward_hack_signal_target,
                    beta_floor=tcfg.reward_hack_beta_floor,
                    beta_ceil=tcfg.reward_hack_beta_ceil,
                    integral_clamp=tcfg.reward_hack_integral_clamp,
                )
            callback = RewardHackMitigationCallback(
                mode=mitigation,
                detector=detector,
                log_writer=writer,
                signals=signals,
                buffer=buffer,
                tokenizer=tokenizer,
                task=task,
                bang_bang=bang_bang,
                pid=pid,
                rollback=bool(getattr(tcfg, "reward_hack_rollback", False)),
                rollback_patience=int(
                    getattr(tcfg, "reward_hack_rollback_patience", 3)
                ),
                max_recovery_attempts=int(
                    getattr(tcfg, "reward_hack_max_recovery_attempts", 2)
                ),
                rl_checkpoint_cb=rl_checkpoint_cb,
                smoothing=getattr(tcfg, "reward_hack_signal_smoothing", "none"),
                smoothing_window=int(
                    getattr(tcfg, "reward_hack_smoothing_window", 8)
                ),
                conservative_on_disagreement=bool(
                    getattr(tcfg, "reward_hack_conservative_on_disagreement", False)
                ),
            )
            trainer.add_callback(callback)
            callback.attach(trainer)
            return 1
        except (TypeError, ValueError, OSError) as exc:
            # A user explicitly enabled mitigation — a silent drop would leave
            # them believing a safety controller is active when it is not.
            # Warn LOUDLY (e.g. output dir outside cwd fails the log writer).
            logger.warning(
                "reward-hack mitigation callback NOT attached (%s): %s. "
                "Training will proceed WITHOUT mitigation.",
                type(exc).__name__,
                exc,
            )
            return 0
    if detector is not None:
        from soup_cli.utils.reward_hacking import build_reward_hack_callback

        try:
            trainer.add_callback(
                build_reward_hack_callback(
                    detector=detector,
                    halt_on_hack=bool(getattr(tcfg, "reward_hack_halt", False)),
                    buffer=buffer,
                )
            )
            return 1
        except (TypeError, ValueError) as exc:
            logger.debug("attach reward-hack callback rejected: %s", exc)
            return 0
    return 0


def attach_rl_callbacks(
    trainer: Any,
    tcfg: Any,
    *,
    buffer: Any = None,
    tokenizer: Any = None,
    output_dir: str = ".",
    task: str = "grpo",
) -> int:
    """Attach the v0.71.11 live RL callbacks; return how many were attached.

    Wires (when their schema fields are set):
    - reward-hacking detector (#235) — reads ``buffer``.
    - echo-trap detector (#240) — reads ``buffer`` + ``tokenizer``.
    - mid-epoch RL checkpoint (#238) — saves under ``output_dir``.

    The schema cross-validators already gate these fields to RL tasks on
    non-mlx backends, so this helper trusts the caller's config.
    """
    attached = 0
    # Build the RL-checkpoint callback FIRST so the pid_lagrangian rollback
    # ladder can be handed a reference to restore from.
    ckpt_cb = _build_rl_checkpoint_cb(tcfg, output_dir=output_dir, task=task)
    if ckpt_cb is not None:
        trainer.add_callback(ckpt_cb)
        attached += 1

    attached += _attach_reward_hack(
        trainer,
        tcfg,
        buffer=buffer,
        tokenizer=tokenizer,
        output_dir=output_dir,
        task=task,
        rl_checkpoint_cb=ckpt_cb,
    )

    if bool(getattr(tcfg, "echo_trap_enabled", False)):
        from soup_cli.utils.echo_trap import build_echo_trap_callback

        try:
            trainer.add_callback(
                build_echo_trap_callback(
                    threshold=float(getattr(tcfg, "echo_trap_threshold", 0.6)),
                    halt_on_trap=bool(getattr(tcfg, "echo_trap_halt", False)),
                    tokenizer_aware=bool(
                        getattr(tcfg, "echo_trap_tokenizer_aware", False)
                    ),
                    buffer=buffer,
                    tokenizer=tokenizer,
                )
            )
            attached += 1
        except (TypeError, ValueError) as exc:
            logger.debug("attach echo-trap callback rejected: %s", exc)

    return attached


def _build_rl_checkpoint_cb(tcfg: Any, *, output_dir: str, task: str) -> Any:
    """Build the mid-epoch RL-checkpoint callback (or None if not configured)."""
    save_every = getattr(tcfg, "rl_checkpoint_save_every_steps", None)
    if save_every is None:
        return None
    from soup_cli.utils.rl_checkpoint import (
        RLCheckpointConfig,
        build_rl_checkpoint_callback,
    )

    try:
        ckpt_cfg = RLCheckpointConfig(
            save_every_steps=int(save_every),
            include_optimizer_state=bool(
                getattr(tcfg, "rl_checkpoint_include_optimizer", True)
            ),
            include_ref_model=bool(
                getattr(tcfg, "rl_checkpoint_include_ref_model", False)
            ),
            include_rollout_buffer=bool(
                getattr(tcfg, "rl_checkpoint_include_rollout_buffer", False)
            ),
            keep_last=int(getattr(tcfg, "rl_checkpoint_keep_last", 3)),
        )
        return build_rl_checkpoint_callback(
            ckpt_cfg, output_dir=output_dir, task=task
        )
    except (TypeError, ValueError) as exc:
        logger.debug("build RL-checkpoint callback rejected: %s", exc)
        return None


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
