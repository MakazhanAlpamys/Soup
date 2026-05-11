"""PRM (Process Reward Model) — v0.50.0 Part E.

Schema helpers for the new ``task='prm'`` stepwise-supervised trainer.
The PRM data format (``data.format='prm'``) was schema-locked in v0.42.0
Part A; v0.50.0 promotes it to a first-class task with cross-validators.

The actual PRM trainer wrapper (``soup_cli/trainer/prm.py``) is deferred
to v0.50.1 — mirrors v0.27.0 MII / v0.37.0 multipack / v0.41.0 LLaMA Pro /
v0.45.0 plugins / v0.49.0 LongLoRA stub-then-live pattern.

Security:
- Pure schema-time validation; no filesystem touch.
- All validators raise ``ValueError`` with actionable messages.
"""

from __future__ import annotations


def validate_prm_compat(
    *,
    task: str,
    data_format: str,
    backend: str,
    modality: str,
) -> None:
    """Schema-time gate for ``task='prm'``.

    Rejects:
    - non-PRM task (the function is intended to be called only when
      ``task == 'prm'``; defence-in-depth).
    - ``data.format`` not in ``{'prm', 'auto'}`` — PRM requires the
      stepwise-supervised data shape from v0.42.0 Part A.
    - ``backend='mlx'`` — PRM trainer is HF Trainer-specific.
    - ``modality != 'text'`` — vision/audio PRM not modelled.
    """
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    if task != "prm":
        raise ValueError(
            f"validate_prm_compat called with task={task!r} (expected 'prm')"
        )
    if not isinstance(data_format, str) or not data_format:
        raise ValueError("data.format must be a non-empty string")
    if data_format not in ("prm", "auto"):
        raise ValueError(
            f"task='prm' requires data.format in ('prm', 'auto'); "
            f"got data.format={data_format!r}"
        )
    if backend == "mlx":
        raise ValueError(
            "task='prm' is not supported on backend=mlx in v0.50.0"
        )
    if modality != "text":
        raise ValueError(
            f"task='prm' requires modality='text'; got modality={modality!r}"
        )


def validate_vision_grpo_compat(
    *,
    task: str,
    modality: str,
    backend: str,
) -> None:
    """Schema-time gate for ``vision_grpo=True``.

    Rejects on:
    - task not in {'grpo', 'ppo'} (vision RL is only meaningful for RL);
    - modality != 'vision' (the whole point of the flag);
    - backend == 'mlx' (no VLM-RL on MLX).
    """
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    if task not in ("grpo", "ppo"):
        raise ValueError(
            f"vision_grpo requires task in ('grpo', 'ppo'); got task={task!r}"
        )
    if modality != "vision":
        raise ValueError(
            f"vision_grpo requires modality='vision'; got modality={modality!r}"
        )
    if backend == "mlx":
        raise ValueError(
            "vision_grpo is not supported on backend=mlx in v0.50.0"
        )


def build_prm_trainer() -> None:
    """Live PRM trainer factory — deferred to v0.50.1.

    Planned v0.50.1 signature:
    ``build_prm_trainer(*, config, model, tokenizer, train_dataset, eval_dataset)``.

    Raises ``NotImplementedError`` so callers cannot silently train an
    SFT model when they asked for PRM.
    """
    raise NotImplementedError(
        "PRM trainer (task='prm') live wiring deferred to v0.50.1. "
        "Schema accepts the value but no trainer wrapper is registered yet."
    )
