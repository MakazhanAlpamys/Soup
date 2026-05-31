"""GRPO long-context + memory-efficient RL — v0.50.0 Part B.

Schema helpers for two unsloth/axolotl features:

1. ``long_context_grpo`` — wires Tiled MLP (v0.56.0 Part A) when available,
   otherwise accepts as-is so users can opt in early. Schema-only in v0.50.0;
   live Tiled MLP integration deferred to v0.56.0.
2. ``vllm_sleep_mode`` — between-rollouts vLLM standby (memory savings during
   the optimisation step). Requires vLLM ≥ 0.7 and is delegated to the vLLM
   AsyncEngineArgs ``enable_sleep_mode`` flag.

Security:
- Both flags are strict bools (project bool-as-int policy enforced at the
  schema layer via Pydantic).
- ``validate_long_context_grpo_compat`` rejects multi-stage combinations
  that would silently no-op (e.g. ring-attention + long_context_grpo on a
  non-Llama base).
"""

from __future__ import annotations

# Backends where vLLM is genuinely available (mirrors v0.30.0 vllm.py policy).
_VLLM_SUPPORTED_BACKENDS: frozenset[str] = frozenset({"transformers", "unsloth"})


def validate_long_context_grpo_compat(
    *,
    task: str,
    backend: str,
    use_ring_attention: bool,
) -> None:
    """Schema-time gate for ``long_context_grpo=True``.

    Rejects on:
    - non-GRPO task (long_context_grpo only meaningful for RL rollouts);
    - mlx backend (no Tiled MLP support);
    - ``use_ring_attention=True`` (both rewrite attention — pick one;
      mirrors v0.49.0 LongLoRA + ring-attention exclusivity).
    """
    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    if "\x00" in task:
        raise ValueError("task must not contain null bytes")
    if task != "grpo":
        raise ValueError(
            f"long_context_grpo requires task='grpo'; got task={task!r}"
        )
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    if "\x00" in backend:
        raise ValueError("backend must not contain null bytes")
    if backend == "mlx":
        raise ValueError(
            "long_context_grpo is not supported on backend=mlx in v0.50.0"
        )
    if not isinstance(use_ring_attention, bool):
        raise ValueError(
            "use_ring_attention must be a bool, got "
            f"{type(use_ring_attention).__name__}"
        )
    if use_ring_attention:
        raise ValueError(
            "long_context_grpo and use_ring_attention are mutually "
            "exclusive — pick one (both rewrite the attention kernel)"
        )


def validate_vllm_sleep_mode_compat(*, backend: str) -> None:
    """Schema-time gate for ``vllm_sleep_mode=True``.

    Sleep mode is a vLLM-only feature; rejected on backends that do not
    use vLLM for rollouts.
    """
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    if "\x00" in backend:
        raise ValueError("backend must not contain null bytes")
    if backend not in _VLLM_SUPPORTED_BACKENDS:
        raise ValueError(
            f"vllm_sleep_mode requires backend in {sorted(_VLLM_SUPPORTED_BACKENDS)}; "
            f"got backend={backend!r}"
        )


def apply_vllm_sleep_mode(engine_args: object) -> None:  # noqa: ARG001
    """Live wiring for vLLM sleep mode — deferred to v0.50.1.

    The schema flag is accepted in v0.50.0 but the actual
    ``AsyncEngineArgs.enable_sleep_mode=True`` plumbing is wired in
    v0.50.1 (mirrors v0.27.0 MII stub-then-live pattern).
    """
    raise NotImplementedError(
        "vllm_sleep_mode live wiring deferred to v0.50.1; "
        "schema flag accepts the value but rollout-time sleep/wake "
        "is not yet wired into the vLLM engine factory."
    )
