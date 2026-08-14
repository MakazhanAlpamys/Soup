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

import contextlib
import logging
from collections.abc import Collection, Iterator
from typing import Any

logger = logging.getLogger(__name__)

# Backends where vLLM is genuinely available (mirrors v0.30.0 vllm.py policy).
_VLLM_SUPPORTED_BACKENDS: frozenset[str] = frozenset({"transformers", "unsloth"})

# vLLM grew ``AsyncEngineArgs.enable_sleep_mode`` + ``engine.sleep()`` /
# ``wake_up()`` in the 0.7 line.
_MIN_VLLM_SLEEP_VERSION: tuple[int, ...] = (0, 7)


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


def _parse_version_tuple(value: str) -> tuple[int, ...]:
    """Parse leading-int dot-chunks: ``"0.7.0.dev0"`` -> ``(0, 7, 0)``.

    Mirrors the v0.40.1 ``_version_ge`` leading-int policy — a chunk that
    does not start with digits terminates the parse, so pre-release
    suffixes never break the comparison.
    """
    if not isinstance(value, str):
        return ()
    parts: list[int] = []
    for chunk in value.split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _installed_vllm_version() -> tuple[int, ...] | None:
    """Return the installed vLLM version tuple, or None when missing."""
    try:
        import vllm
    except ImportError:
        return None
    return _parse_version_tuple(getattr(vllm, "__version__", "") or "")


def vllm_supports_sleep_mode() -> bool:
    """True when vLLM is installed and >= 0.7 (sleep/wake API present)."""
    version = _installed_vllm_version()
    return version is not None and version >= _MIN_VLLM_SLEEP_VERSION


def apply_vllm_sleep_mode(engine_args: object) -> object:
    """Enable vLLM sleep mode on an ``AsyncEngineArgs``-like object.

    Live since v0.71.21 (#124): sets ``enable_sleep_mode=True`` so the
    engine can be put on standby between rollouts (``engine.sleep()`` /
    ``engine.wake_up()`` — see :func:`vllm_sleep_cycle`).

    Raises:
        TypeError: ``engine_args`` is None.
        RuntimeError: vLLM is missing or older than 0.7 (friendly,
            actionable message — BETA gate; vLLM is not installable on
            every platform, e.g. Windows).
    """
    if engine_args is None:
        raise TypeError("engine_args must not be None")
    version = _installed_vllm_version()
    if version is None:
        raise RuntimeError(
            "vllm_sleep_mode requires vLLM >= 0.7 but vLLM is not "
            "installed (pip install \"soup-cli[serve-fast]\")."
        )
    if version < _MIN_VLLM_SLEEP_VERSION:
        found = ".".join(str(part) for part in version) or "unknown"
        raise RuntimeError(
            f"vllm_sleep_mode requires vLLM >= 0.7; found {found}. "
            "Upgrade with: pip install -U vllm"
        )
    engine_args.enable_sleep_mode = True  # type: ignore[attr-defined]
    return engine_args


@contextlib.contextmanager
def vllm_sleep_cycle(engine: object, *, level: int = 1) -> Iterator[object]:
    """Put a vLLM engine to sleep for the duration of the ``with`` body.

    Live since v0.71.21 (#124). Intended to wrap the optimisation step
    between rollouts: ``with vllm_sleep_cycle(engine): optimizer_step()``.
    The engine is woken in a ``finally`` block so an exception inside the
    body never leaves it asleep. An engine without the sleep/wake API logs
    a WARNING and the body runs unchanged — memory instrumentation must
    never crash a training loop (matches the v0.71.11 callback policy).

    ``level`` is vLLM's sleep level: 1 = offload weights to CPU, 2 = also
    discard the KV cache. Bool-rejected and bounded to [1, 2].
    """
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError(f"level must be an int, got {type(level).__name__}")
    if not 1 <= level <= 2:
        raise ValueError(f"level must be 1 or 2, got {level}")
    sleep_fn = getattr(engine, "sleep", None)
    wake_fn = getattr(engine, "wake_up", None)
    if not callable(sleep_fn) or not callable(wake_fn):
        logger.warning(
            "vllm_sleep_cycle: engine %s does not expose sleep()/wake_up() "
            "— running the body without engine standby.",
            type(engine).__name__,
        )
        yield engine
        return
    sleep_fn(level=level)
    try:
        yield engine
    finally:
        wake_fn()


def maybe_enable_trl_sleep_mode(
    grpo_kwargs: "dict[str, Any]",
    grpo_param_names: "Collection[str]",
    console: object = None,
) -> bool:
    """Thread ``vllm_sleep_mode`` into TRL's GRPOConfig when it has the hook.

    Live since v0.71.21 (#124). TRL manages its own vLLM engine inside
    ``GRPOTrainer`` (server / colocate modes), so Soup cannot reach the
    engine to call ``sleep()``/``wake_up()`` directly. When the installed
    TRL exposes a ``vllm_enable_sleep_mode`` GRPOConfig parameter it is
    set here; otherwise a yellow advisory explains that Soup's own vLLM
    engine factory (``create_vllm_engine(sleep_mode=True)``) honors the
    flag for serve / custom rollout loops.

    Returns True when the TRL kwarg was set.
    """
    if "vllm_enable_sleep_mode" in grpo_param_names:
        grpo_kwargs["vllm_enable_sleep_mode"] = True
        if console is not None:
            console.print(
                "[green]vLLM sleep mode enabled via TRL GRPOConfig[/]"
            )
        return True
    if console is not None:
        console.print(
            "[yellow]vllm_sleep_mode: the installed TRL GRPOConfig does not "
            "expose a sleep-mode option; Soup's vLLM engine factory honors "
            "sleep_mode=True for serve / custom rollout loops. Upgrade trl "
            "when it ships the hook.[/]"
        )
    return False
