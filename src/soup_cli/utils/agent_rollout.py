"""Multi-turn agent rollout backends — v0.50.0 Part C.

Closed allowlist for multi-turn agent RL rollout backends shipped for
unsloth + axolotl parity:

- ``art``      — OpenPipe ART (multi-turn agent rollouts; unsloth)
- ``ruler``    — RULER agent eval framework (unsloth)
- ``nemo_gym`` — NVIDIA NeMo Gym (unsloth, axolotl)
- ``openenv``  — Generic ``rollout_func`` protocol (unsloth, axolotl)

Schema-only in v0.50.0; live launcher wiring deferred to v0.50.1 (mirrors
v0.27.0 MII / v0.37.0 multipack / v0.41.0 LLaMA Pro / v0.45.0 plugins /
v0.46.0 deploy autopilot / v0.48.0 curriculum / v0.49.0 LongLoRA
stub-then-live pattern).

Security:
- Closed allowlist; arbitrary string at schema level rejected.
- ``_BACKEND_METADATA`` wrapped in MappingProxyType (matches v0.36.0
  _REGISTRY policy).
- ``validate_rollout_backend`` rejects empty / null-byte / non-string /
  oversize inputs.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

_MAX_BACKEND_NAME_LEN = 32

SUPPORTED_ROLLOUT_BACKENDS: frozenset[str] = frozenset({
    "art",
    "ruler",
    "nemo_gym",
    "openenv",
})


@dataclass(frozen=True)
class RolloutBackendSpec:
    """Metadata for a multi-turn agent rollout backend."""

    name: str
    description: str
    required_package: str | None
    live_wired: bool


_BACKEND_METADATA = types.MappingProxyType({
    "art": RolloutBackendSpec(
        name="art",
        description="OpenPipe ART — multi-turn agent rollouts",
        required_package="openpipe-art",
        live_wired=False,
    ),
    "ruler": RolloutBackendSpec(
        name="ruler",
        description="RULER agent evaluation framework",
        required_package="ruler-eval",
        live_wired=False,
    ),
    "nemo_gym": RolloutBackendSpec(
        name="nemo_gym",
        description="NVIDIA NeMo Gym single/multi-turn rollout backend",
        required_package="nemo-gym",
        live_wired=False,
    ),
    "openenv": RolloutBackendSpec(
        name="openenv",
        description="Generic OpenEnv rollout_func protocol",
        required_package=None,
        live_wired=False,
    ),
})


def validate_rollout_backend(name: object) -> str:
    """Validate and normalise a rollout backend name."""
    if isinstance(name, bool):
        raise ValueError("rollout_backend must be a string, got bool")
    if not isinstance(name, str):
        raise ValueError(
            f"rollout_backend must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("rollout_backend must be a non-empty string")
    if "\x00" in name:
        raise ValueError("rollout_backend must not contain null bytes")
    if len(name) > _MAX_BACKEND_NAME_LEN:
        raise ValueError(
            f"rollout_backend exceeds {_MAX_BACKEND_NAME_LEN} chars"
        )
    normalised = name.lower()
    if normalised not in SUPPORTED_ROLLOUT_BACKENDS:
        raise ValueError(
            f"rollout_backend={name!r} is not supported. "
            f"Valid: {sorted(SUPPORTED_ROLLOUT_BACKENDS)}"
        )
    return normalised


def get_rollout_backend_spec(name: str) -> RolloutBackendSpec:
    """Return the :class:`RolloutBackendSpec` for ``name``."""
    normalised = validate_rollout_backend(name)
    return _BACKEND_METADATA[normalised]


def required_rollout_package(name: str) -> str | None:
    """Return the pip-installable package name (or None for openenv)."""
    spec = get_rollout_backend_spec(name)
    return spec.required_package


def list_rollout_backends() -> tuple[str, ...]:
    """Return a sorted tuple of supported rollout backend names."""
    return tuple(sorted(SUPPORTED_ROLLOUT_BACKENDS))


def launch_rollout(name: str) -> None:
    """Live launcher for the rollout backend — deferred to v0.50.1.

    Planned v0.50.1 signature:
    ``launch_rollout(name, *, prompts, model, reward_fn, max_steps)``.
    """
    validate_rollout_backend(name)
    raise NotImplementedError(
        f"rollout_backend={name!r} live launcher deferred to v0.50.1. "
        "The schema accepts the value but the actual rollout/optimization "
        "loop is not yet wired."
    )
