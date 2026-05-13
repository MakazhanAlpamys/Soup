"""v0.45.0 Part D — Advanced trainer-plugin allowlist (schema-only).

Closed allowlist of optional trainer plugins so a future
``training.trainer_plugins: [grokfast, spectrum, ...]`` Pydantic field
can validate against a stable surface. Live wiring (callbacks, kernel
swaps, LLMCompressor passes) is deferred to v0.45.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

_PLUGIN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")
_MAX_DESCRIPTION = 256
_MAX_PLUGINS_PER_RUN = 8


@dataclass(frozen=True)
class TrainerPluginSpec:
    """One advanced trainer plugin descriptor."""

    name: str
    description: str
    required_package: Optional[str]


def _make(
    name: str,
    description: str,
    required_package: Optional[str],
) -> TrainerPluginSpec:
    if not _PLUGIN_NAME_RE.match(name):
        raise ValueError(
            "trainer-plugin name must be snake_case ([a-z0-9][a-z0-9_]{0,31})"
        )
    if not isinstance(description, str) or "\x00" in description:
        raise ValueError("description must be a NUL-free string")
    if len(description) > _MAX_DESCRIPTION:
        raise ValueError(f"description exceeds {_MAX_DESCRIPTION} chars")
    if required_package is not None:
        if not isinstance(required_package, str) or not required_package:
            raise ValueError("required_package must be a non-empty string")
    return TrainerPluginSpec(
        name=name, description=description, required_package=required_package
    )


_BUILTIN: Mapping[str, TrainerPluginSpec] = MappingProxyType(
    {
        "cce_plugin": _make(
            "cce_plugin",
            "Cut Cross-Entropy plugin variant (axolotl binding)",
            "cut-cross-entropy",
        ),
        "grokfast": _make(
            "grokfast",
            "Gradient-grokking accelerator (axolotl)",
            "grokfast",
        ),
        "spectrum": _make(
            "spectrum",
            "Gradient-norm-based layer freezing (axolotl)",
            None,
        ),
        "llmcompressor": _make(
            "llmcompressor",
            "Post-training compression (axolotl)",
            "llmcompressor",
        ),
        "sonicmoe": _make(
            "sonicmoe",
            "Alternative MoE kernel (axolotl)",
            None,
        ),
        "math_verify": _make(
            "math_verify",
            "Standalone math reward verifier (promoted v0.25.0)",
            None,
        ),
    }
)


def list_trainer_plugins() -> Mapping[str, TrainerPluginSpec]:
    """Return an immutable view of the registry."""
    return _BUILTIN


def get_trainer_plugin(name: str) -> TrainerPluginSpec:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    canonical = name.strip().lower()
    if not canonical or "\x00" in canonical:
        raise ValueError("name must be a non-empty NUL-free string")
    if canonical not in _BUILTIN:
        raise KeyError(canonical)
    return _BUILTIN[canonical]


def validate_trainer_plugin_list(names: Sequence[str]) -> Tuple[str, ...]:
    """Validate a list of trainer-plugin names. Returns canonical names."""
    if isinstance(names, str) or not isinstance(names, (list, tuple)):
        raise TypeError("names must be a list or tuple")
    if len(names) > _MAX_PLUGINS_PER_RUN:
        raise ValueError(
            f"too many trainer plugins (max {_MAX_PLUGINS_PER_RUN})"
        )
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        if not isinstance(raw, str):
            raise TypeError("plugin name must be a string")
        canonical = raw.strip().lower()
        if not canonical or "\x00" in canonical:
            raise ValueError("plugin name must be non-empty NUL-free")
        if canonical not in _BUILTIN:
            raise ValueError(
                f"unknown trainer plugin: {canonical!r}. supported: "
                f"{sorted(_BUILTIN)}"
            )
        if canonical in seen:
            raise ValueError(f"duplicate trainer plugin: {canonical!r}")
        seen.add(canonical)
        out.append(canonical)
    return tuple(out)


def instantiate_trainer_plugins(names: Sequence[str]) -> Tuple[Any, ...]:
    """v0.53.6 #105 — instantiate live upstream callbacks (stub-then-live).

    Validates ``names`` against :func:`validate_trainer_plugin_list` then
    raises :class:`NotImplementedError` with a v0.53.7 marker. Live lazy
    imports + per-plugin callback construction (``grokfast``,
    ``spectrum``, ``llmcompressor``, ``sonicmoe``, ``cce_plugin``,
    ``math_verify``) land in v0.53.7. Same stub-then-live pattern as
    v0.27.0 MII / v0.37.0 multipack / v0.41.0 LLaMA Pro.

    Raises:
        TypeError: per :func:`validate_trainer_plugin_list`.
        ValueError: per :func:`validate_trainer_plugin_list`.
        NotImplementedError: always (after validation) — live wiring
            ships in v0.53.7.
    """
    canonical = validate_trainer_plugin_list(names)
    raise NotImplementedError(
        f"Trainer-plugin live instantiation deferred to v0.53.7. "
        f"Validated names: {canonical!r}"
    )


__all__ = [
    "TrainerPluginSpec",
    "list_trainer_plugins",
    "get_trainer_plugin",
    "validate_trainer_plugin_list",
    "instantiate_trainer_plugins",
]
