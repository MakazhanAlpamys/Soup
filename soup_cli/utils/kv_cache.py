"""v0.53.0 Part C — KV cache types schema helpers.

Closed allowlist of ``kv_cache_type`` strings exposed to
``soup serve --kv-cache-type <type>`` and YAML ``training.kv_cache_type`` (the
field is reused by serve / chat runtime in v0.53.1). Mirrors the unsloth
serve recipe.

* ``q8_0`` — 8-bit (default for the unsloth runtime)
* ``bf16`` — bfloat16
* ``f16``  — float16
* ``fp8``  — FP8 on Hopper+ (gated by a separate runtime check)

Live wiring into the vLLM / SGLang / transformers serve loops is deferred
to v0.53.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

KV_CACHE_TYPES: frozenset[str] = frozenset({"q8_0", "bf16", "f16", "fp8"})

_MAX_KV_CACHE_LEN: int = 16


@dataclass(frozen=True)
class KVCacheSpec:
    """Frozen metadata for a KV-cache type. Immutable by construction."""

    name: str
    bits: int
    requires_hopper: bool
    description: str
    live_wired: bool


_KV_CACHE_METADATA: Mapping[str, KVCacheSpec] = MappingProxyType({
    "q8_0": KVCacheSpec(
        name="q8_0", bits=8, requires_hopper=False,
        description="8-bit KV cache (default for unsloth runtime)",
        live_wired=False,
    ),
    "bf16": KVCacheSpec(
        name="bf16", bits=16, requires_hopper=False,
        description="bfloat16 KV cache",
        live_wired=False,
    ),
    "f16": KVCacheSpec(
        name="f16", bits=16, requires_hopper=False,
        description="float16 KV cache",
        live_wired=False,
    ),
    "fp8": KVCacheSpec(
        name="fp8", bits=8, requires_hopper=True,
        description="FP8 KV cache (Hopper+ only)",
        live_wired=False,
    ),
})


def validate_kv_cache_type(value: object) -> str:
    """Validate a ``kv_cache_type`` string and return the canonical form.

    Mirrors v0.52.0 ``validate_reasoning_effort`` policy: bool-first /
    null-byte / oversize / case-insensitive normalisation.
    """
    if isinstance(value, bool):
        raise TypeError(f"kv_cache_type must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(
            f"kv_cache_type must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("kv_cache_type must be non-empty")
    if "\x00" in value:
        raise ValueError("kv_cache_type must not contain null bytes")
    if len(value) > _MAX_KV_CACHE_LEN:
        raise ValueError(
            f"kv_cache_type too long (max {_MAX_KV_CACHE_LEN} chars)"
        )
    canonical = value.lower()
    if canonical not in KV_CACHE_TYPES:
        supported = ", ".join(sorted(KV_CACHE_TYPES))
        raise ValueError(
            f"kv_cache_type {value!r} not supported. Supported: {supported}"
        )
    return canonical


def get_kv_cache_spec(name: str) -> KVCacheSpec:
    """Return the frozen :class:`KVCacheSpec` for ``name`` (canonical)."""
    canonical = validate_kv_cache_type(name)
    return _KV_CACHE_METADATA[canonical]


def requires_hopper(name: object) -> bool:
    """Return True iff ``name`` needs a Hopper+ GPU.

    Single source of truth: reads ``requires_hopper`` from the
    ``_KV_CACHE_METADATA`` spec. Adding a Hopper-only type means flipping
    the spec field only — no separate update here (code-review MEDIUM fix).
    """
    if isinstance(name, bool) or not isinstance(name, str):
        return False
    canonical = name.lower()
    spec = _KV_CACHE_METADATA.get(canonical)
    return spec is not None and spec.requires_hopper


def apply_kv_cache_type() -> None:
    """Live KV-cache-type wiring — deferred to v0.53.1.

    Mirrors v0.50.0 ``apply_vllm_sleep_mode`` and v0.52.0
    ``apply_moe_expert_quant`` stub-then-live pattern.
    """
    raise NotImplementedError(
        "kv_cache_type live wiring deferred to v0.53.1. Schema accepts "
        "q8_0 / bf16 / f16 / fp8 but no serve backend is routing the flag yet."
    )
