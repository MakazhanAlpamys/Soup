"""v0.53.0 Part C — KV cache types schema helpers + v0.71.14 #140 live wiring.

Closed allowlist of ``kv_cache_type`` strings exposed to
``soup serve --kv-cache-type <type>`` and YAML ``training.kv_cache_type``.
Mirrors the unsloth serve recipe.

* ``q8_0`` — 8-bit quantized KV cache (transformers: HQQ quantized cache)
* ``bf16`` — bfloat16 KV cache (transformers: model + cache dtype)
* ``f16``  — float16 KV cache (transformers: model + cache dtype)
* ``fp8``  — FP8 on Hopper+ (vLLM-backend only; transformers has no fp8 path)

v0.71.14 (#140) lifts the v0.53.0 ``apply_kv_cache_type`` stub: it returns a
:class:`KvCacheRuntime` plan for the **transformers** backend (model dtype +
``generate`` kwargs) + a Hopper SM gate for ``fp8``. vLLM / SGLang stay in the
infra-blocked tail (need a Hopper / Linux box to validate honestly).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

KV_CACHE_TYPES: frozenset[str] = frozenset({"q8_0", "bf16", "f16", "fp8"})

# Live serve wiring exists for the transformers backend only. vLLM / SGLang
# KV-cache-dtype routing needs a Hopper / Linux box to validate honestly →
# blocked tail (v0.71.14 #140).
_LIVE_BACKENDS: frozenset[str] = frozenset({"transformers"})

_MAX_KV_CACHE_LEN: int = 16

# Hopper is compute capability (9, x); fp8 KV cache needs SM >= 9.0.
_HOPPER_MAJOR: int = 9


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


@dataclass(frozen=True)
class KvCacheRuntime:
    """Resolved KV-cache plan for a serve backend.

    * ``model_dtype`` — ``"bfloat16"`` / ``"float16"`` for the dtype-based
      types (the transformers DynamicCache inherits the model's compute
      dtype), else ``None``.
    * ``generate_kwargs`` — kwargs threaded into ``model.generate`` for the
      quantized-cache types (``cache_implementation`` / ``cache_config``),
      else empty.
    * ``requires_quant_backend`` — True for ``q8_0`` (needs ``hqq`` or
      ``optimum-quanto`` installed for the quantized cache to run).
    """

    kv_cache_type: str
    backend: str
    model_dtype: Optional[str]
    generate_kwargs: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    requires_quant_backend: bool = False
    note: str = ""


def quantized_cache_backend_available() -> Optional[str]:
    """Return the installed quantized-KV-cache backend name, or ``None``.

    Probes (via :func:`importlib.util.find_spec`, no import) for the optional
    libraries the transformers quantized cache needs: ``hqq`` (8-bit) or
    ``optimum-quanto`` / ``quanto`` (2/4-bit). Returns the first found, else
    ``None`` (the ``q8_0`` path then surfaces a friendly install advisory).
    """
    if importlib.util.find_spec("hqq") is not None:
        return "hqq"
    if importlib.util.find_spec("optimum.quanto") is not None:
        return "quanto"
    if importlib.util.find_spec("quanto") is not None:
        return "quanto"
    return None


def _validate_compute_capability(
    cc: object,
) -> Optional[Tuple[int, int]]:
    """Validate an optional ``(major, minor)`` compute-capability tuple."""
    if cc is None:
        return None
    if isinstance(cc, bool) or not isinstance(cc, tuple) or len(cc) != 2:
        raise TypeError(
            "compute_capability must be a (major, minor) int 2-tuple or None"
        )
    major, minor = cc
    if isinstance(major, bool) or isinstance(minor, bool):
        raise TypeError("compute_capability entries must be int, not bool")
    if not isinstance(major, int) or not isinstance(minor, int):
        raise TypeError("compute_capability entries must be int")
    return (major, minor)


def apply_kv_cache_type(
    kv_cache_type: object,
    *,
    backend: str = "transformers",
    compute_capability: Optional[Tuple[int, int]] = None,
) -> KvCacheRuntime:
    """Resolve a ``kv_cache_type`` into a serve runtime plan (v0.71.14 #140).

    Args:
        kv_cache_type: one of ``q8_0`` / ``bf16`` / ``f16`` / ``fp8``.
        backend: serve backend. Only ``transformers`` is live; ``vllm`` /
            ``sglang`` raise :class:`NotImplementedError` (blocked tail).
        compute_capability: optional ``(major, minor)`` CUDA compute
            capability, used to gate ``fp8`` (Hopper SM >= 9.0).

    Raises:
        TypeError / ValueError: invalid ``kv_cache_type`` / ``backend`` /
            ``compute_capability``.
        NotImplementedError: ``backend`` is vLLM / SGLang (deferred).
        RuntimeError: ``fp8`` (transformers has no fp8 KV-cache path; it is a
            vLLM + Hopper feature — friendly error names the requirement).
    """
    canonical = validate_kv_cache_type(kv_cache_type)
    if isinstance(backend, bool) or not isinstance(backend, str):
        raise TypeError(
            f"backend must be str, got {type(backend).__name__}"
        )
    backend_l = backend.lower()
    if backend_l not in _LIVE_BACKENDS:
        raise NotImplementedError(
            f"kv_cache_type live wiring for backend {backend!r} is deferred "
            "(vLLM / SGLang KV-cache-dtype routing needs a Hopper / Linux box "
            "to validate — tracked in the infra-blocked tail). "
            "Use --backend transformers."
        )
    cc = _validate_compute_capability(compute_capability)
    spec = _KV_CACHE_METADATA[canonical]

    if spec.requires_hopper:
        # fp8: transformers has NO fp8 KV-cache path — it is a vLLM feature
        # that additionally needs a Hopper+ GPU. Always raise on transformers,
        # with a message that names the Hopper requirement (so the non-Hopper
        # case is explicit) and the vLLM-only support.
        if cc is not None and cc[0] < _HOPPER_MAJOR:
            raise RuntimeError(
                f"kv_cache_type 'fp8' needs a Hopper+ GPU "
                f"(compute capability >= {_HOPPER_MAJOR}.0); "
                f"detected {cc[0]}.{cc[1]}. fp8 KV cache is also vLLM-only — "
                "the transformers backend has no fp8 KV-cache path."
            )
        raise RuntimeError(
            "kv_cache_type 'fp8' is only available on the vLLM backend with a "
            f"Hopper+ GPU (compute capability >= {_HOPPER_MAJOR}.0); the "
            "transformers backend has no fp8 KV-cache path. Use --backend "
            "transformers with q8_0 / bf16 / f16 instead."
        )

    if canonical in ("bf16", "f16"):
        dtype = "bfloat16" if canonical == "bf16" else "float16"
        return KvCacheRuntime(
            kv_cache_type=canonical,
            backend=backend_l,
            model_dtype=dtype,
            generate_kwargs=MappingProxyType({}),
            requires_quant_backend=False,
            note=(
                f"KV cache runs in {dtype} (the transformers DynamicCache "
                "inherits the model's compute dtype)."
            ),
        )

    # q8_0 — 8-bit quantized KV cache. transformers routes this through the
    # HQQ backend (quanto only does 2/4-bit); needs `pip install hqq`.
    return KvCacheRuntime(
        kv_cache_type=canonical,
        backend=backend_l,
        model_dtype=None,
        generate_kwargs=MappingProxyType({
            "cache_implementation": "quantized",
            "cache_config": MappingProxyType({
                "backend": "hqq",
                "nbits": 8,
                "axis_key": 0,
                "axis_value": 0,
            }),
        }),
        requires_quant_backend=True,
        note=(
            "8-bit quantized KV cache via the HQQ backend "
            "(install with `pip install hqq`)."
        ),
    )
