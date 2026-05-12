"""v0.52.0 Part D — BitNet 1.58-bit fine-tuning + export schema helpers.

Schema-only support for ``quantization='bitnet_1.58'`` and the new
``soup export --format bitnet`` / ``--format tq1_0`` GGUF flavours.

Live ``onebitllms`` wrapping + llama.cpp ``TQ1_0`` export wiring are
deferred to v0.52.1 (mirrors v0.50.0 stub-then-live pattern).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# Closed allowlist of BitNet-flavoured quant strings exposed to YAML.
BITNET_QUANT_FORMATS: frozenset[str] = frozenset({"bitnet_1.58"})

# Closed allowlist of BitNet export targets (the actual export formats).
BITNET_EXPORT_FORMATS: frozenset[str] = frozenset({"bitnet", "tq1_0"})

_BITNET_FAMILY_RE_PREFIXES: tuple[str, ...] = (
    "bitnet", "falcon-e", "falcone", "1bitllm", "onebit",
)


@dataclass(frozen=True)
class BitNetSpec:
    """Metadata for the BitNet quant path. Frozen — immutable."""

    name: str
    description: str
    bits: float
    live_wired: bool


_BITNET_METADATA: Mapping[str, BitNetSpec] = MappingProxyType({
    "bitnet_1.58": BitNetSpec(
        name="bitnet_1.58",
        description="BitNet 1.58-bit ternary weights (axolotl + onebitllms)",
        bits=1.58,
        live_wired=False,
    ),
})


def is_bitnet_quant(value: object) -> bool:
    """Return True iff ``value`` is a BitNet quant string."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, str):
        return False
    return value in BITNET_QUANT_FORMATS


def is_bitnet_export_format(value: object) -> bool:
    """Return True iff ``value`` is a BitNet export-format string."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, str):
        return False
    return value in BITNET_EXPORT_FORMATS


def get_bitnet_spec(name: str) -> BitNetSpec:
    """Return the frozen :class:`BitNetSpec` for ``name`` or raise."""
    if not is_bitnet_quant(name):
        supported = ", ".join(sorted(BITNET_QUANT_FORMATS))
        raise ValueError(
            f"BitNet quant {name!r} not supported. Supported: {supported}"
        )
    return _BITNET_METADATA[name]


def is_bitnet_model(model_name: object) -> bool:
    """Best-effort detect whether ``model_name`` references a BitNet family.

    Checks every slash-delimited component (lowercased) against the
    ``_BITNET_FAMILY_RE_PREFIXES`` prefix list. This is intentionally more
    permissive than v0.39.0 ``is_gemma4_model`` because BitNet families
    typically live under namespaced orgs (``1bitllm/...``, ``OneBitLLM/...``)
    rather than being identifiable by repo name alone.
    """
    if isinstance(model_name, bool):
        return False
    if not isinstance(model_name, str):
        return False
    if not model_name or "\x00" in model_name:
        return False
    # Check each path component so an org name like "1bitllm/foo" matches
    # while still rejecting unrelated substrings (e.g. an SFT model that
    # happens to embed "bitnet" inside a description path).
    for part in model_name.lower().split("/"):
        if any(part.startswith(prefix) for prefix in _BITNET_FAMILY_RE_PREFIXES):
            return True
    return False


def validate_bitnet_compat(*, task: str, backend: str, modality: str) -> None:
    """Schema-time gate for ``quantization='bitnet_1.58'``.

    Rejects:
    - non-string / bool args (defence-in-depth).
    - ``backend == 'mlx'`` — onebitllms is CUDA-only in v0.52.0.
    - ``modality != 'text'`` — vision/audio BitNet not modelled.
    - ``task`` outside {sft, pretrain, dpo} — BitNet wiring is text-LM
      training only this release.
    """
    for name, value in (("task", task), ("backend", backend), ("modality", modality)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if backend == "mlx":
        raise ValueError(
            "quantization='bitnet_1.58' is not supported on backend=mlx "
            "(onebitllms is CUDA-only). Use backend='transformers'."
        )
    if modality != "text":
        raise ValueError(
            f"quantization='bitnet_1.58' is wired for modality='text' only; "
            f"got modality={modality!r}"
        )
    if task not in ("sft", "pretrain", "dpo"):
        raise ValueError(
            f"quantization='bitnet_1.58' is only wired for "
            f"task in (sft, pretrain, dpo); got task={task!r}"
        )


def validate_bitnet_export(format_name: object) -> str:
    """Validate a BitNet export-format string. Returns canonical form."""
    if isinstance(format_name, bool):
        raise TypeError(
            f"bitnet export format must not be bool, got {format_name!r}"
        )
    if not isinstance(format_name, str):
        raise TypeError(
            f"bitnet export format must be str, "
            f"got {type(format_name).__name__}"
        )
    if not format_name:
        raise ValueError("bitnet export format must be non-empty")
    if "\x00" in format_name:
        raise ValueError(
            "bitnet export format must not contain null bytes"
        )
    canonical = format_name.lower()
    if canonical not in BITNET_EXPORT_FORMATS:
        supported = ", ".join(sorted(BITNET_EXPORT_FORMATS))
        raise ValueError(
            f"bitnet export format {format_name!r} not supported. "
            f"Supported: {supported}"
        )
    return canonical


def build_bitnet_trainer() -> None:
    """Live BitNet trainer factory — deferred to v0.52.1."""
    raise NotImplementedError(
        "BitNet 1.58-bit fine-tuning live wiring deferred to v0.52.1. "
        "Schema accepts quantization='bitnet_1.58' but no trainer integration "
        "is registered yet."
    )


def export_bitnet_gguf() -> None:
    """Live BitNet GGUF export — deferred to v0.52.1."""
    raise NotImplementedError(
        "BitNet / TQ1_0 GGUF export live wiring deferred to v0.52.1. "
        "Schema accepts --format bitnet / --format tq1_0 but no export "
        "pipeline is registered yet."
    )
