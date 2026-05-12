"""v0.53.0 Part F — Advanced save / merge formats schema helpers.

Two new save-format surfaces ship this release (schema-only):

* ``soup merge --save-format <fmt>`` where fmt ∈ {fp16, 4bit, 4bit_forced}.
  ``4bit`` writes a single BNB-4bit-quantized merged checkpoint without
  the dequant → merge → requant cycle (unsloth ``merged_4bit`` recipe).
  ``4bit_forced`` is the unsloth ``4bit_forced`` shortcut.
* ``soup export --format torchao --quant-config <yaml>`` — invokes
  ``torchao.quantize_`` then ``save_pretrained`` for the
  ``Int4WeightOnly`` / ``Int8DynActInt4`` / ``Float8DynActFloat8`` /
  ``NVFP4`` PTQ schemes (unsloth + axolotl parity).

Live wiring deferred to v0.53.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# --- Merge save formats ------------------------------------------------------
MERGE_SAVE_FORMATS: frozenset[str] = frozenset({
    "fp16", "4bit", "4bit_forced",
})

# --- TorchAO PTQ schemes (a closed allowlist, mirrors the v0.38.0 Quant Menu
# string convention so YAML round-trips cleanly).
TORCHAO_PTQ_SCHEMES: frozenset[str] = frozenset({
    "Int4WeightOnly",
    "Int8DynActInt4",
    "Float8DynActFloat8",
    "NVFP4",
})

_MAX_SAVE_FORMAT_LEN: int = 32
_MAX_TORCHAO_SCHEME_LEN: int = 48


@dataclass(frozen=True)
class MergeSaveSpec:
    """Frozen metadata for a merge-save format."""

    name: str
    bits: int
    description: str
    live_wired: bool


_MERGE_METADATA: Mapping[str, MergeSaveSpec] = MappingProxyType({
    "fp16": MergeSaveSpec(
        name="fp16", bits=16,
        description="Standard FP16 merged checkpoint (default — pre-v0.53.0)",
        live_wired=True,
    ),
    "4bit": MergeSaveSpec(
        name="4bit", bits=4,
        description="Single BNB-4bit-quantized merged checkpoint",
        live_wired=False,
    ),
    "4bit_forced": MergeSaveSpec(
        name="4bit_forced", bits=4,
        description="Forced BNB-4bit merge (unsloth 4bit_forced)",
        live_wired=False,
    ),
})


@dataclass(frozen=True)
class TorchAOPTQSpec:
    """Frozen metadata for a TorchAO PTQ scheme."""

    name: str
    bits: int
    description: str
    live_wired: bool


_TORCHAO_METADATA: Mapping[str, TorchAOPTQSpec] = MappingProxyType({
    "Int4WeightOnly": TorchAOPTQSpec(
        name="Int4WeightOnly", bits=4,
        description="TorchAO Int4WeightOnly (weight-only int4)",
        live_wired=False,
    ),
    "Int8DynActInt4": TorchAOPTQSpec(
        name="Int8DynActInt4", bits=4,
        description="TorchAO Int8DynActInt4 (dynamic int8 act + int4 weight)",
        live_wired=False,
    ),
    "Float8DynActFloat8": TorchAOPTQSpec(
        name="Float8DynActFloat8", bits=8,
        description="TorchAO FP8 dynamic activations + FP8 weight",
        live_wired=False,
    ),
    "NVFP4": TorchAOPTQSpec(
        name="NVFP4", bits=4,
        description="TorchAO NVFP4 (Blackwell FP4 PTQ)",
        live_wired=False,
    ),
})


def _validate_string_field(value: object, field: str, max_len: int) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} too long (max {max_len} chars)")
    return value


def validate_merge_save_format(value: object) -> str:
    """Validate ``--save-format`` arg. Case-insensitive normalisation.

    Mirrors v0.41.0 ``optimizer`` policy. Returns the lowercase canonical
    form so ``_MERGE_METADATA`` (all-lowercase keys) lookups are O(1).
    """
    validated = _validate_string_field(value, "save_format", _MAX_SAVE_FORMAT_LEN)
    canonical = validated.lower()
    if canonical not in MERGE_SAVE_FORMATS:
        supported = ", ".join(sorted(MERGE_SAVE_FORMATS))
        raise ValueError(
            f"save_format {value!r} not supported. Supported: {supported}"
        )
    return canonical


def validate_torchao_scheme(value: object) -> str:
    """Validate a TorchAO PTQ scheme name.

    INTENTIONALLY CASE-SENSITIVE — these are PyTorch class names
    (``Int4WeightOnly`` / ``NVFP4`` etc.) that ``torchao.quantize_`` looks up
    by exact name. Diverges from ``validate_merge_save_format`` (which
    lowercase-normalises) and ``validate_kv_cache_type`` (also lowercase)
    on purpose: TorchAO uses CapWords, the others are operator-facing flags.
    """
    validated = _validate_string_field(
        value, "torchao_scheme", _MAX_TORCHAO_SCHEME_LEN,
    )
    if validated not in TORCHAO_PTQ_SCHEMES:
        supported = ", ".join(sorted(TORCHAO_PTQ_SCHEMES))
        raise ValueError(
            f"torchao_scheme {value!r} not supported. Supported: {supported}"
        )
    return validated


def get_merge_save_spec(name: str) -> MergeSaveSpec:
    """Return the frozen :class:`MergeSaveSpec` for ``name`` (case-insensitive)."""
    canonical = validate_merge_save_format(name)
    return _MERGE_METADATA[canonical]


def get_torchao_spec(name: str) -> TorchAOPTQSpec:
    """Return the frozen :class:`TorchAOPTQSpec` for ``name`` (case-sensitive)."""
    canonical = validate_torchao_scheme(name)
    return _TORCHAO_METADATA[canonical]


def validate_quant_config_path(path: object) -> str:
    """Validate ``--quant-config <yaml>`` argument shape.

    Boundary contract — what's enforced HERE vs at CLI dispatch (v0.53.1):

    THIS helper enforces non-empty ``str``, no null bytes, length <= 4096.

    CLI dispatch in v0.53.1 MUST additionally apply (mirrors v0.43.0 /
    v0.46.0 / v0.47.0 TOCTOU policy):
    * ``os.path.realpath`` + ``os.path.commonpath`` cwd containment
    * ``os.lstat`` + ``stat.S_ISLNK`` rejection before any ``open()``
    * existence + extension check (``.yaml`` / ``.yml``)
    * ``yaml.safe_load`` only.
    """
    if isinstance(path, bool):
        raise TypeError(f"quant_config must not be bool, got {path!r}")
    if not isinstance(path, str):
        raise TypeError(
            f"quant_config must be str, got {type(path).__name__}"
        )
    if not path:
        raise ValueError("quant_config must be non-empty")
    if "\x00" in path:
        raise ValueError("quant_config must not contain null bytes")
    if len(path) > 4096:
        raise ValueError("quant_config path too long (max 4096 chars)")
    return path


def merge_4bit() -> None:
    """Live 4bit-merge wiring — deferred to v0.53.1."""
    raise NotImplementedError(
        "soup merge --save-format 4bit live wiring deferred to v0.53.1. "
        "Schema accepts the flag but no merged-4bit writer is registered yet."
    )


def export_torchao() -> None:
    """Live TorchAO PTQ export — deferred to v0.53.1."""
    raise NotImplementedError(
        "soup export --format torchao live wiring deferred to v0.53.1. "
        "Schema accepts the format but no torchao.quantize_ call is wired."
    )
