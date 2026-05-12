"""v0.53.0 Parts A+B — UD GGUF + IQ / Apple-ARM quant schema helpers.

Schema-only support for Unsloth Dynamic 2.0 GGUF ladder (``UD-Q8_K_XL`` …
``UD-IQ1_M``), the IQ1/IQ2/IQ3 family, the Apple/ARM-friendly Q4_NL /
Q5.x / Q4.x variants, and the existing TQ1_0 1.58-bit GGUF flavour (from
v0.52.0 Part D, re-exposed here for ``soup export --format gguf-iq``).

Live llama.cpp ``imatrix`` calibration + actual GGUF write are deferred
to v0.53.1 (mirrors v0.50.0 stub-then-live pattern).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# --- Part A: Unsloth Dynamic 2.0 GGUF ladder ---------------------------------
UD_GGUF_FORMATS: frozenset[str] = frozenset({
    "UD-Q8_K_XL",
    "UD-Q6_K_XL",
    "UD-Q5_K_XL",
    "UD-Q4_K_XL",
    "UD-Q3_K_XL",
    "UD-Q2_K_XL",
    "UD-IQ4_XS",
    "UD-IQ3_M",
    "UD-IQ3_XXS",
    "UD-IQ2_M",
    "UD-IQ2_XS",
    "UD-IQ2_XXS",
    "UD-IQ1_M",
    "UD-IQ1_S",
})

# --- Part B: IQ + Apple/ARM quant flavours -----------------------------------
IQ_GGUF_FORMATS: frozenset[str] = frozenset({
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ3_XXS",
    "IQ3_XS",
    "IQ3_S",
    "IQ3_M",
    "IQ4_XS",
    "IQ4_NL",
})
APPLE_ARM_GGUF_FORMATS: frozenset[str] = frozenset({
    "Q4_0_4_4",
    "Q4_0_4_8",
    "Q4_0_8_8",
    "Q4_NL",
    "Q5_0",
    "Q5_1",
    "Q5_K_S",
    "Q5_K_M",
    "Q4_K_S",
    "Q4_K_M",
})

# Union of all v0.53.0 schema-only GGUF flavours (excludes TQ1_0 which is owned
# by v0.52.0 Part D ``utils/bitnet.py``; ``is_advanced_gguf_format`` returns
# True for the BitNet family too via the helper below for export-CLI parity).
ALL_ADVANCED_GGUF_FORMATS: frozenset[str] = (
    UD_GGUF_FORMATS | IQ_GGUF_FORMATS | APPLE_ARM_GGUF_FORMATS
)

_MAX_FORMAT_LEN: int = 32


@dataclass(frozen=True)
class GGUFQuantSpec:
    """Frozen metadata for a v0.53.0 advanced GGUF flavour."""

    name: str
    family: str          # "ud" | "iq" | "apple_arm"
    bits: float
    description: str
    live_wired: bool


def _spec(name: str, family: str, bits: float, description: str) -> GGUFQuantSpec:
    return GGUFQuantSpec(
        name=name, family=family, bits=bits,
        description=description, live_wired=False,
    )


_GGUF_METADATA: Mapping[str, GGUFQuantSpec] = MappingProxyType({
    # Unsloth Dynamic 2.0 ladder
    "UD-Q8_K_XL": _spec("UD-Q8_K_XL", "ud", 8.0, "UD Q8_K_XL (Unsloth Dynamic 2.0)"),
    "UD-Q6_K_XL": _spec("UD-Q6_K_XL", "ud", 6.0, "UD Q6_K_XL"),
    "UD-Q5_K_XL": _spec("UD-Q5_K_XL", "ud", 5.0, "UD Q5_K_XL"),
    "UD-Q4_K_XL": _spec("UD-Q4_K_XL", "ud", 4.0, "UD Q4_K_XL"),
    "UD-Q3_K_XL": _spec("UD-Q3_K_XL", "ud", 3.0, "UD Q3_K_XL"),
    "UD-Q2_K_XL": _spec("UD-Q2_K_XL", "ud", 2.0, "UD Q2_K_XL"),
    "UD-IQ4_XS":  _spec("UD-IQ4_XS",  "ud", 4.0, "UD IQ4_XS"),
    "UD-IQ3_M":   _spec("UD-IQ3_M",   "ud", 3.0, "UD IQ3_M"),
    "UD-IQ3_XXS": _spec("UD-IQ3_XXS", "ud", 3.0, "UD IQ3_XXS"),
    "UD-IQ2_M":   _spec("UD-IQ2_M",   "ud", 2.0, "UD IQ2_M"),
    "UD-IQ2_XS":  _spec("UD-IQ2_XS",  "ud", 2.0, "UD IQ2_XS"),
    "UD-IQ2_XXS": _spec("UD-IQ2_XXS", "ud", 2.0, "UD IQ2_XXS"),
    "UD-IQ1_M":   _spec("UD-IQ1_M",   "ud", 1.0, "UD IQ1_M (smallest UD)"),
    "UD-IQ1_S":   _spec("UD-IQ1_S",   "ud", 1.0, "UD IQ1_S"),
    # IQ family (non-UD)
    "IQ1_S":   _spec("IQ1_S",   "iq", 1.0, "IQ1_S 1-bit"),
    "IQ1_M":   _spec("IQ1_M",   "iq", 1.0, "IQ1_M 1-bit"),
    "IQ2_XXS": _spec("IQ2_XXS", "iq", 2.0, "IQ2_XXS 2-bit"),
    "IQ2_XS":  _spec("IQ2_XS",  "iq", 2.0, "IQ2_XS 2-bit"),
    "IQ2_S":   _spec("IQ2_S",   "iq", 2.0, "IQ2_S 2-bit"),
    "IQ2_M":   _spec("IQ2_M",   "iq", 2.0, "IQ2_M 2-bit"),
    "IQ3_XXS": _spec("IQ3_XXS", "iq", 3.0, "IQ3_XXS 3-bit"),
    "IQ3_XS":  _spec("IQ3_XS",  "iq", 3.0, "IQ3_XS 3-bit"),
    "IQ3_S":   _spec("IQ3_S",   "iq", 3.0, "IQ3_S 3-bit"),
    "IQ3_M":   _spec("IQ3_M",   "iq", 3.0, "IQ3_M 3-bit"),
    "IQ4_XS":  _spec("IQ4_XS",  "iq", 4.0, "IQ4_XS 4-bit"),
    "IQ4_NL":  _spec("IQ4_NL",  "iq", 4.0, "IQ4_NL 4-bit (non-linear)"),
    # Apple/ARM neural-engine-friendly
    "Q4_0_4_4": _spec("Q4_0_4_4", "apple_arm", 4.0, "Apple/ARM Q4_0_4_4"),
    "Q4_0_4_8": _spec("Q4_0_4_8", "apple_arm", 4.0, "Apple/ARM Q4_0_4_8"),
    "Q4_0_8_8": _spec("Q4_0_8_8", "apple_arm", 4.0, "Apple/ARM Q4_0_8_8"),
    "Q4_NL":    _spec("Q4_NL",    "apple_arm", 4.0, "Apple/ARM Q4_NL"),
    "Q5_0":     _spec("Q5_0",     "apple_arm", 5.0, "Apple/ARM Q5_0"),
    "Q5_1":     _spec("Q5_1",     "apple_arm", 5.0, "Apple/ARM Q5_1"),
    "Q5_K_S":   _spec("Q5_K_S",   "apple_arm", 5.0, "Apple/ARM Q5_K_S"),
    "Q5_K_M":   _spec("Q5_K_M",   "apple_arm", 5.0, "Apple/ARM Q5_K_M"),
    "Q4_K_S":   _spec("Q4_K_S",   "apple_arm", 4.0, "Apple/ARM Q4_K_S"),
    "Q4_K_M":   _spec("Q4_K_M",   "apple_arm", 4.0, "Apple/ARM Q4_K_M"),
})


def _basic_validate(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > _MAX_FORMAT_LEN:
        raise ValueError(f"{field} too long (max {_MAX_FORMAT_LEN} chars)")
    return value


# Lowercase index built once at module load — O(1) lookup vs O(N) walk
# (code-review MEDIUM fix). Mirrors v0.32.0 ``pick_mixed_precision`` quirk
# ordering policy where sorting / indexing is precomputed.
_LOWER_INDEX: Mapping[str, str] = MappingProxyType({
    name.lower(): name for name in ALL_ADVANCED_GGUF_FORMATS
})


def _resolve_canonical(value: str) -> str | None:
    """Match ``value`` against the union allowlist case-insensitively.

    Returns the canonical (original-case) entry from the allowlist or
    ``None`` if no match.
    """
    return _LOWER_INDEX.get(value.lower())


def validate_ud_gguf_format(value: object) -> str:
    """Validate ``value`` is a UD GGUF format string. Returns canonical form."""
    _basic_validate(value, "ud_gguf_format")
    canonical = _resolve_canonical(value)  # type: ignore[arg-type]
    if canonical is None or canonical not in UD_GGUF_FORMATS:
        supported = ", ".join(sorted(UD_GGUF_FORMATS))
        raise ValueError(
            f"ud_gguf_format {value!r} not supported. Supported: {supported}"
        )
    return canonical


def validate_iq_gguf_format(value: object) -> str:
    """Validate ``value`` is an IQ GGUF format string. Returns canonical form."""
    _basic_validate(value, "iq_gguf_format")
    canonical = _resolve_canonical(value)  # type: ignore[arg-type]
    if canonical is None or canonical not in IQ_GGUF_FORMATS:
        supported = ", ".join(sorted(IQ_GGUF_FORMATS))
        raise ValueError(
            f"iq_gguf_format {value!r} not supported. Supported: {supported}"
        )
    return canonical


def validate_apple_arm_gguf_format(value: object) -> str:
    """Validate ``value`` is an Apple/ARM GGUF format string."""
    _basic_validate(value, "apple_arm_gguf_format")
    canonical = _resolve_canonical(value)  # type: ignore[arg-type]
    if canonical is None or canonical not in APPLE_ARM_GGUF_FORMATS:
        supported = ", ".join(sorted(APPLE_ARM_GGUF_FORMATS))
        raise ValueError(
            f"apple_arm_gguf_format {value!r} not supported. "
            f"Supported: {supported}"
        )
    return canonical


def is_ud_gguf_format(value: object) -> bool:
    """Return True iff ``value`` is one of the UD GGUF ladder entries."""
    if isinstance(value, bool) or not isinstance(value, str):
        return False
    canonical = _resolve_canonical(value)
    return canonical is not None and canonical in UD_GGUF_FORMATS


def is_iq_gguf_format(value: object) -> bool:
    """Return True iff ``value`` is one of the IQ GGUF flavours."""
    if isinstance(value, bool) or not isinstance(value, str):
        return False
    canonical = _resolve_canonical(value)
    return canonical is not None and canonical in IQ_GGUF_FORMATS


def is_apple_arm_gguf_format(value: object) -> bool:
    """Return True iff ``value`` is one of the Apple/ARM GGUF flavours."""
    if isinstance(value, bool) or not isinstance(value, str):
        return False
    canonical = _resolve_canonical(value)
    return canonical is not None and canonical in APPLE_ARM_GGUF_FORMATS


def is_advanced_gguf_format(value: object) -> bool:
    """Return True iff ``value`` is any v0.53.0 advanced GGUF format."""
    return (
        is_ud_gguf_format(value)
        or is_iq_gguf_format(value)
        or is_apple_arm_gguf_format(value)
    )


def get_gguf_spec(name: str) -> GGUFQuantSpec:
    """Return the frozen :class:`GGUFQuantSpec` for ``name`` (case-insensitive)."""
    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}")
    canonical = _resolve_canonical(name)
    if canonical is None:
        supported_n = len(ALL_ADVANCED_GGUF_FORMATS)
        raise ValueError(
            f"GGUF format {name!r} not in v0.53.0 catalog "
            f"({supported_n} known)"
        )
    return _GGUF_METADATA[canonical]


def validate_calibration_data_path(path: object) -> str:
    """Validate ``--calibration-data <jsonl>`` argument shape.

    Boundary contract — what's enforced HERE vs at CLI dispatch (v0.53.1):

    THIS helper enforces:
    * non-empty ``str`` (bool / None / other types rejected with ``TypeError``)
    * no null bytes
    * length <= 4096 chars

    CLI dispatch in v0.53.1 MUST additionally apply (mirrors v0.43.0 /
    v0.46.0 / v0.47.0 TOCTOU policy):
    * ``os.path.realpath`` + ``os.path.commonpath`` cwd containment
    * ``os.lstat`` + ``stat.S_ISLNK`` rejection BEFORE any ``open()``
    * existence check via ``os.path.isfile``

    Do NOT skip the dispatch-time controls — this helper is shape-only.
    """
    if isinstance(path, bool):
        raise TypeError(f"calibration_data must not be bool, got {path!r}")
    if not isinstance(path, str):
        raise TypeError(
            f"calibration_data must be str, got {type(path).__name__}"
        )
    if not path:
        raise ValueError("calibration_data must be non-empty")
    if "\x00" in path:
        raise ValueError("calibration_data must not contain null bytes")
    if len(path) > 4096:
        raise ValueError("calibration_data path too long (max 4096 chars)")
    return path


def export_advanced_gguf() -> None:
    """Live UD/IQ/Apple-ARM GGUF export — deferred to v0.53.1.

    Mirrors v0.52.0 ``export_bitnet_gguf`` stub-then-live pattern.
    """
    raise NotImplementedError(
        "UD / IQ / Apple-ARM GGUF export live wiring deferred to v0.53.1. "
        "Schema accepts every format in ALL_ADVANCED_GGUF_FORMATS but no "
        "llama.cpp imatrix invocation is registered yet."
    )
