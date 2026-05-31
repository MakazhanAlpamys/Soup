"""v0.62.0 Part C — Activation steering (CAA / ITI / RepE).

Three control-vector backends for inference-time intervention:

* ``caa`` — Contrastive Activation Addition (Panickssery et al., 2023).
  Add a contrastive vector to the residual stream.
* ``iti`` — Inference-Time Intervention (Li et al., 2023). Shift specific
  attention heads.
* ``repe`` — Representation Engineering (Zou et al., 2023). PCA-based
  direction in the residual stream.

Schema-only release: validators + frozen dataclasses + CLI surface ship
in v0.62.0. The live forward-hook + decode-time intervention land in
v0.62.1, mirroring the v0.50.0 / v0.52.0 / v0.61.0 stub-then-live cadence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

SUPPORTED_STEERING_METHODS: frozenset[str] = frozenset({"caa", "iti", "repe"})

_MAX_METHOD_LEN: int = 32
_MAX_NAME_LEN: int = 128
_MAX_STRENGTH_ABS: float = 10.0  # |strength| <= 10 sanity cap.

# Kebab-case + underscore + dots only. Path-separators / whitespace /
# shell-metacharacters all rejected so the name can be safely embedded in
# CLI args, filenames, and Rich markup. Mirrors v0.57.0 adapter-branch
# policy (alphanumeric + `._-`).
_NAME_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")


@dataclass(frozen=True)
class SteeringMethodSpec:
    """Metadata for a single steering backend. Frozen post-construction."""

    name: str
    description: str
    needs_contrastive_pairs: bool
    needs_attention_heads: bool
    live_wired: bool


_STEERING_METHOD_METADATA: Mapping[str, SteeringMethodSpec] = MappingProxyType({
    "caa": SteeringMethodSpec(
        name="caa",
        description=(
            "Contrastive Activation Addition — add a contrastive vector "
            "to the residual stream during decoding. Trains on "
            "(positive, negative) prompt pairs (Panickssery et al., 2023)."
        ),
        needs_contrastive_pairs=True,
        needs_attention_heads=False,
        live_wired=False,
    ),
    "iti": SteeringMethodSpec(
        name="iti",
        description=(
            "Inference-Time Intervention — shift specific attention heads "
            "along a learned direction (Li et al., 2023). Needs per-head "
            "calibration."
        ),
        needs_contrastive_pairs=True,
        needs_attention_heads=True,
        live_wired=False,
    ),
    "repe": SteeringMethodSpec(
        name="repe",
        description=(
            "Representation Engineering — PCA over hidden states to "
            "extract a behavioural direction (Zou et al., 2023)."
        ),
        needs_contrastive_pairs=True,
        needs_attention_heads=False,
        live_wired=False,
    ),
})


def validate_steering_method(value: object) -> str:
    """Normalise + validate a steering-method name.

    Mirrors v0.41.0 / v0.51.0 / v0.61.0 validator policy: bool-rejected,
    null-byte-rejected, oversize-rejected, case-insensitive normalisation,
    unknown rejected with friendly actionable message.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"steering method must not be bool, got {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            f"steering method must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("steering method must be non-empty")
    if "\x00" in value:
        raise ValueError("steering method must not contain null bytes")
    if len(value) > _MAX_METHOD_LEN:
        raise ValueError(
            f"steering method must be <= {_MAX_METHOD_LEN} chars"
        )
    canonical = value.lower()
    if canonical not in SUPPORTED_STEERING_METHODS:
        supported = ", ".join(sorted(SUPPORTED_STEERING_METHODS))
        raise ValueError(
            f"unknown steering method {value!r}; supported: {supported}"
        )
    return canonical


def validate_steering_name(value: object) -> str:
    """Validate an operator-supplied steering-vector name.

    Returns the value unchanged on success. Closed regex allowlist
    (alphanumeric + ``._-``, leading-alnum, ≤128 chars) so the name can
    be safely used as a Registry artifact id, CLI flag, and filename
    fragment on every platform.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"steering name must not be bool, got {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            f"steering name must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("steering name must be non-empty")
    if "\x00" in value:
        raise ValueError("steering name must not contain null bytes")
    if len(value) > _MAX_NAME_LEN:
        raise ValueError(
            f"steering name must be <= {_MAX_NAME_LEN} chars"
        )
    if not _NAME_RE.match(value):
        raise ValueError(
            f"steering name {value!r} must match {_NAME_RE.pattern!r} "
            "(alphanumeric + `._-`, leading alnum)."
        )
    return value


def validate_steering_strength(value: object) -> float:
    """Validate a steering strength multiplier.

    Bool-rejected (bool is a subclass of int), NaN/Inf-rejected via
    ``math.isfinite``, bounded ``|strength| <= 10.0`` as a sanity cap.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"steering strength must not be bool, got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"steering strength must be a number, got {type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError("steering strength must be finite (no NaN / Inf)")
    if abs(fval) > _MAX_STRENGTH_ABS:
        raise ValueError(
            f"steering strength must satisfy |s| <= {_MAX_STRENGTH_ABS}; "
            f"got {fval}"
        )
    return fval


def get_steering_method_spec(name: str) -> SteeringMethodSpec:
    """Return the frozen :class:`SteeringMethodSpec` for ``name`` or raise."""
    canonical = validate_steering_method(name)
    return _STEERING_METHOD_METADATA[canonical]


def apply_steering(method: str) -> None:
    """Apply a steering vector during decoding — deferred to v0.62.1.

    Validates the method name first so the deferred-live error
    distinguishes between "unknown method" and "method is on the
    allowlist but not yet wired". Mirrors v0.50.0 ``apply_variant_loss`` /
    v0.61.0 ``apply_unlearn_loss`` policy.
    """
    canonical = validate_steering_method(method)
    raise NotImplementedError(
        f"apply_steering({canonical!r}) is deferred to v0.62.1. "
        "Schema accepts the method now so callers can write soup.yaml "
        "today, but the live forward-hook + decode-time intervention "
        "land in v0.62.1."
    )


def build_steering_vector(
    *,
    method: str,
    name: str,
    pairs_path: Optional[str] = None,
    layer: Optional[int] = None,
) -> None:
    """Train a steering vector from contrastive pairs — deferred to v0.62.1.

    Validates inputs first; the deferred-live error fires only after the
    method + name + (optional) layer all pass shape/range checks.
    """
    canonical = validate_steering_method(method)
    canonical_name = validate_steering_name(name)
    if pairs_path is not None:
        if not isinstance(pairs_path, str):
            raise TypeError(
                f"pairs_path must be str, got {type(pairs_path).__name__}"
            )
        if not pairs_path:
            raise ValueError("pairs_path must be non-empty")
        if "\x00" in pairs_path:
            raise ValueError("pairs_path must not contain null bytes")
    if layer is not None:
        if isinstance(layer, bool):
            raise TypeError("layer must not be bool")
        if not isinstance(layer, int):
            raise TypeError(
                f"layer must be int, got {type(layer).__name__}"
            )
        if layer < 0 or layer > 2048:
            raise ValueError(
                f"layer must satisfy 0 <= layer <= 2048, got {layer}"
            )
    raise NotImplementedError(
        f"build_steering_vector(method={canonical!r}, "
        f"name={canonical_name!r}) is deferred to v0.62.1."
    )
