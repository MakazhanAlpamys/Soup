"""License-conflict matrix at adapter merge (v0.60.0 Part E).

Closed compatibility table for the top open-source / model-license families.
``check_license_compat`` returns a ``LicenseConflictReport`` when the supplied
list contains an incompatible pair; ``adapter_merge`` consults this helper
and refuses to merge unless ``--license-override <reason>`` is passed.

Compatibility is intentionally conservative: when in doubt, flag. Operators
who have legal clearance can override; the override reason is captured in
the audit log so a future legal review can trace the decision.

Public surface:
- ``KNOWN_LICENSES`` frozenset of recognised SPDX-ish ids.
- ``LICENSE_MATRIX`` ``MappingProxyType`` mapping license-id -> tuple of
  compatible counterparts.
- ``LICENSE_KINDS`` ``MappingProxyType`` mapping license-id -> category.
- ``check_license_compat(licenses)`` -> ``LicenseConflictReport``.
- ``validate_license_override_reason(reason)`` -> validated string.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# License "kinds" — broad categories that drive the compatibility matrix.
# Closed allowlist; mixing categories that are flagged "permissive" with
# "copyleft" or "non-commercial" produces a conflict.
_PERMISSIVE = "permissive"
_WEAK_COPYLEFT = "weak-copyleft"
_STRONG_COPYLEFT = "strong-copyleft"
_NON_COMMERCIAL = "non-commercial"
_RESTRICTED_USE = "restricted-use"  # Llama / Gemma / etc. — community licenses
_OPENRAIL = "openrail"             # CreativeML OpenRAIL family — use-based
_PROPRIETARY = "proprietary"        # OpenAI ToS, etc.

_LICENSE_KINDS_RAW: dict[str, str] = {
    # Permissive
    "apache-2.0": _PERMISSIVE,
    "mit": _PERMISSIVE,
    "bsd-2-clause": _PERMISSIVE,
    "bsd-3-clause": _PERMISSIVE,
    "isc": _PERMISSIVE,
    "unlicense": _PERMISSIVE,
    "cc0-1.0": _PERMISSIVE,
    "cc-by-4.0": _PERMISSIVE,
    # Weak copyleft
    "lgpl-2.1": _WEAK_COPYLEFT,
    "lgpl-3.0": _WEAK_COPYLEFT,
    "mpl-2.0": _WEAK_COPYLEFT,
    # Strong copyleft
    "gpl-2.0": _STRONG_COPYLEFT,
    "gpl-3.0": _STRONG_COPYLEFT,
    "agpl-3.0": _STRONG_COPYLEFT,
    # Non-commercial
    "cc-by-nc-4.0": _NON_COMMERCIAL,
    "cc-by-nc-sa-4.0": _NON_COMMERCIAL,
    "cc-by-nc-nd-4.0": _NON_COMMERCIAL,
    # Restricted-use model licenses
    "llama-2": _RESTRICTED_USE,
    "llama-3": _RESTRICTED_USE,
    "llama-3.1": _RESTRICTED_USE,
    "llama-3.2": _RESTRICTED_USE,
    "llama-3.3": _RESTRICTED_USE,
    "llama-community": _RESTRICTED_USE,
    "gemma": _RESTRICTED_USE,
    "qwen-research": _RESTRICTED_USE,
    "qwen-license": _RESTRICTED_USE,
    "mistral-research": _RESTRICTED_USE,
    # OpenRAIL family
    "openrail": _OPENRAIL,
    "creativeml-openrail-m": _OPENRAIL,
    "bigscience-openrail-m": _OPENRAIL,
    "bigcode-openrail-m": _OPENRAIL,
    # Proprietary ToS
    "openai-tos": _PROPRIETARY,
    "anthropic-aup": _PROPRIETARY,
}

KNOWN_LICENSES = frozenset(_LICENSE_KINDS_RAW.keys())
LICENSE_KINDS = types.MappingProxyType(dict(_LICENSE_KINDS_RAW))

# Compatibility rules. Each entry is the set of kinds that the row kind can
# be safely combined with. Conservative-by-design — when legal counsel is
# uncertain, flag the operator (they can `--license-override <reason>`).
_COMPAT_RAW: dict[str, Tuple[str, ...]] = {
    _PERMISSIVE: (_PERMISSIVE,),
    _WEAK_COPYLEFT: (_WEAK_COPYLEFT, _PERMISSIVE),
    _STRONG_COPYLEFT: (_STRONG_COPYLEFT,),
    _NON_COMMERCIAL: (_NON_COMMERCIAL,),
    _RESTRICTED_USE: (_RESTRICTED_USE,),
    _OPENRAIL: (_OPENRAIL,),
    _PROPRIETARY: (_PROPRIETARY,),
}
LICENSE_MATRIX = types.MappingProxyType(dict(_COMPAT_RAW))

_MIN_OVERRIDE_REASON = 8
_MAX_OVERRIDE_REASON = 4096


@dataclass(frozen=True)
class LicenseConflictReport:
    """Outcome of ``check_license_compat``."""

    ok: bool
    licenses: Tuple[str, ...]
    conflict_pair: Optional[Tuple[str, str]]
    reason: str


def normalise_license_id(value: object) -> str:
    """Lowercase + strip + validate a license id."""
    if isinstance(value, bool):
        raise TypeError("license id must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"license id must be str, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError("license id must not contain null bytes")
    stripped = value.strip()
    if not stripped:
        raise ValueError("license id must be non-empty")
    if len(stripped) > 128:
        raise ValueError("license id too long")
    return stripped.lower()


def _kind_of(license_id: str) -> Optional[str]:
    return LICENSE_KINDS.get(license_id)


def check_license_compat(
    licenses: Sequence[object],
) -> LicenseConflictReport:
    """Decide whether a list of licenses can be combined in one merged adapter.

    Decision rule:
    - Normalise + dedup the input.
    - If any license is unknown → ``ok=False`` with reason naming it.
    - Otherwise, compute the categories. The merge is OK iff every
      pair-wise (kind_i, kind_j) satisfies kind_j in LICENSE_MATRIX[kind_i].

    Returns the first failing pair so the operator gets actionable advice.
    """
    if isinstance(licenses, (str, bytes)) or not isinstance(licenses, Sequence):
        raise TypeError("licenses must be a list of strings")
    if len(licenses) == 0:
        raise ValueError("licenses must contain at least one entry")

    norm: list[str] = []
    for entry in licenses:
        norm.append(normalise_license_id(entry))

    # Dedup while preserving order.
    seen: set[str] = set()
    dedup: list[str] = []
    for lic in norm:
        if lic not in seen:
            seen.add(lic)
            dedup.append(lic)

    licenses_tuple = tuple(dedup)

    # Single license — trivially compatible.
    if len(dedup) == 1:
        if _kind_of(dedup[0]) is None:
            return LicenseConflictReport(
                ok=False,
                licenses=licenses_tuple,
                conflict_pair=None,
                reason=f"unknown license: {dedup[0]!r}",
            )
        return LicenseConflictReport(
            ok=True, licenses=licenses_tuple, conflict_pair=None,
            reason="single-license merge",
        )

    # Probe every pair (i, j) and require BOTH directions in the matrix.
    for first in dedup:
        kind_a = _kind_of(first)
        if kind_a is None:
            return LicenseConflictReport(
                ok=False,
                licenses=licenses_tuple,
                conflict_pair=None,
                reason=f"unknown license: {first!r}",
            )
        for second in dedup:
            if first == second:
                continue
            kind_b = _kind_of(second)
            if kind_b is None:
                return LicenseConflictReport(
                    ok=False,
                    licenses=licenses_tuple,
                    conflict_pair=None,
                    reason=f"unknown license: {second!r}",
                )
            compat = LICENSE_MATRIX.get(kind_a, ())
            if kind_b not in compat:
                # Friendlier reason names the category so operators can
                # search the matrix.
                if kind_b == _NON_COMMERCIAL or kind_a == _NON_COMMERCIAL:
                    msg = (
                        f"non-commercial license {kind_b!r} cannot combine "
                        f"with {kind_a!r}: {first} vs {second}"
                    )
                elif _STRONG_COPYLEFT in (kind_a, kind_b):
                    msg = (
                        f"strong copyleft {kind_a!r}/{kind_b!r} incompatible "
                        f"with non-copyleft license: {first} vs {second}"
                    )
                elif _RESTRICTED_USE in (kind_a, kind_b):
                    msg = (
                        f"restricted-use license {kind_a!r}/{kind_b!r} "
                        f"adds acceptable-use clauses that "
                        f"{first} vs {second} cannot satisfy"
                    )
                else:
                    msg = (
                        f"license categories incompatible: "
                        f"{kind_a!r} vs {kind_b!r} "
                        f"({first} vs {second})"
                    )
                return LicenseConflictReport(
                    ok=False,
                    licenses=licenses_tuple,
                    conflict_pair=(first, second),
                    reason=msg,
                )

    return LicenseConflictReport(
        ok=True,
        licenses=licenses_tuple,
        conflict_pair=None,
        reason="all license pairs compatible",
    )


def validate_license_override_reason(reason: object) -> str:
    """Sanity-check the ``--license-override <reason>`` payload.

    Defends against `--license-override y` style placeholder bypasses: the
    reason must be ≥ 8 chars, ≤ 4096 chars, free of null bytes, and a string.
    """
    if isinstance(reason, bool):
        raise TypeError("override reason must be str, not bool")
    if not isinstance(reason, str):
        raise TypeError(f"override reason must be str, got {type(reason).__name__}")
    if "\x00" in reason:
        raise ValueError("override reason must not contain null bytes")
    stripped = reason.strip()
    if len(stripped) < _MIN_OVERRIDE_REASON:
        raise ValueError(
            f"override reason too short (need ≥ {_MIN_OVERRIDE_REASON} chars)"
        )
    if len(stripped) > _MAX_OVERRIDE_REASON:
        raise ValueError(
            f"override reason too long (> {_MAX_OVERRIDE_REASON} chars)"
        )
    return stripped
