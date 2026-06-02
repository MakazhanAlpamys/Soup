"""License advisor (v0.64.0 Part F).

Given a target deployment (``b2c``, ``defense``, ``embedded``),
recommend a license-clean base model + flag downstream risk for a
specific license id. Composes with v0.60 Part E
``license_matrix.check_license_compat`` for the merge-time gate.

The matrix here is deliberately *coarse-grained*: it groups licenses
into ``recommended`` / ``forbidden`` for each deploy target, and the
``flag_downstream_risk`` helper applies a per-license risk heuristic
(e.g. Llama community license + > 700M MAU = block, per Meta's
acceptable-use policy).

Live model-card scraping for license inference lands in v0.64.1;
v0.64.0 takes the license id from the operator (matches the v0.60
``--license <id>`` convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from soup_cli.utils.license_matrix import KNOWN_LICENSES, LICENSE_KINDS

DEPLOY_TARGETS = frozenset({"b2c", "defense", "embedded"})
_MAX_TARGET_LEN = 32
_MAX_LICENSE_LEN = 128
_VALID_SEVERITIES = frozenset({"ok", "warn", "block"})

# Llama community license: Meta requires a separate licence for products
# with > 700M monthly active users. We surface this as a downstream-risk
# block in B2C deployments.
_LLAMA_COMMUNITY_MAU_CAP = 700_000_000

# Tight allowlist of Llama-family community license ids. Defends against
# a future "llama-permissive" id wrongly tripping the MAU gate.
_LLAMA_COMMUNITY_LICENSES = frozenset({
    "llama-2",
    "llama-3",
    "llama-3.1",
    "llama-3.2",
    "llama-3.3",
    "llama-community",
})


def validate_deploy_target(value: object) -> str:
    """Normalise + validate a deploy target name."""
    if isinstance(value, bool):
        raise TypeError("target must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"target must be str, got {type(value).__name__}")
    if not value:
        raise ValueError("target must be non-empty")
    if "\x00" in value:
        raise ValueError("target must not contain null bytes")
    if len(value) > _MAX_TARGET_LEN:
        raise ValueError(f"target too int (> {_MAX_TARGET_LEN} chars)")
    normalised = value.lower().strip()
    if normalised not in DEPLOY_TARGETS:
        allowed = ", ".join(sorted(DEPLOY_TARGETS))
        raise ValueError(f"unknown target {value!r}; known: {allowed}")
    return normalised


@dataclass(frozen=True)
class LicenseRecommendation:
    """Output of ``advise_license_for_target``."""

    target: str
    recommended_licenses: Tuple[str, ...]
    forbidden_licenses: Tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.target, str):
            raise TypeError("target must be str")
        if not isinstance(self.recommended_licenses, tuple):
            raise TypeError("recommended_licenses must be a tuple of str")
        if not isinstance(self.forbidden_licenses, tuple):
            raise TypeError("forbidden_licenses must be a tuple of str")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be str")


# Permissive baseline used as the b2c "recommended" set. Operators
# almost always want these.
_PERMISSIVE_RECOMMENDED: Tuple[str, ...] = (
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "isc",
    "unlicense",
)

# Non-commercial licenses — always forbidden for B2C.
_NC_LICENSES: Tuple[str, ...] = (
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "cc-by-nc-nd-4.0",
)

# Restricted-use model licenses (Llama / Gemma / Qwen / Mistral
# community licenses) — forbidden for defense per acceptable-use clauses.
_RESTRICTED_USE: Tuple[str, ...] = (
    "llama-2",
    "llama-3",
    "llama-3.1",
    "llama-3.2",
    "llama-3.3",
    "llama-community",
    "gemma",
    "qwen-research",
    "qwen-license",
    "mistral-research",
)

# Strong copyleft — forbidden for embedded (driver / firmware tree
# incompatible with GPL/AGPL closed-source distribution).
_STRONG_COPYLEFT: Tuple[str, ...] = (
    "gpl-2.0",
    "gpl-3.0",
    "agpl-3.0",
)


def advise_license_for_target(target: object) -> LicenseRecommendation:
    """Return the per-target ``LicenseRecommendation``."""
    normalised = validate_deploy_target(target)
    if normalised == "b2c":
        return LicenseRecommendation(
            target=normalised,
            recommended_licenses=_PERMISSIVE_RECOMMENDED,
            forbidden_licenses=_NC_LICENSES,
            reason=(
                "Consumer B2C: prefer broadly-permissive licenses; "
                "non-commercial licenses are categorically forbidden."
            ),
        )
    if normalised == "defense":
        # Defense forbids restricted-use clauses on top of NC.
        forbidden = _NC_LICENSES + _RESTRICTED_USE
        return LicenseRecommendation(
            target=normalised,
            recommended_licenses=_PERMISSIVE_RECOMMENDED,
            forbidden_licenses=forbidden,
            reason=(
                "Defense: restricted-use community licenses (Llama / "
                "Gemma / Qwen / Mistral community) carry "
                "acceptable-use clauses that conflict with defense "
                "applications. Pick a permissive license."
            ),
        )
    if normalised == "embedded":
        # Embedded forbids strong copyleft + NC.
        forbidden = _NC_LICENSES + _STRONG_COPYLEFT
        return LicenseRecommendation(
            target=normalised,
            recommended_licenses=_PERMISSIVE_RECOMMENDED,
            forbidden_licenses=forbidden,
            reason=(
                "Embedded: strong copyleft (GPL/AGPL) is incompatible "
                "with closed-source firmware distribution; "
                "non-commercial licenses are also forbidden."
            ),
        )
    # Unreachable thanks to validate; defensive default.
    raise ValueError(f"unhandled target {normalised!r}")


@dataclass(frozen=True)
class DownstreamRisk:
    """Outcome of a per-license risk check."""

    ok: bool
    severity: str  # "ok" / "warn" / "block"
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be bool")
        if not isinstance(self.severity, str):
            raise TypeError("severity must be str")
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {{ok, warn, block}}, got {self.severity!r}"
            )
        if not isinstance(self.reason, str):
            raise TypeError("reason must be str")


def _validate_license_id(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("license_id must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"license_id must be str, got {type(value).__name__}")
    if not value:
        raise ValueError("license_id must be non-empty")
    if "\x00" in value:
        raise ValueError("license_id must not contain null bytes")
    if len(value) > _MAX_LICENSE_LEN:
        raise ValueError(f"license_id too int (> {_MAX_LICENSE_LEN} chars)")
    return value.lower().strip()


_MAX_MAU = 10_000_000_000_000  # 10 trillion — more humans than ever existed


def _validate_mau(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("monthly_active_users must be int, not bool")
    if not isinstance(value, int):
        raise TypeError(
            f"monthly_active_users must be int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(
            f"monthly_active_users must be >= 0, got {value}"
        )
    if value > _MAX_MAU:
        raise ValueError(
            f"monthly_active_users too large (> {_MAX_MAU}), got {value}"
        )
    return value


def flag_downstream_risk(
    *,
    license_id: object,
    target: object,
    monthly_active_users: object,
) -> DownstreamRisk:
    """Per-license risk check for a target deployment.

    Decision matrix:
    - Unknown license -> ``warn`` (operator should review).
    - Non-commercial license on B2C/defense/embedded -> ``block``.
    - Llama-family community license on B2C with MAU > 700M -> ``block``.
    - Llama-family on defense -> ``block``.
    - Strong copyleft on embedded -> ``block``.
    - Otherwise: ``ok``.
    """
    lic = _validate_license_id(license_id)
    tgt = validate_deploy_target(target)
    mau = _validate_mau(monthly_active_users)

    if lic not in KNOWN_LICENSES:
        return DownstreamRisk(
            ok=False,
            severity="warn",
            reason=(
                f"unknown license {lic!r}; "
                "verify with legal before shipping."
            ),
        )

    kind = LICENSE_KINDS.get(lic)
    if kind == "non-commercial":
        return DownstreamRisk(
            ok=False,
            severity="block",
            reason=(
                f"non-commercial license {lic!r} categorically forbidden "
                f"on target {tgt!r}."
            ),
        )

    if kind == "restricted-use":
        if tgt == "defense":
            return DownstreamRisk(
                ok=False,
                severity="block",
                reason=(
                    f"restricted-use license {lic!r} carries an "
                    "acceptable-use clause incompatible with defense."
                ),
            )
        # Llama community license: > 700M MAU = block per Meta AUP.
        # Tight allowlist — defends against a future non-community
        # "llama-permissive" id falsely tripping the MAU gate.
        if lic in _LLAMA_COMMUNITY_LICENSES and mau > _LLAMA_COMMUNITY_MAU_CAP:
            return DownstreamRisk(
                ok=False,
                severity="block",
                reason=(
                    f"Llama community license requires a separate licence from "
                    f"Meta for MAU > {_LLAMA_COMMUNITY_MAU_CAP:,} "
                    f"(your MAU: {mau:,})."
                ),
            )
        # Restricted-use on B2C with low MAU is a warn, not a block.
        return DownstreamRisk(
            ok=False,
            severity="warn",
            reason=(
                f"restricted-use license {lic!r}: review acceptable-use "
                "clauses before shipping. "
                "Per Meta AUP, MAU > 700M needs a separate licence."
            ),
        )

    if kind == "strong-copyleft" and tgt == "embedded":
        return DownstreamRisk(
            ok=False,
            severity="block",
            reason=(
                f"strong copyleft {lic!r} is incompatible with "
                "closed-source embedded firmware."
            ),
        )

    return DownstreamRisk(
        ok=True,
        severity="ok",
        reason=f"license {lic!r} ({kind}) is OK for target {tgt!r}.",
    )


__all__ = [
    "DEPLOY_TARGETS",
    "DownstreamRisk",
    "LicenseRecommendation",
    "advise_license_for_target",
    "flag_downstream_risk",
    "validate_deploy_target",
]
