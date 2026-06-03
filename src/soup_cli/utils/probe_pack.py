"""v0.66.0 Part E — Activation probe pack.

``soup probe pack <base>`` assembles a per-base manifest of calibrated
linear probes (sleeper, SAE, truth, harm…). The pack composes with:

- Part C (``soup_cli.utils.sleeper_probe``) — the live sleeper probe
- Part A (``soup_cli.utils.sae_diff``) — the SAE feature-diff surface
- Part D (``soup_cli.utils.interference``) — adapter pairwise eval

The pack manifest is metadata only — probe weights are derived on demand
by the relevant ``utils/<probe>.py`` module. This release intentionally
does not auto-download from HF Hub (operators bring their own weights);
``soup probe pack`` exposes the manifest so future v0.66.x can land a
real fetcher via ``utils/hubs.py``.

Public surface:

- ``PROBE_KINDS`` closed allowlist
- ``ProbeEntry`` / ``ProbePack`` frozen dataclasses
- ``BUNDLED_PACKS`` ``MappingProxyType``
- ``validate_pack_base(name)`` allowlist canonicaliser
- ``get_probe_pack(base)`` typed lookup
- ``list_probe_bases()`` sorted list
- ``render_pack_json`` / ``render_pack_markdown``
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Tuple

# Closed allowlist of probe kinds. Each maps to a utility module:
#  - sleeper -> sleeper_probe.py (v0.66.0 Part C)
#  - sae     -> sae_diff.py      (v0.66.0 Part A)
#  - truth   -> truth_probe.py   (v0.71.8 #217 — TruthfulQA-style honesty probe)
#  - harm    -> harm_probe.py    (v0.71.8 #217 — HarmBench-style misuse probe)
PROBE_KINDS: frozenset[str] = frozenset({"sleeper", "sae", "truth", "harm"})

_MAX_BASE_LEN = 200
_MAX_NAME_LEN = 256
_MAX_HIDDEN_DIM = 65_536
_MAX_PROBES_PER_PACK = 32
_MIN_PROBES_PER_PACK = 1
_MAX_DESCRIPTION_LEN = 4096  # M5 review fix — operator-controlled field cap


def _md_escape(value: object) -> str:
    """Escape Rich-markup metacharacters for safe markdown render (review M5)."""
    return str(value).replace("[", "\\[").replace("]", "\\]")


def _validate_name(value: object, field: str, max_len: int = _MAX_NAME_LEN) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} must be ≤{max_len} chars")
    return value


@dataclass(frozen=True)
class ProbeEntry:
    """One probe entry in a pack."""

    name: str
    kind: str
    hidden_dim: int
    description: str

    def __post_init__(self) -> None:
        _validate_name(self.name, "probe name")
        if self.kind not in PROBE_KINDS:
            raise ValueError(
                f"kind must be in {sorted(PROBE_KINDS)}, got {self.kind!r}"
            )
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
        ):
            raise TypeError("hidden_dim must be int")
        if self.hidden_dim <= 0 or self.hidden_dim > _MAX_HIDDEN_DIM:
            raise ValueError(
                f"hidden_dim must be in (0, {_MAX_HIDDEN_DIM}]"
            )
        if not isinstance(self.description, str):
            raise TypeError("description must be str")
        if "\x00" in self.description:
            raise ValueError("description must not contain null bytes")
        if len(self.description) > _MAX_DESCRIPTION_LEN:
            # M5 review fix: cap operator-controlled description so it
            # cannot blow up Rich render or downstream JSON sinks.
            raise ValueError(
                f"description must be ≤{_MAX_DESCRIPTION_LEN} chars"
            )


@dataclass(frozen=True)
class ProbePack:
    """A per-base bundle of probes."""

    base: str
    probes: Tuple[ProbeEntry, ...]
    soup_version: str

    def __post_init__(self) -> None:
        _validate_name(self.base, "base", max_len=_MAX_BASE_LEN)
        if not isinstance(self.probes, tuple):
            raise TypeError("probes must be tuple")
        if (
            len(self.probes) < _MIN_PROBES_PER_PACK
            or len(self.probes) > _MAX_PROBES_PER_PACK
        ):
            raise ValueError(
                f"pack must have at least {_MIN_PROBES_PER_PACK} probe "
                f"(max {_MAX_PROBES_PER_PACK})"
            )
        for entry in self.probes:
            if not isinstance(entry, ProbeEntry):
                raise TypeError("probes entries must be ProbeEntry")
        names = [e.name for e in self.probes]
        if len(set(names)) != len(names):
            raise ValueError("duplicate probe names in pack")
        if not isinstance(self.soup_version, str):
            raise TypeError("soup_version must be str")


# ---------------------------------------------------------------------------
# Bundled packs — one per base in v0.66.0 Part C sleeper catalogue.
# ---------------------------------------------------------------------------


def _make_bundled_packs() -> Mapping[str, ProbePack]:
    from soup_cli import __version__
    from soup_cli.utils.harm_probe import BUNDLED_HARM_PROBES
    from soup_cli.utils.sleeper_probe import BUNDLED_PROBES
    from soup_cli.utils.truth_probe import BUNDLED_TRUTH_PROBES

    catalogue: dict[str, ProbePack] = {}
    for base, sleeper_spec in BUNDLED_PROBES.items():
        # v0.71.8 #217 — each base now ships sleeper + truth + harm probes.
        entries: list[ProbeEntry] = [
            ProbeEntry(
                name=f"sleeper:{base}",
                kind="sleeper",
                hidden_dim=sleeper_spec.hidden_dim,
                description=(
                    f"Defection probe for {base} "
                    f"(threshold={sleeper_spec.threshold:.2f})"
                ),
            )
        ]
        truth_spec = BUNDLED_TRUTH_PROBES.get(base)
        if truth_spec is not None:
            entries.append(
                ProbeEntry(
                    name=f"truth:{base}",
                    kind="truth",
                    hidden_dim=truth_spec.hidden_dim,
                    description=f"Honesty probe for {base} (5% / 20% bands)",
                )
            )
        harm_spec = BUNDLED_HARM_PROBES.get(base)
        if harm_spec is not None:
            entries.append(
                ProbeEntry(
                    name=f"harm:{base}",
                    kind="harm",
                    hidden_dim=harm_spec.hidden_dim,
                    description=f"Misuse probe for {base} (5% / 20% bands)",
                )
            )
        catalogue[base] = ProbePack(
            base=base,
            probes=tuple(entries),
            soup_version=__version__,
        )
    return MappingProxyType(catalogue)


BUNDLED_PACKS: Mapping[str, ProbePack] = _make_bundled_packs()


def list_probe_bases() -> Tuple[str, ...]:
    """Sorted tuple of supported base model names."""
    return tuple(sorted(BUNDLED_PACKS))


def validate_pack_base(name: object) -> str:
    """Validate a base name; returns canonical entry from BUNDLED_PACKS."""
    if isinstance(name, bool):
        raise TypeError("base must be str, got bool")
    if not isinstance(name, str):
        raise TypeError(f"base must be str, got {type(name).__name__}")
    if not name:
        raise ValueError("base must be non-empty")
    if "\x00" in name:
        raise ValueError("base must not contain null bytes")
    if len(name) > _MAX_BASE_LEN:
        raise ValueError(f"base must be ≤{_MAX_BASE_LEN} chars")
    lower = name.lower()
    for known in BUNDLED_PACKS:
        if known.lower() == lower:
            return known
    raise ValueError(
        f"no probe pack for base {name!r} "
        f"(known: {sorted(BUNDLED_PACKS)})"
    )


def get_probe_pack(base: str) -> ProbePack:
    """Typed lookup with allowlist + case-insensitive normalisation."""
    canonical = validate_pack_base(base)
    return BUNDLED_PACKS[canonical]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_pack_json(pack: ProbePack) -> str:
    """Canonical JSON for CI / Registry artifact."""
    if not isinstance(pack, ProbePack):
        raise TypeError("pack must be ProbePack")
    payload = {
        "base": pack.base,
        "soup_version": pack.soup_version,
        "probes": [asdict(p) for p in pack.probes],
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def render_pack_markdown(pack: ProbePack) -> str:
    """Human-readable markdown for `soup probe pack <base>`.

    All operator-controlled string fields (``base`` / probe ``name`` /
    ``kind`` / ``description``) are routed through ``_md_escape`` so a
    crafted probe entry cannot inject Rich markup (review M5 fix).
    """
    if not isinstance(pack, ProbePack):
        raise TypeError("pack must be ProbePack")
    lines = [
        f"# Probe pack: {_md_escape(pack.base)}",
        "",
        f"- Soup version: **{_md_escape(pack.soup_version)}**",
        f"- Probes: **{len(pack.probes)}**",
        "",
        "| Name | Kind | Hidden dim | Description |",
        "| --- | --- | --- | --- |",
    ]
    for p in pack.probes:
        lines.append(
            f"| `{_md_escape(p.name)}` | {_md_escape(p.kind)} | "
            f"{p.hidden_dim} | {_md_escape(p.description)} |"
        )
    return "\n".join(lines) + "\n"
