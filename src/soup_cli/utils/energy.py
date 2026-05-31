"""CodeCarbon + electricityMap energy/CO2 capture (v0.59.0 Part F).

Lazy-imports ``codecarbon`` so the module loads cleanly without it. When
codecarbon is absent the public API returns ``None`` from ``measure_run_energy``
so callers can fall back gracefully.

The electricityMap endpoint is SSRF-validated with full parity to v0.51.0
``utils/hubs.validate_hub_endpoint``: scheme allowlist + loopback-only HTTP
+ RFC1918 / link-local / cloud-metadata rejection + null-byte / control
char / oversize rejection.
"""

from __future__ import annotations

import ipaddress
import math
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

_MAX_ENDPOINT_LEN = 2048
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_SCHEMES = frozenset({"http", "https"})


@dataclass(frozen=True)
class EnergyMeasurement:
    """One per-run energy + CO2 reading."""

    energy_kwh: float
    co2_kg: float
    pue: float
    grid_intensity_g_per_kwh: float
    source: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.energy_kwh, "energy_kwh"),
            (self.co2_kg, "co2_kg"),
            (self.pue, "pue"),
            (self.grid_intensity_g_per_kwh, "grid_intensity_g_per_kwh"),
        ):
            if isinstance(value, bool):
                raise ValueError(f"{name} must not be bool")
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            f = float(value)
            if not math.isfinite(f):
                raise ValueError(f"{name} must be finite")
            if f < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.pue < 1.0:
            raise ValueError("pue must be >= 1.0")
        if not isinstance(self.source, str) or "\x00" in self.source:
            raise ValueError("source must be a null-byte-free str")


def validate_electricity_map_endpoint(endpoint: str) -> str:
    """SSRF-harden the electricityMap query endpoint.

    Mirrors v0.51.0 ``validate_hub_endpoint``: scheme allowlist (http/https),
    loopback-only HTTP, private-IP rejection, no control chars / null bytes.
    """
    if not isinstance(endpoint, str):
        raise ValueError("endpoint must be str")
    if not endpoint:
        raise ValueError("endpoint must be non-empty")
    if "\x00" in endpoint:
        raise ValueError("endpoint must not contain null bytes")
    if len(endpoint) > _MAX_ENDPOINT_LEN:
        raise ValueError(f"endpoint too long (> {_MAX_ENDPOINT_LEN})")
    if _CTRL_RE.search(endpoint):
        raise ValueError("endpoint must not contain control chars")
    try:
        parts = urlsplit(endpoint)
    except ValueError as exc:
        raise ValueError(f"endpoint unparseable: {exc}") from exc
    scheme = parts.scheme.lower()
    if scheme not in _SCHEMES:
        raise ValueError(
            f"endpoint scheme must be http or https, got {scheme!r}"
        )
    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError("endpoint must have a host")
    if host == "0.0.0.0":
        raise ValueError("0.0.0.0 endpoints are rejected")
    is_loopback = host in _LOOPBACK
    if scheme == "http" and not is_loopback:
        # Reject plain HTTP except for loopback.
        raise ValueError(
            "http:// only permitted for loopback hosts; use https:// for remote"
        )
    # Reject private / link-local / cloud-metadata IPs explicitly.
    # ``parts.hostname`` already strips IPv6 brackets, so feed it directly.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and not is_loopback:
        if ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(
                f"endpoint host {host!r} resolves to a private/link-local IP"
            )
    return endpoint


def adjust_for_pue(energy_kwh: float, pue: float) -> float:
    """Multiply raw energy by PUE (Power Usage Effectiveness).

    PUE must be >= 1.0 (a data centre that does no overhead-cooling at all has
    PUE == 1.0; typical hyperscale is 1.1–1.5).
    """
    if isinstance(energy_kwh, bool) or isinstance(pue, bool):
        raise ValueError("inputs must not be bool")
    if not isinstance(energy_kwh, (int, float)) or not isinstance(pue, (int, float)):
        raise ValueError("inputs must be numeric")
    if not math.isfinite(float(energy_kwh)) or not math.isfinite(float(pue)):
        raise ValueError("inputs must be finite")
    if energy_kwh < 0:
        raise ValueError("energy_kwh must be >= 0")
    if pue < 1.0:
        raise ValueError("pue must be >= 1.0")
    return float(energy_kwh) * float(pue)


def measure_run_energy(
    duration_seconds: float = 0.0,
    *,
    grid_intensity_g_per_kwh: float = 400.0,
    pue: float = 1.1,
) -> Optional[EnergyMeasurement]:
    """Best-effort capture of a single run's energy + CO2.

    Returns ``None`` when ``codecarbon`` is not installed AND ``duration_seconds``
    is ``<= 0`` (degenerate inputs surface as None rather than a fake zero).

    Live CodeCarbon hook wiring into trainer wrappers lands in v0.59.1.
    """
    if isinstance(duration_seconds, bool) or isinstance(grid_intensity_g_per_kwh, bool):
        raise ValueError("numeric inputs must not be bool")
    if not isinstance(duration_seconds, (int, float)):
        raise ValueError("duration_seconds must be numeric")
    if not math.isfinite(float(duration_seconds)) or duration_seconds < 0:
        raise ValueError("duration_seconds must be a finite non-negative number")
    if not isinstance(grid_intensity_g_per_kwh, (int, float)):
        raise ValueError("grid_intensity_g_per_kwh must be numeric")
    if (
        not math.isfinite(float(grid_intensity_g_per_kwh))
        or grid_intensity_g_per_kwh < 0
    ):
        raise ValueError("grid_intensity_g_per_kwh must be a finite non-negative number")
    try:
        adjust_for_pue(1.0, pue)
    except ValueError as exc:
        raise ValueError(f"invalid pue: {exc}") from exc

    try:
        import codecarbon  # noqa: F401, PLC0415
    except ImportError:
        # No live codecarbon — return None so the caller (typically the BOM
        # builder) can decide whether to omit the energy properties.
        return None

    # Live wiring deferred to v0.59.1; return None until the codecarbon
    # `EmissionsTracker` hook lands. The schema + endpoint validator + PUE
    # math are live; the actual measurement is the v0.59.1 deliverable.
    return None
