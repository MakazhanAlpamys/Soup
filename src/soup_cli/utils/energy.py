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
import logging
import math
import re
from dataclasses import dataclass
from types import TracebackType
from typing import Optional
from urllib.parse import urlsplit

_LOG = logging.getLogger(__name__)

_MAX_ENDPOINT_LEN = 2048
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
_SCHEMES = frozenset({"http", "https"})
_COUNTRY_RE = re.compile(r"^[A-Za-z]{3}$")
_DEFAULT_COUNTRY = "USA"


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
        raise ValueError(f"endpoint too int (> {_MAX_ENDPOINT_LEN})")
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

    # ``measure_run_energy`` is the duration-only fallback: with only a wall-
    # clock number it cannot read instantaneous power draw, so it returns None
    # even when codecarbon IS installed. Use ``EnergyTracker`` (v0.71.3 #180)
    # for a real start()/stop() measurement around the training window.
    return None


class EnergyTracker:
    """Context manager that measures a training window's energy + CO2.

    Wraps codecarbon's ``OfflineEmissionsTracker`` (offline = no network /
    no IP-geolocation call, so the privacy guarantee is preserved). Lazy-
    imports codecarbon; when it is absent the tracker is a graceful no-op and
    ``measurement`` stays ``None``.

    The energy reading (kWh) is country-independent — it is measured power ×
    time. Only the CO2 conversion uses the chosen country's grid intensity
    (``country_iso_code``, default ``"USA"``). Both energy and CO2 are scaled
    by ``pue`` (Power Usage Effectiveness) so data-centre overhead is counted.

    Usage::

        with EnergyTracker(pue=1.1) as tracker:
            trainer.train()
        m = tracker.measurement  # EnergyMeasurement or None
    """

    def __init__(
        self,
        *,
        pue: float = 1.1,
        grid_intensity_g_per_kwh: float = 400.0,
        country_iso_code: str = _DEFAULT_COUNTRY,
    ) -> None:
        # Validate PUE up front (reuses the shared bounds checker).
        adjust_for_pue(1.0, pue)
        if isinstance(grid_intensity_g_per_kwh, bool) or not isinstance(
            grid_intensity_g_per_kwh, (int, float)
        ):
            raise ValueError("grid_intensity_g_per_kwh must be numeric")
        if (
            not math.isfinite(float(grid_intensity_g_per_kwh))
            or grid_intensity_g_per_kwh < 0
        ):
            raise ValueError("grid_intensity_g_per_kwh must be finite and >= 0")
        if not isinstance(country_iso_code, str) or not _COUNTRY_RE.match(
            country_iso_code
        ):
            raise ValueError(
                "country_iso_code must be a 3-letter ISO 3166-1 alpha-3 code "
                f"(got {country_iso_code!r})"
            )
        self._pue = float(pue)
        self._grid_default = float(grid_intensity_g_per_kwh)
        self._country = country_iso_code.upper()
        self._tracker = None
        self._measurement: Optional[EnergyMeasurement] = None

    @property
    def measurement(self) -> Optional[EnergyMeasurement]:
        return self._measurement

    def __enter__(self) -> "EnergyTracker":
        try:
            from codecarbon import OfflineEmissionsTracker  # noqa: PLC0415

            self._tracker = OfflineEmissionsTracker(
                country_iso_code=self._country,
                save_to_file=False,
                log_level="error",
            )
            self._tracker.start()
        except Exception as exc:  # noqa: BLE001 — never crash training
            _LOG.debug("EnergyTracker: codecarbon unavailable/failed: %s", exc)
            self._tracker = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        tracker = self._tracker
        self._tracker = None
        if tracker is None:
            return False
        try:
            emissions = tracker.stop()
            data = getattr(tracker, "final_emissions_data", None)
            energy_raw = float(getattr(data, "energy_consumed", 0.0) or 0.0)
            co2_raw = float(
                emissions
                if emissions is not None
                else getattr(data, "emissions", 0.0) or 0.0
            )
            if not math.isfinite(energy_raw) or energy_raw < 0:
                energy_raw = 0.0
            if not math.isfinite(co2_raw) or co2_raw < 0:
                co2_raw = 0.0
            grid = (
                (co2_raw / energy_raw * 1000.0)
                if energy_raw > 0
                else self._grid_default
            )
            if not math.isfinite(grid) or grid < 0:
                grid = self._grid_default
            self._measurement = EnergyMeasurement(
                energy_kwh=adjust_for_pue(energy_raw, self._pue),
                co2_kg=co2_raw * self._pue,
                pue=self._pue,
                grid_intensity_g_per_kwh=grid,
                source="codecarbon-offline",
            )
        except Exception as exc:  # noqa: BLE001 — never crash training
            _LOG.debug("EnergyTracker: stop()/measurement failed: %s", exc)
            self._measurement = None
        # Always return False so a body exception propagates unmasked.
        return False
