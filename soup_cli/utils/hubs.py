"""v0.51.0 Part E — Alternative model hubs (ModelScope / Modelers).

Schema-only support for selecting a non-HF model hub for downloads + pushes.
Each hub gets a closed allowlist + SSRF-hardened endpoint validator that
mirrors the v0.29.0 ``HF_ENDPOINT`` policy in ``utils/hf.py``:

* scheme allowlist (http/https only)
* null-byte rejection
* ``0.0.0.0`` explicitly rejected
* plain HTTP only permitted for loopback hosts
  (``localhost`` / ``127.0.0.1`` / ``::1``)
* private / link-local / cloud-metadata IPs (RFC1918, 169.254.x) rejected
  for plain HTTP.

Live download / upload wiring is deferred to v0.51.1 — this module ships the
schema lock-in (``hub`` Literal on ``TrainingConfig``) plus the validators
that the live wiring will call.
"""
from __future__ import annotations

import ipaddress
import os
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlparse

# Closed allowlist of supported hubs. Wrapped in MappingProxyType so callers
# cannot mutate the registry at runtime (matches v0.36.0 _REGISTRY policy).
SUPPORTED_HUBS: frozenset[str] = frozenset({"hf", "modelscope", "modelers"})

_HUB_DEFAULT_ENDPOINTS: Mapping[str, str] = MappingProxyType({
    "hf": "https://huggingface.co",
    "modelscope": "https://modelscope.cn",
    "modelers": "https://modelers.cn",
})

# Per-hub env var that overrides the default endpoint (mirrors HF_ENDPOINT).
_HUB_ENDPOINT_ENV: Mapping[str, str] = MappingProxyType({
    "hf": "HF_ENDPOINT",
    "modelscope": "MODELSCOPE_ENDPOINT",
    "modelers": "MODELERS_ENDPOINT",
})

# Per-hub pip-install hint, surfaced when the live downloader complains.
_HUB_PACKAGE: Mapping[str, str] = MappingProxyType({
    "hf": "huggingface-hub",
    "modelscope": "modelscope",
    "modelers": "openmind-hub",
})

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

_MAX_HUB_NAME_LEN: int = 32


def validate_hub_name(name: str) -> str:
    """Validate ``name`` against ``SUPPORTED_HUBS`` and return canonical form.

    Mirrors v0.41.0 ``validate_optimizer_name`` policy:
    rejects non-string / bool / empty / null-byte / oversize / unknown, and
    lower-cases the input for deterministic lookup.
    """
    if isinstance(name, bool):
        raise TypeError(f"hub name must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(f"hub name must be str, got {type(name).__name__}")
    if not name:
        raise ValueError("hub name must be non-empty")
    if "\x00" in name:
        raise ValueError("hub name must not contain null bytes")
    if len(name) > _MAX_HUB_NAME_LEN:
        raise ValueError(
            f"hub name too long (max {_MAX_HUB_NAME_LEN} chars)"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_HUBS:
        supported = ", ".join(sorted(SUPPORTED_HUBS))
        raise ValueError(
            f"hub {name!r} not supported. Supported: {supported}"
        )
    return canonical


def required_hub_package(hub: str) -> str | None:
    """Return the pip-installable package name for ``hub``, or ``None``.

    Non-string / unknown returns ``None`` (no advisory available).
    """
    if not isinstance(hub, str):
        return None
    return _HUB_PACKAGE.get(hub.lower())


def default_endpoint(hub: str) -> str:
    """Return the canonical default endpoint URL for ``hub``.

    Raises ``ValueError`` if ``hub`` is not in :data:`SUPPORTED_HUBS`.
    """
    canonical = validate_hub_name(hub)
    return _HUB_DEFAULT_ENDPOINTS[canonical]


def endpoint_env_var(hub: str) -> str:
    """Return the env-var name that overrides the default endpoint."""
    canonical = validate_hub_name(hub)
    return _HUB_ENDPOINT_ENV[canonical]


def _is_private_or_link_local(host: str) -> bool:
    """Whether ``host`` is a private / link-local / loopback IP.

    DNS resolution intentionally not performed (matches v0.29.0 hf.py policy).
    """
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_private or addr.is_link_local or addr.is_loopback


def validate_hub_endpoint(endpoint: str, *, hub: str | None = None) -> str:
    """SSRF-hardened endpoint validator. Returns the stripped endpoint.

    Mirrors ``utils/hf.resolve_endpoint`` exactly so all three hubs share the
    same security posture. Raises ``ValueError`` on bad input.

    Args:
        endpoint: candidate URL.
        hub: optional hub name, threaded into the error messages.
    """
    label = (hub or "hub_endpoint").strip() or "hub_endpoint"

    if isinstance(endpoint, bool):
        raise TypeError(f"{label} must not be bool, got {endpoint!r}")
    if not isinstance(endpoint, str):
        raise TypeError(
            f"{label} must be str, got {type(endpoint).__name__}"
        )
    if not endpoint:
        raise ValueError(f"{label} must be a non-empty string")
    if "\x00" in endpoint:
        raise ValueError(f"{label} must not contain null bytes")
    # Defence-in-depth: control characters (CR/LF/etc.) inside a URL would
    # be a CRLF-injection hazard if the URL ever flowed into a raw HTTP
    # client. Reject them here even though urlparse silently strips them.
    if any(ord(c) < 0x20 for c in endpoint):
        raise ValueError(f"{label} must not contain control characters")

    stripped = endpoint.rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"{label} must use http/https scheme, got: {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise ValueError(f"{label} is missing a host")

    host = parsed.hostname or ""
    if host == "0.0.0.0":
        raise ValueError(
            f"{label} 0.0.0.0 is ambiguous; use 127.0.0.1 or localhost"
        )
    if parsed.scheme == "http" and host not in _LOOPBACK_HOSTS:
        if _is_private_or_link_local(host):
            raise ValueError(
                f"{label} plain HTTP is only allowed for loopback "
                f"(localhost / 127.0.0.1 / ::1); private/link-local hosts "
                f"require HTTPS"
            )
        raise ValueError(
            f"{label} for remote hosts must use HTTPS "
            f"(localhost HTTP allowed)"
        )
    return stripped


def resolve_endpoint(hub: str, *, env: Mapping[str, str] | None = None) -> str:
    """Return the active endpoint for ``hub`` after env-override + validation.

    Looks up the per-hub env var (e.g. ``MODELSCOPE_ENDPOINT``) — if set, runs
    it through :func:`validate_hub_endpoint` and returns the stripped URL.
    Otherwise returns :func:`default_endpoint`.

    The default endpoints are baked-in HTTPS URLs so they don't need
    re-validation on every call.
    """
    canonical = validate_hub_name(hub)
    source = env if env is not None else os.environ
    raw = source.get(_HUB_ENDPOINT_ENV[canonical])
    if not raw:
        return _HUB_DEFAULT_ENDPOINTS[canonical]
    return validate_hub_endpoint(raw, hub=canonical)


def is_hf(hub: str) -> bool:
    """Convenience: True iff ``hub`` canonicalises to ``'hf'``.

    Rejects ``bool`` explicitly (matches v0.30.0 ``Candidate`` /
    v0.34.0 ``estimate_run_cost_usd`` policy) so a stray ``True`` cannot
    silently pretend to be a hub name.
    """
    if isinstance(hub, bool):
        return False
    if not isinstance(hub, str):
        return False
    return hub.lower() == "hf"
