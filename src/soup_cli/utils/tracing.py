"""OpenTelemetry request tracing for soup serve (v0.30.0).

All imports are lazy; tracing is a no-op when the SDK is missing.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_otel_available() -> bool:
    try:
        import opentelemetry  # noqa: F401

        return True
    except ImportError:
        return False


def _is_private_ip(host: str) -> bool:
    """True if host resolves to a private, link-local, or unspecified IP."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        addr.is_private
        or addr.is_link_local
        or addr.is_unspecified
        or addr.is_reserved
        or addr.is_multicast
    )


def validate_otlp_endpoint(endpoint: str) -> str:
    """SSRF-hardened OTLP endpoint validation.

    Rules (mirrors utils/hf.py):
    - scheme must be http/https
    - plain http only permitted for loopback hosts
    - null byte rejected
    - RFC1918 / link-local / cloud-metadata IPs rejected (defence-in-depth
      so an operator can't point the exporter at 169.254.169.254 etc.)
    """
    if not isinstance(endpoint, str):
        raise ValueError("OTLP endpoint must be a string")
    if "\x00" in endpoint:
        raise ValueError("OTLP endpoint contains null byte")

    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"OTLP endpoint scheme must be http/https, got {parsed.scheme!r}"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("OTLP endpoint missing host")
    # Explicit unspecified host 0.0.0.0 always rejected (valid for bind, not connect)
    if host == "0.0.0.0":
        raise ValueError("OTLP endpoint host 0.0.0.0 is not a valid destination")
    if parsed.scheme == "http" and host not in _LOOPBACK_HOSTS:
        raise ValueError(
            f"plain HTTP is only allowed for loopback hosts; use HTTPS for {host}"
        )
    if host not in _LOOPBACK_HOSTS and _is_private_ip(host):
        raise ValueError(
            f"OTLP endpoint host {host} is a private / link-local IP; "
            "use a public endpoint or a loopback host"
        )
    return endpoint


def build_tracer(
    enabled: bool,
    endpoint: Optional[str] = None,
    service_name: str = "soup-serve",
) -> Optional[Any]:
    """Build an OpenTelemetry tracer provider.

    Returns None when disabled, when the SDK is missing, or when initialisation
    fails — callers should fall through without crashing. The server still
    accepts requests; it just doesn't emit spans.
    """
    if not enabled:
        return None
    if not is_otel_available():
        logger.warning(
            "--trace was requested but opentelemetry-sdk is not installed; "
            "tracing disabled. Install with: pip install opentelemetry-sdk "
            "opentelemetry-exporter-otlp"
        )
        return None
    if endpoint is not None:
        validate_otlp_endpoint(endpoint)

    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
            BatchSpanProcessor,
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # Exporter is optional — user may rely on their own instrumentation.
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
                    OTLPSpanExporter,
                )

                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
                )
            except ImportError:
                logger.warning(
                    "opentelemetry-exporter-otlp not installed; spans "
                    "will not be exported to %s",
                    endpoint,
                )

        # Idempotent: only install our provider if no user-supplied one is
        # already set. Otherwise we'd silently drop the operator's
        # instrumentation (and any unflushed spans).
        existing = trace.get_tracer_provider()
        existing_type = type(existing).__name__
        if existing_type in ("ProxyTracerProvider", "NoOpTracerProvider"):
            trace.set_tracer_provider(provider)
        else:
            logger.info(
                "TracerProvider already set (%s); reusing without override.",
                existing_type,
            )
        return trace.get_tracer("soup.serve")
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to initialise OpenTelemetry tracer")
        return None
