"""Shared Slack / Discord webhook helpers (v0.71.5 #207).

Lifts the SSRF-hardened ``validate_webhook_url`` + best-effort ``post_webhook``
out of ``utils/drift_alarm.py`` (v0.63.0 Part E) so every production-trace
command can offer ``--slack-url`` / ``--discord-url`` without re-implementing
the SSRF gate. ``drift_alarm`` now re-exports these for back-compat.

SSRF policy — full parity with v0.29.0 ``HF_ENDPOINT`` / v0.30.0 OTLP /
v0.51.0 ``validate_hub_endpoint`` / v0.63.0 drift-alarm:
- scheme allowlist {http, https}
- null-byte / control-char rejection
- ``0.0.0.0`` rejected
- plain HTTP only permitted for loopback hosts
- private / link-local / reserved / multicast IPs rejected

``post_webhook`` NEVER raises — webhook delivery must not crash the command
that triggered it. ``httpx`` is lazy-imported so the runtime cost is paid only
when an alarm actually fires.
"""

from __future__ import annotations

import ipaddress
from typing import List, Mapping, Optional, Tuple
from urllib.parse import urlparse

_MAX_WEBHOOK_URL_LEN = 4096
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_private_or_link_local(host: str) -> bool:
    """Return True iff ``host`` resolves to a non-loopback private/reserved IP.

    Explicit parentheses on the final clause (mirrors v0.63.0 drift-alarm
    code-review MEDIUM fix): Python binds ``and`` tighter than ``or``, but
    the SSRF gate is safety-critical and a future edit should not need to
    re-derive the precedence rules to verify the logic.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_link_local
        or (ip.is_loopback is False and (ip.is_reserved or ip.is_multicast))
    )


def validate_webhook_url(url: object) -> str:
    """SSRF-hardened webhook URL validator (returns the canonical URL)."""
    if isinstance(url, bool):
        raise TypeError("webhook URL must be str, not bool")
    if not isinstance(url, str):
        raise TypeError(f"webhook URL must be str, got {type(url).__name__}")
    if not url:
        raise ValueError("webhook URL must be non-empty")
    if "\x00" in url:
        raise ValueError("webhook URL must not contain null bytes")
    if any(ord(c) < 0x20 for c in url):
        raise ValueError("webhook URL must not contain control characters")
    if len(url) > _MAX_WEBHOOK_URL_LEN:
        raise ValueError(f"webhook URL must be <= {_MAX_WEBHOOK_URL_LEN} chars")
    stripped = url.rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"webhook URL must use http/https scheme, got {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise ValueError("webhook URL is missing a host")
    host = parsed.hostname or ""
    if host == "0.0.0.0":
        raise ValueError(
            "webhook URL 0.0.0.0 is ambiguous; use 127.0.0.1 or localhost"
        )
    if parsed.scheme == "http" and host not in _LOOPBACK_HOSTS:
        if _is_private_or_link_local(host):
            raise ValueError(
                "webhook URL plain HTTP is only allowed for loopback; "
                "private/link-local hosts require HTTPS"
            )
        raise ValueError("webhook URL for remote hosts must use HTTPS")
    return stripped


def post_webhook(
    *,
    url: Optional[str],
    payload: Mapping[str, object],
    timeout_seconds: float = 5.0,
) -> bool:
    """POST ``payload`` as JSON to ``url``. Returns True on 2xx, False otherwise.

    Never raises — webhook delivery must NOT crash the calling command.
    """
    if url is None:
        return False
    try:
        validated = validate_webhook_url(url)
    except (TypeError, ValueError):
        return False
    try:
        import httpx  # type: ignore[import-untyped]
    except ImportError:
        return False
    try:
        response = httpx.post(
            validated,
            json=dict(payload),
            timeout=timeout_seconds,
        )
        return 200 <= response.status_code < 300
    except Exception:  # noqa: BLE001 — webhook must never crash the command
        return False


def send_webhooks(
    payload: Mapping[str, object],
    *,
    slack_url: Optional[str] = None,
    discord_url: Optional[str] = None,
    timeout_seconds: float = 5.0,
) -> List[Tuple[str, bool]]:
    """POST ``payload`` to each provided webhook; return per-target delivery.

    Returns a list of ``(label, delivered)`` for every non-``None`` URL,
    in ``slack`` then ``discord`` order. ``None`` URLs are skipped (not
    attempted). Never raises (delegates to the never-raising
    :func:`post_webhook`).
    """
    results: List[Tuple[str, bool]] = []
    for label, url in (("slack", slack_url), ("discord", discord_url)):
        if url:
            results.append(
                (
                    label,
                    post_webhook(
                        url=url,
                        payload=payload,
                        timeout_seconds=timeout_seconds,
                    ),
                )
            )
    return results


__all__ = [
    "post_webhook",
    "send_webhooks",
    "validate_webhook_url",
]
