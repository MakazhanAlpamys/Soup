"""v0.45.0 Part B — Server-side tools registry (schema-only).

Exposes a closed allowlist of safe tools that the inference server can offer
to a model: ``python``, ``bash``, ``web_search``. Live wiring (HTTP tool
endpoints) is deferred to v0.45.1; this module locks the public schema +
domain allowlist so the configuration surface is stable now.

Defence-in-depth choices:

- ``python`` and ``bash`` reuse the v0.25.0 RLVR sandbox (5 s timeout,
  RLIMIT_AS, ephemeral cwd, socket patch) — schema only here.
- ``web_search`` requires every domain on a closed allowlist; subdomain
  matches require a leading dot (``foo.example.com`` matches
  ``example.com`` only when the allow entry is ``.example.com``).
- ``rate_limit`` per tool uses the v0.20.0 [1, 600] requests-per-minute
  window so a misbehaving plugin cannot DoS the server.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

# Tool name allowlist — closed set, used as map keys.
_TOOL_NAMES = ("python", "bash", "web_search")
SUPPORTED_TOOLS: frozenset = frozenset(_TOOL_NAMES)

# Default web_search domain allowlist. Operators must override / extend
# explicitly via ``WebSearchConfig.domain_allowlist``; the empty default
# is intentionally a hard "deny all".
_DEFAULT_WEB_DOMAINS: Tuple[str, ...] = ()

_DOMAIN_RE = re.compile(r"^\.?[a-z0-9]([a-z0-9.\-]{0,253})$")
_MAX_DOMAINS = 64
_MAX_DOMAIN_LEN = 253
_MIN_RPM = 1
_MAX_RPM = 600


@dataclass(frozen=True)
class WebSearchConfig:
    """Web-search tool configuration."""

    domain_allowlist: Tuple[str, ...] = _DEFAULT_WEB_DOMAINS
    rate_limit_per_minute: int = 30


@dataclass(frozen=True)
class ToolSpec:
    """One server-side tool registration."""

    name: str
    description: str
    rate_limit_per_minute: int
    web_search: Optional[WebSearchConfig] = None


def validate_tool_name(name: str) -> str:
    """Validate ``name`` is one of the allowlisted tools."""
    if not isinstance(name, str):
        raise TypeError("tool name must be a string")
    canonical = name.strip().lower()
    if not canonical:
        raise ValueError("tool name must be non-empty")
    if "\x00" in canonical:
        raise ValueError("tool name must not contain null bytes")
    if canonical not in SUPPORTED_TOOLS:
        raise ValueError(
            f"unknown tool: {canonical!r}. supported: {sorted(SUPPORTED_TOOLS)}"
        )
    return canonical


def validate_rate_limit(rpm: int) -> int:
    if isinstance(rpm, bool) or not isinstance(rpm, int):
        raise TypeError("rate_limit_per_minute must be an int")
    if rpm < _MIN_RPM or rpm > _MAX_RPM:
        raise ValueError(
            f"rate_limit_per_minute must be in [{_MIN_RPM}, {_MAX_RPM}]"
        )
    return rpm


def validate_domain(domain: str) -> str:
    if not isinstance(domain, str):
        raise TypeError("domain must be a string")
    candidate = domain.strip().lower()
    if not candidate:
        raise ValueError("domain must be non-empty")
    if "\x00" in candidate:
        raise ValueError("domain must not contain null bytes")
    if len(candidate) > _MAX_DOMAIN_LEN:
        raise ValueError(f"domain exceeds {_MAX_DOMAIN_LEN} chars")
    if "/" in candidate or " " in candidate:
        raise ValueError("domain must not contain '/' or whitespace")
    if not _DOMAIN_RE.match(candidate):
        raise ValueError(f"invalid domain syntax: {candidate!r}")
    return candidate


def validate_web_search_config(config: WebSearchConfig) -> WebSearchConfig:
    if not isinstance(config, WebSearchConfig):
        raise TypeError("config must be a WebSearchConfig")
    if len(config.domain_allowlist) > _MAX_DOMAINS:
        raise ValueError(f"domain_allowlist exceeds {_MAX_DOMAINS} entries")
    seen = set()
    for raw in config.domain_allowlist:
        canonical = validate_domain(raw)
        if canonical in seen:
            raise ValueError(f"duplicate domain: {canonical!r}")
        seen.add(canonical)
    validate_rate_limit(config.rate_limit_per_minute)
    return config


def is_domain_allowed(host: str, allowlist: Tuple[str, ...]) -> bool:
    """Return True iff ``host`` is permitted by ``allowlist``.

    Bare entries (``example.com``) match the host exactly. Entries with a
    leading dot (``.example.com``) also match any subdomain
    (``a.example.com``). Both forms are validated by ``validate_domain``.
    """
    if not isinstance(host, str) or not host:
        return False
    canonical = host.strip().lower()
    if not canonical or "\x00" in canonical:
        return False
    # Strip optional ``:port`` suffix and bracketed IPv6 noise so an
    # ``Authority``-form value like ``api.example.com:443`` matches the
    # bare allowlist entry ``api.example.com`` instead of silently
    # missing.
    if canonical.startswith("[") and "]" in canonical:
        # IPv6 literal — never matches a domain allowlist; deny.
        return False
    if ":" in canonical:
        canonical = canonical.split(":", 1)[0]
    if not canonical:
        return False
    for raw in allowlist:
        rule = raw.strip().lower()
        if not rule:
            continue
        if rule.startswith("."):
            base = rule[1:]
            if canonical == base or canonical.endswith(rule):
                return True
        elif canonical == rule:
            return True
    return False


_TOOL_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "python": "Sandboxed Python execution (RLVR sandbox)",
        "bash": "Sandboxed bash execution (RLVR sandbox)",
        "web_search": "Web search constrained to a domain allowlist",
    }
)


def tool_description(name: str) -> str:
    """Return the canonical description for a supported tool."""
    canonical = validate_tool_name(name)
    return _TOOL_DESCRIPTIONS[canonical]


__all__ = [
    "SUPPORTED_TOOLS",
    "ToolSpec",
    "WebSearchConfig",
    "is_domain_allowed",
    "tool_description",
    "validate_domain",
    "validate_rate_limit",
    "validate_tool_name",
    "validate_web_search_config",
]
