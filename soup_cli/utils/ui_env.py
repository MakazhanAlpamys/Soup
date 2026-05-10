"""v0.44.0 Part C — Web UI environment-variable knobs.

`API_HOST` / `API_PORT` / `API_KEY` for the FastAPI server, plus
`GRADIO_HOST` / `GRADIO_PORT` for any Gradio sub-UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UiEnv:
    """Resolved Web UI environment overrides."""

    api_host: Optional[str]
    api_port: Optional[int]
    api_key: Optional[str]
    gradio_host: Optional[str]
    gradio_port: Optional[int]


_VALID_HOST_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-:"
)


def _parse_host(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError("host must be str")
    cleaned = raw.strip()
    if not cleaned:
        return None
    if "\x00" in cleaned or len(cleaned) > 253:
        raise ValueError("invalid host string")
    if any(char not in _VALID_HOST_CHARS for char in cleaned):
        raise ValueError(
            "host contains characters outside [a-zA-Z0-9.-:]"
        )
    return cleaned


def _parse_port(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError("port must be str")
    cleaned = raw.strip()
    if not cleaned:
        return None
    try:
        port = int(cleaned)
    except ValueError as exc:
        raise ValueError(f"port must be int; got {cleaned!r}") from exc
    if not (1 <= port <= 65535):
        raise ValueError("port must be in [1, 65535]")
    return port


def _parse_key(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError("key must be str")
    cleaned = raw.strip()
    if not cleaned:
        return None
    if "\x00" in cleaned or len(cleaned) > 256:
        raise ValueError("invalid key string")
    return cleaned


def resolve_ui_env(env: Optional[dict] = None) -> UiEnv:
    """Read the documented env knobs, applying validation. `env=None` reads
    from `os.environ`."""
    source = os.environ if env is None else env
    return UiEnv(
        api_host=_parse_host(source.get("API_HOST")),
        api_port=_parse_port(source.get("API_PORT")),
        api_key=_parse_key(source.get("API_KEY")),
        gradio_host=_parse_host(source.get("GRADIO_HOST")),
        gradio_port=_parse_port(source.get("GRADIO_PORT")),
    )
