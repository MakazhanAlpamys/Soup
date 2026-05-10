"""v0.44.0 Part A — llama-server timings + KV-cache fill % parser.

Pure-Python: takes the JSON dict that llama-server returns under the `timings`
field on `/v1/chat/completions`. Returns a frozen summary that the dashboard
can render. Never raises on malformed input — returns None values instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LlamaServerTimings:
    """Subset of llama-server `timings` block we surface in the dashboard."""

    prompt_tokens: Optional[int]
    prompt_ms: Optional[float]
    prompt_per_token_ms: Optional[float]
    predicted_tokens: Optional[int]
    predicted_ms: Optional[float]
    predicted_per_token_ms: Optional[float]
    kv_cache_used: Optional[int]
    kv_cache_size: Optional[int]
    kv_cache_pct: Optional[float]


def _coerce_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0:
        return int(value)
    return None


def _coerce_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isfinite(f) and f >= 0:
            return f
    return None


def _kv_pct(used: Optional[int], size: Optional[int]) -> Optional[float]:
    if used is None or size is None or size <= 0:
        return None
    pct = 100.0 * float(used) / float(size)
    if pct < 0.0:
        return 0.0
    if pct > 100.0:
        return 100.0
    return pct


def parse_timings(payload: Dict[str, Any]) -> LlamaServerTimings:
    """Extract a `LlamaServerTimings` from a llama-server response dict.

    Tolerates missing keys / wrong types — every field defaults to None.
    """
    if not isinstance(payload, dict):
        raise TypeError("payload must be dict")
    timings = payload.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}
    used = _coerce_int(payload.get("kv_cache_used"))
    size = _coerce_int(payload.get("kv_cache_size"))
    return LlamaServerTimings(
        prompt_tokens=_coerce_int(timings.get("prompt_n")),
        prompt_ms=_coerce_float(timings.get("prompt_ms")),
        prompt_per_token_ms=_coerce_float(timings.get("prompt_per_token_ms")),
        predicted_tokens=_coerce_int(timings.get("predicted_n")),
        predicted_ms=_coerce_float(timings.get("predicted_ms")),
        predicted_per_token_ms=_coerce_float(
            timings.get("predicted_per_token_ms")
        ),
        kv_cache_used=used,
        kv_cache_size=size,
        kv_cache_pct=_kv_pct(used, size),
    )


def format_kv_bar(pct: Optional[float], *, width: int = 20) -> str:
    """Render a single-line KV-cache fill bar, e.g. `[████░░░░░] 42%`."""
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be int")
    if width <= 0 or width > 200:
        raise ValueError("width must be in (0, 200]")
    if pct is None:
        return f"[{' ' * width}]   --%"
    if isinstance(pct, bool) or not isinstance(pct, (int, float)):
        raise TypeError("pct must be a number or None")
    pct_f = max(0.0, min(100.0, float(pct)))
    filled = int(round(width * pct_f / 100.0))
    return f"[{'█' * filled}{' ' * (width - filled)}] {pct_f:5.1f}%"
