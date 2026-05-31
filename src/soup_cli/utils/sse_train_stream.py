"""v0.44.0 Part A — SSE training-stream payload schema.

Pure-Python: serialises a TrainEvent as a single SSE-frame string that the
FastAPI endpoint can write into a streaming response. No FastAPI/dep import.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Closed allowlist — drop any payload key not in this set.
_ALLOWED_KEYS = frozenset(
    {
        "type",
        "ts",
        "step",
        "epoch",
        "loss",
        "lr",
        "grad_norm",
        "tokens_per_s",
        "ema_loss",
        "p95_step_ms",
        "p99_step_ms",
        "eta_seconds",
        "message",
    }
)

_VALID_TYPES = frozenset({"metric", "status", "log", "eval"})

_MAX_MESSAGE_LEN = 1024


@dataclass(frozen=True)
class TrainEvent:
    """One SSE-streamed training event."""

    type: str
    ts: float = field(default_factory=lambda: time.time())
    step: Optional[int] = None
    epoch: Optional[float] = None
    loss: Optional[float] = None
    lr: Optional[float] = None
    grad_norm: Optional[float] = None
    tokens_per_s: Optional[float] = None
    ema_loss: Optional[float] = None
    p95_step_ms: Optional[float] = None
    p99_step_ms: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in _VALID_TYPES:
            raise ValueError(
                f"type must be one of {sorted(_VALID_TYPES)}; got {self.type!r}"
            )
        if isinstance(self.ts, bool) or not isinstance(self.ts, (int, float)):
            raise ValueError("ts must be a number")
        if not math.isfinite(float(self.ts)):
            raise ValueError("ts must be finite")
        if self.message is not None:
            if not isinstance(self.message, str):
                raise ValueError("message must be str or None")
            if "\x00" in self.message:
                raise ValueError("message contains NUL byte")
            if len(self.message) > _MAX_MESSAGE_LEN:
                raise ValueError(
                    f"message exceeds {_MAX_MESSAGE_LEN} chars"
                )


def to_payload(event: TrainEvent) -> Dict[str, Any]:
    """Convert a TrainEvent into a JSON-serialisable dict, omitting None."""
    raw: Dict[str, Any] = {
        "type": event.type,
        "ts": float(event.ts),
    }
    for key in (
        "step",
        "epoch",
        "loss",
        "lr",
        "grad_norm",
        "tokens_per_s",
        "ema_loss",
        "p95_step_ms",
        "p99_step_ms",
        "eta_seconds",
        "message",
    ):
        value = getattr(event, key)
        if value is not None:
            raw[key] = value
    # Defence-in-depth: filter out any keys that drifted in from refactors.
    return {key: value for key, value in raw.items() if key in _ALLOWED_KEYS}


def format_sse_frame(event: TrainEvent) -> str:
    """Serialise the event as a single Server-Sent-Events frame.

    `data: {json}\\n\\n` — the standard W3C SSE wire format.
    """
    payload = to_payload(event)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n"
