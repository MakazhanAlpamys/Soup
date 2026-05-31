"""Online-eval drift alarm — rolling KL on output-token distribution.

v0.63.0 Part E — pages when the FT-time reference distribution diverges
from the live production distribution by more than `threshold` KL. Optional
webhook (`--slack-url` / `--discord-url`) composes with v0.30 OpenTelemetry
export so drift events flow into existing observability pipelines.

Why token-distribution KL (not perplexity):
- Perplexity needs the live model — too expensive at every request.
- A whitespace-tokenised output distribution is cheap to maintain, runs in
  ms over a day of traces, and surfaces both behavioural drift ("model now
  outputs JSON when it used to output prose") AND vocabulary drift ("model
  has started repeating the same 20 phrases").

Signal interpretation:
- KL < 0.05: normal drift band, no alert.
- KL 0.05 - 0.2: minor drift, worth a glance.
- KL >= 0.2: major drift, page on-call.

We pick a default threshold of 0.2 to match v0.43.0 Part B `classify_kl_delta`
quant-check thresholds; operators can tune via --threshold.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple
from urllib.parse import urlparse

from soup_cli.utils.paths import is_under_cwd

_MAX_REFERENCE_ROWS = 1_000_000
_MAX_WEBHOOK_URL_LEN = 4096
_MAX_TEXT_LEN = 1_000_000  # 1 MB / row
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# Default smoothing constant for the KL kernel (Laplace-style add-epsilon
# to defend against `log(0)` when a token in `p` is absent from `q`).
_EPS = 1e-9


@dataclass(frozen=True)
class DriftReport:
    """Result of one drift-check pass."""

    kl_divergence: float
    threshold: float
    drift_detected: bool
    n_reference: int
    n_live: int
    top_drift_tokens: Tuple[Tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.kl_divergence < 0.0:
            raise ValueError("kl_divergence must be >= 0")
        if not math.isfinite(self.kl_divergence):
            raise ValueError("kl_divergence must be finite")
        if self.threshold <= 0:
            raise ValueError("threshold must be > 0")
        if self.n_reference < 0:
            raise ValueError("n_reference must be >= 0")
        if self.n_live < 0:
            raise ValueError("n_live must be >= 0")


def validate_threshold(value: object) -> float:
    """Validate ``threshold`` is a finite positive float."""
    if isinstance(value, bool):
        raise TypeError("threshold must be number, not bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"threshold must be number, got {type(value).__name__}"
        )
    f_val = float(value)
    if not math.isfinite(f_val):
        raise ValueError("threshold must be finite (no NaN / Inf)")
    if f_val <= 0.0:
        raise ValueError(f"threshold must be > 0, got {f_val}")
    if f_val > 100.0:
        raise ValueError(f"threshold must be <= 100, got {f_val}")
    return f_val


def _is_private_or_link_local(host: str) -> bool:
    """Return True iff ``host`` resolves to a non-loopback private/reserved IP.

    Explicit parentheses on the final clause (code-review MEDIUM fix
    v0.63.0): Python binds `and` tighter than `or`, but the SSRF gate is
    safety-critical and a future edit should not need to re-derive the
    precedence rules to verify the logic.
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
    """SSRF-hardened webhook URL validator.

    Mirrors v0.29.0 `HF_ENDPOINT` / v0.30.0 OTLP / v0.51.0 `validate_hub_endpoint`
    policy:
    - scheme allowlist {http, https}
    - null-byte / control-char rejection
    - ``0.0.0.0`` rejected
    - plain HTTP only permitted for loopback hosts
    - private / link-local / cloud-metadata IPs rejected
    """
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
        raise ValueError(
            "webhook URL for remote hosts must use HTTPS"
        )
    return stripped


def compute_token_distribution(rows: Iterable[object]) -> Mapping[str, float]:
    """Compute a normalised whitespace-token frequency distribution.

    Returns a dict mapping token -> probability summing to 1.0 (empty dict
    when no usable tokens). Skips non-string rows silently to tolerate
    messy JSONL inputs.
    """
    if isinstance(rows, str):
        raise TypeError("rows must be an iterable of strings, not str")
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise TypeError(
            f"rows must be iterable, got {type(rows).__name__}"
        ) from exc

    counts: dict[str, int] = {}
    total = 0
    for row in iterator:
        if not isinstance(row, str):
            continue
        text = row if len(row) <= _MAX_TEXT_LEN else row[:_MAX_TEXT_LEN]
        for token in text.split():
            counts[token] = counts.get(token, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {tok: cnt / total for tok, cnt in counts.items()}


def _validate_distribution(dist: object, *, name: str) -> Mapping[str, float]:
    if not isinstance(dist, Mapping):
        raise TypeError(
            f"{name} must be a Mapping, got {type(dist).__name__}"
        )
    for tok, prob in dist.items():
        if isinstance(prob, bool):
            raise TypeError(f"{name}[{tok!r}] must be number, not bool")
        if not isinstance(prob, (int, float)):
            raise TypeError(
                f"{name}[{tok!r}] must be number, got {type(prob).__name__}"
            )
        f_prob = float(prob)
        if not math.isfinite(f_prob):
            raise ValueError(f"{name}[{tok!r}] must be finite")
        if f_prob < 0.0:
            raise ValueError(
                f"{name}[{tok!r}] must be >= 0, got {f_prob}"
            )
    return dist


def rolling_kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Compute KL(p || q) over the union of token vocabularies.

    Laplace-smoothed: missing tokens in q get ``_EPS`` so divergence stays
    finite even on disjoint vocabularies.
    """
    _validate_distribution(p, name="p")
    _validate_distribution(q, name="q")

    kl = 0.0
    for token, p_prob in p.items():
        if p_prob == 0.0:
            continue
        q_prob = float(q.get(token, 0.0))
        q_smooth = q_prob if q_prob > 0.0 else _EPS
        kl += float(p_prob) * math.log(float(p_prob) / q_smooth)
    # Clamp to non-negative (numerical artefacts can drop the tally below 0
    # for near-identical distributions).
    return max(0.0, kl)


def _top_drift_tokens(
    p: Mapping[str, float],
    q: Mapping[str, float],
    *,
    n: int = 5,
) -> Tuple[Tuple[str, float], ...]:
    """Return the top-N tokens by absolute probability delta."""
    keys = set(p) | set(q)
    deltas = []
    for tok in keys:
        p_prob = float(p.get(tok, 0.0))
        q_prob = float(q.get(tok, 0.0))
        deltas.append((tok, abs(p_prob - q_prob)))
    deltas.sort(key=lambda kv: (-kv[1], kv[0]))
    return tuple(deltas[:n])


def _read_jsonl_outputs(path: str) -> list[str]:
    rows: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            text = obj.get("output") or obj.get("response") or obj.get("text")
            if isinstance(text, str):
                rows.append(text)
                if len(rows) >= _MAX_REFERENCE_ROWS:
                    break
    return rows


def _check_path(path: str, *, label: str) -> None:
    if not isinstance(path, str):
        raise TypeError(
            f"{label} must be str, got {type(path).__name__}"
        )
    if not path:
        raise ValueError(f"{label} must be non-empty")
    if "\x00" in path:
        raise ValueError(f"{label} must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(f"{label} {path!r} is outside cwd")


def run_drift_check(
    *,
    reference_path: str,
    live_path: str,
    threshold: float,
) -> DriftReport:
    """Compare reference vs live output-token distributions via rolling KL."""
    thr = validate_threshold(threshold)
    _check_path(reference_path, label="reference_path")
    _check_path(live_path, label="live_path")
    if not os.path.isfile(reference_path):
        raise FileNotFoundError(reference_path)
    if not os.path.isfile(live_path):
        raise FileNotFoundError(live_path)

    ref_rows = _read_jsonl_outputs(reference_path)
    live_rows = _read_jsonl_outputs(live_path)
    ref_dist = compute_token_distribution(ref_rows)
    live_dist = compute_token_distribution(live_rows)
    kl = rolling_kl(live_dist, ref_dist) if (ref_dist and live_dist) else 0.0
    return DriftReport(
        kl_divergence=kl,
        threshold=thr,
        drift_detected=kl > thr,
        n_reference=len(ref_rows),
        n_live=len(live_rows),
        top_drift_tokens=_top_drift_tokens(live_dist, ref_dist),
    )


def post_webhook(
    *,
    url: Optional[str],
    payload: Mapping[str, object],
    timeout_seconds: float = 5.0,
) -> bool:
    """POST ``payload`` as JSON to ``url``. Returns True on 2xx, False otherwise.

    Never raises — webhook delivery must NOT crash the drift-check run.
    Lazy-imports ``httpx`` so the runtime cost is paid only when an alarm
    actually fires.
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
    except Exception:  # noqa: BLE001 — webhook must never crash drift check
        return False


__all__ = [
    "DriftReport",
    "compute_token_distribution",
    "post_webhook",
    "rolling_kl",
    "run_drift_check",
    "validate_threshold",
    "validate_webhook_url",
]
