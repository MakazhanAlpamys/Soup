"""Canary router (v0.58.0 Part B).

Pure-Python deterministic routing of inference requests between a stable
adapter and a canary adapter. The router is *deterministic* on a hashed
request key — so a given conversation always lands in the same bucket
within an iteration — and *sticky on rollback* so a flaky verdict can't
ping-pong traffic between adapters.

Why this lives in `utils/` and not inside `commands/serve.py`: the
canary policy is a pure math kernel exercised by `soup loop watch`
without needing a live FastAPI app. The HTTP middleware in `serve.py`
plugs into `route()` directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

from soup_cli.utils.loop_state import LoopState
from soup_cli.utils.paths import atomic_write_text, enforce_under_cwd_and_no_symlink


@dataclass(frozen=True)
class CanaryPolicy:
    """Frozen rollout policy: stable vs canary + traffic split + verdict."""

    stable: str
    canary: Optional[str] = None
    traffic_pct: float = 0.0  # in [0, 100]
    sticky_on_rollback: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.stable, str) or not self.stable or "\x00" in self.stable:
            raise ValueError("stable must be a non-empty NUL-free string")
        if len(self.stable) > 256:
            raise ValueError("stable name exceeds 256 chars")
        if self.canary is not None:
            if not isinstance(self.canary, str) or not self.canary or "\x00" in self.canary:
                raise ValueError("canary must be a non-empty NUL-free string or None")
            if len(self.canary) > 256:
                raise ValueError("canary name exceeds 256 chars")
            if self.canary == self.stable:
                raise ValueError("canary must differ from stable")
        v = self.traffic_pct
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("traffic_pct must be a finite number")
        if not (0.0 <= float(v) <= 100.0):
            raise ValueError("traffic_pct must be in [0, 100]")
        if self.canary is None and float(v) > 0.0:
            raise ValueError("cannot route traffic to None canary")
        if not isinstance(self.sticky_on_rollback, bool):
            raise ValueError("sticky_on_rollback must be bool")


@dataclass(frozen=True)
class RouteDecision:
    """Result of one routing decision: which adapter + which bucket."""

    adapter: str
    bucket: str  # "stable" | "canary"
    rolled_back: bool = False


_HASH_MOD = 10_000  # buckets — gives ±0.01 % granularity on the split
_DEFAULT_STATS_PATH = os.path.join(".soup", "canary-stats.json")
_MAX_STATS_BYTES = 64 * 1024
_STATS_LOCK = threading.Lock()


class CanaryStateCache:
    """App-local state cache invalidated by atomic replacements, not just mtime."""

    def __init__(self, path: Optional[str] = None) -> None:
        from soup_cli.utils.loop_state import default_state_path

        self.path = path if path is not None else default_state_path()
        self._key: Optional[tuple[int, int, int]] = None
        self._state: Optional[LoopState] = None
        self._lock = threading.Lock()

    def get(self) -> Optional[LoopState]:
        from soup_cli.utils.loop_state import read_state

        with self._lock:
            enforce_under_cwd_and_no_symlink(self.path, "canary state path")
            try:
                info = os.stat(self.path)
            except FileNotFoundError:
                self._key = self._state = None
                return None
            key = (info.st_mtime_ns, info.st_size, info.st_ino)
            if key != self._key:
                # Cache only successful reads. A concurrent replacement causes
                # another refresh on the next request instead of a stale policy.
                state = read_state(self.path)
                self._state, self._key = state, key
            return self._state

    def matches(self, policy: CanaryPolicy, rollout_id: Optional[str]) -> bool:
        """Whether these observations still belong to the active promotion."""
        state = self.get()
        return state is not None and (
            state.served_model, state.canary_active, state.canary_rollout_id
        ) == (policy.stable, policy.canary, rollout_id)


def _bucket_for_key(key: str) -> int:
    """Deterministic 4-hex-digit bucket via SHA-256 (key fingerprint)."""
    if not isinstance(key, str):
        raise TypeError("key must be a string")
    if not key:
        raise ValueError("key must not be empty")
    if "\x00" in key:
        raise ValueError("key must not contain NUL")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Take 4 bytes → 32-bit unsigned, modulo bucket count.
    val = int.from_bytes(digest[:4], "big", signed=False)
    return val % _HASH_MOD


def route(policy: CanaryPolicy, request_key: str) -> RouteDecision:
    """Decide which adapter serves a request given its fingerprint key.

    Deterministic: the same ``(policy, request_key)`` always returns the
    same bucket. Stickiness comes from the caller building ``request_key``
    from a conversation id (not a per-message timestamp).
    """
    if not isinstance(policy, CanaryPolicy):
        raise TypeError("policy must be CanaryPolicy")
    bucket = _bucket_for_key(request_key)
    # `math.ceil` is more predictable than `round` at sub-bucket fractions:
    # `traffic_pct=0.005` → 1 bucket out of 10 000 (0.01%), not 0 (silent
    # truncation per code-review MEDIUM #5).
    threshold = math.ceil(policy.traffic_pct / 100.0 * _HASH_MOD)
    if policy.canary is None or bucket >= threshold:
        return RouteDecision(adapter=policy.stable, bucket="stable")
    return RouteDecision(adapter=policy.canary, bucket="canary")


def rollback(policy: CanaryPolicy, *, reason: str = "regression") -> CanaryPolicy:
    """Return a policy with canary cleared (traffic forced to stable).

    Sticky-on-rollback means subsequent calls to ``route`` return the
    stable adapter even if a noisy re-evaluation later flips the verdict
    — the operator must explicitly re-promote a canary to clear the
    sticky bit (by calling ``CanaryPolicy(...)`` afresh).
    """
    if not isinstance(policy, CanaryPolicy):
        raise TypeError("policy must be CanaryPolicy")
    if not isinstance(reason, str) or not reason or "\x00" in reason:
        raise ValueError("reason must be a non-empty NUL-free string")
    return CanaryPolicy(
        stable=policy.stable,
        canary=None,
        traffic_pct=0.0,
        sticky_on_rollback=policy.sticky_on_rollback,
    )


# ---------------------------------------------------------------------------
# Verdict bucket aggregation — used by `soup loop watch` to decide whether to
# roll back. Each per-bucket result is a {0, 1} OK/MAJOR signal (matches the
# v0.26.0 Quant-Lobotomy verdict surface).
# ---------------------------------------------------------------------------

@dataclass
class BucketStats:
    """Mutable per-bucket counters. NOT thread-safe — call ``aggregate``
    under a single thread or wrap externally with ``threading.Lock``."""

    stable_ok: int = 0
    stable_major: int = 0
    canary_ok: int = 0
    canary_major: int = 0
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def record(self, bucket: str, ok: bool) -> None:
        if bucket not in ("stable", "canary"):
            raise ValueError("bucket must be 'stable' or 'canary'")
        if not isinstance(ok, bool):
            raise ValueError("ok must be bool")
        with self._lock:
            if bucket == "stable":
                if ok:
                    self.stable_ok += 1
                else:
                    self.stable_major += 1
            else:
                if ok:
                    self.canary_ok += 1
                else:
                    self.canary_major += 1

    def verdict(self, *, min_samples: int = 30, regression_threshold: float = 0.05) -> str:
        """Return ``"OK"`` / ``"MAJOR"`` / ``"UNKNOWN"``.

        - ``UNKNOWN``: fewer than ``min_samples`` total samples in the
          canary bucket. Defends against early-rollback on insufficient
          evidence (matches v0.26.0 Quant-Lobotomy policy).
        - ``MAJOR``: canary OK rate is below stable's by more than
          ``regression_threshold`` (default 5 percentage points).
        - ``OK``: otherwise.
        """
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive int")
        v = regression_threshold
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("regression_threshold must be a finite number")
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("regression_threshold must be in [0, 1]")
        with self._lock:
            canary_total = self.canary_ok + self.canary_major
            stable_total = self.stable_ok + self.stable_major
            if canary_total < min_samples:
                return "UNKNOWN"
            stable_rate = (self.stable_ok / stable_total) if stable_total > 0 else 1.0
            canary_rate = self.canary_ok / canary_total
            if stable_rate - canary_rate > regression_threshold:
                return "MAJOR"
            return "OK"

    def snapshot(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(
                {
                    "stable_ok": self.stable_ok,
                    "stable_major": self.stable_major,
                    "canary_ok": self.canary_ok,
                    "canary_major": self.canary_major,
                }
            )


def default_stats_path() -> str:
    """Return the canary outcome path next to the default loop state."""
    return _DEFAULT_STATS_PATH


def read_bucket_stats(
    *,
    stable: str,
    canary: str,
    rollout_id: Optional[str] = None,
    path: Optional[str] = None,
) -> BucketStats:
    """Load counters for one rollout, returning empty stats for another rollout.

    The adapter identities prevent samples from an older promotion being reused
    after an operator selects a different canary.
    """
    policy = CanaryPolicy(stable=stable, canary=canary, traffic_pct=0.0)
    target = path or default_stats_path()
    enforce_under_cwd_and_no_symlink(target, "canary stats path")
    try:
        if os.path.getsize(target) > _MAX_STATS_BYTES:
            raise ValueError("canary stats file exceeds 64 KiB cap")
        with open(target, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return BucketStats()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("canary stats file is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("canary stats root must be an object")
    if (
        payload.get("stable") != policy.stable
        or payload.get("canary") != policy.canary
        or payload.get("rollout_id") != rollout_id
    ):
        return BucketStats()
    values = {}
    for name in ("stable_ok", "stable_major", "canary_ok", "canary_major"):
        value = payload.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative int")
        values[name] = value
    return BucketStats(**values)


def record_bucket_outcome(
    policy: CanaryPolicy,
    bucket: str,
    ok: bool,
    *,
    rollout_id: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    """Atomically add one served request outcome to the active rollout."""
    if not isinstance(policy, CanaryPolicy):
        raise TypeError("policy must be CanaryPolicy")
    if policy.canary is None:
        return
    if rollout_id is not None and (
        not isinstance(rollout_id, str) or not rollout_id or "\x00" in rollout_id
    ):
        raise ValueError("rollout_id must be a non-empty NUL-free string or None")
    delta = BucketStats()
    delta.record(bucket, ok)
    _record_bucket_counts(policy, delta.snapshot(), rollout_id=rollout_id, path=path)


def _record_bucket_counts(
    policy: CanaryPolicy, counts: Mapping[str, int], *,
    rollout_id: Optional[str], path: Optional[str],
    state_cache: Optional[CanaryStateCache] = None,
) -> None:
    """Merge one batch under the existing process-local persistence lock."""
    target = path or default_stats_path()
    with _STATS_LOCK:
        if state_cache is not None and not state_cache.matches(policy, rollout_id):
            return
        stats = read_bucket_stats(
            stable=policy.stable,
            canary=policy.canary,
            rollout_id=rollout_id,
            path=target,
        )
        totals = {name: value + counts[name] for name, value in stats.snapshot().items()}
        payload = {
            "stable": policy.stable,
            "canary": policy.canary,
            "rollout_id": rollout_id,
            **totals,
        }
        atomic_write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
            target,
            prefix=".canary_stats_",
            field="canary stats path",
        )


class BufferedCanaryOutcomes:
    """Persist at 64 outcomes or two seconds, whichever comes first.

    One buffer per serving app. Promotion changes retire the previous buffer
    before accepting the next; shutdown cancels the timer and drains the tail.
    """

    def __init__(self, path: Optional[str] = None, *, batch_size: int = 64,
                 interval: float = 2.0, state_cache: Optional[CanaryStateCache] = None) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive int")
        if isinstance(interval, bool) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite")
        self.path = path if path is not None else default_stats_path()
        self.batch_size = batch_size
        self.interval = interval
        self._state_cache = state_cache
        self._lock = threading.Lock()
        self._policy: Optional[CanaryPolicy] = None
        self._rollout_id: Optional[str] = None
        self._counts = BucketStats()
        self._pending = 0
        self._timer: Optional[threading.Timer] = None
        self._closed = False
        self._warned = False

    def record(self, policy: CanaryPolicy, bucket: str, ok: bool, *,
               rollout_id: Optional[str] = None) -> None:
        if not isinstance(policy, CanaryPolicy):
            raise TypeError("policy must be CanaryPolicy")
        if policy.canary is None:
            return
        if rollout_id is not None and (
            not isinstance(rollout_id, str) or not rollout_id or "\x00" in rollout_id
        ):
            raise ValueError("rollout_id must be a non-empty NUL-free string or None")
        # Validate before changing the buffer or flushing another rollout.
        delta = BucketStats()
        delta.record(bucket, ok)
        with self._lock:
            if self._closed:
                raise ValueError("canary outcome buffer is closed")
            if not self._is_current(policy, rollout_id):
                return
            if self._policy is not None and (
                (self._policy.stable, self._policy.canary, self._rollout_id)
                != (policy.stable, policy.canary, rollout_id)
            ):
                self._flush_locked()
            if self._pending >= self.batch_size:
                self._flush_locked()
            self._policy, self._rollout_id = policy, rollout_id
            self._counts.record(bucket, ok)
            self._pending += 1
            if self._timer is None:
                self._timer = threading.Timer(self.interval, self._timed_flush)
                self._timer.daemon = True
                self._timer.start()
            if self._pending >= self.batch_size:
                self._flush_locked()

    def _is_current(self, policy: CanaryPolicy, rollout_id: Optional[str]) -> bool:
        if self._state_cache is None:
            return True
        return self._state_cache.matches(policy, rollout_id)

    def _flush_locked(self) -> None:
        if self._pending:
            assert self._policy is not None
            # Check under the buffer lock, both when accepting and when flushing:
            # an older in-flight request/timer cannot replace a new rollout's file.
            _record_bucket_counts(self._policy, self._counts.snapshot(),
                                  rollout_id=self._rollout_id, path=self.path,
                                  state_cache=self._state_cache)
            self._counts = BucketStats()
            self._pending = 0
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _timed_flush(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._flush_locked()
            except (OSError, TypeError, ValueError):
                if not self._warned:
                    logging.getLogger(__name__).warning("canary batch flush failed", exc_info=True)
                    self._warned = True
                # Keep the batch for retry; a failed write never clears counters.
                self._timer = threading.Timer(self.interval, self._timed_flush)
                self._timer.daemon = True
                self._timer.start()

    def flush(self) -> None:
        """Make the pending window durable (also useful before a local watch)."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Stop the timer and drain counters during orderly server shutdown."""
        with self._lock:
            self._closed = True
            try:
                self._flush_locked()
            finally:
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
