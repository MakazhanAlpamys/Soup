"""Closed-loop reward-hacking auto-mitigation — v0.71.26.

The trainer DETECTS reward hacking mid-run (reusing the v0.70.0
:mod:`soup_cli.utils.reward_hacking` detector) and SELF-CORRECTS instead of
only halting:

- ``log_only`` (Stage 0): instrument + append a mitigation log; NO control action.
- ``kl_control`` (Stage 1): reversible bang-bang + hysteresis KL/β controller.
- ``pid_lagrangian`` (Stage 2): PID-Lagrangian controller + rollback ladder.

Design:
- Pure controller pieces (``ControllerState``, ``BangBangPolicy``,
  ``PIDLagrangianPolicy``, ``combine_signals``, ``smooth_signal``, the step
  functions) are frozen dataclasses / free functions with NO torch import at
  module scope — CPU-testable and constructible on a torch-less interpreter.
- ``RewardHackMitigationCallback`` lazily resolves ``transformers.TrainerCallback``
  (falls back to ``object``), mirroring :mod:`soup_cli.utils.reward_hacking`.

Security:
- ``MitigationLogWriter`` mirrors :class:`soup_cli.monitoring.trace_logger.TraceLogWriter`:
  thread-safe append, size-based rotation, secret redaction (reuses the shared
  ``redact_value``), cwd containment, symlink-reject on rotate.
- Every public validator rejects bool-for-int, non-finite floats, and
  out-of-bounds values with actionable messages (mirrors ``reward_hacking.py``).
- HARD INVARIANT: β is clamped to ``[beta_floor > 0, beta_ceil]`` and never
  crosses 0 (β=0 gates off the ref-log-prob path at generation time).
"""

from __future__ import annotations

import json
import math
import os
import stat
import statistics
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soup_cli.monitoring.trace_logger import redact_value
from soup_cli.utils.paths import is_under_cwd

_DEFAULT_CAP_MB = 100
_MIN_CAP_MB = 1
_MAX_CAP_MB = 10_000

# Closed allowlists (frozenset — runtime-immutable, mirrors reward_hacking.py).
MITIGATION_MODES: frozenset[str] = frozenset(
    {"off", "log_only", "kl_control", "pid_lagrangian"}
)
SIGNAL_NAMES: frozenset[str] = frozenset(
    {"info_rm", "rm_ensemble", "length_trend", "repetition"}
)
SMOOTHING_METHODS: frozenset[str] = frozenset({"none", "ema", "median"})

_EMA_ALPHA = 0.5  # fixed EMA weight on the new sample (documented, Stage 3)


# --- pure validators (mirror reward_hacking.py bool-before-int policy) ---


def _check_nonneg_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must not be bool")
    if not isinstance(value, int):
        raise TypeError(f"{field} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be non-negative, got {value}")
    return value


def _check_finite_float(value: object, field: str, *, nonneg: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must not be bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number, got {type(value).__name__}")
    fvalue = float(value)
    if not math.isfinite(fvalue):
        raise ValueError(f"{field} must be finite (no NaN/Inf)")
    if nonneg and fvalue < 0.0:
        raise ValueError(f"{field} must be non-negative, got {fvalue}")
    return fvalue


@dataclass(frozen=True)
class ControllerState:
    """Frozen controller state; evolve via :func:`dataclasses.replace`.

    - ``beta`` / ``kl_coef``: current ABSOLUTE coefficient applied (0.0 is the
      uninitialised sentinel; the callback seeds it from the trainer on step 1).
    - ``tripped``: currently in the raised band.
    - ``dwell_count`` / ``release_count``: consecutive want-raise / want-relax
      steps (hysteresis counters).
    - ``integral`` / ``prev_error``: PID accumulator + derivative memory (Stage 2).
    - ``last_signal``: last smoothed hacking signal.
    - ``recovery_attempts``: rollbacks performed so far (Stage 2).
    """

    step: int = 0
    beta: float = 0.0
    kl_coef: float = 0.0
    tripped: bool = False
    dwell_count: int = 0
    release_count: int = 0
    integral: float = 0.0
    prev_error: float = 0.0
    last_signal: float = 0.0
    recovery_attempts: int = 0

    def __post_init__(self) -> None:
        _check_nonneg_int(self.step, "step")
        _check_nonneg_int(self.dwell_count, "dwell_count")
        _check_nonneg_int(self.release_count, "release_count")
        _check_nonneg_int(self.recovery_attempts, "recovery_attempts")
        _check_finite_float(self.beta, "beta", nonneg=True)
        _check_finite_float(self.kl_coef, "kl_coef", nonneg=True)
        _check_finite_float(self.integral, "integral")
        _check_finite_float(self.prev_error, "prev_error")
        _check_finite_float(self.last_signal, "last_signal")
        if not isinstance(self.tripped, bool):
            raise TypeError("tripped must be bool")


def combine_signals(signals: Mapping[str, Any], names: Sequence[str]) -> float:
    """Multi-signal vote → a single drop_pct in ``[0, 1]``.

    Clamps each enabled, finite per-signal value to ``[0, 1]`` then returns
    their mean. Missing / non-finite / non-numeric values are dropped; an empty
    result is ``0.0`` (no evidence of hacking).
    """
    values: list[float] = []
    for name in names:
        if name not in signals:
            continue
        raw = signals[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        fval = float(raw)
        if not math.isfinite(fval):
            continue
        values.append(min(1.0, max(0.0, fval)))
    if not values:
        return 0.0
    return sum(values) / len(values)


def smooth_signal(new: float, window: Sequence[float], *, method: str) -> float:
    """Smooth a scalar signal. ``none`` → new; ``ema`` → 0.5·prev + 0.5·new
    (prev = ``window[-1]``, or ``new`` when the window is empty); ``median`` →
    median of ``window + [new]``.
    """
    if method not in SMOOTHING_METHODS:
        raise ValueError(
            f"method must be one of {sorted(SMOOTHING_METHODS)}, got {method!r}"
        )
    fnew = float(new)
    if method == "none":
        return fnew
    win = [float(w) for w in window]
    if method == "ema":
        if not win:
            return fnew
        return _EMA_ALPHA * win[-1] + (1.0 - _EMA_ALPHA) * fnew
    return float(statistics.median(win + [fnew]))


# --- telemetry helpers for the log_only stream (pure) ---


def mean_token_len(completions: Sequence[Any]) -> float:
    """Mean whitespace-token count over completion strings; empty → 0.0."""
    lengths: list[int] = []
    for comp in completions:
        text = comp if isinstance(comp, str) else str(comp)
        lengths.append(len(text.split()))
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)


def reward_mean_std(rewards: Sequence[Any]) -> tuple[float, float]:
    """Population ``(mean, std)`` over finite numeric rewards; empty → ``(0.0, 0.0)``."""
    values = [
        float(reward)
        for reward in rewards
        if not isinstance(reward, bool)
        and isinstance(reward, (int, float))
        and math.isfinite(float(reward))
    ]
    if not values:
        return (0.0, 0.0)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return (mean, math.sqrt(variance))


def mean_repetition(completions: Sequence[Any]) -> float:
    """Mean echo-trap repetition score over completions; empty → 0.0.

    Reuses :func:`soup_cli.utils.echo_trap.score_echo_signal` (lazy import —
    keeps this module torch-free at scope).
    """
    trajectories: list[list[str]] = []
    for comp in completions:
        text = comp if isinstance(comp, str) else str(comp)
        trajectories.append(text.split())
    if not trajectories:
        return 0.0
    from soup_cli.utils.echo_trap import score_echo_signal

    try:
        return float(score_echo_signal(trajectories))
    except (TypeError, ValueError):
        return 0.0


class MitigationLogWriter:
    """Thread-safe append-only JSONL log for the reward-hack controller.

    One JSON object per :meth:`record` call with shape
    ``{"ts": ..., "step": ..., **snapshot}``. Hard rotation cap of ``cap_mb``
    (default 100 MB): when the active file would exceed the cap it is renamed
    ``<path>.1`` (one backup retained) and a fresh active file is started.
    Mirrors :class:`soup_cli.monitoring.trace_logger.TraceLogWriter`.
    """

    def __init__(self, path: str, *, cap_mb: int = _DEFAULT_CAP_MB) -> None:
        if isinstance(cap_mb, bool) or not isinstance(cap_mb, int):
            raise TypeError("cap_mb must be int")
        if not (_MIN_CAP_MB <= cap_mb <= _MAX_CAP_MB):
            raise ValueError(
                f"cap_mb must be in [{_MIN_CAP_MB}, {_MAX_CAP_MB}], got {cap_mb}"
            )
        if not isinstance(path, str) or not path or "\x00" in path:
            raise ValueError("path must be a non-empty string with no null bytes")
        if not is_under_cwd(path):
            raise ValueError(f"mitigation log path must stay under cwd: {path}")
        resolved = Path(os.path.realpath(path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._path = resolved
        self._cap_bytes = cap_mb * 1024 * 1024
        # Single-process lock: protects rotate + write within ONE run process.
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def cap_bytes(self) -> int:
        return self._cap_bytes

    def record(self, *, step: int, snapshot: Mapping[str, Any]) -> None:
        """Append one telemetry entry. Never raises on serialisation issues."""
        entry: dict[str, Any] = {"ts": time.time(), "step": int(step)}
        if isinstance(snapshot, Mapping):
            for key, value in snapshot.items():
                skey = str(key)
                if skey in entry:
                    continue
                entry[skey] = redact_value(value)
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return  # drop unserialisable entries silently — passive log
        line_bytes = (line + "\n").encode("utf-8")
        with self._lock:
            self._maybe_rotate(extra=len(line_bytes))
            try:
                with self._path.open("ab") as handle:
                    handle.write(line_bytes)
            except OSError:
                return

    def _maybe_rotate(self, *, extra: int) -> None:
        try:
            current = self._path.stat().st_size
        except OSError:
            return
        if current + extra <= self._cap_bytes:
            return
        backup = self._path.with_suffix(self._path.suffix + ".1")
        try:
            # Refuse to overwrite a symlink at the backup path (TOCTOU guard —
            # mirrors trace_logger). An attacker pre-placing <log>.1 as a
            # symlink would otherwise have the active log renamed onto it.
            try:
                backup_stat = os.lstat(backup)
                if stat.S_ISLNK(backup_stat.st_mode):
                    return
                backup.unlink()
            except FileNotFoundError:
                pass
            self._path.rename(backup)
        except OSError:
            return


def _get_trainer_callback_base():
    """Lazy-resolve ``transformers.TrainerCallback`` (mirror reward_hacking.py).

    Resolved once at class-definition time so this module imports on a
    torch-less interpreter (falls back to ``object``).
    """
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except ImportError:
        return object


_TrainerCallbackBase = _get_trainer_callback_base()


class RewardHackMitigationCallback(_TrainerCallbackBase):  # type: ignore[misc, valid-type]
    """Closed-loop reward-hacking mitigation HF TrainerCallback (v0.71.26).

    Reads the shared :class:`~soup_cli.utils.rl_signal_buffer.RLSignalBuffer`
    snapshot each step, computes a multi-signal hacking score (reusing the
    v0.70.0 :class:`~soup_cli.utils.reward_hacking.RewardHackCallback` for the
    info_rm / rm_ensemble component), and — depending on ``mode`` — either only
    logs telemetry (``log_only``) or drives a controller that mutates the
    trainer's β / kl_coef (``kl_control`` / ``pid_lagrangian``, wired in later
    stages).

    ``attach(trainer)`` stores the trainer reference so the controller can read
    and mutate the live β (mirrors ``dpo_variants.BetaScheduleCallback``).
    """

    def __init__(
        self,
        *,
        mode: str,
        detector: str,
        log_writer: MitigationLogWriter | None,
        signals: Sequence[str] = ("info_rm",),
        buffer: Any = None,
        tokenizer: Any = None,
        task: str = "grpo",
    ) -> None:
        if mode not in MITIGATION_MODES:
            raise ValueError(
                f"mode must be one of {sorted(MITIGATION_MODES)}, got {mode!r}"
            )
        from soup_cli.utils.reward_hacking import (
            RewardHackCallback,
            validate_hack_detector,
        )

        self.mode = mode
        self.detector = validate_hack_detector(detector)
        self.log_writer = log_writer
        self.signals = tuple(signals)
        for name in self.signals:
            if name not in SIGNAL_NAMES:
                raise ValueError(
                    f"signal {name!r} not in {sorted(SIGNAL_NAMES)}"
                )
        self.buffer = buffer
        self.tokenizer = tokenizer
        self.task = task
        # Compose the v0.70.0 detector callback for the info_rm/rm_ensemble
        # baseline + drop_pct logic (DRY — no re-implementation).
        self._detector_cb = RewardHackCallback(
            detector=self.detector, halt_on_hack=False, buffer=buffer
        )
        self._trainer: Any = None
        self._state = ControllerState()

    def attach(self, trainer: Any) -> None:
        """Store the trainer reference (the β / kl_coef mutation target)."""
        self._trainer = trainer

    def _current_coefficient(self) -> float | None:
        """Read the live coefficient the controller governs (β for GRPO,
        kl_coef for PPO)."""
        trainer = self._trainer
        if trainer is None:
            return None
        if self.task == "ppo":
            args = getattr(trainer, "args", None)
            return getattr(args, "kl_coef", None) if args is not None else None
        return getattr(trainer, "beta", None)

    def _build_telemetry(self, snapshot: Mapping[str, Any], step: int) -> dict[str, Any]:
        """Compute the per-step telemetry entry from a buffer snapshot."""
        raw = self._detector_cb.compute_signal(snapshot)
        drop_pct = 0.0
        verdict = "OK"
        if raw is not None:
            report = self._detector_cb.observe_signal(raw, step)
            drop_pct = self._detector_cb.last_drop_pct()
            verdict = report.verdict
        completions = snapshot.get("completions", []) or []
        rewards = snapshot.get("rewards", []) or []
        reward_mean, reward_std = reward_mean_std(rewards)
        return {
            "mode": self.mode,
            "detector": self.detector,
            "raw_signal": raw,
            "drop_pct": drop_pct,
            "verdict": verdict,
            "beta": self._current_coefficient(),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "completion_length_mean": mean_token_len(completions),
            "repetition": mean_repetition(completions),
        }

    def on_step_end(self, args, state, control, **kwargs):
        """Per-step hook — read the buffer, compute telemetry, act by mode.

        Instrumentation must NEVER crash training: a broad except returns the
        unmodified control on any error.
        """
        if self.buffer is None or self.log_writer is None:
            return control
        try:
            snapshot = self.buffer.snapshot()
            step = int(getattr(state, "global_step", 0) or 0)
            telemetry = self._build_telemetry(snapshot, step)
            # Stage 0: log_only observes; control modes (kl_control /
            # pid_lagrangian) are wired in later stages. In every mode we
            # record the telemetry line.
            self.log_writer.record(step=step, snapshot=telemetry)
            return control
        except Exception:  # noqa: BLE001 — instrumentation must never crash
            return control
