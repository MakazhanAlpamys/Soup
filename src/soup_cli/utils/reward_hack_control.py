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
from dataclasses import dataclass, replace
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
SHAPING_KINDS: frozenset[str] = frozenset({"length", "repetition", "sentinel"})

_EMA_ALPHA = 0.5  # fixed EMA weight on the new sample (documented, Stage 3)
_CONSERVATIVE_DISAGREE_TOL = 0.2  # detectors differ beyond this → stay cautious


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


def combine_conservative(votes: Sequence[float], *, disagree_tol: float) -> float:
    """Conservative-on-disagreement vote (Stage 3).

    Clamps each finite vote to ``[0, 1]``. When the detectors disagree beyond
    ``disagree_tol`` (``max - min > tol``), return the MAX (stay cautious — keep
    KL high, don't relax on a possibly-fooled detector). Otherwise the mean.
    Empty → ``0.0``.
    """
    finite = [
        min(1.0, max(0.0, float(v)))
        for v in votes
        if not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if not finite:
        return 0.0
    if max(finite) - min(finite) > float(disagree_tol):
        return max(finite)
    return sum(finite) / len(finite)


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def detect_reward_distribution_drift(
    rewards: Sequence[Any], *, degenerate_frac: float = 0.1
) -> bool:
    """Flag a bimodal reward-distribution collapse (Stage 3 anti-gaming).

    A policy can shift the reward distribution into two tight, well-separated
    clusters to fool the InfoRM top/bottom-half split into reporting healthy
    separation while it games the proxy. Heuristic: split sorted rewards in
    half; if the within-half spread is near-degenerate (< ``degenerate_frac`` of
    the between-half gap), the distribution has collapsed to two clusters →
    drift. Natural unimodal spreads and constant rewards are NOT flagged.
    Needs ≥ 4 finite rewards.
    """
    values = [
        float(r)
        for r in rewards
        if not isinstance(r, bool) and isinstance(r, (int, float)) and math.isfinite(float(r))
    ]
    if len(values) < 4:
        return False
    ordered = sorted(values)
    half = len(ordered) // 2
    low, high = ordered[:half], ordered[half:]
    gap = (sum(high) / len(high)) - (sum(low) / len(low))
    if gap <= 0.0:
        return False
    within = (_std(low) + _std(high)) / 2.0
    return within < degenerate_frac * gap


# --- reward-shaping shim (Stage 3) ---

_SHAPING_LENGTH_SAT = 32.0
_DEFAULT_SENTINEL = "GOLD"


def _completion_text(completion: Any) -> str:
    """Assistant text from a string / dict / list-of-message-dicts completion."""
    from soup_cli.utils.rl_signal_buffer import _completion_to_text

    return _completion_to_text(completion)


def _extract_completions(args: tuple, kwargs: dict) -> Any:
    """Pull the ``completions`` arg from a TRL reward-fn call."""
    completions = kwargs.get("completions")
    if completions is None:
        if len(args) >= 2:
            completions = args[1]
        elif len(args) == 1:
            completions = args[0]
    return completions


def _shaping_penalty(kind: str, text: str, *, sentinel: str) -> float:
    """Bounded ``[0, 1]`` penalty on the gamed proxy for one completion."""
    if kind == "length":
        return min(1.0, len(text.split()) / _SHAPING_LENGTH_SAT)
    if kind == "repetition":
        tokens = text.split()
        if not tokens:
            return 0.0
        from soup_cli.utils.echo_trap import score_trajectory_repetition

        try:
            return float(score_trajectory_repetition(tokens))
        except (TypeError, ValueError):
            return 0.0
    # sentinel
    return 1.0 if sentinel in text else 0.0


def shape_reward_fn(
    inner: Any, *, kind: str, strength: float, sentinel: str = _DEFAULT_SENTINEL
) -> Any:
    """Wrap a reward fn to subtract a bounded penalty on a gamed proxy.

    The inner fn is called exactly once (verbatim) and its reward is reduced by
    ``strength · penalty`` where ``penalty ∈ [0, 1]`` — so the reduction never
    exceeds ``strength``. ``strength=0`` is a pure passthrough. A shim error can
    never corrupt training: on any exception the verbatim reward is returned.
    ``__name__`` is preserved so TRL's per-function logging keys stay correct.
    """
    if kind not in SHAPING_KINDS:
        raise ValueError(f"kind must be one of {sorted(SHAPING_KINDS)}, got {kind!r}")
    strength_val = _check_finite_float(strength, "strength", nonneg=True)
    if strength_val > 1.0:
        raise ValueError(f"strength must be <= 1, got {strength_val}")

    def _shaped(*args: Any, **kwargs: Any) -> Any:
        rewards = inner(*args, **kwargs)  # verbatim — inner runs exactly once
        if strength_val <= 0.0:
            return rewards
        try:
            completions = _extract_completions(args, kwargs)
            if completions is None:
                return rewards
            comp_list = list(completions)
            out = []
            for idx, reward in enumerate(rewards):
                if (
                    idx < len(comp_list)
                    and isinstance(reward, (int, float))
                    and not isinstance(reward, bool)
                ):
                    penalty = _shaping_penalty(
                        kind, _completion_text(comp_list[idx]), sentinel=sentinel
                    )
                    out.append(float(reward) - strength_val * penalty)
                else:
                    out.append(reward)
            return out
        except Exception:  # noqa: BLE001 — shim MUST NOT corrupt the reward
            return rewards

    _shaped.__name__ = getattr(inner, "__name__", "reward")
    return _shaped


def apply_reward_shaping(reward_funcs: Any, tcfg: Any) -> Any:
    """Wrap the reward fn(s) with the shaping shim when ``reward_hack_reward_shaping``
    is set; otherwise return them unchanged. Preserves the single-vs-list shape."""
    if not getattr(tcfg, "reward_hack_reward_shaping", False):
        return reward_funcs
    kind = getattr(tcfg, "reward_hack_shaping_kind", "length")
    strength = float(getattr(tcfg, "reward_hack_shaping_strength", 0.0))
    if isinstance(reward_funcs, (list, tuple)):
        return [shape_reward_fn(fn, kind=kind, strength=strength) for fn in reward_funcs]
    return shape_reward_fn(reward_funcs, kind=kind, strength=strength)


def explain_giveup(
    state: ControllerState, *, signal_name: str, action_history: Sequence[str]
) -> str:
    """Plain-English explanation of why the controller gave up (mirrors why.py).

    Names the signal, how long it stayed elevated, and the actions tried, so
    ``soup diagnose`` / ``soup why`` can surface it to the operator.
    """
    lines = [
        "Reward-hacking mitigation gave up: the controller could not suppress "
        "the hacking signal.",
        f"It exhausted {state.recovery_attempts} recovery attempt(s) and then "
        "early-stopped training.",
        f"The '{signal_name}' signal stayed elevated (last smoothed drop_pct="
        f"{state.last_signal:.3f}).",
    ]
    recent = [str(a) for a in list(action_history)[-5:]]
    if recent:
        lines.append("Recent actions tried: " + " | ".join(recent))
    lines.append(
        "Next steps: use a stronger / ensemble reward model, enable reward "
        "shaping on the gamed proxy, or review the reward function for a "
        "gameable shortcut."
    )
    return "\n".join(lines)


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


# --- bang-bang controller (Stage 1) ---


@dataclass(frozen=True)
class MitigationAction:
    """One controller step's decision (frozen).

    - ``new_beta``: the β / kl_coef the trainer should apply next step.
    - ``tripped``: whether the controller is in the raised band.
    - ``verdict``: OK / WARN / HACK from the classifier over the vote.
    - ``reason``: human-readable summary for the mitigation log.
    """

    new_beta: float
    tripped: bool
    verdict: str
    reason: str


@dataclass(frozen=True)
class BangBangPolicy:
    """Reversible bang-bang + hysteresis controller policy.

    Two bands with a dead-zone between them: the controller wants to RAISE β
    when the vote is at/above ``trip_band`` and RELAX when at/below
    ``release_band``. ``dwell_steps`` consecutive want-raise steps are required
    before the first trip, and ``release_patience`` consecutive want-relax
    steps before relaxing — so a signal that flaps across the bands does not
    flap β. β moves geometrically by ``kl_gain`` and is clamped to
    ``[beta_floor, beta_ceil]`` (never crossing 0).
    """

    beta_floor: float
    beta_ceil: float
    trip_band: float
    release_band: float
    dwell_steps: int
    release_patience: int
    kl_gain: float

    def __post_init__(self) -> None:
        _check_finite_float(self.beta_floor, "beta_floor", nonneg=True)
        _check_finite_float(self.beta_ceil, "beta_ceil", nonneg=True)
        _check_finite_float(self.trip_band, "trip_band", nonneg=True)
        _check_finite_float(self.release_band, "release_band", nonneg=True)
        _check_finite_float(self.kl_gain, "kl_gain", nonneg=True)
        _check_nonneg_int(self.dwell_steps, "dwell_steps")
        _check_nonneg_int(self.release_patience, "release_patience")
        if self.beta_floor <= 0.0:
            raise ValueError(f"beta_floor must be > 0, got {self.beta_floor}")
        if self.beta_floor >= self.beta_ceil:
            raise ValueError(
                f"beta_floor ({self.beta_floor}) must be < "
                f"beta_ceil ({self.beta_ceil})"
            )
        if not 0.0 <= self.release_band < self.trip_band <= 1.0:
            raise ValueError(
                "require 0 <= release_band < trip_band <= 1, got "
                f"release_band={self.release_band}, trip_band={self.trip_band}"
            )
        if self.dwell_steps < 1:
            raise ValueError("dwell_steps must be >= 1")
        if self.release_patience < 1:
            raise ValueError("release_patience must be >= 1")
        if self.kl_gain <= 1.0:
            raise ValueError(f"kl_gain must be > 1, got {self.kl_gain}")


def _verdict_for(vote: float) -> str:
    from soup_cli.utils.reward_hacking import classify_hack_signal

    return classify_hack_signal(min(1.0, max(0.0, float(vote))))


def bang_bang_step(
    policy: BangBangPolicy, state: ControllerState, *, vote: float
) -> tuple[ControllerState, MitigationAction]:
    """Advance the bang-bang controller one step given the multi-signal ``vote``.

    ``vote`` is the combined hacking signal in ``[0, 1]`` (see
    :func:`combine_signals`). Returns the next :class:`ControllerState` and the
    :class:`MitigationAction` (the β the trainer should apply).
    """
    fvote = float(vote)
    beta = state.beta if state.beta > 0.0 else policy.beta_floor
    tripped = state.tripped
    dwell = state.dwell_count
    release = state.release_count
    reason = "hold"

    if fvote >= policy.trip_band:
        release = 0
        if not tripped:
            dwell += 1
            if dwell >= policy.dwell_steps:
                beta = min(policy.beta_ceil, beta * policy.kl_gain)
                tripped = True
                dwell = 0
                reason = f"trip: raise beta to {beta:.4f} (vote={fvote:.3f})"
        else:
            new_beta = min(policy.beta_ceil, beta * policy.kl_gain)
            if new_beta != beta:
                reason = f"raise beta to {new_beta:.4f} (vote={fvote:.3f})"
            beta = new_beta
    elif fvote <= policy.release_band:
        dwell = 0
        if tripped:
            release += 1
            if release >= policy.release_patience:
                new_beta = max(policy.beta_floor, beta / policy.kl_gain)
                if new_beta != beta:
                    reason = f"relax beta to {new_beta:.4f} (vote={fvote:.3f})"
                beta = new_beta
                if beta <= policy.beta_floor:
                    tripped = False
                    release = 0
        else:
            release = 0
    else:
        # dead-band: hold and decay both counters (hysteresis).
        dwell = 0
        release = 0

    new_state = replace(
        state,
        step=state.step + 1,
        beta=beta,
        tripped=tripped,
        dwell_count=dwell,
        release_count=release,
        last_signal=min(1.0, max(0.0, fvote)),
    )
    action = MitigationAction(
        new_beta=beta,
        tripped=tripped,
        verdict=_verdict_for(fvote),
        reason=reason,
    )
    return new_state, action


# --- PID-Lagrangian controller (Stage 2) ---


@dataclass(frozen=True)
class PIDLagrangianPolicy:
    """PID-Lagrangian controller policy (Stooke et al. 2020).

    Treats "hacking signal ≤ ``signal_target``" as a constraint whose Lagrange
    multiplier (the β / kl_coef) is updated by a PID law on the constraint
    violation ``error = signal - target``. The integral term is clamped
    (anti-windup) and the output is clamped to ``[beta_floor, beta_ceil]`` and
    never crosses 0.
    """

    kp: float
    ki: float
    kd: float
    signal_target: float
    beta_floor: float
    beta_ceil: float
    integral_clamp: float

    def __post_init__(self) -> None:
        _check_finite_float(self.kp, "kp", nonneg=True)
        _check_finite_float(self.ki, "ki", nonneg=True)
        _check_finite_float(self.kd, "kd", nonneg=True)
        _check_finite_float(self.signal_target, "signal_target", nonneg=True)
        _check_finite_float(self.beta_floor, "beta_floor", nonneg=True)
        _check_finite_float(self.beta_ceil, "beta_ceil", nonneg=True)
        _check_finite_float(self.integral_clamp, "integral_clamp", nonneg=True)
        if not 0.0 <= self.signal_target < 1.0:
            raise ValueError(
                f"signal_target must be in [0, 1), got {self.signal_target}"
            )
        if self.beta_floor <= 0.0:
            raise ValueError(f"beta_floor must be > 0, got {self.beta_floor}")
        if self.beta_floor >= self.beta_ceil:
            raise ValueError(
                f"beta_floor ({self.beta_floor}) must be < "
                f"beta_ceil ({self.beta_ceil})"
            )
        if self.integral_clamp <= 0.0:
            raise ValueError(
                f"integral_clamp must be > 0, got {self.integral_clamp}"
            )


def pid_step(
    policy: PIDLagrangianPolicy, state: ControllerState, *, signal: float
) -> tuple[ControllerState, MitigationAction]:
    """Advance the PID-Lagrangian controller one step for the hacking ``signal``.

    ``error = signal - target``; the integral accumulates (clamped ±
    ``integral_clamp``); the multiplier β = clamp(floor..ceil, floor + Kp·error
    + Ki·∫error + Kd·Δerror). β never crosses 0.
    """
    fsignal = float(signal)
    error = fsignal - policy.signal_target
    integral = state.integral + error
    integral = max(-policy.integral_clamp, min(policy.integral_clamp, integral))
    derivative = error - state.prev_error
    control = policy.kp * error + policy.ki * integral + policy.kd * derivative
    beta = max(policy.beta_floor, min(policy.beta_ceil, policy.beta_floor + control))
    tripped = beta > policy.beta_floor

    new_state = replace(
        state,
        step=state.step + 1,
        beta=beta,
        tripped=tripped,
        integral=integral,
        prev_error=error,
        last_signal=min(1.0, max(0.0, fsignal)),
    )
    action = MitigationAction(
        new_beta=beta,
        tripped=tripped,
        verdict=_verdict_for(fsignal),
        reason=f"pid: error={error:.3f} integral={integral:.3f} beta={beta:.4f}",
    )
    return new_state, action


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
        bang_bang: BangBangPolicy | None = None,
        pid: PIDLagrangianPolicy | None = None,
        rollback: bool = False,
        rollback_patience: int = 3,
        max_recovery_attempts: int = 2,
        rl_checkpoint_cb: Any = None,
        smoothing: str = "none",
        smoothing_window: int = 8,
        conservative_on_disagreement: bool = False,
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
        self.bang_bang = bang_bang
        if mode == "kl_control" and bang_bang is None:
            raise ValueError("kl_control mode requires a BangBangPolicy")
        self.pid = pid
        if mode == "pid_lagrangian" and pid is None:
            raise ValueError("pid_lagrangian mode requires a PIDLagrangianPolicy")
        self.rollback = bool(rollback)
        self.rollback_patience = int(rollback_patience)
        self.max_recovery_attempts = int(max_recovery_attempts)
        self.rl_checkpoint_cb = rl_checkpoint_cb
        if smoothing not in SMOOTHING_METHODS:
            raise ValueError(
                f"smoothing must be one of {sorted(SMOOTHING_METHODS)}, "
                f"got {smoothing!r}"
            )
        self.smoothing = smoothing
        self.smoothing_window = int(smoothing_window)
        self.conservative_on_disagreement = bool(conservative_on_disagreement)
        # Compose the v0.70.0 detector callback for the info_rm/rm_ensemble
        # baseline + drop_pct logic (DRY — no re-implementation).
        self._detector_cb = RewardHackCallback(
            detector=self.detector, halt_on_hack=False, buffer=buffer
        )
        self._trainer: Any = None
        self._state = ControllerState()
        self._length_baseline: float | None = None
        self._hack_streak = 0
        self._last_good_step: int | None = None
        self._signal_windows: dict[str, list[float]] = {}
        self._action_history: list[str] = []
        self._last_drift = False

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

    def _length_trend(self, length_mean: float) -> float:
        """Relative growth of the mean completion length vs its baseline, in
        ``[0, 1]``. Rising = the policy is padding output (length hacking)."""
        if self._length_baseline is None:
            if length_mean > 0:
                self._length_baseline = length_mean
            return 0.0
        base = self._length_baseline
        if base <= 0:
            return 0.0
        return min(1.0, max(0.0, (length_mean - base) / base))

    def _observe(
        self, snapshot: Mapping[str, Any], step: int
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Compute the telemetry entry + the per-signal vote inputs."""
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
        length_mean = mean_token_len(completions)
        repetition = mean_repetition(completions)
        length_trend = self._length_trend(length_mean)
        signals = {
            self.detector: drop_pct,
            "length_trend": length_trend,
            "repetition": repetition,
        }
        telemetry = {
            "mode": self.mode,
            "detector": self.detector,
            "raw_signal": raw,
            "drop_pct": drop_pct,
            "verdict": verdict,
            "beta": self._current_coefficient(),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "completion_length_mean": length_mean,
            "repetition": repetition,
            "length_trend": length_trend,
        }
        # Stage 3 — reward-distribution drift guard is opt-in (conservative
        # mode) since the snapshot detector can't distinguish a healthy
        # well-separated distribution from a gamed bimodal collapse.
        self._last_drift = False
        if self.conservative_on_disagreement:
            self._last_drift = detect_reward_distribution_drift(rewards)
            telemetry["drift"] = self._last_drift
        return telemetry, signals

    def _compute_vote(self, signals: Mapping[str, float]) -> float:
        """Combine the enabled per-signal drops into the controller vote,
        applying smoothing + conservative-on-disagreement + the drift guard."""
        signals_for_vote = dict(signals)
        if self.smoothing != "none":
            for name in self.signals:
                if name not in signals:
                    continue
                window = self._signal_windows.setdefault(name, [])
                signals_for_vote[name] = smooth_signal(
                    signals[name], window, method=self.smoothing
                )
                window.append(float(signals[name]))
                if len(window) > self.smoothing_window:
                    del window[0]
        if self.conservative_on_disagreement:
            votes = [signals_for_vote[n] for n in self.signals if n in signals_for_vote]
            vote = combine_conservative(votes, disagree_tol=_CONSERVATIVE_DISAGREE_TOL)
            if self._last_drift:
                # suspected distribution-shift attack — don't relax.
                vote = max(vote, self._state.last_signal)
        else:
            vote = combine_signals(signals_for_vote, self.signals)
        return vote

    def _apply_coefficient(self, value: float) -> None:
        """Write the controller's coefficient to the trainer.

        GRPO: β must be dual-written — stock ``GRPOTrainer.compute_loss`` reads
        ``self.beta`` (the instance) while Soup's ``_GRPOTrainerVariant`` reads
        ``self.args.beta`` (the config). PPO: ``args.kl_coef``.
        """
        trainer = self._trainer
        if trainer is None:
            return
        args = getattr(trainer, "args", None)
        if self.task == "ppo":
            if args is not None:
                try:
                    args.kl_coef = value
                except Exception:  # noqa: BLE001 — never crash training
                    pass
            return
        try:
            trainer.beta = value
        except Exception:  # noqa: BLE001
            pass
        if args is not None:
            try:
                args.beta = value
            except Exception:  # noqa: BLE001
                pass

    def _seed_coefficient(self, floor: float) -> None:
        """Seed the controller β from the live trainer coefficient on step 1."""
        if self._state.beta > 0.0:
            return
        current = self._current_coefficient()
        seed = (
            float(current)
            if isinstance(current, (int, float))
            and not isinstance(current, bool)
            and current > 0
            else floor
        )
        self._state = replace(self._state, beta=seed)

    def _run_bang_bang(
        self, telemetry: dict[str, Any], signals: Mapping[str, float]
    ) -> None:
        """kl_control: vote → bang-bang step → mutate the trainer coefficient."""
        policy = self.bang_bang
        if policy is None:
            return
        self._seed_coefficient(policy.beta_floor)
        vote = self._compute_vote(signals)
        new_state, action = bang_bang_step(policy, self._state, vote=vote)
        self._state = new_state
        self._apply_coefficient(action.new_beta)
        self._action_history.append(action.reason)
        telemetry["vote"] = vote
        telemetry["new_beta"] = action.new_beta
        telemetry["tripped"] = action.tripped
        telemetry["action"] = action.reason

    def _request_stop(self, control: Any) -> None:
        if control is not None:
            try:
                control.should_training_stop = True
            except Exception:  # noqa: BLE001
                pass

    def _escalate(
        self, model: Any, optimizer: Any, control: Any, telemetry: dict[str, Any]
    ) -> Any:
        """Escalation ladder rung: rollback to last-good, else early-stop."""
        if self._state.recovery_attempts >= self.max_recovery_attempts:
            telemetry["escalation"] = "early_stop"
            telemetry["explanation"] = explain_giveup(
                self._state,
                signal_name=self.detector,
                action_history=self._action_history,
            )
            self._request_stop(control)
            return control
        target = self._last_good_step
        restored = False
        if target is not None and self.rl_checkpoint_cb is not None:
            try:
                restored = bool(
                    self.rl_checkpoint_cb.restore_checkpoint(
                        step=target, model=model, optimizer=optimizer
                    )
                )
            except Exception:  # noqa: BLE001 — rollback must never crash the run
                restored = False
        self._state = replace(
            self._state, recovery_attempts=self._state.recovery_attempts + 1
        )
        self._hack_streak = 0
        telemetry["escalation"] = f"rollback to step {target} (restored={restored})"
        return control

    def _run_pid(
        self,
        telemetry: dict[str, Any],
        signals: Mapping[str, float],
        model: Any,
        optimizer: Any,
        control: Any,
    ) -> Any:
        """pid_lagrangian: PID β update + rollback escalation ladder."""
        policy = self.pid
        if policy is None:
            return control
        self._seed_coefficient(policy.beta_floor)
        vote = self._compute_vote(signals)
        new_state, action = pid_step(policy, self._state, signal=vote)
        self._state = new_state
        self._apply_coefficient(action.new_beta)
        self._action_history.append(action.reason)
        telemetry["vote"] = vote
        telemetry["new_beta"] = action.new_beta
        telemetry["tripped"] = action.tripped
        telemetry["action"] = action.reason
        # Escalation ladder: raise (above) → rollback → early-stop.
        if action.verdict == "HACK":
            self._hack_streak += 1
        else:
            self._hack_streak = 0
            saved = getattr(self.rl_checkpoint_cb, "_saved", None)
            if saved:
                self._last_good_step = max(saved)
        if self.rollback and self._hack_streak >= self.rollback_patience:
            control = self._escalate(model, optimizer, control, telemetry)
        return control

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
            telemetry, signals = self._observe(snapshot, step)
            # log_only observes; kl_control drives bang-bang; pid_lagrangian
            # drives the PID controller + rollback escalation ladder.
            if self.mode == "kl_control":
                self._run_bang_bang(telemetry, signals)
            elif self.mode == "pid_lagrangian":
                control = self._run_pid(
                    telemetry,
                    signals,
                    kwargs.get("model"),
                    kwargs.get("optimizer"),
                    control,
                )
            self.log_writer.record(step=step, snapshot=telemetry)
            return control
        except Exception:  # noqa: BLE001 — instrumentation must never crash
            return control
