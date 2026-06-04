"""Live echo-trap detector — v0.70.0 Part F.

RAGEN-style detection of trajectory degeneration during multi-turn
agent RL (Zhu et al. 2025, arXiv:2504.14437). When the policy collapses
to self-repeating outputs, the reward saturates and the policy drifts
without learning. This module ships the math kernels + report schema;
the live HF Trainer callback is deferred to v0.70.1.

Composes with v0.53.11 #127 ``GRPOStabilityCallback`` — the live
echo-trap callback shares the per-step instrumentation hook so both
detectors can fire in the same training step without duplicating
trajectory collection.

Security:
- Pure-Python math (no torch import at module top).
- Bool / NaN / Inf / range rejection on every numeric input.
- Whitespace-mode tokens must be strings; tokeniser-aware mode accepts
  integer token ids through the dedicated ``*_tokenized`` helpers.
- ``_MAX_BATCH_TRAJECTORIES = 100_000`` DoS cap (matches v0.55 /
  v0.65 / v0.66 cap policy).
- ``_MAX_NGRAM_N = 32`` keeps the n-gram counter bounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

VERDICTS: tuple[str, ...] = ("OK", "WARN", "TRAP")
_VALID_VERDICTS: frozenset[str] = frozenset(VERDICTS)

# OK / WARN / TRAP boundaries on the aggregate echo signal. Mirrors v0.26
# Quant-Lobotomy + v0.56 diagnose three-band taxonomy.
_ECHO_OK_BAND = 0.30  # signal < 0.30 → OK
_ECHO_TRAP_BAND = 0.60  # signal >= 0.60 → TRAP; in between → WARN

_MAX_NGRAM_N = 32
_MAX_TRAJECTORY_TOKENS = 1_000_000
_MAX_BATCH_TRAJECTORIES = 100_000


def _check_ngram_n(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("ngram_n must not be bool")
    if not isinstance(value, int):
        raise ValueError(f"ngram_n must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"ngram_n must be >= 1, got {value}")
    if value > _MAX_NGRAM_N:
        raise ValueError(f"ngram_n={value} exceeds {_MAX_NGRAM_N} cap")
    return value


def _check_tokens(tokens: object) -> tuple[str, ...]:
    if isinstance(tokens, (str, bytes)):
        raise TypeError("tokens must be a sequence of strings, not str/bytes")
    try:
        iterator = list(tokens)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"tokens must be iterable, got {type(tokens).__name__}"
        ) from exc
    if len(iterator) > _MAX_TRAJECTORY_TOKENS:
        raise ValueError(
            f"trajectory has {len(iterator)} tokens, exceeds "
            f"{_MAX_TRAJECTORY_TOKENS} cap"
        )
    for idx, t in enumerate(iterator):
        if not isinstance(t, str):
            raise TypeError(
                f"tokens[{idx}] must be str, got {type(t).__name__}"
            )
    return tuple(iterator)


def _check_token_ids(token_ids: object) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)):
        raise TypeError("token_ids must be a sequence of ints, not str/bytes")
    try:
        iterator = list(token_ids)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"token_ids must be iterable, got {type(token_ids).__name__}"
        ) from exc
    if len(iterator) > _MAX_TRAJECTORY_TOKENS:
        raise ValueError(
            f"trajectory has {len(iterator)} token ids, exceeds "
            f"{_MAX_TRAJECTORY_TOKENS} cap"
        )
    for idx, token_id in enumerate(iterator):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(
                f"token_ids[{idx}] must be int, got {type(token_id).__name__}"
            )
    return tuple(iterator)


def _score_repetition(units: Sequence[object], *, ngram_n: int) -> float:
    if len(units) < ngram_n:
        return 0.0
    counts: dict[tuple[object, ...], int] = {}
    for i in range(len(units) - ngram_n + 1):
        gram = tuple(units[i : i + ngram_n])
        counts[gram] = counts.get(gram, 0) + 1
    if not counts:
        return 0.0
    repeating = sum(1 for count in counts.values() if count > 1)
    return repeating / len(counts)


def score_trajectory_repetition(tokens: object, *, ngram_n: object = 2) -> float:
    """Per-trajectory repetition score.

    Returns the fraction of n-grams whose count exceeds 1 (the
    "repeating n-grams" rate). Range ``[0, 1]``. 0 = every n-gram
    unique; closer to 1 = many n-grams repeat.

    Edge cases:
    - ``len(tokens) < ngram_n`` returns 0.0 (no n-grams possible).
    - Empty input returns 0.0.
    """
    n = _check_ngram_n(ngram_n)
    tok = _check_tokens(tokens)
    return _score_repetition(tok, ngram_n=n)


def score_trajectory_repetition_tokenized(
    token_ids: object,
    *,
    ngram_n: object = 2,
) -> float:
    """Per-trajectory repetition score over tokenizer ids.

    This mirrors :func:`score_trajectory_repetition`, but operates on
    integer token ids before decoding/whitespace splitting can hide
    subword repetition. It is intentionally separate so the existing
    string-token API keeps rejecting accidental tensor-id input.
    """
    n = _check_ngram_n(ngram_n)
    ids = _check_token_ids(token_ids)
    return _score_repetition(ids, ngram_n=n)


def score_echo_signal(
    trajectories: object,
    *,
    ngram_n: object = 2,
) -> float:
    """Mean repetition score across a batch of trajectories.

    Higher = more trajectory degeneration = closer to echo trap.
    Returns 0.0 on empty input (no signal = nothing to flag).
    """
    n = _check_ngram_n(ngram_n)
    if isinstance(trajectories, (str, bytes)):
        raise TypeError(
            "trajectories must be a sequence of sequences, not str/bytes"
        )
    try:
        batch = list(trajectories)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"trajectories must be iterable, got "
            f"{type(trajectories).__name__}"
        ) from exc
    if len(batch) > _MAX_BATCH_TRAJECTORIES:
        raise ValueError(
            f"batch has {len(batch)} trajectories, exceeds "
            f"{_MAX_BATCH_TRAJECTORIES} cap"
        )
    if not batch:
        return 0.0
    scores: list[float] = []
    for traj in batch:
        scores.append(score_trajectory_repetition(traj, ngram_n=n))
    return sum(scores) / len(scores)


def score_echo_signal_tokenized(
    trajectories: object,
    *,
    ngram_n: object = 2,
) -> float:
    """Mean repetition score across token-id trajectories.

    Use this when the caller has access to the trainer tokenizer and can
    pass ``tokenizer.encode(text)`` output rather than decoded strings.
    """
    n = _check_ngram_n(ngram_n)
    if isinstance(trajectories, (str, bytes)):
        raise TypeError(
            "trajectories must be a sequence of token-id sequences, not str/bytes"
        )
    try:
        batch = list(trajectories)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"trajectories must be iterable, got "
            f"{type(trajectories).__name__}"
        ) from exc
    if len(batch) > _MAX_BATCH_TRAJECTORIES:
        raise ValueError(
            f"batch has {len(batch)} trajectories, exceeds "
            f"{_MAX_BATCH_TRAJECTORIES} cap"
        )
    if not batch:
        return 0.0
    scores: list[float] = []
    for traj in batch:
        scores.append(score_trajectory_repetition_tokenized(traj, ngram_n=n))
    return sum(scores) / len(scores)


def classify_echo_signal(signal: object) -> str:
    """Map a signal in ``[0, 1]`` to OK / WARN / TRAP.

    - signal in [0.0, _ECHO_OK_BAND=0.30): OK
    - signal in [_ECHO_OK_BAND, _ECHO_TRAP_BAND=0.60): WARN
    - signal >= _ECHO_TRAP_BAND: TRAP
    """
    if isinstance(signal, bool):
        raise ValueError("signal must not be bool")
    if not isinstance(signal, (int, float)):
        raise ValueError(
            f"signal must be a number, got {type(signal).__name__}"
        )
    fv = float(signal)
    if not math.isfinite(fv):
        raise ValueError("signal must be finite (no NaN/Inf)")
    if not (0.0 <= fv <= 1.0):
        raise ValueError(f"signal must be in [0.0, 1.0], got {fv}")
    if fv < _ECHO_OK_BAND:
        return "OK"
    if fv < _ECHO_TRAP_BAND:
        return "WARN"
    return "TRAP"


@dataclass(frozen=True)
class EchoTrapReport:
    """Frozen result of an echo-trap probe.

    - ``signal``: aggregate echo signal in ``[0, 1]``.
    - ``verdict``: OK / WARN / TRAP per :func:`classify_echo_signal`.
    - ``step``: training step at which the probe fired. Non-negative
      int (bool rejected per project policy).
    - ``trajectories_seen``: count of trajectories that contributed to
      the signal. Non-negative.
    - ``details``: tuple of human-readable lines for the report panel.
    """

    signal: float
    verdict: str
    step: int
    trajectories_seen: int
    details: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.signal, bool):
            raise ValueError("signal must not be bool")
        if not isinstance(self.signal, (int, float)):
            raise TypeError(
                f"signal must be a number, got {type(self.signal).__name__}"
            )
        fv = float(self.signal)
        if not math.isfinite(fv) or not (0.0 <= fv <= 1.0):
            raise ValueError(f"signal must be in [0.0, 1.0], got {self.signal}")
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(
                f"verdict={self.verdict!r} must be one of {sorted(_VALID_VERDICTS)}"
            )
        if isinstance(self.step, bool):
            raise ValueError("step must not be bool")
        if not isinstance(self.step, int):
            raise TypeError(f"step must be int, got {type(self.step).__name__}")
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
        if isinstance(self.trajectories_seen, bool):
            raise ValueError("trajectories_seen must not be bool")
        if not isinstance(self.trajectories_seen, int):
            raise TypeError(
                "trajectories_seen must be int, got "
                f"{type(self.trajectories_seen).__name__}"
            )
        if self.trajectories_seen < 0:
            raise ValueError(
                f"trajectories_seen must be non-negative, got "
                f"{self.trajectories_seen}"
            )
        if not isinstance(self.details, tuple):
            raise TypeError(
                f"details must be a tuple, got {type(self.details).__name__}"
            )


def _split_whitespace(text: str) -> list[str]:
    """Whitespace tokenisation for the string echo path."""
    return text.split()


def _get_trainer_callback_base():
    """Lazy-resolve ``transformers.TrainerCallback`` (mirror v0.53.11)."""
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except ImportError:
        return object


_TrainerCallbackBase = _get_trainer_callback_base()


class EchoTrapCallback(_TrainerCallbackBase):  # type: ignore[misc, valid-type]
    """Live HF TrainerCallback for echo-trap detection (v0.71.11 #240).

    Reads the GRPO step's generated completions (via the shared
    :class:`~soup_cli.utils.rl_signal_buffer.RLSignalBuffer`), scores
    trajectory repetition, and classifies OK / WARN / TRAP. When
    ``tokenizer_aware`` and a tokenizer are supplied, scores over
    tokenizer ids (subword-repetition sensitive); otherwise whitespace
    tokens.

    The aggregate echo signal is surfaced to ``state.log_history``;
    ``halt_on_trap`` sets ``control.should_training_stop`` on TRAP.
    """

    def __init__(
        self,
        *,
        threshold: float,
        halt_on_trap: bool = True,
        ngram_n: int = 2,
        tokenizer_aware: bool = False,
        buffer: object = None,
        tokenizer: object = None,
    ) -> None:
        if isinstance(threshold, bool):
            raise ValueError("threshold must not be bool")
        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"threshold must be a number, got {type(threshold).__name__}"
            )
        fv = float(threshold)
        if not math.isfinite(fv) or not (0.0 <= fv <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        if not isinstance(halt_on_trap, bool):
            raise TypeError(
                f"halt_on_trap must be bool, got {type(halt_on_trap).__name__}"
            )
        if not isinstance(tokenizer_aware, bool):
            raise TypeError(
                "tokenizer_aware must be bool, got "
                f"{type(tokenizer_aware).__name__}"
            )
        self.threshold = fv
        self.halt_on_trap = halt_on_trap
        self.ngram_n = _check_ngram_n(ngram_n)
        self.tokenizer_aware = tokenizer_aware
        self.buffer = buffer
        self.tokenizer = tokenizer
        self._last_report: Optional[EchoTrapReport] = None
        self._traps_seen = 0

    def compute_signal(self, snapshot: dict) -> Optional[float]:
        """Compute the aggregate echo signal from a buffer snapshot.

        Returns ``None`` when no completions are available.
        """
        completions = snapshot.get("completions", []) or []
        if not completions:
            return None
        if self.tokenizer_aware and self.tokenizer is not None:
            id_trajectories: list[list[int]] = []
            for text in completions:
                try:
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                except (TypeError, ValueError):
                    ids = []
                id_trajectories.append([int(i) for i in ids])
            return score_echo_signal_tokenized(id_trajectories, ngram_n=self.ngram_n)
        trajectories = [_split_whitespace(text) for text in completions]
        return score_echo_signal(trajectories, ngram_n=self.ngram_n)

    def observe_signal(
        self, signal: float, step: int, n_trajectories: int
    ) -> EchoTrapReport:
        """Classify a signal and build the :class:`EchoTrapReport`."""
        clamped = max(0.0, min(1.0, float(signal)))
        verdict = classify_echo_signal(clamped)
        if verdict == "TRAP":
            self._traps_seen += 1
        report = EchoTrapReport(
            signal=clamped,
            verdict=verdict,
            step=max(0, int(step)),
            trajectories_seen=max(0, int(n_trajectories)),
            details=(
                f"ngram_n={self.ngram_n}",
                f"tokenizer_aware={self.tokenizer_aware}",
                f"signal={clamped:.4f} verdict={verdict} thr={self.threshold}",
            ),
        )
        self._last_report = report
        return report

    def last_report(self) -> Optional[EchoTrapReport]:
        """Return the most recent :class:`EchoTrapReport` (or None)."""
        return self._last_report

    def on_step_end(self, args, state, control, **kwargs):
        """Per-step probe — read completions from the capture buffer."""
        if self.buffer is None:
            return control
        try:
            snapshot = self.buffer.snapshot()
            signal = self.compute_signal(snapshot)
            if signal is None:
                return control
            n_traj = len(snapshot.get("completions", []) or [])
            step = int(getattr(state, "global_step", 0) or 0)
            report = self.observe_signal(signal, step, n_traj)
            log_history = getattr(state, "log_history", None)
            if log_history is not None:
                log_history.append({
                    "echo_trap_signal": report.signal,
                    "echo_trap_verdict": report.verdict,
                })
            # TRAP verdict + over-threshold → optional halt.
            if (
                report.verdict == "TRAP"
                and report.signal >= self.threshold
                and self.halt_on_trap
                and control is not None
            ):
                try:
                    control.should_training_stop = True
                except Exception:  # noqa: BLE001 — never crash training
                    pass
            return control
        except Exception:  # noqa: BLE001 — instrumentation must never crash
            return control


def build_echo_trap_callback(
    *,
    threshold: float,
    halt_on_trap: bool = True,
    ngram_n: int = 2,
    tokenizer_aware: bool = False,
    buffer: object = None,
    tokenizer: object = None,
) -> "EchoTrapCallback":
    """Build the live echo-trap HF Trainer callback (v0.71.11 #240).

    Lifts the v0.70.0 ``NotImplementedError`` stub. Validates every input
    at the public boundary (mirrors v0.50.0 / v0.61.0 fail-fast policy),
    then returns an :class:`EchoTrapCallback`.
    """
    return EchoTrapCallback(
        threshold=threshold,
        halt_on_trap=halt_on_trap,
        ngram_n=ngram_n,
        tokenizer_aware=tokenizer_aware,
        buffer=buffer,
        tokenizer=tokenizer,
    )


# Public re-exports — type hints for the v0.70.1 callback signature so
# external consumers (e.g. the GRPO stability callback) can import them
# without circular dependencies.
__all__ = [
    "VERDICTS",
    "EchoTrapCallback",
    "EchoTrapReport",
    "build_echo_trap_callback",
    "classify_echo_signal",
    "score_echo_signal",
    "score_echo_signal_tokenized",
    "score_trajectory_repetition",
    "score_trajectory_repetition_tokenized",
]


# Type aliases retained for the v0.70.1 wiring.
TrajectoryTokens = Sequence[str]
TrajectoryBatch = Iterable[TrajectoryTokens]
TokenIdTrajectory = Sequence[int]
TokenIdTrajectoryBatch = Iterable[TokenIdTrajectory]
