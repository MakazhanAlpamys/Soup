"""Reward-hacking detector — v0.70.0 Part A.

Closed allowlist of reward-hacking detectors for GRPO/PPO training.
Surfaces an early-warning signal when the policy starts gaming the
reward model rather than improving the underlying capability.

Two detectors ship in v0.70.0 (schema-only; live trainer callback wired
in v0.70.1):

- ``info_rm``: Cluster-Separation Index over (good, bad) response
  reward distributions. Drops in cluster separation across training =
  reward model losing its grip. Inspired by Wang et al. 2024
  "InfoRM: Mitigating Reward Hacking via Information-Theoretic Reward
  Modeling" (arXiv 2402.09345).

- ``rm_ensemble``: pairwise variance across an ensemble of RMs.
  Rising disagreement = unreliable reward signal.

Live wiring deferred to v0.70.1 — mirrors v0.27.0 MII / v0.50.0
GRPO Plus / v0.61.0 unlearning stub-then-live pattern.

Security:
- Closed allowlist (frozenset); arbitrary detector name rejected.
- Bool / null-byte / non-string / oversize rejection on every public
  validator (mirrors v0.41.0 / v0.51.0 / v0.62.0 policy).
- ``_DETECTOR_METADATA`` wrapped in ``MappingProxyType`` for runtime
  immutability (matches v0.36.0 / v0.41.0 / v0.50.0 / v0.62.0 policy).
- Math kernels never import torch at module top — lazy imports only.
- All bound-check rejections include actionable messages naming the
  field + offending value.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Optional, Sequence

_MAX_DETECTOR_NAME_LEN = 32
_MAX_RM_ENSEMBLE_SIZE = 32
_EPS = 1e-9

SUPPORTED_HACK_DETECTORS: frozenset[str] = frozenset({"info_rm", "rm_ensemble"})

VERDICTS: tuple[str, ...] = ("OK", "WARN", "HACK")
_VALID_VERDICTS: frozenset[str] = frozenset(VERDICTS)

# OK / WARN / HACK boundaries on the relative drop in cluster separation.
# These match the v0.26 Quant-Lobotomy + v0.56 diagnose three-band taxonomy
# (OK / MINOR / MAJOR) but renamed for the RL setting.
_HACK_OK_BAND = 0.10  # drop < 10% → OK
_HACK_WARN_BAND = 0.30  # 10% ≤ drop < 30% → WARN; else HACK


@dataclass(frozen=True)
class HackDetectorSpec:
    """Static metadata for a reward-hacking detector."""

    name: str
    description: str
    paper: str


_DETECTOR_METADATA = types.MappingProxyType({
    "info_rm": HackDetectorSpec(
        name="info_rm",
        description=(
            "InfoRM Cluster-Separation Index over (good, bad) reward "
            "distributions. Drop across training = reward-hacking signal."
        ),
        paper="Wang et al. 2024 — arXiv:2402.09345",
    ),
    "rm_ensemble": HackDetectorSpec(
        name="rm_ensemble",
        description=(
            "Pairwise variance across an RM ensemble. Rising disagreement "
            "= unreliable reward signal."
        ),
        paper="Coste et al. 2024 — arXiv:2312.09244",
    ),
})


def validate_hack_detector(name: object) -> str:
    """Validate and normalise a reward-hacking detector name.

    Returns the canonical (lower-cased) name on success. Raises
    ``ValueError`` with an actionable message on any failure.
    """
    if isinstance(name, bool):
        raise ValueError("reward_hack_detector must be a string, got bool")
    if not isinstance(name, str):
        raise ValueError(
            f"reward_hack_detector must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("reward_hack_detector must be a non-empty string")
    if "\x00" in name:
        raise ValueError("reward_hack_detector must not contain null bytes")
    if len(name) > _MAX_DETECTOR_NAME_LEN:
        raise ValueError(
            f"reward_hack_detector exceeds {_MAX_DETECTOR_NAME_LEN} chars"
        )
    normalised = name.lower()
    if normalised not in SUPPORTED_HACK_DETECTORS:
        raise ValueError(
            f"reward_hack_detector={name!r} is not supported. "
            f"Valid: {sorted(SUPPORTED_HACK_DETECTORS)}"
        )
    return normalised


def get_detector_spec(name: str) -> HackDetectorSpec:
    """Return the :class:`HackDetectorSpec` for ``name``."""
    normalised = validate_hack_detector(name)
    return _DETECTOR_METADATA[normalised]


def _check_finite_float_sequence(values: object, field: str) -> tuple[float, ...]:
    """Validate a numeric sequence: non-empty, finite, no bool."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of numbers, not str/bytes")
    try:
        iterator = iter(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{field} must be iterable, got {type(values).__name__}"
        ) from exc
    out: list[float] = []
    for idx, v in enumerate(iterator):
        if isinstance(v, bool):
            raise ValueError(
                f"{field}[{idx}] must not be bool"
            )
        if not isinstance(v, (int, float)):
            raise ValueError(
                f"{field}[{idx}] must be a number, got {type(v).__name__}"
            )
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"{field}[{idx}] must be finite (no NaN/Inf)")
        out.append(fv)
    if not out:
        raise ValueError(f"{field} must not be empty")
    return tuple(out)


def _mean(seq: Sequence[float]) -> float:
    return sum(seq) / len(seq)


def _variance(seq: Sequence[float]) -> float:
    """Population variance (n divisor, not n-1) for stability with N=1."""
    m = _mean(seq)
    return sum((x - m) ** 2 for x in seq) / len(seq)


def compute_cluster_separation(
    good_scores: object,
    bad_scores: object,
) -> float:
    """InfoRM-style cluster-separation index.

    Returns ``(mean_good - mean_bad) / sqrt(var_good + var_bad + eps)``.
    Larger = better-separated reward clusters. Watch for a sharp drop
    across training steps — that signals the RM losing its grip.

    Both arguments must be non-empty iterables of finite numbers.
    ``bool`` rejected per project bool-as-int policy.
    """
    good = _check_finite_float_sequence(good_scores, "good_scores")
    bad = _check_finite_float_sequence(bad_scores, "bad_scores")
    delta = _mean(good) - _mean(bad)
    pooled = _variance(good) + _variance(bad) + _EPS
    return delta / math.sqrt(pooled)


def compute_rm_ensemble_divergence(rm_scores: object) -> float:
    """Mean pairwise variance across a small RM ensemble.

    Input is a sequence of per-RM score lists; every inner list must be
    the same length (one score per prompt, aligned across RMs).
    Returns the mean of the per-prompt variance over RMs. Higher =
    RMs disagree more = reward signal less reliable.

    Bounds:
    - ensemble size in [2, _MAX_RM_ENSEMBLE_SIZE=32]
    - all inner lists same length (≥ 1)
    - every value finite + non-bool
    """
    if isinstance(rm_scores, (str, bytes)):
        raise TypeError("rm_scores must be a sequence of sequences, not str/bytes")
    try:
        outer = list(rm_scores)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"rm_scores must be iterable, got {type(rm_scores).__name__}"
        ) from exc
    if len(outer) < 2:
        raise ValueError("rm_scores requires at least 2 RMs (ensemble divergence)")
    if len(outer) > _MAX_RM_ENSEMBLE_SIZE:
        raise ValueError(
            f"rm_scores has too many RMs (>{_MAX_RM_ENSEMBLE_SIZE} cap)"
        )
    per_rm: list[tuple[float, ...]] = []
    expected_len: Optional[int] = None
    for idx, inner in enumerate(outer):
        seq = _check_finite_float_sequence(inner, f"rm_scores[{idx}]")
        if expected_len is None:
            expected_len = len(seq)
        elif len(seq) != expected_len:
            raise ValueError(
                f"rm_scores[{idx}] has length {len(seq)} but expected "
                f"{expected_len} (all RM score lists must share length)"
            )
        per_rm.append(seq)
    # Per-prompt variance over RMs, then mean across prompts.
    assert expected_len is not None
    per_prompt_var: list[float] = []
    for j in range(expected_len):
        column = [rm[j] for rm in per_rm]
        per_prompt_var.append(_variance(column))
    return _mean(per_prompt_var) if per_prompt_var else 0.0


def classify_hack_signal(drop_pct: object) -> str:
    """Map a relative drop in cluster-separation to OK / WARN / HACK.

    ``drop_pct`` is the *relative* drop:
    ``(baseline_signal - current_signal) / baseline_signal``. Non-negative
    by definition; the caller clamps to 0 when the RM improves.

    Boundaries (matches v0.26 Quant-Lobotomy / v0.56 diagnose taxonomy):
    - drop_pct ∈ [0, _HACK_OK_BAND=0.10): OK
    - drop_pct ∈ [_HACK_OK_BAND, _HACK_WARN_BAND=0.30): WARN
    - drop_pct >= _HACK_WARN_BAND: HACK
    """
    if isinstance(drop_pct, bool):
        raise ValueError("drop_pct must not be bool")
    if not isinstance(drop_pct, (int, float)):
        raise ValueError(
            f"drop_pct must be a number, got {type(drop_pct).__name__}"
        )
    fv = float(drop_pct)
    if not math.isfinite(fv):
        raise ValueError("drop_pct must be finite (no NaN/Inf)")
    if fv < 0.0:
        raise ValueError(f"drop_pct must be non-negative, got {fv}")
    if fv < _HACK_OK_BAND:
        return "OK"
    if fv < _HACK_WARN_BAND:
        return "WARN"
    return "HACK"


@dataclass(frozen=True)
class RewardHackReport:
    """Frozen result of a reward-hacking probe.

    - ``detector``: which detector produced the signal (allowlist).
    - ``signal``: the raw scalar (cluster-sep value or ensemble variance).
      Non-negative + finite.
    - ``verdict``: OK / WARN / HACK per :func:`classify_hack_signal`.
    - ``step``: training step at which the probe fired. Non-negative int,
      bool rejected per project bool-as-int policy.
    - ``baseline_signal``: the reference signal recorded at the start of
      training. Used to compute the relative drop. Finite, non-negative.
    - ``details``: tuple of human-readable lines for the report panel.
    """

    detector: str
    signal: float
    verdict: str
    step: int
    baseline_signal: float
    details: tuple[str, ...]

    def __post_init__(self) -> None:
        # validate_hack_detector normalises; re-write via object.__setattr__
        # so frozen + canonical-case invariants stay together.
        normalised = validate_hack_detector(self.detector)
        if normalised != self.detector:
            object.__setattr__(self, "detector", normalised)
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(
                f"verdict={self.verdict!r} must be one of {sorted(_VALID_VERDICTS)}"
            )
        if isinstance(self.signal, bool):
            raise ValueError("signal must not be bool")
        if not isinstance(self.signal, (int, float)):
            raise TypeError(
                f"signal must be a number, got {type(self.signal).__name__}"
            )
        if not math.isfinite(float(self.signal)):
            raise ValueError("signal must be finite")
        if float(self.signal) < 0.0:
            raise ValueError(f"signal must be non-negative, got {self.signal}")
        if isinstance(self.step, bool):
            raise ValueError("step must not be bool")
        if not isinstance(self.step, int):
            raise TypeError(f"step must be int, got {type(self.step).__name__}")
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
        if isinstance(self.baseline_signal, bool):
            raise ValueError("baseline_signal must not be bool")
        if not isinstance(self.baseline_signal, (int, float)):
            raise TypeError("baseline_signal must be a number")
        if not math.isfinite(float(self.baseline_signal)):
            raise ValueError("baseline_signal must be finite")
        if float(self.baseline_signal) < 0.0:
            raise ValueError("baseline_signal must be non-negative")
        if not isinstance(self.details, tuple):
            raise TypeError(
                f"details must be a tuple, got {type(self.details).__name__}"
            )


def build_reward_hack_callback(
    *,
    detector: str,
    halt_on_hack: bool = True,
    baseline_signal: Optional[float] = None,
):
    """Live HF Trainer callback factory — deferred to v0.70.1.

    Validates inputs at construction time so misconfigured runs fail
    fast even though the live callback is not yet wired. Raises
    ``NotImplementedError`` with explicit v0.70.1 marker after
    validation (mirrors v0.50.0 ``apply_variant_loss`` / v0.61.0
    ``apply_unlearn_loss`` policy).
    """
    # Validation order: name first (cheap, allowlist) then bool guard.
    validate_hack_detector(detector)
    if not isinstance(halt_on_hack, bool):
        raise TypeError(
            f"halt_on_hack must be bool, got {type(halt_on_hack).__name__}"
        )
    if baseline_signal is not None:
        if isinstance(baseline_signal, bool):
            raise TypeError("baseline_signal must not be bool")
        if not isinstance(baseline_signal, (int, float)):
            raise TypeError("baseline_signal must be a number or None")
        if not math.isfinite(float(baseline_signal)) or float(baseline_signal) < 0.0:
            raise ValueError("baseline_signal must be finite and non-negative")
    raise NotImplementedError(
        f"Live reward-hack callback for detector={detector!r} is deferred to "
        "v0.70.1. v0.70.0 ships the schema + math kernels only."
    )
