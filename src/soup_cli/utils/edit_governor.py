"""v0.61.0 Part D — Sequential edit governor (norm-blowup detection).

Knowledge-editing methods (ROME / MEMIT / AlphaEdit) accumulate weight
deltas with each successive edit. Past a threshold, the model's
parameter norm grows quadratically and downstream capability collapses
("norm blowup" pathology — see R-ROME / ENCORE / AlphaEdit literature).

This module ships:

* :class:`NormBlowupPolicy` — frozen thresholds + max-edit cap.
* :func:`classify_norm_blowup` — OK / WARN / BLOWUP taxonomy from a
  measured ``||W - W_base||_F`` delta.
* :func:`governor_recommend_method` — auto-switch ROME → AlphaEdit at
  the edit-count threshold or on detected blowup. AlphaEdit is already
  the survival-mode method so it's never switched away from.
* :class:`EditGovernor` — stateful per-base-model tracker. Refuses
  further edits when ``edit_count >= max_sequential_edits`` or the
  last verdict was BLOWUP.
* :class:`GovernedEditError` — raised by :meth:`EditGovernor.check_can_edit`
  on refusal so callers can distinguish governance refusals from other
  errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

# Pure-Python module without heavy deps — lift the validator import to
# module top (review MEDIUM M3) so the hot governor path doesn't pay
# repeated lazy-import cost.
from soup_cli.utils.knowledge_edit import validate_edit_method

VERDICTS: Tuple[str, ...] = ("OK", "WARN", "BLOWUP")

_MAX_THRESHOLD: float = 1e6  # Sanity cap; no reason to want a higher norm delta.
_MAX_SEQ_EDITS: int = 10_000
_MAX_BASE_LEN: int = 512


@dataclass(frozen=True)
class NormBlowupPolicy:
    """Frozen norm-blowup detection policy.

    Defaults (tuned against ROME / MEMIT 2024 reproductions):

    * ``warn_threshold`` — ``||W - W_base||_F`` Frobenius delta above
      which we surface a yellow advisory.
    * ``blowup_threshold`` — delta above which we refuse further edits.
    * ``max_sequential_edits`` — absolute upper bound on edits per
      base model before the governor refuses (defence-in-depth in case
      the norm-delta probe is unavailable).
    * ``auto_switch_at`` — ROME → AlphaEdit auto-switch at this edit
      count regardless of norm delta.
    """

    warn_threshold: float = 1.0
    blowup_threshold: float = 5.0
    max_sequential_edits: int = 50
    auto_switch_at: int = 10

    def __post_init__(self) -> None:
        for name, value in (
            ("warn_threshold", self.warn_threshold),
            ("blowup_threshold", self.blowup_threshold),
        ):
            if isinstance(value, bool):
                raise TypeError(f"{name} must not be bool")
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be a number, got {type(value).__name__}"
                )
            fval = float(value)
            if not math.isfinite(fval):
                raise ValueError(f"{name} must be finite")
            if fval < 0.0 or fval > _MAX_THRESHOLD:
                raise ValueError(
                    f"{name} must be in [0, {_MAX_THRESHOLD}], got {fval}"
                )
        if self.warn_threshold >= self.blowup_threshold:
            raise ValueError(
                f"warn_threshold ({self.warn_threshold}) must be < "
                f"blowup_threshold ({self.blowup_threshold})"
            )
        for name, value in (
            ("max_sequential_edits", self.max_sequential_edits),
            ("auto_switch_at", self.auto_switch_at),
        ):
            if isinstance(value, bool):
                raise TypeError(f"{name} must not be bool")
            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be int, got {type(value).__name__}"
                )
        if self.max_sequential_edits < 1:
            raise ValueError("max_sequential_edits must be >= 1")
        if self.max_sequential_edits > _MAX_SEQ_EDITS:
            raise ValueError(
                f"max_sequential_edits must be <= {_MAX_SEQ_EDITS}"
            )
        if self.auto_switch_at < 0:
            raise ValueError("auto_switch_at must be >= 0")


DEFAULT_BLOWUP_POLICY: NormBlowupPolicy = NormBlowupPolicy()


def classify_norm_blowup(
    delta: float, policy: NormBlowupPolicy = DEFAULT_BLOWUP_POLICY,
) -> str:
    """Classify a Frobenius norm delta as OK / WARN / BLOWUP.

    Bool-rejected, NaN/Inf-rejected, negative-rejected. Matches project
    bool-before-numeric policy.
    """
    if isinstance(delta, bool):
        raise TypeError("delta must not be bool")
    if not isinstance(delta, (int, float)):
        raise TypeError(
            f"delta must be a number, got {type(delta).__name__}"
        )
    fval = float(delta)
    if not math.isfinite(fval):
        raise ValueError("delta must be finite (no NaN / Inf)")
    if fval < 0.0:
        raise ValueError(f"delta must be >= 0, got {fval}")
    if fval >= policy.blowup_threshold:
        return "BLOWUP"
    if fval >= policy.warn_threshold:
        return "WARN"
    return "OK"


@dataclass(frozen=True)
class MethodRecommendation:
    """Output of :func:`governor_recommend_method`."""

    method: str
    switched: bool
    reason: str


def governor_recommend_method(
    *,
    current_method: str,
    edit_count: int,
    norm_delta: float,
    policy: NormBlowupPolicy = DEFAULT_BLOWUP_POLICY,
) -> MethodRecommendation:
    """Recommend the next method given accumulated state.

    Switching rules:

    1. ``alphaedit`` is the survival-mode method — never switched away.
    2. On BLOWUP, switch to ``alphaedit`` regardless of current method.
    3. When ``current_method == 'rome'`` AND ``edit_count >=
       auto_switch_at``, switch to ``alphaedit`` (MEMIT's
       multi-edit-capable but still suffers blowup at high counts;
       AlphaEdit is the projection-based survivor).
    4. Otherwise keep ``current_method``.
    """
    canonical = validate_edit_method(current_method)

    if isinstance(edit_count, bool):
        raise TypeError("edit_count must not be bool")
    if not isinstance(edit_count, int):
        raise TypeError(
            f"edit_count must be int, got {type(edit_count).__name__}"
        )
    if edit_count < 0:
        raise ValueError(f"edit_count must be >= 0, got {edit_count}")
    if edit_count > _MAX_SEQ_EDITS:
        raise ValueError(
            f"edit_count must be <= {_MAX_SEQ_EDITS}"
        )
    verdict = classify_norm_blowup(norm_delta, policy)

    # Rule 1: AlphaEdit stays.
    if canonical == "alphaedit":
        return MethodRecommendation(
            method="alphaedit",
            switched=False,
            reason="alphaedit is already the survival-mode method",
        )

    # Rule 2: blowup forces switch.
    if verdict == "BLOWUP":
        return MethodRecommendation(
            method="alphaedit",
            switched=True,
            reason=f"norm_delta={norm_delta:.4f} crossed BLOWUP threshold",
        )

    # Rule 3: ROME → AlphaEdit at auto_switch_at.
    if canonical == "rome" and edit_count >= policy.auto_switch_at:
        return MethodRecommendation(
            method="alphaedit",
            switched=True,
            reason=(
                f"edit_count={edit_count} >= auto_switch_at="
                f"{policy.auto_switch_at} — switching ROME to AlphaEdit"
            ),
        )

    # Rule 4: keep.
    return MethodRecommendation(
        method=canonical,
        switched=False,
        reason="below switch / blowup thresholds",
    )


class GovernedEditError(RuntimeError):
    """Raised by :meth:`EditGovernor.check_can_edit` on refusal."""


@dataclass
class EditGovernor:
    """Stateful per-base-model edit governor.

    Tracks ``edit_count`` and the last observed verdict so subsequent
    edits can be refused once the model crosses BLOWUP or hits the
    per-base ``max_sequential_edits`` cap.

    Mutable counters are declared as real dataclass fields (review
    HIGH H1 fix — slots-safe, ``replace`` / ``asdict`` compatible).
    """

    base_model: str
    policy: NormBlowupPolicy = field(default_factory=NormBlowupPolicy)
    max_sequential_edits: int = 50
    edit_count: int = 0
    last_method: str = ""
    last_verdict: str = "OK"
    last_norm_delta: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.base_model, str):
            raise TypeError(
                f"base_model must be str, got {type(self.base_model).__name__}"
            )
        if not self.base_model:
            raise ValueError("base_model must be non-empty")
        if "\x00" in self.base_model:
            raise ValueError("base_model must not contain null bytes")
        if len(self.base_model) > _MAX_BASE_LEN:
            raise ValueError(
                f"base_model must be <= {_MAX_BASE_LEN} chars"
            )
        if isinstance(self.max_sequential_edits, bool):
            raise TypeError("max_sequential_edits must not be bool")
        if not isinstance(self.max_sequential_edits, int):
            raise TypeError(
                f"max_sequential_edits must be int, got "
                f"{type(self.max_sequential_edits).__name__}"
            )
        if self.max_sequential_edits < 1:
            raise ValueError("max_sequential_edits must be >= 1")
        if self.max_sequential_edits > _MAX_SEQ_EDITS:
            raise ValueError(
                f"max_sequential_edits must be <= {_MAX_SEQ_EDITS}"
            )

    def record_edit(self, *, method: str, norm_delta: float) -> None:
        """Append a completed edit to the governor's history."""
        canonical_method = validate_edit_method(method)
        # Canonicalise norm_delta once so the verdict and the stored
        # last_norm_delta always agree (review MEDIUM M8 — prevents
        # int-passed-as-float drift in display).
        canonical_delta = float(norm_delta)
        verdict = classify_norm_blowup(canonical_delta, self.policy)
        self.edit_count += 1
        self.last_method = canonical_method
        self.last_verdict = verdict
        self.last_norm_delta = canonical_delta

    def check_can_edit(self) -> None:
        """Raise :class:`GovernedEditError` if a new edit would be refused."""
        if self.edit_count >= self.max_sequential_edits:
            raise GovernedEditError(
                f"max_sequential_edits cap ({self.max_sequential_edits}) "
                f"reached for base {self.base_model!r}; refuse further edits"
            )
        if self.last_verdict == "BLOWUP":
            raise GovernedEditError(
                f"last edit produced norm blowup "
                f"(delta={self.last_norm_delta:.4f}); refuse further "
                f"edits on base {self.base_model!r}"
            )

    def recommend_next_method(
        self, *, current_method: str,
    ) -> MethodRecommendation:
        """Recommend the next method given accumulated state."""
        return governor_recommend_method(
            current_method=current_method,
            edit_count=self.edit_count,
            norm_delta=self.last_norm_delta,
            policy=self.policy,
        )

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of governor state."""
        return {
            "base_model": self.base_model,
            "edit_count": self.edit_count,
            "last_method": self.last_method,
            "last_verdict": self.last_verdict,
            "last_norm_delta": self.last_norm_delta,
            "max_sequential_edits": self.max_sequential_edits,
        }
