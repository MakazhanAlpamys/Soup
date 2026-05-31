"""MiniLLM — reverse-KL on-policy distillation — v0.70.0 Part C.

MiniLLM (Gu et al. 2024, arXiv:2306.08543) extends knowledge
distillation with three stability tricks:

1. **Teacher-mixed sampling** — at rollout time, with probability
   ``teacher_mix_ratio`` sample from teacher logits instead of student
   to keep the student near a known-good distribution.
2. **Length normalisation** — divide the rollout log-probability by
   the completion length so longer completions don't dominate the
   gradient.
3. **Pretrain-loss anchor** — add a small SFT-on-pretrain term to the
   loss to prevent the student from drifting away from coherent
   language during the on-policy distillation.

Bundles stability tricks scattered across §3 of the paper. Extends
v0.53.2 :class:`DistillTrainerWrapper`. Live wiring deferred to v0.70.1
— mirrors v0.50.0 / v0.62.0 / v0.69.0 stub-then-live pattern.

Security:
- Bool / NaN / Inf / range rejection on every numeric validator.
- Null-byte + 4096-char cap on ``pretrain_anchor_path``.
- ``length_normalize`` must be a real bool (no str/int coercion).
- Cross-validators reject silent-no-op combinations
  (anchor_weight=0 + anchor_path set, anchor_weight > 0 + path None).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

_MAX_ANCHOR_PATH_LEN = 4096


def _check_unit_float(value: object, field: str) -> float:
    """Validate a finite float in [0.0, 1.0]. Bool rejected."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must not be bool")
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field} must be a number, got {type(value).__name__}"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{field} must be finite (no NaN/Inf)")
    if not (0.0 <= fv <= 1.0):
        raise ValueError(f"{field} must be in [0.0, 1.0], got {fv}")
    return fv


def validate_teacher_mix_ratio(value: object) -> float:
    """Validate the teacher-mix sampling ratio. Range [0.0, 1.0].

    0.0 = student-only rollouts; 1.0 = teacher-only rollouts. Typical
    MiniLLM recipes use 0.2-0.5 to balance exploration against
    proximity to the teacher's distribution.
    """
    return _check_unit_float(value, "teacher_mix_ratio")


def validate_pretrain_anchor_weight(value: object) -> float:
    """Validate the pretrain-anchor loss coefficient. Range [0.0, 1.0].

    0.0 = no anchor; small positive (e.g. 0.1) adds the SFT-on-pretrain
    term as a regulariser. Capped at 1.0 — values above would dominate
    the distillation loss (silent regression to vanilla SFT).
    """
    return _check_unit_float(value, "pretrain_anchor_weight")


def _check_path_shape(value: Optional[str]) -> Optional[str]:
    """Validate a string path field for shape only (cwd containment is
    deferred to the v0.70.1 runtime hook — schema permits relative
    paths for the same reason v0.69.0 build_dag does)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("pretrain_anchor_path must not be bool")
    if not isinstance(value, str):
        raise TypeError(
            f"pretrain_anchor_path must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("pretrain_anchor_path must be non-empty")
    if "\x00" in value:
        raise ValueError("pretrain_anchor_path must not contain null bytes")
    if len(value) > _MAX_ANCHOR_PATH_LEN:
        raise ValueError(
            f"pretrain_anchor_path exceeds {_MAX_ANCHOR_PATH_LEN} chars"
        )
    return value


@dataclass(frozen=True)
class MiniLLMConfig:
    """Frozen MiniLLM configuration.

    Cross-validation:
    - ``pretrain_anchor_weight > 0`` requires ``pretrain_anchor_path`` to
      be set (otherwise the anchor term has nothing to anchor against).
    - ``pretrain_anchor_path is not None`` requires
      ``pretrain_anchor_weight > 0`` (otherwise the path is a silent
      no-op).
    """

    teacher_mix_ratio: float = 0.0
    length_normalize: bool = True
    pretrain_anchor_weight: float = 0.0
    pretrain_anchor_path: Optional[str] = None

    def __post_init__(self) -> None:
        validate_teacher_mix_ratio(self.teacher_mix_ratio)
        if not isinstance(self.length_normalize, bool):
            raise TypeError(
                f"length_normalize must be bool, got "
                f"{type(self.length_normalize).__name__}"
            )
        validate_pretrain_anchor_weight(self.pretrain_anchor_weight)
        _check_path_shape(self.pretrain_anchor_path)
        if self.pretrain_anchor_weight > 0.0 and self.pretrain_anchor_path is None:
            raise ValueError(
                "pretrain_anchor_weight > 0 requires pretrain_anchor_path "
                "to be set"
            )
        if (
            self.pretrain_anchor_weight == 0.0
            and self.pretrain_anchor_path is not None
        ):
            raise ValueError(
                "pretrain_anchor_path is set but pretrain_anchor_weight is "
                "0 (silent no-op); set anchor_weight > 0 or clear the path"
            )


def build_minillm_callback(config):
    """Build the MiniLLM HF Trainer callback. Deferred to v0.70.1.

    Validates the config type at the public boundary so misconfigured
    callers fail fast (mirrors v0.50.0 / v0.62.0 / v0.67.0 / v0.69.0
    deferred-live policy).
    """
    if not isinstance(config, MiniLLMConfig):
        raise TypeError(
            f"config must be MiniLLMConfig, got {type(config).__name__}"
        )
    raise NotImplementedError(
        "Live MiniLLM HF Trainer callback is deferred to v0.70.1. "
        "v0.70.0 ships the schema + validators only."
    )
