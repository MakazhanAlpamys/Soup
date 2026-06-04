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
_MAX_ANCHOR_ROW_BYTES = 1_000_000


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


def minillm_distill_term(
    student_logits,
    teacher_logits,
    labels,
    *,
    config: "MiniLLMConfig",
    temperature: float = 1.0,
):
    """MiniLLM reverse-KL distillation term (v0.71.11 #237).

    Computes ``KL(student || target)`` where ``target`` is the
    teacher-mixed rollout distribution
    ``ratio * teacher + (1 - ratio) * student_detached`` — the offline
    analog of MiniLLM's teacher-mixed sampling (Gu et al. 2024 §3.1). At
    ``ratio = 1`` this reduces to standard reverse-KL distillation; at
    ``ratio < 1`` the target stays closer to the student's current
    distribution (keeps the student near a known-good policy).

    When ``config.length_normalize`` is set, the per-token reverse-KL is
    averaged over the *valid* (``labels != -100``) tokens per sequence so
    long completions don't dominate the gradient.

    Differentiable w.r.t. the student logits. Returns a scalar.
    """
    import torch

    if not isinstance(config, MiniLLMConfig):
        raise TypeError(
            f"config must be MiniLLMConfig, got {type(config).__name__}"
        )
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a non-bool number")
    t = float(temperature)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError("temperature must be finite and positive")

    s = student_logits / t
    teach = teacher_logits / t
    log_p_s = torch.log_softmax(s, dim=-1)
    p_s = log_p_s.exp()
    p_t = torch.softmax(teach, dim=-1)
    ratio = float(config.teacher_mix_ratio)
    target = ratio * p_t + (1.0 - ratio) * p_s.detach()
    log_target = target.clamp(min=1e-12).log()
    # reverse KL(student || target) per token.
    rkl = (p_s * (log_p_s - log_target)).sum(dim=-1)  # [B, T]

    if config.length_normalize and labels is not None:
        valid = (labels != -100).to(rkl.dtype)
        per_seq = (rkl * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)
        loss = per_seq.mean()
    else:
        loss = rkl.mean()
    return loss * (t * t)


def _get_trainer_callback_base():
    """Lazy-resolve ``transformers.TrainerCallback`` (mirror v0.53.11)."""
    try:
        from transformers import TrainerCallback

        return TrainerCallback
    except ImportError:
        return object


_TrainerCallbackBase = _get_trainer_callback_base()


class MiniLLMCallback(_TrainerCallbackBase):  # type: ignore[misc, valid-type]
    """Live MiniLLM helper + HF TrainerCallback (v0.71.11 #237).

    Carries the :class:`MiniLLMConfig` and provides the loss terms the
    distill trainer applies:

    - :meth:`distill_term` — teacher-mixed length-normalised reverse-KL.
    - :meth:`anchor_term` — pretrain-anchor SFT CE on a small batch read
      lazily from ``pretrain_anchor_path``, scaled by
      ``pretrain_anchor_weight`` (prevents the student drifting away from
      coherent language).

    Honest scope: the teacher-mix is the offline distribution-blend
    analog of MiniLLM's on-policy teacher-mixed *sampling*; a full
    autoregressive rollout loop (sample → teacher-score) is a larger
    follow-up documented as a known limitation.
    """

    def __init__(
        self,
        config: MiniLLMConfig,
        *,
        tokenizer: Optional[object] = None,
        temperature: float = 1.0,
        max_anchor_rows: int = 8,
        anchor_max_length: int = 128,
    ) -> None:
        if not isinstance(config, MiniLLMConfig):
            raise TypeError(
                f"config must be MiniLLMConfig, got {type(config).__name__}"
            )
        self.config = config
        self.tokenizer = tokenizer
        self.temperature = float(temperature)
        self.max_anchor_rows = int(max_anchor_rows)
        self.anchor_max_length = int(anchor_max_length)
        self._anchor_inputs: Optional[dict] = None
        self._anchor_loaded = False

    def distill_term(self, student_logits, teacher_logits, labels):
        """Compute the teacher-mixed length-normalised reverse-KL term."""
        return minillm_distill_term(
            student_logits,
            teacher_logits,
            labels,
            config=self.config,
            temperature=self.temperature,
        )

    def _load_anchor(self) -> Optional[dict]:
        """Lazily tokenise a small pretrain-anchor batch (cwd-contained)."""
        if self._anchor_loaded:
            return self._anchor_inputs
        self._anchor_loaded = True
        path = self.config.pretrain_anchor_path
        if (
            path is None
            or self.config.pretrain_anchor_weight <= 0.0
            or self.tokenizer is None
        ):
            return None
        from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

        # cwd-containment + TOCTOU symlink rejection (security review fix —
        # mirrors v0.53.7 / v0.65 reader policy; was a bare is_under_cwd).
        enforce_under_cwd_and_no_symlink(path, "minillm_pretrain_anchor_path")
        import json

        texts: list[str] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                # Per-line byte cap — defends against a single multi-MB line
                # blowing up tokenisation memory (matches v0.53.7 #106 caps).
                if len(line) > _MAX_ANCHOR_ROW_BYTES:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content") or ""
                except (ValueError, AttributeError):
                    text = line
                if text:
                    texts.append(str(text))
                if len(texts) >= self.max_anchor_rows:
                    break
        if not texts:
            return None
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.anchor_max_length,
        )
        self._anchor_inputs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        return self._anchor_inputs

    def anchor_term(self, model):
        """Pretrain-anchor SFT cross-entropy scaled by the anchor weight.

        Returns ``None`` when the anchor is disabled / unavailable so the
        caller can skip the term cleanly.
        """
        weight = float(self.config.pretrain_anchor_weight)
        if weight <= 0.0:
            return None
        anchor = self._load_anchor()
        if anchor is None:
            return None
        import torch

        device = next(model.parameters()).device
        input_ids = anchor["input_ids"].to(device)
        attention_mask = anchor["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        # Mask padding so the anchor CE only counts real tokens.
        pad_mask = attention_mask[:, 1:].contiguous().bool()
        shift_labels = shift_labels.masked_fill(~pad_mask, -100)
        ce = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return weight * ce


def build_minillm_callback(
    config,
    *,
    tokenizer: Optional[object] = None,
    temperature: float = 1.0,
) -> "MiniLLMCallback":
    """Build the live MiniLLM callback / loss helper (v0.71.11 #237).

    Lifts the v0.70.0 ``NotImplementedError`` stub. Validates the config
    type at the public boundary (fail-fast policy), then returns a
    :class:`MiniLLMCallback`.
    """
    if not isinstance(config, MiniLLMConfig):
        raise TypeError(
            f"config must be MiniLLMConfig, got {type(config).__name__}"
        )
    return MiniLLMCallback(config, tokenizer=tokenizer, temperature=temperature)
