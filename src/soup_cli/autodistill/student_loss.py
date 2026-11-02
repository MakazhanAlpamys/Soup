"""Proposed internal Torch objective for v1 teacher-expert capture rows.

This numerical slice neither loads a model nor authorizes a cache for training.
Its caller must verify publication, identities and teacher-process exit first.
No task, CLI, default loss weights or default top-k are introduced here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from soup_cli.autodistill.contract import CaptureToken


@dataclass(frozen=True)
class CapturedTokenLoss:
    """Token means; weight microbatch means by their counts when accumulating."""

    total: Any
    ce: Any
    kd: Any
    token_count: int


def _weight(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _chunk_sums(logits: Any, captures: tuple[CaptureToken, ...]) -> Any:
    import torch

    # Keep the double-precision oracle path, and promote fp16/bf16 probability
    # arithmetic to float32. These are logits for captured target positions only.
    values = logits if logits.dtype == torch.float64 else logits.float()
    ce_terms = []
    kd_terms = []
    for value, row in zip(values, captures):
        # Normalize after removing a common offset: subtracting a raw logit
        # from logsumexp(raw_logits) loses log-probabilities at large offsets.
        # This offset is constant for differentiation; the objective is shift invariant.
        centered = value - value.detach().amax()
        ce_terms.append(torch.logsumexp(centered, dim=0) - centered[row.target_token_id])
        scaled = centered / row.temperature
        log_normalizer = torch.logsumexp(scaled, dim=0)
        ids = torch.tensor(row.selected_token_ids, dtype=torch.long, device=value.device)
        log_q = scaled.index_select(0, ids) - log_normalizer
        log_p = value.new_tensor(row.teacher_log_probabilities)
        kd = (log_p.exp() * (log_p - log_q)).sum()
        if len(row.selected_token_ids) < row.vocab_size:
            # Subtracting sum(q_selected) from one rounds to zero when the
            # student is confident. Direct logsumexp over omitted logits keeps
            # a finite tail and its gradient even at extreme finite logits.
            omitted = scaled.scatter(0, ids, float("-inf"))
            log_q_tail = torch.logsumexp(omitted, dim=0) - log_normalizer
            kd = kd + row.tail_mass * (math.log(row.tail_mass) - log_q_tail)
        kd_terms.append(kd)
    sums = torch.stack((torch.stack(ce_terms).sum(), torch.stack(kd_terms).sum()))
    if not bool(torch.isfinite(sums).all()):
        raise ValueError("objective is not finite at the working dtype and temperature")
    return sums


def captured_token_loss(
    student_logits: Any,
    captures: Sequence[CaptureToken],
    *,
    ce_weight: float,
    kd_weight: float,
    token_chunk_size: int,
    checkpoint_chunks: bool,
) -> CapturedTokenLoss:
    """Differentiate CE plus coarse-tail forward KL from frozen teacher rows.

    ``student_logits[n]`` must predict the next token after exactly the recorded
    ``captures[n].context_token_ids``. Prompt/pad/masked positions are absent.
    CE uses untempered logits; KD uses each capture's declared temperature with
    no implicit temperature-squared multiplier. Loss weights are explicit.

    The caller supplies student logits of shape [captured_tokens, vocabulary].
    Chunking bounds additional objective intermediates; it does not remove
    already-materialized student logits or model activations. There is no dense
    teacher tensor. Optional non-reentrant checkpointing recomputes this pure
    objective during backward, rather than retaining per-chunk intermediates.

    This is not a shard consumer or a training/resume controller: cache and
    process-isolation checks must precede the call. Student rollouts are refused.
    """
    import torch

    ce_weight = _weight(ce_weight, "ce_weight")
    kd_weight = _weight(kd_weight, "kd_weight")
    if ce_weight == kd_weight == 0:
        raise ValueError("at least one objective weight must be positive")
    if isinstance(token_chunk_size, bool) or not isinstance(token_chunk_size, int):
        raise TypeError("token_chunk_size must be a positive integer")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be a positive integer")
    if not isinstance(checkpoint_chunks, bool):
        raise TypeError("checkpoint_chunks must be bool")
    if not isinstance(student_logits, torch.Tensor) or not student_logits.is_floating_point():
        raise TypeError("student_logits must be a floating-point tensor")
    if student_logits.ndim != 2 or min(student_logits.shape) <= 0:
        raise ValueError("student_logits must have non-empty [tokens, vocabulary] shape")
    if not bool(torch.isfinite(student_logits).all()):
        raise ValueError("student_logits must be finite")
    rows = tuple(captures)
    if len(rows) != student_logits.shape[0]:
        raise ValueError("one capture row is required per student logit position")
    validated_rows = []
    for row in rows:
        if not isinstance(row, CaptureToken):
            raise TypeError("captures must contain CaptureToken instances")
        # model_copy/update bypasses Pydantic validation; preserve the immutable
        # artifact contract even for an in-process caller using that API.
        row = CaptureToken.model_validate(row.model_dump(by_alias=True), strict=True)
        if row.trajectory_kind != "teacher_expert":
            raise ValueError("this objective accepts teacher_expert captures only")
        if row.vocab_size != student_logits.shape[1]:
            raise ValueError("student vocabulary does not match the capture")
        if len(row.selected_token_ids) == row.vocab_size and row.tail_mass != 0:
            raise ValueError("full-vocabulary capture must have zero tail mass")
        validated_rows.append(row)
    rows = tuple(validated_rows)

    sums = student_logits.new_zeros(2, dtype=(
        torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    ))
    for start in range(0, len(rows), token_chunk_size):
        chunk_logits = student_logits[start:start + token_chunk_size]
        chunk_rows = rows[start:start + token_chunk_size]
        if checkpoint_chunks and student_logits.requires_grad:
            from torch.utils.checkpoint import checkpoint

            chunk = checkpoint(_chunk_sums, chunk_logits, chunk_rows,
                               use_reentrant=False, preserve_rng_state=False)
        else:
            chunk = _chunk_sums(chunk_logits, chunk_rows)
        sums = sums + chunk
    ce, kd = sums / len(rows)
    total = ce_weight * ce + kd_weight * kd
    if not bool(torch.isfinite(total)):
        raise ValueError("weighted objective is not finite at the working dtype")
    return CapturedTokenLoss(total=total, ce=ce, kd=kd, token_count=len(rows))
