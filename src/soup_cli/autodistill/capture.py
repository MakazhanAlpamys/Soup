"""Deterministic, model-free construction of AutoDistill capture rows.

This module turns already-computed teacher logits into the versioned Milestone A
artifact.  It deliberately does not import or load an ML runtime; backend workers
remain responsible for producing one logit vector per target position.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from soup_cli.autodistill.contract import CaptureToken, ProbabilityPolicy


def _validate_token_ids(
    token_ids: Sequence[int],
    *,
    field: str,
    vocab_size: int,
    allow_empty: bool,
) -> tuple[int, ...]:
    result = tuple(token_ids)
    if not allow_empty and not result:
        raise ValueError(f"{field} must not be empty")
    if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in result):
        raise TypeError(f"{field} must contain integer token ids")
    if any(token_id < 0 or token_id >= vocab_size for token_id in result):
        raise ValueError(f"{field} contains an id outside the vocabulary")
    return result


@dataclass(frozen=True)
class TeacherExpertExample:
    """One tokenized supervised example presented to a teacher backend."""

    example_id: str
    prompt_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]


def _teacher_log_probabilities(
    logits: Sequence[float],
    *,
    vocab_size: int,
    temperature: float,
) -> tuple[float, ...]:
    if len(logits) != vocab_size:
        raise ValueError("teacher logits length must equal vocab_size")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in logits):
        raise TypeError("teacher logits must contain finite numbers")
    scaled = tuple(float(value) / temperature for value in logits)
    if any(not math.isfinite(value) for value in scaled):
        raise ValueError("teacher logits must contain finite numbers")

    maximum = max(scaled)
    log_normalizer = maximum + math.log(math.fsum(math.exp(value - maximum) for value in scaled))
    return tuple(value - log_normalizer for value in scaled)


def build_teacher_expert_capture_token(
    *,
    example_id: str,
    position: int,
    context_token_ids: Sequence[int],
    target_token_id: int,
    teacher_logits: Sequence[float],
    vocab_size: int,
    probability_policy: ProbabilityPolicy,
) -> CaptureToken:
    """Build one sparse teacher-expert row without losing selected probabilities."""

    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size must be an integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if probability_policy.top_k > vocab_size:
        raise ValueError("top_k must not exceed vocab_size")
    context = _validate_token_ids(
        context_token_ids,
        field="context_token_ids",
        vocab_size=vocab_size,
        allow_empty=True,
    )
    target = _validate_token_ids(
        (target_token_id,),
        field="target_token_id",
        vocab_size=vocab_size,
        allow_empty=False,
    )[0]

    log_probabilities = _teacher_log_probabilities(
        teacher_logits,
        vocab_size=vocab_size,
        temperature=probability_policy.temperature,
    )
    ranked_ids = sorted(
        range(vocab_size),
        key=lambda token_id: (-log_probabilities[token_id], token_id),
    )
    top_k_ids = tuple(sorted(ranked_ids[: probability_policy.top_k]))
    forced_ids = (target,)
    selected_ids = tuple(sorted(set(top_k_ids) | set(forced_ids)))
    selected_log_probabilities = tuple(log_probabilities[token_id] for token_id in selected_ids)
    omitted_ids = set(range(vocab_size)) - set(selected_ids)
    tail_mass = math.fsum(math.exp(log_probabilities[token_id]) for token_id in omitted_ids)
    entropy = math.fsum(
        -math.exp(log_probability) * log_probability
        for log_probability in log_probabilities
    )

    return CaptureToken(
        schema="soup.autodistill.capture-token.v1",
        example_id=example_id,
        trajectory_kind="teacher_expert",
        position=position,
        vocab_size=vocab_size,
        context_token_ids=context,
        target_token_id=target,
        student_sampled_token_id=None,
        top_k_token_ids=top_k_ids,
        forced_token_ids=forced_ids,
        selected_token_ids=selected_ids,
        teacher_log_probabilities=selected_log_probabilities,
        tail_mass=tail_mass,
        teacher_entropy=entropy,
        temperature=probability_policy.temperature,
    )


def capture_teacher_expert_trajectory(
    *,
    example: TeacherExpertExample,
    teacher_logits_by_position: Sequence[Sequence[float]],
    vocab_size: int,
    probability_policy: ProbabilityPolicy,
    max_sequence_length: int,
    truncation: Literal["left", "right", "none"],
) -> tuple[CaptureToken, ...]:
    """Capture every supervised target position with its exact inference context."""

    if isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int):
        raise TypeError("max_sequence_length must be an integer")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if truncation not in {"left", "right", "none"}:
        raise ValueError("truncation must be left, right, or none")
    prompt = _validate_token_ids(
        example.prompt_token_ids,
        field="prompt_token_ids",
        vocab_size=vocab_size,
        allow_empty=True,
    )
    targets = _validate_token_ids(
        example.target_token_ids,
        field="target_token_ids",
        vocab_size=vocab_size,
        allow_empty=False,
    )
    if len(teacher_logits_by_position) != len(targets):
        raise ValueError("one teacher logit vector is required per target position")

    captures: list[CaptureToken] = []
    for position, (target, logits) in enumerate(zip(targets, teacher_logits_by_position)):
        context = prompt + targets[:position]
        if len(context) > max_sequence_length:
            if truncation == "none":
                raise ValueError("teacher context exceeds max_sequence_length")
            if truncation == "left":
                context = context[-max_sequence_length:]
            else:
                context = context[:max_sequence_length]
        captures.append(
            build_teacher_expert_capture_token(
                example_id=example.example_id,
                position=position,
                context_token_ids=context,
                target_token_id=target,
                teacher_logits=logits,
                vocab_size=vocab_size,
                probability_policy=probability_policy,
            )
        )
    return tuple(captures)
