"""Model-free trajectory capture tests for AutoDistill Milestone B1."""

from __future__ import annotations

import math

import pytest

from soup_cli.autodistill.capture import (
    TeacherExpertExample,
    build_teacher_expert_capture_token,
    capture_teacher_expert_trajectory,
)
from soup_cli.autodistill.contract import ProbabilityPolicy


def _policy(*, top_k: int, temperature: float = 1.0) -> ProbabilityPolicy:
    return ProbabilityPolicy(
        name="topk_union_forced_tail.v1",
        top_k=top_k,
        forced_token_sources=("target", "student_sample"),
        token_id_bytes=4,
        log_probability_bytes=4,
        tail_mass_bytes=8,
        entropy_bytes=8,
        temperature=temperature,
        renormalize_selected=False,
    )


def _dense_log_softmax(logits: tuple[float, ...], temperature: float = 1.0) -> tuple[float, ...]:
    scaled = tuple(value / temperature for value in logits)
    maximum = max(scaled)
    denominator = math.fsum(math.exp(value - maximum) for value in scaled)
    normalizer = maximum + math.log(denominator)
    return tuple(value - normalizer for value in scaled)


def test_sparse_capture_forces_target_without_renormalizing_top_k():
    logits = (4.0, 3.0, 2.0, -4.0)
    token = build_teacher_expert_capture_token(
        example_id="example-1",
        position=0,
        context_token_ids=(1,),
        target_token_id=3,
        teacher_logits=logits,
        vocab_size=4,
        probability_policy=_policy(top_k=2),
    )
    dense = _dense_log_softmax(logits)

    assert token.top_k_token_ids == (0, 1)
    assert token.forced_token_ids == (3,)
    assert token.selected_token_ids == (0, 1, 3)
    assert token.teacher_log_probabilities == pytest.approx((dense[0], dense[1], dense[3]))
    assert token.tail_mass == pytest.approx(math.exp(dense[2]))
    selected_mass = math.fsum(math.exp(value) for value in token.teacher_log_probabilities)
    assert selected_mass + token.tail_mass == pytest.approx(1.0)


def test_top_k_ties_are_broken_by_token_id_before_canonical_sorting():
    token = build_teacher_expert_capture_token(
        example_id="example-1",
        position=0,
        context_token_ids=(),
        target_token_id=3,
        teacher_logits=(2.0, 2.0, 2.0, 0.0),
        vocab_size=4,
        probability_policy=_policy(top_k=2),
    )

    assert token.top_k_token_ids == (0, 1)


def test_dense_capture_matches_full_distribution_and_has_no_tail():
    logits = (1.0, -2.0, 0.5)
    token = build_teacher_expert_capture_token(
        example_id="example-1",
        position=0,
        context_token_ids=(0,),
        target_token_id=2,
        teacher_logits=logits,
        vocab_size=3,
        probability_policy=_policy(top_k=3, temperature=0.5),
    )

    assert token.selected_token_ids == (0, 1, 2)
    assert token.teacher_log_probabilities == pytest.approx(_dense_log_softmax(logits, 0.5))
    assert token.tail_mass == 0.0


def test_full_target_trajectory_records_each_autoregressive_context():
    example = TeacherExpertExample(
        example_id="example-1",
        prompt_token_ids=(0, 1),
        target_token_ids=(2, 3, 4),
    )
    rows = capture_teacher_expert_trajectory(
        example=example,
        teacher_logits_by_position=(
            (0.0, 0.0, 3.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 3.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 3.0),
        ),
        vocab_size=5,
        probability_policy=_policy(top_k=2),
        max_sequence_length=8,
        truncation="none",
    )

    assert [row.position for row in rows] == [0, 1, 2]
    assert [row.context_token_ids for row in rows] == [(0, 1), (0, 1, 2), (0, 1, 2, 3)]
    assert [row.target_token_id for row in rows] == [2, 3, 4]


def test_trajectory_applies_declared_left_truncation_to_recorded_context():
    rows = capture_teacher_expert_trajectory(
        example=TeacherExpertExample(
            example_id="example-1",
            prompt_token_ids=(0, 1, 2),
            target_token_ids=(3, 4),
        ),
        teacher_logits_by_position=((0.0,) * 5, (0.0,) * 5),
        vocab_size=5,
        probability_policy=_policy(top_k=1),
        max_sequence_length=2,
        truncation="left",
    )

    assert [row.context_token_ids for row in rows] == [(1, 2), (2, 3)]


@pytest.mark.parametrize(
    ("logits", "match"),
    [
        ((0.0, 1.0), "length must equal"),
        ((0.0, math.inf, 1.0), "finite numbers"),
    ],
)
def test_capture_rejects_invalid_teacher_logits(logits: tuple[float, ...], match: str):
    with pytest.raises(ValueError, match=match):
        build_teacher_expert_capture_token(
            example_id="example-1",
            position=0,
            context_token_ids=(),
            target_token_id=1,
            teacher_logits=logits,
            vocab_size=3,
            probability_policy=_policy(top_k=1),
        )


def test_trajectory_rejects_missing_positions_and_undeclared_truncation():
    example = TeacherExpertExample(
        example_id="example-1",
        prompt_token_ids=(0, 1),
        target_token_ids=(2, 1),
    )
    with pytest.raises(ValueError, match="one teacher logit vector"):
        capture_teacher_expert_trajectory(
            example=example,
            teacher_logits_by_position=((0.0, 0.0, 0.0),),
            vocab_size=3,
            probability_policy=_policy(top_k=1),
            max_sequence_length=8,
            truncation="none",
        )
    with pytest.raises(ValueError, match="exceeds max_sequence_length"):
        capture_teacher_expert_trajectory(
            example=example,
            teacher_logits_by_position=((0.0, 0.0, 0.0),) * 2,
            vocab_size=3,
            probability_policy=_policy(top_k=1),
            max_sequence_length=1,
            truncation="none",
        )
