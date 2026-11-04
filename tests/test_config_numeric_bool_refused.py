"""Regression tests for booleans being coerced into numeric config fields (#2375)."""

from types import UnionType
from typing import Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from soup_cli.config.schema import LoraConfig, TrainingConfig

# Any numeric field that intentionally accepts bool must be listed here with
# a reason. A new bool/numeric union should trigger an explicit review.
_BOOL_NUMERIC_UNION_EXEMPTIONS: dict[tuple[type[BaseModel], str], str] = {}

# These TrainingConfig fields already have dedicated bool-rejection validators.
# Keep their existing exception types and field-specific messages unchanged.
_EXISTING_BOOL_VALIDATORS = {
    "seed",
    "data_seed",
    "expand_layers",
    "freeze_trainable_layers",
    "eval_steps",
    "grpo_beta",
    "grpo_judge_max_tokens",
    "grpo_judge_max_calls",
    "grpo_judge_max_failure_rate",
    "distill_chunk_size",
    "distill_temperature",
    "mod_capacity_factor",
    "ebft_temperature",
    "num_labels",
    "minillm_rollout_length",
    "mole_top_k",
    "mole_temperature",
    "lisa_num_layers",
    "lisa_interval_steps",
    "stream_buffers",
    "stream_read_ahead",
    "stream_vram_override",
    "stream_loss_chunk",
    "unlearn_alpha",
    "ref_model_ema_alpha",
    "tis_threshold",
    "off_policy_mask_threshold",
    "replay_buffer_size",
    "grpo_delta",
    "yarn_factor",
    "yarn_attn_factor",
    "yarn_beta_fast",
    "yarn_beta_slow",
    "citation_recall_threshold",
    "grace_codebook_size",
    "grace_codebook_dim",
    "reward_hack_dwell_steps",
    "reward_hack_release_patience",
    "reward_hack_rollback_patience",
    "reward_hack_max_recovery_attempts",
    "reward_hack_smoothing_window",
    "reward_hack_beta_floor",
    "reward_hack_beta_ceil",
    "reward_hack_trip_band",
    "reward_hack_release_band",
    "reward_hack_kl_gain",
    "reward_hack_pid_kp",
    "reward_hack_pid_ki",
    "reward_hack_pid_kd",
    "reward_hack_signal_target",
    "reward_hack_integral_clamp",
    "reward_hack_shaping_strength",
}


def _union_members(annotation):
    """Walk union members without treating list[int] or dict[str, int] as scalars."""
    if get_origin(annotation) in (Union, UnionType):
        for member in get_args(annotation):
            yield from _union_members(member)
    else:
        yield annotation


def _is_bool_literal(annotation):
    return get_origin(annotation) is Literal and any(
        type(value) is bool for value in get_args(annotation)
    )


def _numeric_field_cases():
    cases = []
    discovered_exemptions = set()

    for model in (LoraConfig, TrainingConfig):
        for name, field in model.model_fields.items():
            members = tuple(_union_members(field.annotation))

            if not any(member in (int, float) for member in members):
                continue

            key = (model, name)

            if model is TrainingConfig and name in _EXISTING_BOOL_VALIDATORS:
                continue

            permits_bool = any(member is bool or _is_bool_literal(member) for member in members)

            if permits_bool:
                discovered_exemptions.add(key)
                reason = _BOOL_NUMERIC_UNION_EXEMPTIONS.get(key)
                assert reason and reason.strip(), (
                    f"{model.__name__}.{name} explicitly permits bool; "
                    "review it and document an exemption reason."
                )
                continue

            assert key not in _BOOL_NUMERIC_UNION_EXEMPTIONS, (
                f"Stale boolean exemption for {model.__name__}.{name}"
            )
            cases.append(pytest.param(model, name, id=f"{model.__name__}.{name}"))

    assert discovered_exemptions == set(_BOOL_NUMERIC_UNION_EXEMPTIONS), (
        "The numeric boolean exemption list does not match the schema."
    )
    return cases


_NUMERIC_FIELDS = _numeric_field_cases()


@pytest.mark.parametrize(("model", "field_name"), _NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [True, False], ids=["true", "false"])
def test_numeric_fields_reject_booleans(model, field_name, value):
    with pytest.raises(ValidationError) as exc_info:
        model(**{field_name: value})

    error = exc_info.value.errors()[0]
    assert error["loc"] == (field_name,)

    annotation = model.model_fields[field_name].annotation
    members = tuple(_union_members(annotation))
    expected = (
        f"{field_name} must be an int, not a bool"
        if int in members and float not in members
        else f"{field_name} must be a number, not a bool"
    )
    assert expected in error["msg"]


def test_valid_numeric_controls_remain_valid():
    assert LoraConfig(r=0).r == 0
    assert LoraConfig(r=8, alpha=16).r == 8

    training = TrainingConfig(lr=2e-5, batch_size=4, epochs=1)
    assert training.lr == 2e-5
    assert training.batch_size == 4

    assert TrainingConfig(batch_size="auto").batch_size == "auto"


def test_actual_boolean_fields_still_accept_booleans():
    assert LoraConfig(use_dora=True).use_dora is True

    training = TrainingConfig(
        mask_truncated_completions=True,
        tis_threshold=2.0,
    )
    assert training.mask_truncated_completions is True
