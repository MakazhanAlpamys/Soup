"""Regression tests for booleans being coerced into numeric config fields (#2375)."""

from types import UnionType
from typing import Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from soup_cli.config.schema import (
    _TRAINING_NUMERIC_BOOL_VALIDATOR_EXEMPTIONS,
    LoraConfig,
    TrainingConfig,
)

# Any numeric field that intentionally accepts bool must be listed here with
# a reason. A new bool/numeric union should trigger an explicit review.
_BOOL_NUMERIC_UNION_EXEMPTIONS: dict[tuple[type[BaseModel], str], str] = {}

# These TrainingConfig fields already have dedicated bool-rejection validators.
# Keep their existing exception types and field-specific messages unchanged.


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

            if model is TrainingConfig and name in _TRAINING_NUMERIC_BOOL_VALIDATOR_EXEMPTIONS:
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


@pytest.mark.parametrize(
    "field_name",
    sorted(_TRAINING_NUMERIC_BOOL_VALIDATOR_EXEMPTIONS),
)
@pytest.mark.parametrize("value", [True, False], ids=["true", "false"])
def test_fields_with_their_own_check_still_refuse_booleans(field_name, value):
    with pytest.raises((ValidationError, TypeError)):
        TrainingConfig(**{field_name: value})
