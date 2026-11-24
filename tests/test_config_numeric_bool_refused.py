"""Regression tests for booleans being coerced into numeric config fields."""

import inspect
from types import UnionType
from typing import Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

import soup_cli.config.schema as schema
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import (
    _TRAINING_NUMERIC_BOOL_VALIDATOR_EXEMPTIONS,
    DataConfig,
    EvalGateConfig,
    LoraConfig,
    ShipConfig,
    TrainingConfig,
)

# An explicit bool arm next to a numeric arm is allowed only with a reason.
# Keep this empty until the schema has a field whose contract really needs it.
_BOOL_NUMERIC_UNION_EXEMPTIONS: dict[tuple[type[BaseModel], str], str] = {}

# These fields already have their own before validators. The shared guard must
# leave those field-specific errors intact instead of changing their messages.
_EXISTING_BOOL_REFUSAL_REASONS = {
    (DataConfig, "raft_shuffle_seed"): "keeps the RAFT seed validator's message",
    (DataConfig, "replay_ratio"): "keeps the replay ratio validator's message",
    (DataConfig, "replay_seed"): "keeps the replay seed validator's message",
    (ShipConfig, "noise_floor"): "keeps the noise floor validator's message",
}
_EXISTING_BOOL_REFUSAL_REASONS.update(
    {
        (TrainingConfig, name): "keeps the existing training field validator"
        for name in _TRAINING_NUMERIC_BOOL_VALIDATOR_EXEMPTIONS
    }
)


def _union_members(annotation):
    """Walk unions without treating list[int] or dict[str, int] as scalars."""
    if get_origin(annotation) in (Union, UnionType):
        for member in get_args(annotation):
            yield from _union_members(member)
    else:
        yield annotation


def _is_bool_literal(annotation):
    return get_origin(annotation) is Literal and any(
        type(value) is bool for value in get_args(annotation)
    )


def _schema_models():
    """Discover models declared in schema.py so new models join this audit."""
    return tuple(
        model
        for model in vars(schema).values()
        if inspect.isclass(model)
        and model.__module__ == schema.__name__
        and issubclass(model, BaseModel)
    )


def _numeric_field_cases():
    cases = []
    discovered_union_exemptions = set()
    discovered_existing_refusals = set()

    for model in _schema_models():
        for name, field in model.model_fields.items():
            members = tuple(_union_members(field.annotation))
            if not any(member in (int, float) for member in members):
                continue

            key = (model, name)
            permits_bool = any(member is bool or _is_bool_literal(member) for member in members)
            if permits_bool:
                discovered_union_exemptions.add(key)
                reason = _BOOL_NUMERIC_UNION_EXEMPTIONS.get(key)
                assert reason and reason.strip(), (
                    f"{model.__name__}.{name} explicitly permits bool; "
                    "review it and document an exemption reason."
                )
                continue

            if key in _EXISTING_BOOL_REFUSAL_REASONS:
                discovered_existing_refusals.add(key)

            cases.append(pytest.param(model, name, id=f"{model.__name__}.{name}"))

    assert discovered_union_exemptions == set(_BOOL_NUMERIC_UNION_EXEMPTIONS), (
        "The numeric boolean union exemption list does not match the schema."
    )
    assert discovered_existing_refusals == set(_EXISTING_BOOL_REFUSAL_REASONS), (
        "The existing field-validator exemptions do not match the schema."
    )
    return cases


_NUMERIC_FIELDS = _numeric_field_cases()


def _model_input(model, field_name, value):
    payload = {field_name: value}
    if model is DataConfig:
        payload["train"] = "unused.jsonl"
        if field_name == "buffer_size":
            # Otherwise a model-level dependency error could hide bool coercion.
            payload["streaming"] = True
    return payload


@pytest.mark.parametrize(("model", "field_name"), _NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [True, False], ids=["true", "false"])
def test_numeric_fields_reject_booleans_by_field(model, field_name, value):
    with pytest.raises(ValidationError) as exc_info:
        model(**_model_input(model, field_name, value))

    error = next(
        (item for item in exc_info.value.errors() if item["loc"] == (field_name,)),
        None,
    )
    assert error is not None, exc_info.value.errors()

    key = (model, field_name)
    if key in _EXISTING_BOOL_REFUSAL_REASONS:
        assert "bool" in error["msg"].lower(), (key, error)
        return

    members = tuple(_union_members(model.model_fields[field_name].annotation))
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

    data = DataConfig(train="unused.jsonl", streaming=True, buffer_size=128, shards=4)
    assert data.buffer_size == 128
    assert data.shards == 4
    assert EvalGateConfig(every_n_epochs=2, regression_threshold=0.05).every_n_epochs == 2
    assert ShipConfig(forgetting_threshold=0.1).forgetting_threshold == 0.1


def test_actual_boolean_fields_still_accept_booleans():
    assert LoraConfig(use_dora=True).use_dora is True
    assert DataConfig(train="unused.jsonl", streaming=True).streaming is True

    training = TrainingConfig(mask_truncated_completions=True, tis_threshold=2.0)
    assert training.mask_truncated_completions is True


@pytest.mark.parametrize("yaml_value", ["true", "yes"], ids=["true", "yaml-yes"])
@pytest.mark.parametrize(
    ("section", "field_name"),
    [
        ("data", "val_split"),
        ("data", "shards"),
        ("data", "image_min_pixels"),
        ("data", "image_max_pixels"),
        ("data", "video_fps"),
        ("data", "video_maxlen"),
        ("data", "buffer_size"),
        ("training.eval_gate", "every_n_epochs"),
        ("training.eval_gate", "regression_threshold"),
        ("eval.ship", "forgetting_threshold"),
    ],
)
def test_yaml_boolean_spellings_are_rejected_for_reported_fields(section, field_name, yaml_value):
    lines = ["data:", "  train: unused.jsonl"]
    if section == "data" and field_name == "buffer_size":
        lines.append("  streaming: true")
    if section != "data":
        parts = section.split(".")
        for depth, part in enumerate(parts):
            lines.append(f"{'  ' * depth}{part}:")
        field_indent = "  " * len(parts)
    else:
        field_indent = "  "
    lines.append(f"{field_indent}{field_name}: {yaml_value}")

    with pytest.raises(ValueError, match=field_name) as exc_info:
        load_config_from_string("\n".join(lines) + "\n")
    if section == "data" and field_name == "buffer_size":
        assert "buffer_size must be an int, not a bool" in str(exc_info.value)


@pytest.mark.parametrize("field_name", sorted(_TRAINING_NUMERIC_BOOL_VALIDATOR_EXEMPTIONS))
@pytest.mark.parametrize("value", [True, False], ids=["true", "false"])
def test_training_fields_with_their_own_checks_still_refuse_booleans(field_name, value):
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfig(**{field_name: value})

    assert any(error["loc"] == (field_name,) for error in exc_info.value.errors())
