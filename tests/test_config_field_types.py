"""A wrong-typed value in any config field is a ValidationError, never a TypeError (#2366)."""

import typing
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from tests.conftest import strip_ansi

WRONG_VALUES = [0, True, "false", [], {}, 1.5]

# Required fields, so that each model reaches the validator of the field under test.
REQUIRED = {
    "SoupConfig": {"base": "Qwen/Qwen2.5-0.5B-Instruct", "data": {"train": "train.jsonl"}},
    "DataConfig": {"train": "train.jsonl"},
}


def _reachable_models(annotation, found):
    if typing.get_origin(annotation) is not None:
        for arg in typing.get_args(annotation):
            _reachable_models(arg, found)
        return found
    if (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and annotation not in found
    ):
        found.append(annotation)
        for field in annotation.model_fields.values():
            _reachable_models(field.annotation, found)
    return found


MODELS = _reachable_models(SoupConfig, [])
FIELDS = [
    pytest.param(model, name, id=f"{model.__name__}.{name}")
    for model in MODELS
    for name in model.model_fields
]


def test_walk_reaches_every_section():
    names = {model.__name__ for model in MODELS}
    assert {"SoupConfig", "DataConfig", "TrainingConfig", "LoraConfig"} <= names


@pytest.mark.parametrize(("model", "name"), FIELDS)
def test_wrong_typed_value_is_a_validation_error(model, name):
    for value in WRONG_VALUES:
        payload = {**REQUIRED.get(model.__name__, {}), name: value}
        try:
            model.model_validate(payload)
        except ValidationError:
            pass


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minillm_enabled", "0", "minillm_enabled must be bool, got int"),
        ("minillm_enabled", '"false"', "minillm_enabled must be bool, got str"),
        ("fp8_attention", "1", "fp8_attention must be bool, got int"),
        ("echo_trap_halt", "1", "echo_trap_halt must be bool, got int"),
        ("rl_checkpoint_include_optimizer", "1", "rl_checkpoint_include_optimizer must be bool"),
        ("reward_hack_rollback", "1", "reward_hack_rollback must be bool, got int"),
        ("num_labels", "true", "num_labels must not be bool"),
    ],
)
def test_load_config_from_string_raises_value_error(field, value, message):
    yaml_text = (
        "base: Qwen/Qwen2.5-0.5B-Instruct\n"
        "data:\n  train: train.jsonl\n"
        f"training:\n  {field}: {value}\n"
    )
    with pytest.raises(ValueError, match=message) as excinfo:
        load_config_from_string(yaml_text)
    assert not isinstance(excinfo.value, TypeError)
    assert f"training -> {field}:" in str(excinfo.value)


@pytest.mark.parametrize("command", ["train", "plan"])
def test_cli_names_the_field_and_exits_1(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    Path("train.jsonl").write_text('{"instruction": "hi", "output": "hello"}\n')
    Path("soup.yaml").write_text(
        "base: Qwen/Qwen2.5-0.5B-Instruct\n"
        "task: sft\n"
        "data:\n  train: ./train.jsonl\n  format: alpaca\n"
        "training:\n  minillm_enabled: 0\n"
    )

    result = CliRunner().invoke(app, [command, "--config", "soup.yaml"])

    text = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 1, (text, repr(result.exception))
    assert "Config validation error:" in text
    assert "training -> minillm_enabled" in text
    assert "TypeError" not in text
