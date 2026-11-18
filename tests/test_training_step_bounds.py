"""``training.save_steps``/``logging_steps`` bounds (#2892).

``save_steps`` refuses negative values (``ge=0``) and ``logging_steps`` refuses
0 or below (``ge=1``). ``save_steps: 0`` stays accepted: it is the only way to
switch periodic checkpoints off until #2575 adds one, so what 0 means is #2575's
to say. transformers reads a ``*_steps`` value below 1 as a fraction of
``max_steps`` and the save branch is guarded by ``state.save_steps > 0``, so a
negative ``save_steps`` silently dropped checkpoints and ``logging_steps`` of 0
or below silently dropped logging (or failed late, inside ``TrainingArguments``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig

_YAML = "base: some-model\ndata:\n  train: ./data.jsonl\ntraining:\n  {field}: {value}\n"


def _config(**training):
    return SoupConfig(base="some-model", data={"train": "./data.jsonl"}, training=training)


@pytest.mark.parametrize("value", [-1, -100])
def test_save_steps_refuses_negative(value):
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _config(save_steps=value)


@pytest.mark.parametrize("value", [0, 1, 100])
def test_save_steps_accepts_zero_and_above(value):
    assert _config(save_steps=value).training.save_steps == value


@pytest.mark.parametrize("value", [0, -1, -100])
def test_logging_steps_refuses_zero_and_negative(value):
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _config(logging_steps=value)


@pytest.mark.parametrize("value", [1, 10])
def test_logging_steps_accepts_one_and_above(value):
    assert _config(logging_steps=value).training.logging_steps == value


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("save_steps", -1, "training -> save_steps: Input should be greater than or equal to 0"),
        (
            "logging_steps",
            0,
            "training -> logging_steps: Input should be greater than or equal to 1",
        ),
        (
            "logging_steps",
            -1,
            "training -> logging_steps: Input should be greater than or equal to 1",
        ),
    ],
)
def test_refusal_through_the_yaml_loader_names_the_field(field, value, message):
    with pytest.raises(ValueError) as info:
        load_config_from_string(_YAML.format(field=field, value=value))
    assert message in str(info.value)


@pytest.mark.parametrize("field", ["save_steps", "logging_steps"])
def test_bool_is_refused_by_the_generic_check_not_the_bound(field):
    """``save_steps: true`` must keep saying "an int, not a bool" (#2462)."""
    with pytest.raises(ValidationError, match=f"{field} must be an int, not a bool"):
        _config(**{field: True})


def test_eval_steps_still_refuses_zero():
    """The neighbour keeps its own ``ge=1`` (this fix does not touch it)."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        _config(eval_steps=0)
