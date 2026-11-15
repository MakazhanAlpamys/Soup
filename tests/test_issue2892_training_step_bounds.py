"""``training.save_steps``/``logging_steps`` refuse 0 and negative values (#2892).

Before this, both fields were unbounded while ``training.eval_steps`` already
carried ``ge=1``. transformers' ``TrainerState.compute_steps`` reads a value
``< 1`` as a *fraction of max_steps*, so ``save_steps: 0`` silently stopped
saving intermediate checkpoints and ``logging_steps: -1`` effectively turned
logging off, while ``logging_steps: 0`` only failed later, inside
``TrainingArguments``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig

_YAML = "base: some-model\ndata:\n  train: ./data.jsonl\ntraining:\n  {field}: {value}\n"


def _config(**training):
    return SoupConfig(base="some-model", data={"train": "./data.jsonl"}, training=training)


@pytest.mark.parametrize("field", ["save_steps", "logging_steps"])
@pytest.mark.parametrize("value", [0, -1, -100])
def test_step_fields_refuse_zero_and_negative(field, value):
    with pytest.raises(ValidationError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize("field", ["save_steps", "logging_steps"])
@pytest.mark.parametrize("value", [0, -1])
def test_step_fields_refused_through_the_yaml_loader(field, value):
    with pytest.raises(ValueError, match=field):
        load_config_from_string(_YAML.format(field=field, value=value))


@pytest.mark.parametrize(("field", "default"), [("save_steps", 100), ("logging_steps", 10)])
def test_step_field_default_and_boundary_are_accepted(field, default):
    assert getattr(_config().training, field) == default
    assert getattr(_config(**{field: 1}).training, field) == 1


def test_eval_steps_still_refuses_zero():
    """The neighbour keeps its own ``ge=1`` (this fix does not touch it)."""
    with pytest.raises(ValidationError, match="eval_steps"):
        _config(eval_steps=0)
