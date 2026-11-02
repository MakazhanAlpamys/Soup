"""Every config field either validates or raises ``ValidationError`` -- never ``TypeError``.

Pydantic converts a ``ValueError`` or an ``AssertionError`` raised inside a validator into
a ``ValidationError`` that carries the field's location. A ``TypeError`` is passed through
untouched, so it leaves ``SoupConfig(...)`` as a raw exception: the CLI's generic handler
prints ``Error: TypeError: ...`` with no section and no field name, and ``soup plan``
re-raises it as its own exit-2 path instead of the exit-1 validation path.

Before the wildcard wrap validator on ``TrainingConfig``, 38 of that model's fields let a
``TypeError`` out for at least one of the wrong-typed values below. The walk here is the
regression: it fails on ``main`` with exactly those fields, and deleting the validator
makes it fail again.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from soup_cli.config.schema import SoupConfig, TrainingConfig

# Values a YAML user can plausibly type into a field that expects a number, a bool or a
# string. Enough to reach a ``isinstance`` guard in every helper the schema delegates to.
WRONG_TYPED_VALUES: tuple[Any, ...] = (0, True, "false", [], {}, 1.5)

# Fields that leaked a TypeError before the wrap validator, measured on main. Kept as an
# explicit list so a new leak shows up as a new name rather than a changed count.
_KNOWN_LEAKS_BEFORE_FIX = {
    "bnb_4bit_use_double_quant",
    "citation_recall_threshold",
    "citation_style",
    "distill_divergence",
    "distill_mode",
    "distill_temperature",
    "ebft_temperature",
    "echo_trap_enabled",
    "echo_trap_halt",
    "echo_trap_tokenizer_aware",
    "fp8_attention",
    "grace_codebook_dim",
    "grace_codebook_size",
    "hub",
    "kv_cache_type",
    "llm_int8",
    "minillm_enabled",
    "minillm_length_normalize",
    "minillm_on_policy",
    "minillm_rollout_length",
    "mod_capacity_factor",
    "num_labels",
    "nvfp4",
    "quantize_ref_model",
    "quantize_reward_model",
    "ra_dit_retriever_model",
    "ra_dit_stage",
    "reasoning_effort",
    "reward_hack_conservative_on_disagreement",
    "reward_hack_halt",
    "reward_hack_reward_shaping",
    "reward_hack_rollback",
    "rl_checkpoint_include_optimizer",
    "rl_checkpoint_include_ref_model",
    "rl_checkpoint_include_rollout_buffer",
    "unlearn_alpha",
    "unlearn_method",
    "unsloth_bnb_4bit",
}


def _model_typeings(model: type[BaseModel]) -> set[str]:
    """Field names Pydantic treats as ``bool`` at the top level of ``model``."""
    return {
        name
        for name, field in model.model_fields.items()
        if field.annotation is bool or getattr(field.annotation, "__origin__", None) is bool
    }


def _fields_leaking_typeerror(model: type[BaseModel]) -> dict[str, str]:
    """Map each field that lets a ``TypeError`` escape to the value that triggered it."""
    leaks: dict[str, str] = {}
    for name in model.model_fields:
        for value in WRONG_TYPED_VALUES:
            try:
                model(**{name: value})
            except ValidationError:
                break
            except TypeError as exc:
                leaks[name] = f"{value!r} -> {exc}"
                break
        # Only the first failure per field is interesting; any success is a pass.
    return leaks


@pytest.mark.unit
def test_training_config_never_raises_typeerror() -> None:
    """The 38 fields that used to leak a TypeError now raise ValidationError."""
    leaks = _fields_leaking_typeerror(TrainingConfig)
    assert not leaks, f"TypeError escapes TrainingConfig for {sorted(leaks)}: {leaks}"


@pytest.mark.unit
def test_the_known_leak_set_is_the_one_that_used_to_fail() -> None:
    """Guard the regression's own coverage: the walk still visits the 38 known fields."""
    covered = set(TrainingConfig.model_fields) & _KNOWN_LEAKS_BEFORE_FIX
    assert covered == _KNOWN_LEAKS_BEFORE_FIX, sorted(_KNOWN_LEAKS_BEFORE_FIX - covered)


@pytest.mark.unit
def test_minillm_flag_names_the_field_and_the_section() -> None:
    """``minillm_enabled: 0`` reports ``training -> minillm_enabled``, not a release name."""
    with pytest.raises(ValidationError) as excinfo:
        TrainingConfig(minillm_enabled=0)
    errors = excinfo.value.errors()
    assert len(errors) == 1, errors
    error = errors[0]
    assert error["loc"] == ("minillm_enabled",), error["loc"]
    # The message has to name the field, otherwise the user has to guess which key to fix.
    assert "minillm_enabled" in error["msg"], error["msg"]


@pytest.mark.unit
def test_minillm_rollout_length_names_the_field() -> None:
    """A bool here is refused by the field's own validator, and the message names it."""
    with pytest.raises(ValidationError) as excinfo:
        TrainingConfig(minillm_rollout_length=True)
    assert "minillm_rollout_length" in excinfo.value.errors()[0]["msg"], excinfo.value.errors()


@pytest.mark.unit
def test_num_labels_bool_is_a_validation_error() -> None:
    """``num_labels`` goes through a helper in ``utils/classifier.py``; it still converts."""
    with pytest.raises(ValidationError) as excinfo:
        TrainingConfig(num_labels=True)
    assert "num_labels" in excinfo.value.errors()[0]["msg"]


@pytest.mark.unit
def test_distill_temperature_bool_is_a_validation_error() -> None:
    """``distill_temperature`` goes through a helper in ``trainer/distill.py``."""
    with pytest.raises(ValidationError) as excinfo:
        TrainingConfig(distill_temperature=True)
    assert "distill_temperature" in str(excinfo.value.errors()[0]["loc"])


@pytest.mark.unit
def test_soup_config_surfaces_the_section_and_field() -> None:
    """Through the outer model the location is ``training -> <field>``, which is what prints."""
    with pytest.raises(ValidationError) as excinfo:
        SoupConfig(task="sft", training={"minillm_enabled": 0})
    locations = [
        " -> ".join(str(part) for part in error["loc"]) for error in excinfo.value.errors()
    ]
    assert any("minillm_enabled" in loc for loc in locations), locations
