"""#2044 -- ``training.use_ring_attention: true`` is refused at config load.

The field is validated in ``commands/train.py`` (CUDA, the extra, two GPUs,
max_length >= 8192) and then read by no trainer, so a run that passed those
checks trained every rank on the full sequence under ordinary attention without
a word about the flag being ignored. This file pins the refusal on every task
and backend, the message (flag, missing sequence parallelism, the alternative
and the issue), and the values that must keep loading.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import TypeAdapter, ValidationError

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig, TrainingConfig

TASKS = ("sft", "grpo", "dpo", "pretrain")
BACKENDS = ("transformers", "unsloth", "mlx")


def _yaml(value: str, *, task: str = "sft", backend: str = "transformers") -> str:
    return (
        f"base: org/base\ntask: {task}\nbackend: {backend}\n"
        "data: {train: ./data.jsonl, max_length: 16384}\n"
        "training:\n  quantization: none\n"
        f"  use_ring_attention: {value}\n"
    )


def _message(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_every_task_and_backend_refuses_the_flag(task, backend):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true", task=task, backend=backend))
    assert "training.use_ring_attention" in _message(str(excinfo.value))


@pytest.mark.parametrize(
    "value",
    ["true", "True", "yes", "on", "y", "t", "1", "1.0", "'true'", "'yes'", "'1'"],
)
def test_every_true_spelling_is_refused(value):
    with pytest.raises(ValueError):
        load_config_from_string(_yaml(value))


@pytest.mark.parametrize(
    "value", [True, 1, 1.0, Decimal(1), "true", "yes", "y", "on", "t", "1", b"true"]
)
def test_direct_construction_refuses_every_true_value(value):
    with pytest.raises(ValidationError):
        TrainingConfig(use_ring_attention=value)


def test_the_message_names_the_missing_implementation_and_the_alternative():
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml("true"))
    text = _message(str(excinfo.value))
    assert "training.use_ring_attention" in text
    assert "#2044" in text
    assert "no trainer" in text
    assert "sequence parallelism" in text
    assert "rope_scaling_type" in text


@pytest.mark.parametrize(
    "value", ["false", "False", "no", "off", "n", "f", "0", "0.0", "'false'", "'0'"]
)
def test_false_spellings_still_load(value):
    cfg = load_config_from_string(_yaml(value))
    assert cfg.training.use_ring_attention is False


def test_the_default_is_unchanged():
    cfg = load_config_from_string(
        "base: org/base\ntask: sft\ndata: {train: d.jsonl}\ntraining:\n  quantization: none\n"
    )
    assert cfg.training.use_ring_attention is False


def test_the_recommended_alternative_still_loads():
    cfg = load_config_from_string(
        _yaml("false") + "  rope_scaling_type: dynamic\n"
    )
    assert cfg.training.rope_scaling_type == "dynamic"
    assert cfg.training.use_ring_attention is False


def test_a_full_soup_config_direct_construction_refuses_too():
    with pytest.raises(ValidationError):
        SoupConfig.model_validate(
            {
                "base": "org/base",
                "task": "sft",
                "data": {"train": "d.jsonl"},
                "training": {"use_ring_attention": True},
            }
        )


def test_the_schema_adapter_refuses_the_flag_after_coercion():
    adapter = TypeAdapter(TrainingConfig)
    with pytest.raises(ValidationError):
        adapter.validate_python({"use_ring_attention": "on"})


def test_no_field_annotation_changed():
    field = TrainingConfig.model_fields["use_ring_attention"]
    assert field.annotation is bool
    assert field.default is False


def test_both_refused_flags_report_both_reasons():
    """#1240 and #2044 are both named when both flags are set."""
    text = (
        "base: org/base\ntask: sft\n"
        "data: {train: ./data.jsonl}\n"
        "training:\n  use_longlora: true\n  use_ring_attention: true\n"
    )
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(text)
    message = _message(str(excinfo.value))
    assert "use_longlora" in message and "#1240" in message
    assert "use_ring_attention" in message and "#2044" in message
