"""#1412 - training.moe_aux_loss_coeff refused on backend: unsloth and stream_layers.

_setup_unsloth (sft, pretrain) and _setup_streaming_transformers never apply
the MoE router auxiliary loss coefficient, so non-default values are refused
at config load.
"""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string


def _yaml(
    task: str = "sft",
    backend: str = "transformers",
    stream_layers: bool = False,
    coeff: str | None = None,
) -> str:
    lines = ["base: some/model", f"task: {task}"]
    if backend != "transformers":
        lines.append(f"backend: {backend}")
    lines += ["data:", "  train: ./data.jsonl", "  format: auto"]
    training_lines = []
    if stream_layers:
        training_lines += ["  stream_layers: true", "  batch_size: 1"]
    if coeff is not None:
        training_lines.append(f"  moe_aux_loss_coeff: {coeff}")
    if training_lines:
        lines.append("training:")
        lines.extend(training_lines)
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("task", ["sft", "pretrain"])
def test_non_default_coeff_refused_on_backend_unsloth(task: str) -> None:
    with pytest.raises(ValueError, match=r"not applied on backend='unsloth'") as exc:
        load_config_from_string(_yaml(task=task, backend="unsloth", coeff="0.05"))
    assert "moe_aux_loss_coeff" in str(exc.value)
    assert "backend: transformers" in str(exc.value)


@pytest.mark.parametrize("task", ["sft", "pretrain"])
@pytest.mark.parametrize("coeff", [None, "0.01"])
def test_default_coeff_still_loads_on_backend_unsloth(task: str, coeff: str | None) -> None:
    cfg = load_config_from_string(_yaml(task=task, backend="unsloth", coeff=coeff))
    assert cfg.training.moe_aux_loss_coeff == 0.01


def test_non_default_coeff_refused_with_stream_layers() -> None:
    with pytest.raises(ValueError, match=r"not applied with training\.stream_layers") as exc:
        load_config_from_string(_yaml(task="sft", stream_layers=True, coeff="0.05"))
    assert "moe_aux_loss_coeff" in str(exc.value)
    assert "stream_layers" in str(exc.value)


@pytest.mark.parametrize("coeff", [None, "0.01"])
def test_default_coeff_still_loads_with_stream_layers(coeff: str | None) -> None:
    cfg = load_config_from_string(_yaml(task="sft", stream_layers=True, coeff=coeff))
    assert cfg.training.moe_aux_loss_coeff == 0.01


@pytest.mark.parametrize("task", ["sft", "pretrain"])
def test_transformers_backend_keeps_non_default_coeff(task: str) -> None:
    cfg = load_config_from_string(_yaml(task=task, backend="transformers", coeff="0.05"))
    assert cfg.training.moe_aux_loss_coeff == 0.05
