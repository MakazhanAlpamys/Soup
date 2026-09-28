"""#1394 — ``training.moe_aux_loss_coeff`` is read only by SFT's text setup.

``SFTTrainerWrapper._setup_transformers`` applies it; the vision and audio
setups never do, so a non-default value on those modalities loaded and was
silently ignored. It is refused at load now, like #1179's ``moe_lora``.
"""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string


def _yaml(task: str, modality: str | None, coeff: str | None) -> str:
    lines = ["base: some/model", f"task: {task}"]
    if modality is not None:
        lines.append(f"modality: {modality}")
    lines += ["data:", "  train: ./data.jsonl", "  format: auto"]
    if coeff is not None:
        lines += ["training:", f"  moe_aux_loss_coeff: {coeff}"]
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("modality", ["vision", "audio"])
def test_non_default_coeff_refused_on_vision_and_audio(modality):
    with pytest.raises(ValueError, match=rf"modality='{modality}'") as exc:
        load_config_from_string(_yaml("sft", modality, "0.05"))
    assert "moe_aux_loss_coeff" in str(exc.value)
    assert "modality: text" in str(exc.value)


@pytest.mark.parametrize("modality", [None, "text", "vision", "audio"])
@pytest.mark.parametrize("coeff", [None, "0.01"])
def test_default_coeff_still_loads_everywhere(modality, coeff):
    cfg = load_config_from_string(_yaml("sft", modality, coeff))
    assert cfg.training.moe_aux_loss_coeff == 0.01


@pytest.mark.parametrize("modality", [None, "text"])
def test_text_sft_keeps_a_non_default_coeff(modality):
    cfg = load_config_from_string(_yaml("sft", modality, "0.05"))
    assert cfg.training.moe_aux_loss_coeff == 0.05


def test_pretrain_keeps_a_non_default_coeff():
    cfg = load_config_from_string(_yaml("pretrain", None, "0.05"))
    assert cfg.training.moe_aux_loss_coeff == 0.05
