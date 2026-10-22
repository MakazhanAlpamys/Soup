"""#1811 — ``quantization: 8bit`` on ``backend: unsloth`` is refused, not loaded in 16-bit.

``utils/unsloth.py`` computed ``load_in_4bit = quantization == "4bit"`` and passed
nothing else, so ``8bit`` loaded exactly like ``none``: a full 16-bit model, with
no word, on a config the schema accepted. The declared floor is
``unsloth>=2024.8``, whose ``FastLanguageModel.from_pretrained`` has no
``load_in_8bit``, so the value cannot be passed through; the config is refused at
load instead, naming the two values the Unsloth path does apply.
"""

from __future__ import annotations

import sys
import types

import pytest

from soup_cli.config.loader import load_config_from_string


def _yaml(task: str, quantization: str, backend: str = "unsloth") -> str:
    return (
        f"base: meta-llama/Llama-3.1-8B\ntask: {task}\nbackend: {backend}\n"
        f"data:\n  train: d.jsonl\ntraining:\n  quantization: {quantization}\n"
    )


@pytest.mark.parametrize("task", ["sft", "dpo", "grpo"])
def test_8bit_on_unsloth_is_refused_naming_what_applies(task):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml(task, "8bit"))
    message = " ".join(str(excinfo.value).split())
    assert "quantization: 8bit" in message
    assert "backend: unsloth" in message
    assert "4bit" in message and "none" in message


@pytest.mark.parametrize("quantization", ["4bit", "none"])
def test_control_the_values_unsloth_applies_still_load(quantization):
    cfg = load_config_from_string(_yaml("sft", quantization))
    assert (cfg.backend, cfg.training.quantization) == ("unsloth", quantization)


def test_control_8bit_on_transformers_still_loads():
    cfg = load_config_from_string(_yaml("sft", "8bit", backend="transformers"))
    assert cfg.training.quantization == "8bit"


def test_the_loader_refuses_8bit_rather_than_loading_16bit(monkeypatch):
    # The config refusal is the gate; the loader names the same thing if it is
    # ever reached another way, instead of loading the model in 16-bit.
    seen = []
    fast = types.SimpleNamespace(
        from_pretrained=lambda **kw: seen.append(kw) or (object(), object()),
        get_peft_model=lambda model, **kw: model,
    )
    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=fast))
    from soup_cli.utils.unsloth import load_model_and_tokenizer

    with pytest.raises(ValueError, match="8bit"):
        load_model_and_tokenizer("m", 512, quantization="8bit")
    assert seen == []

    load_model_and_tokenizer("m", 512, quantization="4bit")
    assert seen[-1]["load_in_4bit"] is True
    load_model_and_tokenizer("m", 512, quantization="none")
    assert seen[-1]["load_in_4bit"] is False
