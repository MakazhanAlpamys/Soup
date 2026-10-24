"""Exercise the vision/audio k-bit setup paths with real PEFT preparation.

Only Hub loading and the quantization loader boundary are replaced. The tiny
CPU model advertises a k-bit load to exercise PEFT's checkpointing behavior;
this does not qualify bitsandbytes quantization or multimodal GPU training.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.trainer.sft import SFTTrainerWrapper
from tests._tiny_hf_models import build_llama, build_tokenizer


@pytest.mark.parametrize("modality", ["vision", "audio"])
@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        pytest.param(None, False, id="omitted-default"),
        pytest.param(False, False, id="disabled"),
        pytest.param(True, True, id="enabled"),
        pytest.param("selective", True, id="selective"),
        pytest.param("medium", True, id="medium"),
        pytest.param("full", True, id="full"),
        pytest.param("auto", True, id="auto"),
    ],
)
def test_multimodal_kbit_setup_honors_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
    modality: str,
    quantization: str,
    setting: bool | str | None,
    expected: bool,
) -> None:
    import torch
    import transformers

    torch.manual_seed(797)
    tokenizer = build_tokenizer()
    base_model = build_llama(tokenizer)
    base_model.config.use_cache = False
    setattr(base_model, f"is_loaded_in_{quantization}", True)
    assert base_model.is_gradient_checkpointing is False

    processor = SimpleNamespace(tokenizer=tokenizer)
    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor
    )
    loader = (
        transformers.AutoModelForImageTextToText if modality == "vision" else transformers.AutoModel
    )
    monkeypatch.setattr(loader, "from_pretrained", lambda *args, **kwargs: base_model)
    monkeypatch.setattr(
        "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
        lambda **kwargs: None,
    )

    training = {
        "quantization": quantization,
        "lora": {"r": 2, "alpha": 4, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    }
    if setting is not None:
        training["gradient_checkpointing"] = setting
    cfg = SoupConfig(
        base="test-local-model",
        modality=modality,
        data={"train": "./unused.jsonl", "format": "llava" if modality == "vision" else "audio"},
        training=training,
    )
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    setup = getattr(wrapper, f"_setup_{modality}_transformers")
    setup(cfg, cfg.training)

    assert wrapper.model.is_gradient_checkpointing is expected
    assert wrapper.processor is processor
    assert wrapper.tokenizer is processor

    wrapper.model.train()
    input_ids = tokenizer("hello world", return_tensors="pt")["input_ids"]
    loss = wrapper.model(input_ids=input_ids, labels=input_ids).loss
    assert torch.isfinite(loss)
    loss.backward()

    adapters = {
        name: parameter
        for name, parameter in wrapper.model.named_parameters()
        if parameter.requires_grad
    }
    assert adapters
    assert all("lora_" in name for name in adapters)
    assert all(parameter.grad is not None for parameter in adapters.values())
    assert all(torch.isfinite(parameter.grad).all() for parameter in adapters.values())
    assert any(parameter.grad.abs().sum() > 0 for parameter in adapters.values())
    assert all(
        parameter.grad is None
        for parameter in wrapper.model.parameters()
        if not parameter.requires_grad
    )
