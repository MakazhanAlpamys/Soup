"""#3044: the VRAM estimate prices what an auto fp16 pick actually loads.

On a bf16 card, ``auto_mixed_precision`` can pick fp16 for a model in the fp16
quirk families (Qwen2, Qwen2.5, Phi-3). The trainer then keeps peft's fp32
embeddings, norms and head (``kbit_compute_dtype(..., prefer_fp16=True)``),
but the pre-flight used to price them as bf16 and under-count the run.

No GPU: the card's compute capability, the picker and the model shape are
patched; the assertion is on the ``kbit_fp32`` flag the shared fit input gets.
"""

from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils import hardware_fit, model_shape
from soup_cli.utils.hardware_fit import HardwareFitInput, hardware_fit_input_or_reason

_TINY_QWEN = {
    "model_type": "qwen2",
    "hidden_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "vocab_size": 32000,
}


def _cfg(*, auto: bool = True, base: str = "Qwen/Qwen2.5-7B-Instruct"):
    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "sft",
                "data": {"train": "train.jsonl", "max_length": 512},
                "training": {
                    "batch_size": 2,
                    "quantization": "4bit",
                    "auto_mixed_precision": auto,
                },
            }
        )
    )


@pytest.fixture
def bf16_card_picking_fp16(monkeypatch):
    """A bf16 card (cc 8.6) whose Qwen2.5 pick is fp16; no Hub lookup."""
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 6))
    monkeypatch.setattr(
        model_shape, "_hub_lookup", lambda repo, **kw: (dict(_TINY_QWEN), None)
    )
    monkeypatch.setattr(model_shape, "_hub_param_count", lambda repo, **kw: 7_000_000_000)


def test_an_auto_fp16_pick_on_a_bf16_card_prices_fp32_weights(bf16_card_picking_fp16):
    inp = hardware_fit_input_or_reason(_cfg(), allow_network=False)
    assert isinstance(inp, HardwareFitInput)
    assert inp.kbit_fp32 is True


def test_an_auto_bf16_pick_does_not_change_the_price(monkeypatch, bf16_card_picking_fp16):
    # A model outside the quirk map picks bf16 on the same card: peft's weights
    # are bf16 and the old price was right.
    from soup_cli.utils import mixed_precision

    monkeypatch.setattr(mixed_precision, "pick_mixed_precision", lambda name, cc: "bf16")
    inp = hardware_fit_input_or_reason(_cfg(), allow_network=False)
    assert isinstance(inp, HardwareFitInput)
    assert inp.kbit_fp32 is False


def test_without_auto_mixed_precision_the_price_is_unchanged(bf16_card_picking_fp16):
    inp = hardware_fit_input_or_reason(_cfg(auto=False), allow_network=False)
    assert isinstance(inp, HardwareFitInput)
    assert inp.kbit_fp32 is False


def test_a_card_without_a_readable_pick_keeps_the_old_answer(monkeypatch, bf16_card_picking_fp16):
    import torch

    def broken(device=None):
        raise RuntimeError("no card")

    monkeypatch.setattr(torch.cuda, "get_device_capability", broken)
    inp = hardware_fit_input_or_reason(_cfg(), allow_network=False)
    assert isinstance(inp, HardwareFitInput)
    assert inp.kbit_fp32 is False


def test_the_shared_helper_names_the_question(bf16_card_picking_fp16):
    cfg = _cfg()
    assert hardware_fit._auto_fp16_pick_keeps_fp32(cfg, cfg.training) is True
