"""``training.auto_mixed_precision`` loads a frozen base in the dtype it computes in.

``pick_mixed_precision`` answers fp16 for Qwen2 / Qwen2.5 / Phi-3 / Phi-3.5 on every
card from Pascal up, bf16-capable ones included. The SFT wrapper then loaded the base
at the checkpoint's own dtype, so on an Ampere-or-newer card a bf16 checkpoint ran
under fp16 autocast: every frozen weight was cast to an fp16 copy that the backward
pass kept, and the residual stream was promoted to fp32. A card without bf16 already
loads the base in fp16 for the same pick (#471), and now a card with bf16 does too.

The decision is checked here on a faked card; ``test_real_card_loads_the_base_in_fp16``
checks it on a real one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="torch is only in the [train] extra")

QWEN = "Qwen/Qwen2.5-1.5B-Instruct"
LLAMA = "meta-llama/Llama-3.2-1B-Instruct"


class _FakeCuda:
    """A CUDA card of compute capability ``cc``; native bf16 from 8.0, as the driver says."""

    def __init__(self, cc: tuple[int, int]):
        self._cc = cc

    def is_available(self) -> bool:
        return True

    def is_bf16_supported(self, including_emulation: bool = True) -> bool:
        return self._cc >= (8, 0) or including_emulation

    def get_device_capability(self, device=None):
        return self._cc


@pytest.fixture()
def card(monkeypatch):
    def apply(cc: tuple[int, int]) -> None:
        monkeypatch.setattr(torch, "cuda", _FakeCuda(cc))

    return apply


def _training(**overrides):
    from soup_cli.config.schema import SoupConfig

    training = {"quantization": "none", "auto_mixed_precision": True, **overrides}
    return SoupConfig(
        base=QWEN, task="sft", data={"train": "./data.jsonl"}, training=training
    ).training


def _wrapper(base: str = QWEN, device: str = "cuda"):
    from soup_cli.trainer.sft import SFTTrainerWrapper

    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.device = device
    wrapper.config = SimpleNamespace(base=base)
    return wrapper


@pytest.mark.parametrize("cc", [(8, 0), (8, 6), (8, 9), (9, 0), (12, 0)])
def test_fp16_pick_on_a_bf16_card_loads_the_base_in_fp16(card, cc):
    card(cc)
    wrapper = _wrapper()
    tcfg = _training()

    assert wrapper._resolve_mixed_precision(tcfg, QWEN) == (False, True)
    assert wrapper._resolve_load_dtype(tcfg) is torch.float16


def test_bf16_pick_keeps_the_checkpoint_dtype(card):
    card((12, 0))
    wrapper = _wrapper(base=LLAMA)
    tcfg = _training()

    assert wrapper._resolve_mixed_precision(tcfg, LLAMA) == (True, False)
    assert wrapper._resolve_load_dtype(tcfg) == "auto"


def test_without_auto_mixed_precision_the_checkpoint_dtype_stays(card):
    card((12, 0))
    assert _wrapper()._resolve_load_dtype(_training(auto_mixed_precision=False)) == "auto"


@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
def test_a_quantized_base_is_loaded_as_before(card, quantization):
    """k-bit preparation casts the unquantized weights to fp32 whatever they load as."""
    card((12, 0))
    assert _wrapper()._resolve_load_dtype(_training(quantization=quantization)) == "auto"


def test_full_fine_tuning_keeps_fp32_master_weights(card):
    card((12, 0))
    tcfg = _training(lora={"r": 0})
    assert _wrapper()._resolve_load_dtype(tcfg) is torch.float32


def test_a_card_without_bf16_already_loaded_fp16(card):
    """Control: the pre-Ampere answer the fix extends, unchanged."""
    card((7, 5))
    wrapper = _wrapper()
    tcfg = _training()

    assert wrapper._resolve_mixed_precision(tcfg, QWEN) == (False, True)
    assert wrapper._resolve_load_dtype(tcfg) is torch.float16


def test_cpu_keeps_the_checkpoint_dtype(card):
    card((12, 0))
    assert _wrapper(device="cpu")._resolve_load_dtype(_training()) == "auto"


def test_an_unreadable_card_keeps_the_checkpoint_dtype(monkeypatch, card):
    card((12, 0))

    def broken(device=None):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", broken)
    assert _wrapper()._resolve_load_dtype(_training()) == "auto"


@pytest.mark.gpu
def test_real_card_loads_the_base_in_fp16(tmp_path, monkeypatch):
    """A bf16-saved Qwen2 whose name matches the ``qwen2.5`` quirk, loaded by the SFT
    wrapper on the real card: its frozen weights are fp16, the precision it trains in."""
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("the card has no bf16, where the base already loaded fp16 before")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests._tiny_hf_models import build_tokenizer

    base = tmp_path / "tiny-qwen2.5"
    tok = build_tokenizer()
    config = Qwen2Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    Qwen2ForCausalLM(config).to(torch.bfloat16).save_pretrained(base)
    tok.save_pretrained(base)

    monkeypatch.chdir(tmp_path)
    cfg = SoupConfig(
        base=str(base), task="sft", data={"train": "./data.jsonl"},
        training={"quantization": "none", "auto_mixed_precision": True,
                  "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]}},
    )
    wrapper = SFTTrainerWrapper(cfg, device="cuda")
    wrapper._setup_transformers(cfg, cfg.training)

    frozen = {p.dtype for p in wrapper.model.parameters() if not p.requires_grad}
    assert wrapper._resolve_mixed_precision(cfg.training, cfg.base) == (False, True)
    assert frozen == {torch.float16}
