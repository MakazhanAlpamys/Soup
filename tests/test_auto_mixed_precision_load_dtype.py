"""``training.auto_mixed_precision`` loads a frozen base in the dtype it computes in.

``pick_mixed_precision`` answers fp16 for Qwen2 / Qwen2.5 / Phi-3 / Phi-3.5 on every
card from Pascal up, bf16-capable ones included. The SFT wrapper then loaded the base
at the checkpoint's own dtype, so on an Ampere-or-newer card a bf16 checkpoint ran
under fp16 autocast: every frozen weight was cast to an fp16 copy that the backward
pass kept, and the residual stream was promoted to fp32. A card without bf16 already
loads the base in fp16 for the same pick (#471), and now a card with bf16 does too.

Both dtypes, and the dtype a 4-bit/8-bit base keeps its unquantized weights in (#2683's
``_kbit_compute_dtype``), come from one answer asked once per run, so the "Auto
mixed-precision picked" line prints once; it printed twice on a k-bit run. ``task: tts``
trains through the same wrapper and gets the same load dtype.

The decision, and the ``dtype`` the SFT and TTS loads hand ``from_pretrained``, are
checked here on a faked card; ``test_real_card_loads_the_base_in_fp16`` checks both
wrappers on a real one.
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


def _config(task: str = "sft", base: str = QWEN, **overrides):
    from soup_cli.config.schema import SoupConfig

    training = {"quantization": "none", "auto_mixed_precision": True, **overrides}
    top = {}
    if task == "tts":
        top = {"modality": "audio_out"}
        training.setdefault("tts_family", "orpheus")
    return SoupConfig(
        base=base, task=task, data={"train": "./data.jsonl"}, training=training, **top
    )


def _training(task: str = "sft", **overrides):
    return _config(task, **overrides).training


def _wrapper_class(task: str):
    if task == "tts":
        from soup_cli.trainer.tts import TTSTrainerWrapper

        return TTSTrainerWrapper
    from soup_cli.trainer.sft import SFTTrainerWrapper

    return SFTTrainerWrapper


def _tiny_qwen2(path, dtype=None):
    """A tiny Qwen2 named like the ``qwen2.5`` quirk, saved with its tokenizer. Without
    ``dtype`` only the config is written, enough to reach the loader."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from tests._tiny_hf_models import build_tokenizer

    tok = build_tokenizer()
    config = Qwen2Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    if dtype is None:
        config.save_pretrained(path)
    else:
        Qwen2ForCausalLM(config).to(dtype).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def _wrapper(base: str = QWEN, device: str = "cuda", cls=None):
    from soup_cli.trainer.sft import SFTTrainerWrapper

    wrapper = object.__new__(cls or SFTTrainerWrapper)
    wrapper.device = device
    wrapper.config = SimpleNamespace(base=base)
    return wrapper


@pytest.fixture()
def picked_lines(monkeypatch):
    """The "Auto mixed-precision picked" lines the SFT wrapper prints."""
    from soup_cli.trainer import sft

    lines: list[str] = []

    def record(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if "Auto mixed-precision picked" in text:
            lines.append(text)

    monkeypatch.setattr(sft.console, "print", record)
    return lines


@pytest.fixture()
def picks(monkeypatch):
    """Every call to ``pick_mixed_precision``, the question the card is asked."""
    from soup_cli.utils import mixed_precision

    calls: list[tuple[str, float]] = []
    real = mixed_precision.pick_mixed_precision

    def counted(model_name, cc):
        calls.append((model_name, cc))
        return real(model_name, cc)

    monkeypatch.setattr(mixed_precision, "pick_mixed_precision", counted)
    return calls


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


def test_tts_loads_the_base_in_fp16_as_sft_does(card):
    """``task: tts`` is the other task that reads the field (``AMP_APPLYING_TASKS``) and
    trains through this wrapper's ``_setup_transformers``."""
    from soup_cli.trainer.tts import TTSTrainerWrapper

    card((12, 0))
    wrapper = _wrapper(cls=TTSTrainerWrapper)
    tcfg = _training(task="tts")

    assert wrapper._resolve_mixed_precision(tcfg, QWEN) == (False, True)
    assert wrapper._resolve_load_dtype(tcfg) is torch.float16


@pytest.mark.parametrize("task", ["sft", "tts"])
def test_the_model_load_is_handed_fp16(card, picked_lines, monkeypatch, tmp_path, task):
    """Through ``_setup_transformers`` up to ``from_pretrained``: ``TTSTrainerWrapper``
    overrides it and loads through the SFT text path, so ``task: tts`` gets the same
    ``dtype`` as ``task: sft``. Stopped at the loader, so nothing touches the card."""
    from soup_cli.utils import kimi_k2

    class _Loaded(Exception):
        pass

    seen: dict = {}

    def load(base, *, model_kwargs, console=None):
        seen.update(model_kwargs)
        raise _Loaded

    monkeypatch.setattr(kimi_k2, "load_resident_model", load)
    monkeypatch.chdir(tmp_path)
    # Imported before the fake card: peft imports bitsandbytes, which reads the real one.
    pytest.importorskip("peft")
    cfg = _config(task, base=_tiny_qwen2(tmp_path / "tiny-qwen2.5"))
    wrapper = _wrapper_class(task)(cfg, device="cuda")
    card((12, 0))

    with pytest.raises(_Loaded):
        wrapper._setup_transformers(cfg, cfg.training)
    assert seen["dtype"] is torch.float16
    assert len(picked_lines) == 1


@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
def test_a_kbit_run_asks_once_and_keeps_its_compute_dtype(
    card, picked_lines, picks, quantization
):
    """A 4-bit/8-bit run reaches the pick from the load, from ``_kbit_compute_dtype``
    (#2683) and from ``TrainingArguments``, in that order. One question, one line, and
    the k-bit branch keeps today's answer: fp16 compute, so PEFT's fp32 is kept."""
    card((12, 0))
    wrapper = _wrapper()
    tcfg = _training(quantization=quantization)

    assert wrapper._resolve_load_dtype(tcfg) == "auto"
    assert wrapper._kbit_compute_dtype(tcfg, QWEN) is torch.float16
    assert wrapper._resolve_mixed_precision(tcfg, QWEN) == (False, True)
    assert len(picks) == 1
    assert len(picked_lines) == 1
    assert "fp16" in picked_lines[0]


def test_a_lora_run_asks_once_for_both_dtypes(card, picked_lines, picks):
    card((12, 0))
    wrapper = _wrapper()
    tcfg = _training()

    assert wrapper._resolve_load_dtype(tcfg) is torch.float16
    assert wrapper._resolve_mixed_precision(tcfg, QWEN) == (False, True)
    assert len(picks) == 1
    assert len(picked_lines) == 1


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
@pytest.mark.parametrize("task", ["sft", "tts"])
def test_real_card_loads_the_base_in_fp16(tmp_path, monkeypatch, task):
    """A bf16-saved Qwen2 whose name matches the ``qwen2.5`` quirk, loaded by the SFT or
    TTS wrapper on the real card: its frozen weights are fp16, the precision it trains in."""
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("the card has no bf16, where the base already loaded fp16 before")

    base = _tiny_qwen2(tmp_path / "tiny-qwen2.5", dtype=torch.bfloat16)
    monkeypatch.chdir(tmp_path)
    cfg = _config(
        task, base=base,
        lora={"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    )
    wrapper = _wrapper_class(task)(cfg, device="cuda")
    wrapper._setup_transformers(cfg, cfg.training)

    frozen = {p.dtype for p in wrapper.model.parameters() if not p.requires_grad}
    assert wrapper._resolve_mixed_precision(cfg.training, cfg.base) == (False, True)
    assert frozen == {torch.float16}
