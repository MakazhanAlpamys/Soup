"""`SFTTrainerWrapper` never passed `dtype` to `from_pretrained` (issue #339).

All three modality paths (text/vision/audio) built `model_kwargs` with
`trust_remote_code`/`device_map`/optional `quantization_config` but no
`dtype`, so `from_pretrained` defaulted to float32 regardless of the
checkpoint's own dtype — a bf16 checkpoint cost 2x the VRAM it needed.
Measured on an H100, LoRA, Llama-3.1-8B: 48,241 MiB (fp32) -> 18,658 MiB
(`dtype="auto"`), a 28.9 GB / 2.59x saving, byte-identical across 3 repeats.

The fix is a numerics DECISION, not a one-liner: a frozen base (LoRA/QLoRA)
loads at the checkpoint's own dtype (`dtype="auto"`); a trainable base (full
fine-tuning — Spectrum `unfrozen_parameters`, LISA `lisa_enabled`, or the
#340 `lora.r=0` spelling) explicitly loads `torch.float32` master weights,
deliberately rather than by accident.

**#471 review round added two things**, tested below alongside the original four:

- `_resolve_load_dtype` is now card-aware: an unconditional `"auto"` for a
  frozen base would give bf16 STORAGE on a pre-Ampere CUDA card (T4/P100/V100/
  GTX 16xx/RTX 20xx) while training compute correctly stays fp16 (asked via
  `bf16_fp16_flags`, the same helper `_resolve_mixed_precision` already uses)
  — the exact bf16-storage/fp16-compute split v0.73.1 (#385/#387) removed
  from fourteen other places. Pre-Ampere now gets an explicit `torch.float16`
  override instead of `"auto"`.
- The full-FT discriminator (`unfrozen_parameters` / `lisa_enabled` /
  `lora.r==0`) is now a single shared `is_full_finetune()`, used by both
  `_resolve_load_dtype` and `commands/train.py::_build_hardware_fit_input`'s
  VRAM pre-flight `peft` classifier — previously independent copies that had
  drifted apart in BOTH directions (missing `lisa_enabled`/`lora.r==0`
  under-predicted VRAM; treating bare `freeze_layers`/`freeze_ratio` as
  sufficient on its own, even with LoRA still on, over-predicted it and could
  falsely refuse a launch that would fit).

Layers, matching the issue's four acceptance criteria plus the two additions:

- `TestResolveLoadDtype` — unit tests of `_resolve_load_dtype` in isolation,
  no model load (now an instance method — needs `self.device`).
- `TestResolveLoadDtypeCardAware` — the #471 pre-Ampere/Ampere/CPU card check,
  using the same fake-`torch.cuda` pattern as `test_issue385_stream_dtype.py`.
- `TestModelKwargsCaptureAcrossModalities` — mock `from_pretrained` at all
  three call sites (text/vision/audio) and assert on the captured `dtype` kwarg,
  including the QLoRA case (`dtype` + `quantization_config` coexisting).
- `TestLoadDtypeMatchesCheckpoint` — the literal AC1 claim, with real tiny
  on-disk checkpoints: a LoRA run on a bf16 checkpoint really does load bf16,
  and — the control that makes that mean something — a LoRA run on an fp32
  checkpoint stays fp32 rather than "auto" secretly meaning "always bf16".
- `TestHardwareFitFullFTWeightsBytes` — AC4, the VRAM pre-flight's
  bytes-per-param assumption re-checked against the choice made above.
- `TestIsFullFinetuneSharedDiscriminator` — the #471 unification: direct unit
  tests of `is_full_finetune`, plus `_build_hardware_fit_input`'s corrected
  classification for both the previously under- and over-counted cases.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig


class _FakeCuda:
    """Stands in for ``torch.cuda`` (mirrors tests/test_issue385_stream_dtype.py's
    fixture of the same name/shape — copied rather than imported cross-file,
    matching this project's convention for small test doubles).

    ``is_bf16_supported`` mirrors the real signature: the default
    ``including_emulation=True`` returns True on a T4 (software emulation),
    while ``including_emulation=False`` correctly returns False — the trap
    `cuda_supports_bf16()` is written to avoid.
    """

    def __init__(self, available: bool, bf16: bool, emulated: bool = True):
        self._available = available
        self._bf16 = bf16
        self._emulated = emulated

    def is_available(self) -> bool:
        return self._available

    def is_bf16_supported(self, including_emulation: bool = True) -> bool:
        if not including_emulation:
            return self._bf16
        return self._bf16 or self._emulated

    def get_device_capability(self, device=None):
        return (8, 0) if self._bf16 else (7, 5)


@pytest.fixture()
def fake_torch_cuda(monkeypatch):
    """Patch ``torch.cuda`` in place; bf16_fp16_flags imports torch lazily."""
    import torch

    def apply(available: bool, bf16: bool, emulated: bool = True) -> _FakeCuda:
        fake = _FakeCuda(available, bf16, emulated)
        monkeypatch.setattr(torch, "cuda", fake)
        return fake

    return apply


def _make_config(**overrides):
    base = {
        "base": "test-model",
        "data": {"train": "./data.jsonl", "format": "alpaca"},
    }
    base.update(overrides)
    return SoupConfig(**base)


# ---------------------------------------------------------------------------
# Unit tests of _resolve_load_dtype — no model, no [train] extra required for
# the "auto" cases (torch is only imported inside the fp32 branch).
# ---------------------------------------------------------------------------


class _StubLora:
    def __init__(self, r=8):
        self.r = r


class _StubTcfg:
    def __init__(self, r=8, unfrozen_parameters=None, lisa_enabled=False):
        self.lora = _StubLora(r)
        self.unfrozen_parameters = unfrozen_parameters
        self.lisa_enabled = lisa_enabled


def _wrapper_stub(device="cpu"):
    """A bare SFTTrainerWrapper with only `.device` set — `_resolve_load_dtype`
    is an instance method (needs `self.device` for the #471 card check), so it
    can no longer be called unbound on the class."""
    from soup_cli.trainer.sft import SFTTrainerWrapper

    wrapper = object.__new__(SFTTrainerWrapper)
    wrapper.device = device
    return wrapper


class TestResolveLoadDtype:
    def test_lora_frozen_base_on_cpu_resolves_to_auto(self):
        wrapper = _wrapper_stub(device="cpu")
        assert wrapper._resolve_load_dtype(_StubTcfg()) == "auto"

    def test_lora_r_zero_resolves_to_float32(self):
        torch = pytest.importorskip("torch", reason="torch is only in the [train] extra")
        wrapper = _wrapper_stub(device="cpu")
        assert wrapper._resolve_load_dtype(_StubTcfg(r=0)) is torch.float32

    def test_unfrozen_parameters_resolves_to_float32(self):
        torch = pytest.importorskip("torch", reason="torch is only in the [train] extra")
        wrapper = _wrapper_stub(device="cpu")
        tcfg = _StubTcfg(unfrozen_parameters=[".*"])
        assert wrapper._resolve_load_dtype(tcfg) is torch.float32

    def test_lisa_enabled_resolves_to_float32(self):
        torch = pytest.importorskip("torch", reason="torch is only in the [train] extra")
        wrapper = _wrapper_stub(device="cpu")
        tcfg = _StubTcfg(lisa_enabled=True)
        assert wrapper._resolve_load_dtype(tcfg) is torch.float32


class TestResolveLoadDtypeCardAware:
    """#471 — the frozen-base case must ask the card, not just the checkpoint."""

    def test_ampere_cuda_frozen_base_still_resolves_to_auto(self, fake_torch_cuda):
        """CONTROL — the card check must not regress the common case."""
        fake_torch_cuda(available=True, bf16=True)
        wrapper = _wrapper_stub(device="cuda")
        assert wrapper._resolve_load_dtype(_StubTcfg()) == "auto"

    def test_pre_ampere_cuda_frozen_base_resolves_to_float16(self, fake_torch_cuda):
        """THE fix — a T4 (bf16=False) must not get 'auto' on a bf16 checkpoint."""
        torch = pytest.importorskip("torch", reason="torch is only in the [train] extra")
        fake_torch_cuda(available=True, bf16=False)
        wrapper = _wrapper_stub(device="cuda")
        assert wrapper._resolve_load_dtype(_StubTcfg()) is torch.float16

    def test_emulated_bf16_does_not_count_as_bf16(self, fake_torch_cuda):
        """The exact trap #385 named: the bare is_bf16_supported() call
        returns True on a T4 through software emulation; the real answer must
        come from including_emulation=False (cuda_supports_bf16's job)."""
        torch = pytest.importorskip("torch", reason="torch is only in the [train] extra")
        fake = fake_torch_cuda(available=True, bf16=False, emulated=True)
        assert fake.is_bf16_supported() is True, "the stub must model the trap"
        wrapper = _wrapper_stub(device="cuda")
        assert wrapper._resolve_load_dtype(_StubTcfg()) is torch.float16

    def test_cpu_frozen_base_stays_auto_regardless_of_cuda_state(self, fake_torch_cuda):
        """A device='cpu' run must not consult CUDA capability at all — even
        on a box that also has a (possibly pre-Ampere) GPU."""
        fake_torch_cuda(available=True, bf16=False)
        wrapper = _wrapper_stub(device="cpu")
        assert wrapper._resolve_load_dtype(_StubTcfg()) == "auto"

    def test_card_check_does_not_apply_to_full_finetune(self, fake_torch_cuda):
        """CONTROL — full-FT's explicit fp32 must win regardless of card;
        the card check only ever applies to the frozen-base branch."""
        torch = pytest.importorskip("torch", reason="torch is only in the [train] extra")
        fake_torch_cuda(available=True, bf16=False)
        wrapper = _wrapper_stub(device="cuda")
        assert wrapper._resolve_load_dtype(_StubTcfg(r=0)) is torch.float32


# ---------------------------------------------------------------------------
# Mock-based: model_kwargs["dtype"] capture at all three call sites.
# Fake transformers/peft modules, same shape as tests/test_trainer_init.py.
# ---------------------------------------------------------------------------


class _FakeParam:
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad


class _FakeModel:
    config = SimpleNamespace()

    def __init__(self):
        self._params = [_FakeParam(requires_grad=True)]

    def parameters(self):
        return self._params

    def named_parameters(self):
        return [("weight", p) for p in self._params]

    def resize_token_embeddings(self, size):
        pass


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def get_vocab(self):
        return {}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


def _install_common_trainer_mocks(monkeypatch, quant_config_obj=None):
    """The non-transformers/-peft helper mocks shared by every _setup_*
    mock test in tests/test_trainer_init.py — copied verbatim so
    _setup_transformers/_setup_vision_transformers/_setup_audio_transformers
    can run to completion against fake models without touching real MoE
    detection, block expansion, etc.

    ``quant_config_obj`` defaults to ``None`` (no quantization) but a caller
    can pass a sentinel to simulate an active `quantization_config` — used by
    the QLoRA test below to prove `dtype` and `quantization_config` coexist.
    """
    monkeypatch.setattr(
        "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
        lambda **kwargs: quant_config_obj,
    )
    monkeypatch.setattr("soup_cli.utils.moe.detect_moe_model", lambda _m: False)
    monkeypatch.setattr("soup_cli.utils.moe.get_moe_target_modules", lambda _m: [])
    monkeypatch.setattr(
        "soup_cli.utils.block_expansion.apply_block_expansion_if_configured",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "soup_cli.utils.moe_quant.apply_moe_expert_quant_if_configured",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "soup_cli.utils.peft_wiring.apply_pre_lora_patches",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "soup_cli.utils.peft_wiring.apply_post_lora_patches",
        lambda *args, **kwargs: None,
    )


def _capturing_from_pretrained(model, captured):
    def _fake(*args, **kwargs):
        captured.update(kwargs)
        return model

    return _fake


class TestModelKwargsCaptureAcrossModalities:
    def test_text_lora_dtype_is_auto(self, monkeypatch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config(training={"quantization": "none"})
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        captured = {}

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=_capturing_from_pretrained(model, captured)
            ),
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=lambda model_obj, _cfg: model_obj,
            prepare_model_for_kbit_training=lambda model_obj: model_obj,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        _install_common_trainer_mocks(monkeypatch)

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False
        wrapper.model = None
        wrapper.tokenizer = None

        wrapper._setup_transformers(cfg, cfg.training)

        assert captured["dtype"] == "auto"

    def test_text_lora_dtype_is_float16_on_pre_ampere_cuda(self, monkeypatch, fake_torch_cuda):
        """#471 — the card check reaches all the way into model_kwargs, not
        just _resolve_load_dtype in isolation."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        torch = pytest.importorskip("torch", reason="torch is only in the [train] extra")
        fake_torch_cuda(available=True, bf16=False)  # a T4

        cfg = _make_config(training={"quantization": "none"})
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        captured = {}

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=_capturing_from_pretrained(model, captured)
            ),
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=lambda model_obj, _cfg: model_obj,
            prepare_model_for_kbit_training=lambda model_obj: model_obj,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        _install_common_trainer_mocks(monkeypatch)

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cuda"
        wrapper._trust_remote_code = False
        wrapper.model = None
        wrapper.tokenizer = None

        wrapper._setup_transformers(cfg, cfg.training)

        assert captured["dtype"] is torch.float16

    def test_text_qlora_dtype_and_quantization_config_coexist(self, monkeypatch):
        """#471 review — QLoRA now also gets model_kwargs["dtype"] (since a
        quantized run is never is_full_finetune); confirm it lands alongside
        quantization_config rather than one clobbering the other."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        sentinel_quant_config = SimpleNamespace(marker="FAKE_4BIT_CONFIG")
        cfg = _make_config(training={"quantization": "4bit"})
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        captured = {}

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=_capturing_from_pretrained(model, captured)
            ),
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=lambda model_obj, _cfg: model_obj,
            prepare_model_for_kbit_training=lambda model_obj: model_obj,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        _install_common_trainer_mocks(monkeypatch, quant_config_obj=sentinel_quant_config)

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False
        wrapper.model = None
        wrapper.tokenizer = None

        wrapper._setup_transformers(cfg, cfg.training)

        assert captured["dtype"] == "auto"  # frozen base (QLoRA is never full-FT)
        assert captured["quantization_config"] is sentinel_quant_config

    def test_text_full_finetune_dtype_is_float32(self, monkeypatch):
        torch = pytest.importorskip("torch", reason="torch is only in the [train] extra")
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config(training={"quantization": "none", "lora": {"r": 0}})
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        captured = {}

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=_capturing_from_pretrained(model, captured)
            ),
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            TaskType=SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            get_peft_model=lambda model_obj, _cfg: model_obj,
            prepare_model_for_kbit_training=lambda model_obj: model_obj,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        _install_common_trainer_mocks(monkeypatch)

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False
        wrapper.model = None
        wrapper.tokenizer = None

        wrapper._setup_transformers(cfg, cfg.training)

        assert captured["dtype"] is torch.float32

    def test_vision_dtype_is_always_auto(self, monkeypatch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config(
            modality="vision",
            data={"train": "./data.jsonl", "format": "llava"},
            training={"quantization": "none"},
        )
        model = _FakeModel()
        processor = _FakeProcessor()
        captured = {}

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: processor),
            AutoModelForVision2Seq=types.SimpleNamespace(
                from_pretrained=_capturing_from_pretrained(model, captured)
            ),
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            get_peft_model=lambda m, _cfg: m,
            prepare_model_for_kbit_training=lambda m: m,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **kwargs: None,
        )

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False

        wrapper._setup_vision_transformers(cfg, cfg.training)

        assert captured["dtype"] == "auto"

    def test_audio_dtype_is_always_auto(self, monkeypatch):
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = _make_config(
            modality="audio",
            data={"train": "./data.jsonl", "format": "audio"},
            training={"quantization": "none"},
        )
        model = _FakeModel()
        processor = _FakeProcessor()
        captured = {}

        fake_transformers = types.SimpleNamespace(
            AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: processor),
            AutoModel=types.SimpleNamespace(
                from_pretrained=_capturing_from_pretrained(model, captured)
            ),
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            get_peft_model=lambda m, _cfg: m,
            prepare_model_for_kbit_training=lambda m: m,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **kwargs: None,
        )

        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cpu"
        wrapper._trust_remote_code = False

        wrapper._setup_audio_transformers(cfg, cfg.training)

        assert captured["dtype"] == "auto"


# ---------------------------------------------------------------------------
# End-to-end: real tiny on-disk checkpoints, the literal AC1 claim.
# ---------------------------------------------------------------------------


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _tiny_llama_dir_with_dtype(tmp_path, dtype, n_layers=2):
    """A real (tiny) Llama checkpoint on disk, saved at ``dtype``.

    Mirrors tests/test_issue341_seed_and_fullft.py::_tiny_llama_dir, made
    dtype-parametric: this is the whole point of the test — asserting that
    "auto" preserves whatever the checkpoint was actually saved at, not
    just that it produces bf16.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
        architectures=["LlamaForCausalLM"],
    )
    model = LlamaForCausalLM(config).to(dtype).eval()
    weights = tmp_path / f"model-{dtype}".replace(".", "_")
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state.pop("lm_head.weight", None)  # tied
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    _write_tiny_tokenizer(str(weights))
    return str(weights)


def _write_tiny_tokenizer(directory):
    import json
    import os

    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    for word in ("hello", "world", "hi", "yo", "the", "cat", "sat", "on", "mat"):
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(os.path.join(directory, "tokenizer.json"))
    with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": 128,
                "clean_up_tokenization_spaces": False,
            },
            fh,
        )


_ROWS = [
    ("hi", "hello world"),
    ("yo", "the cat sat"),
    ("hello", "on the mat"),
    ("the", "cat sat"),
]


def _dataset():
    return {
        "train": [
            {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            }
            for user, assistant in _ROWS
        ]
    }


def _wrapper(tmp_path, monkeypatch, base, **training_over):
    from soup_cli.trainer.sft import SFTTrainerWrapper

    monkeypatch.chdir(tmp_path)
    training = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "quantization": "none",
        "epochs": 1,
        "lr": 1e-3,
        "logging_steps": 100,
        "save_steps": 10_000,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    }
    lora_over = training_over.pop("lora", None)
    training.update(training_over)
    if lora_over is not None:
        training["lora"] = {**training["lora"], **lora_over}
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
                "training": training,
                "output": str(tmp_path / "out"),
            }
        )
    )
    return SFTTrainerWrapper(cfg, device="cpu"), _dataset()


class TestLoadDtypeMatchesCheckpoint:
    def test_lora_frozen_base_loads_at_checkpoint_bf16(self, tmp_path, monkeypatch):
        _requires_train_extra()
        import torch

        base = _tiny_llama_dir_with_dtype(tmp_path, torch.bfloat16)
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, base=base)
        wrapper.setup(dataset)

        base_params = [
            p for n, p in wrapper.model.named_parameters() if "lora_" not in n
        ]
        assert base_params
        assert all(p.dtype == torch.bfloat16 for p in base_params)

    def test_full_ft_loads_fp32_even_from_a_bf16_checkpoint(self, tmp_path, monkeypatch):
        _requires_train_extra()
        import torch

        base = _tiny_llama_dir_with_dtype(tmp_path, torch.bfloat16)
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, base=base, lora={"r": 0})
        wrapper.setup(dataset)

        assert all(p.dtype == torch.float32 for p in wrapper.model.parameters())

    def test_control_lora_from_an_fp32_checkpoint_stays_fp32(self, tmp_path, monkeypatch):
        """CONTROL. Without this, "auto" could just mean "always bf16" and
        the test above would pass for the wrong reason — AC1 asks for the
        checkpoint's OWN dtype, not a fixed one."""
        _requires_train_extra()
        import torch

        base = _tiny_llama_dir_with_dtype(tmp_path, torch.float32)
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, base=base)
        wrapper.setup(dataset)

        base_params = [
            p for n, p in wrapper.model.named_parameters() if "lora_" not in n
        ]
        assert base_params
        assert all(p.dtype == torch.float32 for p in base_params)


# ---------------------------------------------------------------------------
# hardware_fit.py — AC4, the pre-flight's weights-byte assumption re-checked.
# ---------------------------------------------------------------------------


class TestHardwareFitFullFTWeightsBytes:
    def test_full_ft_weights_use_fp32_bytes_per_param(self):
        from soup_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

        inp = HardwareFitInput(
            params_b=1.0,
            seq_len=512,
            batch_size=1,
            optimizer="adamw_torch",
            quant="none",
            peft="full",
            gradient_checkpointing=False,
        )
        breakdown = estimate_peak_vram_gb(inp)
        assert breakdown.weights_gb == pytest.approx(4.0, rel=1e-6)

    def test_lora_weights_still_use_bf16_bytes_per_param(self):
        from soup_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

        inp = HardwareFitInput(
            params_b=1.0,
            seq_len=512,
            batch_size=1,
            optimizer="adamw_torch",
            quant="none",
            peft="lora",
            gradient_checkpointing=False,
        )
        breakdown = estimate_peak_vram_gb(inp)
        assert breakdown.weights_gb == pytest.approx(2.0, rel=1e-6)

    def test_quantized_weights_unaffected_by_the_full_ft_branch(self):
        """CONTROL — the new branch is gated on quant == 'none'; a QLoRA
        (4bit + peft='qlora') run must not accidentally hit it."""
        from soup_cli.utils.hardware_fit import HardwareFitInput, estimate_peak_vram_gb

        inp = HardwareFitInput(
            params_b=1.0,
            seq_len=512,
            batch_size=1,
            optimizer="adamw_torch",
            quant="4bit",
            peft="qlora",
            gradient_checkpointing=False,
        )
        breakdown = estimate_peak_vram_gb(inp)
        assert breakdown.weights_gb == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# #471 — the shared is_full_finetune() discriminator, and the two
# _build_hardware_fit_input classification bugs it fixes (one in each
# direction — see the module docstring).
# ---------------------------------------------------------------------------


class TestIsFullFinetuneSharedDiscriminator:
    def test_no_flags_is_not_full_finetune(self):
        from soup_cli.trainer.sft import is_full_finetune

        assert is_full_finetune(_StubTcfg()) is False

    def test_lora_r_zero_is_full_finetune(self):
        from soup_cli.trainer.sft import is_full_finetune

        assert is_full_finetune(_StubTcfg(r=0)) is True

    def test_unfrozen_parameters_is_full_finetune(self):
        from soup_cli.trainer.sft import is_full_finetune

        assert is_full_finetune(_StubTcfg(unfrozen_parameters=[".*"])) is True

    def test_lisa_enabled_is_full_finetune(self):
        from soup_cli.trainer.sft import is_full_finetune

        assert is_full_finetune(_StubTcfg(lisa_enabled=True)) is True


_HW_FIT_BASE_YAML = """
base: meta-llama/Llama-2-7b-hf
task: sft
data:
  train: train.jsonl
  max_length: 2048
training:
  batch_size: 8
  quantization: none
"""


class TestBuildHardwareFitInputPeftClassification:
    """Both directions the #471 review found `_build_hardware_fit_input`
    disagreeing with `is_full_finetune` on, now unified through it."""

    def test_control_plain_lora_is_lora(self):
        from soup_cli.commands.train import _build_hardware_fit_input

        inp = _build_hardware_fit_input(load_config_from_string(_HW_FIT_BASE_YAML))
        assert inp is not None
        assert inp.peft == "lora"

    def test_control_unfrozen_parameters_is_still_full(self):
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(
            _HW_FIT_BASE_YAML + "  unfrozen_parameters: ['.*']\n"
        )
        inp = _build_hardware_fit_input(cfg)
        assert inp is not None
        assert inp.peft == "full"

    def test_control_4bit_is_qlora_regardless(self):
        """The quant=='4bit' branch is checked FIRST and is unaffected by
        this fix — confirm that precedence still holds."""
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(
            _HW_FIT_BASE_YAML.replace("quantization: none", "quantization: 4bit")
        )
        inp = _build_hardware_fit_input(cfg)
        assert inp is not None
        assert inp.peft == "qlora"

    def test_lisa_enabled_was_undercounted_now_full(self):
        """The under-count bug: lisa_enabled used to be missed entirely and
        classified as "lora" — never reaching hardware_fit's fp32 correction."""
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(_HW_FIT_BASE_YAML + "  lisa_enabled: true\n")
        inp = _build_hardware_fit_input(cfg)
        assert inp is not None
        assert inp.peft == "full"

    def test_lora_r_zero_was_undercounted_now_full(self):
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(
            _HW_FIT_BASE_YAML + "  lora:\n    r: 0\n"
        )
        inp = _build_hardware_fit_input(cfg)
        assert inp is not None
        assert inp.peft == "full"

    def test_freeze_ratio_with_lora_still_on_was_overcounted_now_lora(self):
        """AmirF194's exact repro: freeze_ratio + lora.r>0 is a LoRA run
        (frozen base, "auto"/2-bytes) that used to be misclassified "full"
        (fp32/4-bytes) purely because freeze_ratio was set — a measured
        16 GB phantom at 8B that could falsely refuse a launch that fits."""
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(
            _HW_FIT_BASE_YAML + "  freeze_ratio: 0.5\n  lora:\n    r: 8\n"
        )
        inp = _build_hardware_fit_input(cfg)
        assert inp is not None
        assert inp.peft == "lora"

    def test_freeze_layers_with_lora_still_on_was_overcounted_now_lora(self):
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(
            _HW_FIT_BASE_YAML + "  freeze_layers: 4\n  lora:\n    r: 8\n"
        )
        inp = _build_hardware_fit_input(cfg)
        assert inp is not None
        assert inp.peft == "lora"

    def test_freeze_ratio_with_lora_r_zero_is_still_full(self):
        """CONTROL — freeze_ratio/freeze_layers legitimately combine with
        r=0 full-FT too (schema allows it: "train everything above layer N"),
        and that combination must still classify as full."""
        from soup_cli.commands.train import _build_hardware_fit_input

        cfg = load_config_from_string(
            _HW_FIT_BASE_YAML + "  freeze_ratio: 0.5\n  lora:\n    r: 0\n"
        )
        inp = _build_hardware_fit_input(cfg)
        assert inp is not None
        assert inp.peft == "full"
