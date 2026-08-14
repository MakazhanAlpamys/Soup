"""Issue #302 — Idefics3 / SmolVLM vision-SFT pad_token routing.

SmolVLM uses an ``Idefics3Processor``. The shared LLaVA vision path sets
``self.tokenizer = <processor>`` and hands it to TRL's ``SFTTrainer`` as
``processing_class``. TRL reads ``processing_class.pad_token`` /
``.eos_token`` / ``.convert_tokens_to_ids`` directly, but HF vision processors
keep the text tokenizer nested at ``processor.tokenizer`` and do NOT forward
token-level attributes (``ProcessorMixin`` has no ``__getattr__``) — so training
crashes with ``AttributeError: 'Idefics3Processor' object has no attribute
'pad_token'``.

This suite pins ``_ensure_vision_processor_pad_token`` — it mirrors the inner
tokenizer's text-token surface onto the processor (setting pad_token = eos_token
when unset), reproducing TRL's exact ``args.pad_token or processing_class.pad_token
or processing_class.eos_token`` access. Both Idefics3 and LLaVA processors share
identical structure (``attributes = ['image_processor', 'tokenizer']``), so the
fix repairs both without regressing a processor that already exposes pad_token.
"""

from __future__ import annotations

import pytest


class _FakeTokenizer:
    """Text tokenizer with the token surface TRL reads off processing_class."""

    def __init__(self, pad_token=None, eos_token="</s>"):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.eos_token_id = 2
        self.bos_token = "<s>"
        self.bos_token_id = 1

    @property
    def pad_token_id(self):
        # Mirrors a real tokenizer: None until pad_token is set.
        return 0 if self.pad_token is not None else None

    def convert_tokens_to_ids(self, token):
        return {"</s>": 2, "<s>": 1, "<pad>": 0}.get(token, 2)


class _FakeIdefics3Processor:
    """Mimics Idefics3Processor: nested .tokenizer, NO pad_token forwarding."""

    attributes = ["image_processor", "tokenizer"]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_processor = object()

    # No __getattr__ — accessing .pad_token raises AttributeError, exactly like
    # the real ProcessorMixin subclass.


class _TokenizerLikeProcessor:
    """A processing_class that already exposes the token surface (regression guard)."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.tokenizer = None

    def convert_tokens_to_ids(self, token):
        return 0


def _trl_pad_token(processing_class, args_pad_token=None):
    """Reproduce TRL SFTTrainer's pad-token resolution (sft_trainer.py:436)."""
    return args_pad_token or processing_class.pad_token or processing_class.eos_token


class TestEnsureVisionProcessorPadToken:
    def test_bare_processor_raises_before_fix(self):
        # Sanity: the un-fixed processor reproduces the reported AttributeError.
        proc = _FakeIdefics3Processor(_FakeTokenizer(pad_token=None))
        with pytest.raises(AttributeError):
            _ = proc.pad_token

    def test_sets_pad_token_from_eos(self):
        from soup_cli.trainer.sft import _ensure_vision_processor_pad_token

        proc = _FakeIdefics3Processor(_FakeTokenizer(pad_token=None, eos_token="</s>"))
        _ensure_vision_processor_pad_token(proc)
        # Inner tokenizer got pad_token = eos_token
        assert proc.tokenizer.pad_token == "</s>"
        # Processor now exposes the surface TRL reads
        assert proc.pad_token == "</s>"
        assert proc.eos_token == "</s>"

    def test_trl_resolution_no_longer_crashes(self):
        from soup_cli.trainer.sft import _ensure_vision_processor_pad_token

        proc = _FakeIdefics3Processor(_FakeTokenizer(pad_token=None))
        _ensure_vision_processor_pad_token(proc)
        pad = _trl_pad_token(proc)
        assert pad == "</s>"
        # convert_tokens_to_ids delegates to the inner tokenizer
        assert proc.convert_tokens_to_ids(pad) == 2

    def test_preserves_existing_pad_token(self):
        from soup_cli.trainer.sft import _ensure_vision_processor_pad_token

        tok = _FakeTokenizer(pad_token="<pad>", eos_token="</s>")
        proc = _FakeIdefics3Processor(tok)
        _ensure_vision_processor_pad_token(proc)
        assert proc.tokenizer.pad_token == "<pad>"  # untouched
        assert proc.pad_token == "<pad>"

    def test_tokenizer_like_processor_unchanged(self):
        # A processing_class that already exposes pad_token must not be clobbered.
        from soup_cli.trainer.sft import _ensure_vision_processor_pad_token

        proc = _TokenizerLikeProcessor()
        _ensure_vision_processor_pad_token(proc)
        assert proc.pad_token == "<pad>"

    def test_no_nested_tokenizer_is_noop(self):
        from soup_cli.trainer.sft import _ensure_vision_processor_pad_token

        class _NoTok:
            pad_token = "<pad>"
            eos_token = "</s>"

        proc = _NoTok()
        _ensure_vision_processor_pad_token(proc)  # must not raise
        assert proc.pad_token == "<pad>"

    def test_convert_tokens_to_ids_mirrored(self):
        from soup_cli.trainer.sft import _ensure_vision_processor_pad_token

        proc = _FakeIdefics3Processor(_FakeTokenizer(pad_token=None))
        _ensure_vision_processor_pad_token(proc)
        assert callable(proc.convert_tokens_to_ids)
        assert proc.convert_tokens_to_ids("</s>") == 2

    def test_eos_token_id_mirrored(self):
        from soup_cli.trainer.sft import _ensure_vision_processor_pad_token

        proc = _FakeIdefics3Processor(_FakeTokenizer(pad_token=None))
        _ensure_vision_processor_pad_token(proc)
        assert proc.eos_token_id == 2

    def test_pad_token_id_mirrored(self):
        from soup_cli.trainer.sft import _ensure_vision_processor_pad_token

        proc = _FakeIdefics3Processor(_FakeTokenizer(pad_token=None))
        _ensure_vision_processor_pad_token(proc)
        # inner tokenizer's pad_token_id property becomes 0 once pad is set
        assert proc.pad_token_id == 0

    def test_readonly_attr_degrades_gracefully(self):
        # A processor whose attributes can't be set (e.g. __slots__) must not
        # make the helper raise — the try/except degrades gracefully.
        from soup_cli.trainer.sft import _ensure_vision_processor_pad_token

        class _SlotsProcessor:
            __slots__ = ("tokenizer",)

            def __init__(self, tok):
                self.tokenizer = tok

        proc = _SlotsProcessor(_FakeTokenizer(pad_token=None))
        # Must not raise even though setattr(proc, "pad_token", ...) fails.
        _ensure_vision_processor_pad_token(proc)
        # Inner tokenizer was still repaired (pad = eos).
        assert proc.tokenizer.pad_token == "</s>"


class TestVisionSetupWiring:
    def test_setup_vision_transformers_invokes_pad_token_mirror(self, monkeypatch):
        # The fix is only useful if _setup_vision_transformers actually calls it.
        # Mock the heavy loads; assert the processor gets pad_token mirrored.
        from unittest.mock import MagicMock

        import transformers

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.sft import SFTTrainerWrapper

        fake_proc = _FakeIdefics3Processor(_FakeTokenizer(pad_token=None))
        monkeypatch.setattr(
            transformers.AutoProcessor, "from_pretrained",
            lambda *a, **k: fake_proc,
        )
        monkeypatch.setattr(
            transformers.AutoModelForVision2Seq, "from_pretrained",
            lambda *a, **k: MagicMock(),
        )
        import peft

        monkeypatch.setattr(peft, "get_peft_model", lambda model, cfg: model)
        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **k: None,
        )
        monkeypatch.setattr(
            "soup_cli.utils.data_pipeline.apply_vocab_expansion",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            SFTTrainerWrapper, "_apply_quantization_aware", lambda self, tcfg: None
        )

        cfg = load_config_from_string(
            "base: fake/vlm\ntask: sft\nmodality: vision\n"
            "data:\n  train: x.jsonl\n  format: llava\n  max_length: 64\n"
            "training:\n  quantization: none\n  lora:\n    target_modules: [q_proj, v_proj]\n"
        )
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        wrapper._setup_vision_transformers(cfg, cfg.training)
        # The helper ran: the Idefics3-style processor now exposes pad_token.
        assert wrapper.processor.pad_token == "</s>"
