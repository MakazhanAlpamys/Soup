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

It also pins the end-to-end half of #302: legacy LLaVA ``<image>`` strings are
converted to structured multimodal content at collation time, where the real
processor can expand image tokens and emit pixel tensors for the model.
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


class _FakeVisionProcessor:
    def __init__(self):
        self.templated_messages = []
        self.call_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.templated_messages.append(messages)
        assert kwargs == {"tokenize": False, "add_generation_prompt": False}
        return "rendered-with-<image>"

    def __call__(self, **kwargs):
        import torch

        self.call_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[11, 12, 13]]),
            "attention_mask": torch.tensor([[1, 0, 1]]),
            "pixel_values": torch.ones((1, 1, 3, 2, 2)),
        }


class TestVisionLanguageCollation:
    def test_legacy_marker_becomes_structured_image_part(self):
        from soup_cli.trainer.sft import VisionLanguageDataCollator

        processor = _FakeVisionProcessor()
        collator = VisionLanguageDataCollator(processor, max_length=128)
        image = object()
        batch = collator(
            [
                {
                    "messages": [
                        {"role": "user", "content": "<image>\nDescribe it."},
                        {"role": "assistant", "content": "A square."},
                    ],
                    "images": [image],
                }
            ]
        )

        user_parts = processor.templated_messages[0][0]["content"]
        assert user_parts == [
            {"type": "image"},
            {"type": "text", "text": "Describe it."},
        ]
        assert processor.call_kwargs["images"] == [[image]]
        assert processor.call_kwargs["text"] == ["rendered-with-<image>"]
        assert processor.call_kwargs["truncation"] is True
        assert processor.call_kwargs["max_length"] == 128
        assert batch["pixel_values"].shape == (1, 1, 3, 2, 2)
        assert batch["labels"].tolist() == [[11, -100, 13]]

    def test_missing_marker_injects_image_into_first_user_turn(self):
        from soup_cli.trainer.sft import _vision_messages_with_image_parts

        messages = _vision_messages_with_image_parts(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Describe it."},
            ],
            image_count=1,
        )

        assert messages[0]["content"] == [{"type": "text", "text": "Be concise."}]
        assert messages[1]["content"] == [
            {"type": "image"},
            {"type": "text", "text": "Describe it."},
        ]

    def test_excess_image_markers_are_rejected_before_processor(self):
        from soup_cli.trainer.sft import _vision_messages_with_image_parts

        with pytest.raises(ValueError, match="2 image placeholder.*1 image"):
            _vision_messages_with_image_parts(
                [{"role": "user", "content": "<image><image>Compare."}],
                image_count=1,
            )

    def test_dataset_keeps_messages_until_collation(self, tmp_path):
        from PIL import Image

        from soup_cli.trainer.sft import SFTTrainerWrapper

        image_path = tmp_path / "sample.png"
        Image.new("RGB", (8, 8), "red").save(image_path)
        row = {
            "messages": [{"role": "user", "content": "<image>\nDescribe."}],
            "image": str(image_path),
        }
        wrapper = object.__new__(SFTTrainerWrapper)
        train_ds, eval_ds = wrapper._prepare_vision_dataset({"train": [row]})

        assert eval_ds is None
        assert train_ds.column_names == ["messages", "images"]
        assert train_ds[0]["messages"] == row["messages"]
        assert len(train_ds[0]["images"]) == 1

    def test_plain_trainer_receives_processor_aware_collator(self, monkeypatch):
        import transformers

        from soup_cli.trainer.sft import (
            VisionLanguageDataCollator,
            _make_vision_trainer,
        )

        captured = {}

        class _FakeTrainer:
            def __init__(
                self,
                model=None,
                args=None,
                data_collator=None,
                train_dataset=None,
                eval_dataset=None,
                processing_class=None,
            ):
                captured.update(
                    model=model,
                    args=args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    processing_class=processing_class,
                )

        monkeypatch.setattr(transformers, "Trainer", _FakeTrainer)
        processor = _FakeVisionProcessor()
        trainer = _make_vision_trainer(
            {
                "model": "model",
                "args": "args",
                "train_dataset": "train",
                "eval_dataset": None,
                "processing_class": processor,
            },
            processor,
            max_length=256,
        )

        assert isinstance(trainer, _FakeTrainer)
        assert isinstance(captured["data_collator"], VisionLanguageDataCollator)
        assert captured["data_collator"].max_length == 256
        assert captured["processing_class"] is processor

    def test_plain_trainer_uses_legacy_tokenizer_keyword_when_required(self, monkeypatch):
        import transformers

        from soup_cli.trainer.sft import _make_vision_trainer

        captured = {}

        class _LegacyTrainer:
            def __init__(
                self,
                model=None,
                args=None,
                data_collator=None,
                train_dataset=None,
                eval_dataset=None,
                tokenizer=None,
            ):
                captured["tokenizer"] = tokenizer

        monkeypatch.setattr(transformers, "Trainer", _LegacyTrainer)
        processor = _FakeVisionProcessor()
        _make_vision_trainer(
            {
                "model": "model",
                "args": "args",
                "train_dataset": "train",
                "eval_dataset": None,
                "processing_class": processor,
            },
            processor,
            max_length=256,
        )

        assert captured["tokenizer"] is processor

    def test_setup_routes_normal_vision_run_to_plain_trainer(self, monkeypatch, tmp_path):
        from datasets import Dataset

        import soup_cli.trainer.sft as sft_module
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.sft import SFTTrainerWrapper

        processor = _FakeVisionProcessor()

        class _FakeModel:
            def get_nb_trainable_parameters(self):
                return 1, 2

        def _fake_vision_setup(self, cfg, tcfg):
            self.model = _FakeModel()
            self.processor = processor
            self.tokenizer = processor

        def _fake_prepare(self, dataset):
            train = Dataset.from_list([{"messages": [], "images": []}])
            return train, None

        sentinel = object()
        captured = {}

        def _fake_make(trainer_kwargs, processor, max_length):
            captured.update(
                trainer_kwargs=trainer_kwargs,
                processor=processor,
                max_length=max_length,
            )
            return sentinel

        monkeypatch.setattr(
            SFTTrainerWrapper, "_setup_vision_transformers", _fake_vision_setup
        )
        monkeypatch.setattr(SFTTrainerWrapper, "_prepare_vision_dataset", _fake_prepare)
        monkeypatch.setattr(sft_module, "_make_vision_trainer", _fake_make)

        cfg = load_config_from_string(
            "base: fake/vlm\ntask: sft\nmodality: vision\n"
            "data:\n  train: x.jsonl\n  format: llava\n  max_length: 321\n"
            "training:\n  epochs: 1\n  batch_size: 1\n"
            "  gradient_accumulation_steps: 1\n  quantization: none\n"
            "  lora:\n    target_modules: [q_proj, v_proj]\n"
            f"output: {tmp_path.as_posix()}/output\n"
        )
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": [{"messages": [], "image": "unused"}]})

        assert wrapper.trainer is sentinel
        assert captured["processor"] is processor
        assert captured["max_length"] == 321
        assert captured["trainer_kwargs"]["train_dataset"].column_names == [
            "messages",
            "images",
        ]
