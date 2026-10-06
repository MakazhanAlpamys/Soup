"""#1393 — DPO loads a vision-language wrapper checkpoint when ``modality: vision``.

``DPOTrainerWrapper`` always built ``AutoModelForCausalLM``, whatever ``modality``
said, so a checkpoint published as a vision-language wrapper (MiniMax-M3,
``model_type: minimax_m3_vl``) could not load at all, while SFT trained its language
tower through ``AutoModelForImageTextToText``. DPO now picks the image-text class and
the processor for ``modality: vision``, the attach preflight mirrors it (the class
``soup recipes verify`` builds is the class the trainer loads), and the one backend
whose DPO setup ignores ``modality`` is refused at config load. A tiny LLaVA stands in
for MiniMax-M3: the same wrapper shape, built locally, no weights downloaded.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from soup_cli.config.loader import load_config_from_string  # noqa: E402
from soup_cli.utils.attach_preflight import (  # noqa: E402
    Verdict,
    build_on_meta,
    check_attach,
    loader_for,
)

_LANGUAGE_TOWER = r".*language_model\..*\.self_attn\.(q_proj|v_proj)"


def _llava_config():
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig

    return LlavaConfig(
        text_config=LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            pad_token_id=3, bos_token_id=1, eos_token_id=2,
        ),
        vision_config=CLIPVisionConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, image_size=32, patch_size=16,
        ),
        image_token_index=63,
    )


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    names = ["<unk>", "<s>", "</s>", "<pad>", "say", "hi", "hello", "world", "yes", "no"]
    names += [f"w{i}" for i in range(len(names), 63)] + ["<image>"]
    tok = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(names)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    # ``<image>`` is registered as a special token so the processor resolves its id
    # (63) after a save/load round trip instead of falling back to the unknown id.
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        pad_token="<pad>", additional_special_tokens=["<image>"],
    )


def _save_tiny_llava(root):
    """A LLaVA checkpoint on disk, the shape MiniMax-M3 is published in."""
    from transformers import CLIPImageProcessor, LlavaForConditionalGeneration, LlavaProcessor

    torch.manual_seed(7)
    LlavaForConditionalGeneration(_llava_config()).save_pretrained(root)
    LlavaProcessor(
        image_processor=CLIPImageProcessor(
            size={"height": 32, "width": 32}, crop_size={"height": 32, "width": 32}
        ),
        tokenizer=_tokenizer(),
        patch_size=16,
    ).save_pretrained(root)
    return root.as_posix()


def _save_tiny_llama(root):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, pad_token_id=3, bos_token_id=1,
        eos_token_id=2,
    )).save_pretrained(root)
    _tokenizer().save_pretrained(root)
    return root.as_posix()


def _yaml(base="org/m", task="dpo", modality=None, backend="transformers", targets=_LANGUAGE_TOWER,
          output=None, **extra):
    lines = [f"base: {base}", f"task: {task}", f"backend: {backend}"]
    if modality is not None:
        lines.append(f"modality: {modality}")
    lines += ["data:", "  train: x.jsonl", "  format: dpo", "  max_length: 64",
              "training:", "  quantization: none", "  epochs: 1", "  batch_size: 2"]
    lines += [f"  {k}: {v}" for k, v in extra.items()]  # training.* knobs such as preference_loss
    lines += ["  lora:", "    r: 4", "    alpha: 8", "    dropout: 0.0",
              f"    target_modules: {json.dumps(targets)}"]
    if output is not None:
        lines.append(f"output: {output}")
    return "\n".join(lines) + "\n"


def _cfg(**kw):
    return load_config_from_string(_yaml(**kw))


class TestTheLoaderMapFollowsTheTrainer:
    """``loader_for`` must build the class ``trainer/dpo.py`` now loads."""

    def test_vision_dpo_uses_the_image_text_class(self):
        assert loader_for(_cfg(modality="vision")) == ("AutoModelForImageTextToText",)

    def test_text_dpo_still_uses_a_causal_lm(self):
        assert loader_for(_cfg()) == ("AutoModelForCausalLM",)
        assert loader_for(_cfg(modality="text")) == ("AutoModelForCausalLM",)

    def test_the_preference_alias_with_the_dpo_loss_follows_dpo(self):
        """``task: preference`` with ``preference_loss: dpo`` runs the DPO wrapper, so
        the preflight must answer for it the same way; the other losses keep their
        own loaders and stay a causal LM."""
        vision_dpo = _cfg(task="preference", modality="vision", preference_loss="dpo")
        vision_simpo = _cfg(task="preference", modality="vision", preference_loss="simpo")
        assert loader_for(vision_dpo) == ("AutoModelForImageTextToText",)
        assert loader_for(vision_simpo) == ("AutoModelForCausalLM",)


class TestTheRealAttach:
    """The acceptance row of the issue, on the real resolution and the real
    ``get_peft_model``: a DPO config with ``modality: vision`` attaches to the
    language tower of a vision-language wrapper, and without the modality the
    causal-LM class refuses the checkpoint, exactly what #1145 measured."""

    @staticmethod
    def _check(cfg):
        return check_attach(
            "vl-dpo", cfg, load_hf_config=lambda _base: _llava_config(), build_model=build_on_meta
        )

    def test_a_vision_dpo_config_attaches_to_the_language_tower(self):
        check = self._check(_cfg(modality="vision"))
        assert check.verdict is Verdict.ATTACHES, check.detail
        assert check.adapted > 0
        assert check.vision_modules == 0

    def test_the_same_config_without_modality_still_cannot_load(self):
        check = self._check(_cfg())
        assert check.verdict is Verdict.CANNOT_ATTACH, check.detail


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestTheWrapperLoadsTheRightClass:
    """Through the real loaders on checkpoints saved to disk, nothing stubbed."""

    @staticmethod
    def _wrapper(tmp_path, monkeypatch, base, modality, targets=_LANGUAGE_TOWER):
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        monkeypatch.setenv("WANDB_DISABLED", "true")
        monkeypatch.chdir(tmp_path)
        cfg = _cfg(base=base, modality=modality, targets=targets,
                   output=(tmp_path / "out").as_posix())
        wrapper = DPOTrainerWrapper(cfg, device="cpu")
        wrapper._setup_transformers(cfg, cfg.training)
        return wrapper

    def test_vision_loads_the_image_text_model_and_its_processor(self, tmp_path, monkeypatch):
        from transformers import LlavaForConditionalGeneration, ProcessorMixin

        base = _save_tiny_llava(tmp_path / "vl")
        wrapper = self._wrapper(tmp_path, monkeypatch, base, "vision")
        assert isinstance(wrapper.model.get_base_model(), LlavaForConditionalGeneration)
        assert isinstance(wrapper.tokenizer, ProcessorMixin)
        assert wrapper.tokenizer.pad_token is not None
        adapted = [n for n, m in wrapper.model.named_modules() if hasattr(m, "lora_A")]
        assert adapted and all("language_model" in n for n in adapted), adapted
        assert not any("vision_tower" in n for n in adapted)

    def test_vision_vocab_expansion_reaches_the_processors_tokenizer(self, tmp_path, monkeypatch):
        """``data.add_new_tokens`` is applied by DPO (#1358), so on the vision path it
        must go through the processor's nested tokenizer and resize the embeddings;
        handing the processor itself to the expansion would skip both."""
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        monkeypatch.setenv("WANDB_DISABLED", "true")
        monkeypatch.chdir(tmp_path)
        base = _save_tiny_llava(tmp_path / "vl")
        text = _yaml(base=base, modality="vision", output=(tmp_path / "out").as_posix())
        text = text.replace("  format: dpo\n", "  format: dpo\n  add_new_tokens: ['<zz>']\n")
        cfg = load_config_from_string(text)
        wrapper = DPOTrainerWrapper(cfg, device="cpu")
        wrapper._setup_transformers(cfg, cfg.training)
        tokenizer = wrapper.tokenizer.tokenizer
        new_id = tokenizer.convert_tokens_to_ids("<zz>")
        assert new_id not in (None, tokenizer.unk_token_id)
        assert wrapper.model.get_input_embeddings().num_embeddings >= len(tokenizer)

    def test_text_still_loads_a_causal_lm_with_a_tokenizer(self, tmp_path, monkeypatch):
        from transformers import LlamaForCausalLM, PreTrainedTokenizerBase

        base = _save_tiny_llama(tmp_path / "lm")
        wrapper = self._wrapper(tmp_path, monkeypatch, base, None, targets=["q_proj", "v_proj"])
        assert isinstance(wrapper.model.get_base_model(), LlamaForCausalLM)
        assert isinstance(wrapper.tokenizer, PreTrainedTokenizerBase)


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestARealDpoStepOnAVisionWrapper:
    """The question the issue left open: what trl's ``DPOTrainer`` does with a
    processor and preference rows that carry no images. Measured: it tokenizes the
    text through the processor and trains. Nothing about trl is mocked."""

    def test_setup_and_one_step(self, tmp_path, monkeypatch):
        pytest.importorskip("trl")
        pytest.importorskip("datasets")
        from soup_cli.trainer.dpo import DPOTrainerWrapper

        monkeypatch.setenv("WANDB_DISABLED", "true")
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.chdir(tmp_path)
        base = _save_tiny_llava(tmp_path / "vl")
        cfg = _cfg(base=base, modality="vision", output=(tmp_path / "out").as_posix())
        # trl concatenates prompt + completion as strings, so the completions lead
        # with a space to stay on the tiny word-level vocabulary.
        rows = [
            {"prompt": "say hi", "chosen": " hi", "rejected": " no"},
            {"prompt": "say hello", "chosen": " hello world", "rejected": " no"},
            {"prompt": "hello", "chosen": " hi", "rejected": " world"},
            {"prompt": "hi", "chosen": " hello", "rejected": " no no"},
        ]
        wrapper = DPOTrainerWrapper(cfg, device="cpu")
        wrapper.setup({"train": rows})
        assert wrapper.trainer.processing_class is wrapper.tokenizer
        result = wrapper.train()
        assert result["total_steps"] >= 1, result
        assert (tmp_path / "out" / "adapter_config.json").is_file()


class TestABackendWhoseDpoIgnoresModalityIsRefused:
    def test_unsloth_vision_dpo_is_refused_at_config_load(self):
        with pytest.raises(ValueError) as excinfo:
            _cfg(modality="vision", backend="unsloth")
        message = " ".join(str(excinfo.value).split())
        assert "modality='vision'" in message and "backend='transformers'" in message
        assert "unsloth" in message

    def test_the_preference_alias_with_the_dpo_loss_is_refused_the_same_way(self):
        """``task: preference`` + ``preference_loss: dpo`` runs the same unsloth setup."""
        with pytest.raises(ValueError) as excinfo:
            _cfg(task="preference", modality="vision", backend="unsloth", preference_loss="dpo")
        message = " ".join(str(excinfo.value).split())
        assert "task='preference'" in message and "backend='transformers'" in message
        alias = _cfg(task="preference", modality="vision", preference_loss="dpo")
        assert alias.backend == "transformers"

    def test_unsloth_text_dpo_and_transformers_vision_dpo_still_load(self):
        assert _cfg(backend="unsloth").backend == "unsloth"
        assert _cfg(modality="vision").modality == "vision"
