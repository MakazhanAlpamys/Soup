"""#797, item 3: distill, classifier, asr and unlearn honour the YAML setting.

These four wrappers never read ``training.gradient_checkpointing``, so ``true`` and
every tier did nothing. Each test drives the real ``setup`` on CPU: a tiny random
Llama (``tests/_tiny_hf_models.py``) or a tiny random Whisper built here, a real
tokenizer or processor, PEFT where the task uses it, and the installed HF
``Trainer``. Only the audio decode is replaced, so the ASR rows need no
``soundfile``.

The checks are the argument the HF trainer holds, the model's
``is_gradient_checkpointing`` after one real optimizer step, and that a decoder
layer runs again in the backward pass (the recompute is what checkpointing is).
``unlearn`` trains in its own loop with no ``Trainer``, so it is checked through
``setup`` and one step of that loop.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from soup_cli.config.loader import load_config_from_string
from tests._tiny_hf_models import tiny_model_dir

_TIERS = ("selective", "medium", "full", "auto")
_STATES = pytest.mark.parametrize(
    "checkpointing", [False, True], ids=["disabled", "enabled"]
)

_WRAPPERS = {
    "distill": ("soup_cli.trainer.distill", "DistillTrainerWrapper"),
    "classifier": ("soup_cli.trainer.classifier", "ClassifierTrainerWrapper"),
    "asr": ("soup_cli.trainer.asr", "AsrTrainerWrapper"),
    # The shape of the three Whisper recipes: the same wrapper with ``asr_lora: true``.
    "asr_lora": ("soup_cli.trainer.asr", "AsrTrainerWrapper"),
}


@pytest.fixture(autouse=True)
def train_stack(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("torch", "transformers", "peft", "datasets"):
        pytest.importorskip(name, reason=f"{name} requires the train extra")
    import torch
    from transformers import TrainingArguments

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # ACCELERATE_USE_CPU alone still lets HF pick MPS on a Mac; set the public flag
    # before HF initialises, without touching the checkpointing value.
    real_post_init = TrainingArguments.__post_init__

    def cpu_post_init(args: TrainingArguments) -> None:
        args.use_cpu = True
        real_post_init(args)

    monkeypatch.setattr(TrainingArguments, "__post_init__", cpu_post_init)
    yield
    torch.set_num_threads(threads)


def _tiny_whisper_dir(root: Path) -> str:
    """A random Whisper small enough to train a step on CPU, with no download.

    One-second feature windows (100 mel frames, 50 encoder positions) and a
    byte-level tokenizer carrying the special tokens Whisper's prefix needs.
    """
    import torch
    from transformers import (
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    vocab = {char: index for index, char in enumerate(bytes_to_unicode().values())}
    tokenizer = WhisperTokenizer(vocab=vocab, merges=[], pad_token="<|endoftext|>")
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|en|>",
            "<|translate|>",
            "<|transcribe|>",
            "<|notimestamps|>",
        ]
    })
    target = root / "whisper"
    processor = WhisperProcessor(
        feature_extractor=WhisperFeatureExtractor(feature_size=8, chunk_length=1),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(str(target))
    # Reload so the special-token ids are the ones the wrapper will see.
    tokenizer = WhisperProcessor.from_pretrained(str(target)).tokenizer
    start = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    config = WhisperConfig(
        vocab_size=len(tokenizer),
        num_mel_bins=8,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=50,
        max_target_positions=32,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=start,
    )
    torch.manual_seed(7)
    WhisperForConditionalGeneration(config).save_pretrained(str(target))
    return str(target)


def _config(task: str, base: str, out: Path, checkpointing: bool | str, **training: Any):
    import yaml

    body: dict[str, Any] = {
        "quantization": "none",
        "gradient_checkpointing": checkpointing,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "warmup_ratio": 0.0,
        "save_steps": 1000,
        "seed": 7,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    body.update(training)
    data: dict[str, Any] = {"train": "train.jsonl", "max_length": 64}
    if task == "distill":
        data["chat_template"] = "chatml"
    if task == "asr":
        data["format"] = "asr"
    if task == "unlearn":
        data.update(forget_set="forget.jsonl", retain_set="retain.jsonl")
    return load_config_from_string(yaml.safe_dump({
        "base": base,
        "task": task,
        "backend": "transformers",
        "data": data,
        "training": body,
        "output": str(out),
    }))


def _build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, checkpointing: bool | str
) -> tuple[Any, Any]:
    """Run the wrapper's real ``setup`` and return it with the HF trainer it built."""
    import numpy as np

    monkeypatch.chdir(tmp_path)
    llama = tiny_model_dir("llama")
    if task == "distill":
        cfg = _config(task, llama, tmp_path / "out", checkpointing, teacher_model=llama)
        rows = [
            {"messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world friend"},
            ]}
        ] * 4
    elif task == "classifier":
        # batch 1: the tiny checkpoint has no pad id, and a sequence-classification
        # head refuses batches > 1 without one.
        cfg = _config(
            task, llama, tmp_path / "out", checkpointing, num_labels=2, batch_size=1
        )
        rows = [{"text": "hello world", "label": index % 2} for index in range(4)]
    else:
        import soup_cli.utils.tts_codec as tts_codec

        def sine(path: str, *, target_sr: int) -> Any:
            seconds = np.arange(target_sr // 2, dtype=np.float32) / target_sr
            return (0.1 * np.sin(2 * np.pi * 220.0 * seconds)).astype(np.float32)

        monkeypatch.setattr(tts_codec, "load_audio_mono", sine)
        extra = {"asr_lora": True} if task == "asr_lora" else {}
        cfg = _config(
            "asr", _tiny_whisper_dir(tmp_path), tmp_path / "out", checkpointing, **extra
        )
        rows = [{"audio": f"clip{index}.wav", "text": "hello world"} for index in range(4)]
    module_name, class_name = _WRAPPERS[task]
    wrapper = getattr(importlib.import_module(module_name), class_name)(cfg, device="cpu")
    wrapper.setup({"train": rows})
    trainer = wrapper.trainer
    if task == "asr_lora":
        assert any("lora_" in name for name, _ in trainer.model.named_parameters())
    assert trainer.args.device.type == "cpu"
    assert next(trainer.model.parameters()).device.type == "cpu"
    return wrapper, trainer


def _count_layer_calls(model: Any) -> list[int]:
    """Count calls of the first checkpointable layer; recompute runs it twice a step."""
    from transformers.modeling_layers import GradientCheckpointingLayer

    layer = next(m for m in model.modules() if isinstance(m, GradientCheckpointingLayer))
    calls = [0]

    def hook(*_args: Any) -> None:
        calls[0] += 1

    layer.register_forward_pre_hook(hook)
    return calls


@pytest.mark.parametrize("task", _WRAPPERS)
@_STATES
def test_yaml_checkpointing_reaches_the_trainer_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, checkpointing: bool
) -> None:
    _, trainer = _build(tmp_path, monkeypatch, task, checkpointing)
    assert trainer.args.gradient_checkpointing is checkpointing, task


@pytest.mark.parametrize("task", _WRAPPERS)
@_STATES
def test_checkpointing_state_after_one_cpu_training_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, checkpointing: bool
) -> None:
    _, trainer = _build(tmp_path, monkeypatch, task, checkpointing)
    trainer.args.max_steps = 1
    trainer.args.save_strategy = "no"
    trainer.args.disable_tqdm = True
    import torch

    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    calls = _count_layer_calls(trainer.model)
    trainer.train()
    assert trainer.state.global_step == 1
    assert trainer.model.is_gradient_checkpointing is checkpointing, task
    # One forward per step, plus the backward recompute when checkpointing is on.
    assert calls[0] == (2 if checkpointing else 1), task
    # The recompute count cannot show this: the step still trained the model.
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, trainable)), task


@pytest.mark.parametrize("task", _WRAPPERS)
@pytest.mark.parametrize("tier", _TIERS)
def test_enabled_tiers_are_full_checkpointing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: str, tier: str
) -> None:
    # No selective hooks on these wrappers: every enabled tier is full checkpointing.
    _, trainer = _build(tmp_path, monkeypatch, task, tier)
    assert trainer.args.gradient_checkpointing is True, (task, tier)


# --- unlearn: its own loop, no HF Trainer -------------------------------------


def _unlearn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpointing: bool | str,
    method: str = "simnpo",
) -> Any:
    from soup_cli.trainer.unlearn import UnlearnTrainerWrapper

    monkeypatch.chdir(tmp_path)
    row = {"prompt": "hello", "completion": "world friend"}
    (tmp_path / "forget.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "retain.jsonl").write_text(
        json.dumps({"prompt": "hi", "completion": "there friend"}) + "\n", encoding="utf-8"
    )
    cfg = _config(
        "unlearn", tiny_model_dir("llama"), Path("out"), checkpointing,
        unlearn_method=method, batch_size=1,
    )
    wrapper = UnlearnTrainerWrapper(cfg, device="cpu")
    wrapper.setup()
    assert wrapper._dev == "cpu"
    return wrapper


@_STATES
def test_unlearn_setup_applies_yaml_checkpointing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpointing: bool
) -> None:
    wrapper = _unlearn(tmp_path, monkeypatch, checkpointing)
    assert wrapper.model.training is True
    assert wrapper.model.is_gradient_checkpointing is checkpointing


@pytest.mark.parametrize("tier", _TIERS)
def test_unlearn_enabled_tiers_are_full_checkpointing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tier: str
) -> None:
    wrapper = _unlearn(tmp_path, monkeypatch, tier)
    assert wrapper.model.is_gradient_checkpointing is True, tier


@pytest.mark.parametrize("method", ["simnpo", "npo", "rmu"])
@_STATES
def test_unlearn_loop_step_recomputes_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, checkpointing: bool
) -> None:
    import torch

    wrapper = _unlearn(tmp_path, monkeypatch, checkpointing, method)
    lora = [p for n, p in wrapper.model.named_parameters() if "lora_B" in n]
    before = [p.detach().clone() for p in lora]
    calls = _count_layer_calls(wrapper.model)
    result = wrapper.train()
    assert result["total_steps"] == 1
    assert wrapper.model.is_gradient_checkpointing is checkpointing
    # The single forget example takes two policy forwards with a retain set (NPO and
    # SimNPO: forget + retain CE; RMU: forget + retain activations). Checkpointing
    # runs each again in the backward pass.
    assert calls[0] == (4 if checkpointing else 2), method
    # The step still trained the adapter through the checkpointed layers.
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, lora))
