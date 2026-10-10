"""Native CSM: config/dispatch, safe audio collation and real multimodal loss."""

from __future__ import annotations

import copy
import json
import math
import wave

import pytest

from tests.conftest import strip_ansi


def csm_config(**training):
    from soup_cli.config.schema import SoupConfig

    return SoupConfig(
        base="sesame/csm-1b", task="tts", modality="audio_out",
        data={"train": "speech.jsonl", "format": "audio", "val_split": 0},
        training={"tts_family": "sesame_csm", "quantization": "none", **training},
    )


def write_audio(path, frames=3840):
    import numpy as np

    samples = (2000 * np.sin(np.arange(frames) * 2 * math.pi * 220 / 24000)).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(samples.tobytes())


def speech_row(path, text="Hello", speaker="0"):
    return {"audio": str(path), "messages": [{"role": speaker, "content": text}]}


def use_cpu_cli(monkeypatch):
    import soup_cli.commands.train as train_command

    monkeypatch.setattr(train_command, "detect_device", lambda **_: ("cpu", "CPU"))
    monkeypatch.setattr(train_command, "get_gpu_info", lambda **_: {
        "memory_total": "0 GB", "memory_total_bytes": 0, "gpu_count": 0,
    })


def test_native_audio_config_and_dispatch():
    from soup_cli.trainer.csm import CSMTrainerWrapper
    from soup_cli.trainer.dispatch import build_trainer

    assert isinstance(build_trainer(csm_config(), device="cpu"), CSMTrainerWrapper)


@pytest.mark.parametrize("fmt", ["chatml", "auto", "pre_tokenized"])
def test_codec_string_configs_still_refused(fmt):
    from soup_cli.config.schema import SoupConfig

    values = csm_config().model_dump()
    values["data"]["format"] = fmt
    if fmt == "pre_tokenized":
        values["data"]["tokenized_path"] = "cache"
    reason = "pre_tokenized" if fmt == "pre_tokenized" else "data.format='audio'"
    with pytest.raises(ValueError, match=reason):
        SoupConfig(**values)


@pytest.mark.parametrize("option", [
    {"quantization": "4bit"}, {"packing": True}, {"multipack": True},
    {"use_liger": True}, {"use_flash_attn": True}, {"freeze_ratio": 0.5},
    {"auto_mixed_precision": True}, {"gradient_checkpointing": "selective"},
    {"neftune_alpha": 5}, {"lr_groups": {"default": 0.0001}},
])
def test_unsupported_training_settings_fail_before_model_load(option):
    with pytest.raises(ValueError):
        csm_config(**option)


@pytest.mark.parametrize("option", [
    {"target_modules": ["q_proj"]}, {"use_dora": True}, {"target_parameters": "auto"},
])
def test_unsupported_lora_settings_are_explicit(option):
    with pytest.raises(ValueError):
        csm_config(lora=option)


@pytest.mark.parametrize("rank", [0, 2])
def test_full_ft_and_lora_are_configurable(rank):
    assert csm_config(lora={"r": rank}).training.lora.r == rank


@pytest.mark.parametrize("speaker", ["0", "1"])
def test_transcript_and_speaker_are_preserved(speaker):
    from soup_cli.utils.csm import validate_csm_row

    row = speech_row("clip.wav", "Exact transcript", speaker)
    assert validate_csm_row(row) == ("clip.wav", speaker, "Exact transcript")


@pytest.mark.parametrize("row", [
    {}, speech_row(""), speech_row("a.wav", ""), speech_row("a.wav", "  "),
    speech_row("a.wav", "Hello", "assistant"), speech_row("a.wav", "Hello", True),
    {"audio": "a.wav", "messages": []},
    {"audio": "a.wav", "messages": [{"role": "0", "content": []}]},
    {"audio": "a.wav", "messages": [{"role": "0", "content": "Hi", "train": False}]},
    {"audio": "a.wav", "messages": [{"role": "0", "content": "A"},
                                      {"role": "1", "content": "B"}]},
])
def test_ambiguous_rows_fail_instead_of_training_wrong_transcript(row):
    from soup_cli.utils.csm import validate_csm_row

    with pytest.raises(ValueError):
        validate_csm_row(row)


@pytest.fixture
def tiny_csm_components():
    """Real CSM + real 32-codebook Mimi, random tiny weights, no Hub access."""
    pytest.importorskip("soundfile")
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        CsmConfig,
        CsmForConditionalGeneration,
        CsmProcessor,
        EncodecFeatureExtractor,
        PreTrainedTokenizerFast,
    )

    torch.manual_seed(42)
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<|AUDIO|>": 3,
             "<|audio_eos|>": 4, "Hello": 5, "longer": 6, "[": 7,
             "]": 8, "0": 9, "1": 10}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>", bos_token="<bos>",
        additional_special_tokens=["<|AUDIO|>", "<|audio_eos|>"],
    )
    template = ("{{ bos_token }}{% for message in messages %}[{{ message.role }}]"
                "{% for part in message.content %}{% if part.type == 'text' %}{{ part.text }}"
                "{% elif part.type == 'audio' %}<|AUDIO|><|audio_eos|>{% endif %}"
                "{% endfor %}{% endfor %}")
    processor = CsmProcessor(
        EncodecFeatureExtractor(feature_size=1, sampling_rate=24000), tokenizer,
        chat_template=template,
    )
    common = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=2)
    cfg = CsmConfig(
        **common, vocab_size=9, text_vocab_size=len(vocab), num_codebooks=32,
        audio_token_id=3, audio_eos_token_id=4, pad_token_id=0, bos_token_id=2,
        codebook_pad_token_id=8,
        depth_decoder_config={**common, "backbone_hidden_size": 16,
                              "vocab_size": 9, "num_codebooks": 32},
        codec_config={"model_type": "mimi", **common, "num_filters": 2,
                      "codebook_size": 8, "codebook_dim": 8,
                      "vector_quantization_hidden_dimension": 8, "num_quantizers": 32,
                      "upsample_groups": 16},
    )
    model = CsmForConditionalGeneration(cfg)
    return processor, model


def test_real_processor_padding_and_both_decoder_gradients(tmp_path, tiny_csm_components):
    import torch

    from soup_cli.trainer.csm import CSMDataCollator, freeze_csm_codec

    processor, model = tiny_csm_components
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    write_audio(a)
    write_audio(b, frames=5760)
    rows = [speech_row(a), speech_row(b, "Hello longer", "1")]
    original = copy.deepcopy(rows)
    batch = CSMDataCollator(processor, max_length=128)(rows)
    assert rows == original
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert batch["input_values_cutoffs"].tolist() == [[3840], [5760]]
    freeze_csm_codec(model)
    model.train()
    model.codec_model.eval()
    out = model(**batch, use_cache=False)
    assert torch.isfinite(out.loss)
    assert torch.isfinite(out.backbone_loss) and torch.isfinite(out.depth_decoder_loss)
    out.loss.backward()
    for decoder in [model.backbone_model, model.depth_decoder]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.codec_model.parameters())


def test_overlength_audio_frames_are_refused_without_truncation(tmp_path, tiny_csm_components):
    from soup_cli.trainer.csm import CSMDataCollator

    processor, _ = tiny_csm_components
    path = tmp_path / "a.wav"
    write_audio(path)
    with pytest.raises(ValueError, match="max_length"):
        CSMDataCollator(processor, max_length=2)([speech_row(path)])


def test_reserved_audio_marker_in_transcript_refused(tmp_path, tiny_csm_components):
    from soup_cli.trainer.csm import CSMDataCollator

    processor, _ = tiny_csm_components
    with pytest.raises(ValueError, match="control token"):
        CSMDataCollator(processor, max_length=128)([speech_row(tmp_path / "a", "<|AUDIO|>")])


@pytest.mark.parametrize("rank", [0, 2])
def test_real_native_setup_train_save_reload_and_resume(tmp_path, tiny_csm_components, rank):
    import torch
    from transformers import CsmForConditionalGeneration, CsmProcessor

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.dispatch import build_trainer

    processor, model = tiny_csm_components
    base, output = tmp_path / "base", tmp_path / "trained"
    model.save_pretrained(base)
    processor.save_pretrained(base)
    path = tmp_path / "clip.wav"
    write_audio(path)
    values = csm_config(
        epochs=1, batch_size=1, gradient_accumulation_steps=1, lr=0.01,
        logging_steps=1, save_steps=1, warmup_ratio=0, seed=42, lora={"r": rank},
        gradient_checkpointing=rank > 0,
    ).model_dump()
    values.update(base=str(base), output=str(output))
    cfg = SoupConfig(**values)
    dataset = {"train": [speech_row(path), speech_row(path, speaker="1")],
               "val": [speech_row(path)]}
    wrapper = build_trainer(cfg, device="cpu")
    wrapper.setup(dataset)
    assert wrapper.trainer.model_accepts_loss_kwargs is False
    if rank:
        assert all("codec_model" not in name for name, p in wrapper.model.named_parameters()
                   if p.requires_grad)
        assert any("depth_decoder" in name for name, p in wrapper.model.named_parameters()
                   if p.requires_grad)
    result = wrapper.train()
    assert wrapper.trainer.state.global_step == 2
    assert math.isfinite(result["final_loss"])
    assert any("eval_loss" in entry for entry in wrapper.trainer.state.log_history)
    assert not wrapper.model.codec_model.training
    assert all(p.grad is None for p in wrapper.model.codec_model.parameters())
    reloaded_processor = CsmProcessor.from_pretrained(output)
    assert reloaded_processor.chat_template == processor.chat_template
    if rank:
        from peft import PeftModel

        reloaded = PeftModel.from_pretrained(
            CsmForConditionalGeneration.from_pretrained(base), output,
        )
        assert any(p.detach().abs().sum() > 0 for name, p in reloaded.named_parameters()
                   if "lora_B" in name)
        for decoder in ("backbone_model", "depth_decoder"):
            assert any(p.detach().abs().sum() > 0 for name, p in reloaded.named_parameters()
                       if decoder in name and "lora_B" in name)
    else:
        reloaded = CsmForConditionalGeneration.from_pretrained(output)
        assert any(not torch.equal(before, after)
                   for before, after in zip(model.backbone_model.parameters(),
                                            reloaded.backbone_model.parameters()))
    cfg.training.epochs = 2
    resumed = build_trainer(cfg, device="cpu")
    resumed.setup(dataset)
    resumed.train(resume_from_checkpoint=str(output / "checkpoint-2"))
    assert resumed.trainer.state.global_step == 4


def test_padding_with_audio_token_does_not_invent_targets(tmp_path, tiny_csm_components):
    import torch

    from soup_cli.trainer.csm import CSMDataCollator, freeze_csm_codec

    processor, model = tiny_csm_components
    processor.tokenizer.pad_token = processor.audio_token
    path = tmp_path / "clip.wav"
    write_audio(path)
    batch = CSMDataCollator(processor, 128)(
        [speech_row(path), speech_row(path, "Hello longer longer longer")],
    )
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    freeze_csm_codec(model)
    assert torch.isfinite(model(**batch).loss)


@pytest.mark.parametrize("speaker,expected", [("0", 0), ("assistant", 1)])
def test_cli_dry_run_validates_native_transcript_shape(tmp_path, monkeypatch, speaker, expected):
    import yaml
    from typer.testing import CliRunner

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    use_cpu_cli(monkeypatch)
    path = tmp_path / "clip.wav"
    write_audio(path)
    (tmp_path / "speech.jsonl").write_text(json.dumps(speech_row(path, speaker=speaker)) + "\n")
    (tmp_path / "csm.yaml").write_text(yaml.safe_dump(csm_config().model_dump()))
    result = CliRunner().invoke(app, ["train", "-c", "csm.yaml", "--dry-run"])
    assert result.exit_code == expected, result.output
    assert ("Ready to train" in strip_ansi(result.output)) == (expected == 0)
    if expected:
        assert "CSM train row 1" in strip_ansi(result.output)


def test_full_cli_trains_local_native_model(tmp_path, monkeypatch, tiny_csm_components):
    import yaml
    from typer.testing import CliRunner

    from soup_cli.cli import app
    from soup_cli.config.schema import SoupConfig

    monkeypatch.chdir(tmp_path)
    processor, model = tiny_csm_components
    base = tmp_path / "base"
    model.save_pretrained(base)
    processor.save_pretrained(base)
    path = tmp_path / "clip.wav"
    write_audio(path)
    (tmp_path / "speech.jsonl").write_text(json.dumps(speech_row(path)) + "\n")
    values = csm_config(
        epochs=1, batch_size=1, gradient_accumulation_steps=1, logging_steps=1,
        lora={"r": 2},
    ).model_dump()
    values.update(base=str(base), output=str(tmp_path / "trained"))
    cfg = SoupConfig(**values)
    (tmp_path / "csm.yaml").write_text(yaml.safe_dump(cfg.model_dump()))
    # The CLI has no --device flag; isolate this portable test from the host's
    # accelerator choice while leaving the data/model/trainer/save paths real.
    use_cpu_cli(monkeypatch)
    result = CliRunner().invoke(app, ["train", "-c", "csm.yaml", "--yes"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert (tmp_path / "trained" / "adapter_model.safetensors").is_file()
    from transformers import CsmProcessor

    saved_processor = CsmProcessor.from_pretrained(tmp_path / "trained")
    assert saved_processor.feature_extractor.sampling_rate == 24000
    assert saved_processor.chat_template == processor.chat_template


def test_csm_imports_keep_heavy_dependencies_lazy():
    import subprocess
    import sys

    command = (
        "import sys; import soup_cli.utils.csm; import soup_cli.trainer.csm; "
        "assert not {'torch','transformers','peft','trl','mlx'}.intersection(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_native_mps_training_step(tmp_path, tiny_csm_components):
    import torch

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.dispatch import build_trainer

    if not torch.backends.mps.is_available():
        pytest.skip("needs Apple Silicon MPS")
    processor, model = tiny_csm_components
    base = tmp_path / "base"
    model.save_pretrained(base)
    processor.save_pretrained(base)
    path = tmp_path / "clip.wav"
    write_audio(path)
    values = csm_config(
        epochs=1, batch_size=1, gradient_accumulation_steps=1, logging_steps=1,
        lora={"r": 2}, seed=42,
    ).model_dump()
    values.update(base=str(base), output=str(tmp_path / "trained"))
    wrapper = build_trainer(SoupConfig(**values), device="mps")
    wrapper.setup({"train": [speech_row(path)]})
    from soup_cli.utils.gpu import mps_supports_bf16

    assert wrapper.trainer.args.device.type == "mps"
    assert wrapper.trainer.args.bf16 == mps_supports_bf16()
    assert wrapper.trainer.args.fp16 is False
    result = wrapper.train()
    assert math.isfinite(result["final_loss"])
    assert wrapper.trainer.state.global_step == 1
    assert all(p.grad is None for p in wrapper.model.codec_model.parameters())


@pytest.mark.parametrize("option", ["loraplus_lr_ratio", "use_lorafa"])
def test_native_csm_refuses_unimplemented_lora_optimizers(option):
    with pytest.raises(ValueError, match=option):
        csm_config(**{option: 16.0 if option == "loraplus_lr_ratio" else True})


@pytest.mark.parametrize("kwargs", [
    {"deepspeed_config": "ds.json"}, {"fsdp_config": {"fsdp": "full_shard"}},
])
def test_native_csm_refuses_distributed_before_loading_any_model(kwargs):
    from soup_cli.trainer.csm import CSMTrainerWrapper

    wrapper = CSMTrainerWrapper(csm_config(), device="cpu", **kwargs)
    with pytest.raises(ValueError, match="single-device"):
        wrapper.setup({})
    assert wrapper.model is None
    assert wrapper.trainer is None


def test_native_csm_refuses_automatic_cuda_data_parallel(monkeypatch):
    import torch

    from soup_cli.trainer.csm import CSMTrainerWrapper

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    wrapper = CSMTrainerWrapper(csm_config(), device="cuda")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        wrapper.setup({})
    assert wrapper.model is None


def test_native_csm_refuses_torchrun_before_loading(monkeypatch):
    from soup_cli.trainer.csm import CSMTrainerWrapper

    monkeypatch.setenv("WORLD_SIZE", "2")
    wrapper = CSMTrainerWrapper(csm_config(), device="cpu")
    with pytest.raises(ValueError, match="single-device"):
        wrapper.setup({})
    assert wrapper.model is None
