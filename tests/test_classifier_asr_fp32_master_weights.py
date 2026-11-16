"""#ISSUE: classifier-family and ASR full fine-tuning trained the base in its checkpoint dtype.

``trainer/classifier.py`` and ``trainer/asr.py`` called ``from_pretrained`` with no
``dtype``. Transformers 5 then loads the checkpoint's own (``"auto"``): bf16 for most
decoder bases, fp16 for ``openai/whisper-large-v3`` and ``-large-v3-turbo``. Both tasks
full fine-tune by default (``classifier_lora`` / ``asr_lora`` are opt-in), so AdamW
stepped half-precision weights with half-precision state and no fp32 master copy, the
defect #1235 fixed for PRM and #1266 for the MoLE gate: an update of about ``lr`` is
below half a bf16 ulp for every ``|w| >= 2**-7`` at ``lr=2e-5``, so those weights never
move. Every other full fine-tune loads fp32 through ``resolve_base_load_dtype`` (#339),
and the VRAM estimate prices these two runs that way.

On the classifier's LoRA path peft trains a copy of the classification head
(``modules_to_save``) in the frozen base's dtype, and its ``autocast_adapter_dtype``
upcasts only ``lora_*``; under mixed precision that copy now gets an fp32 master too.

Each test drives the real ``setup`` (and ``train`` where it says so) on CPU, with a
tiny random Llama saved in bf16 and a tiny random Whisper saved in fp16, so a load that
falls back to the checkpoint's dtype cannot pass for an fp32 one. Mixed precision runs
as CPU autocast through ``use_cpu=True`` (the same Accelerate forward wrapper CUDA
uses), on ATen's kernels (``aten_half_matmuls`` in conftest.py).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

_LR = 2.0e-5  # the schema default

_TEXTS = (
    "the cat sat on the mat",
    "a dog ran in the park",
    "stocks fell sharply today",
    "the team won the final",
)


@pytest.fixture(autouse=True)
def train_stack(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[None]:
    """Real torch/transformers/peft, one thread, and ``TrainingArguments`` on CPU except
    for the ``gpu`` test."""
    for name in ("torch", "transformers", "peft", "datasets"):
        pytest.importorskip(name, reason=f"{name} requires the train extra")
    import torch
    from transformers import TrainingArguments

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    real_post_init = TrainingArguments.__post_init__

    def cpu_post_init(args: TrainingArguments) -> None:
        args.use_cpu = True
        real_post_init(args)

    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(TrainingArguments, "__post_init__", cpu_post_init)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def bf16_llama(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A tiny random Llama with a pad id, saved in bf16 like most published bases."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from tests._tiny_hf_models import build_tokenizer

    target = tmp_path_factory.mktemp("cls-dtype") / "tiny-llama-bf16"
    tokenizer = build_tokenizer(_TEXTS)
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, pad_token_id=tokenizer.pad_token_id,
    )).to(torch.bfloat16).save_pretrained(str(target))
    tokenizer.save_pretrained(str(target))
    return str(target)


@pytest.fixture(scope="module")
def fp16_whisper(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A tiny random Whisper saved in fp16, as whisper-large-v3 is (no download)."""
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
            "<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|translate|>",
            "<|transcribe|>", "<|notimestamps|>",
        ]
    })
    target = tmp_path_factory.mktemp("asr-dtype") / "whisper-fp16"
    WhisperProcessor(
        feature_extractor=WhisperFeatureExtractor(feature_size=8, chunk_length=1),
        tokenizer=tokenizer,
    ).save_pretrained(str(target))
    tokenizer = WhisperProcessor.from_pretrained(str(target)).tokenizer
    config = WhisperConfig(
        vocab_size=len(tokenizer), num_mel_bins=8, d_model=16, encoder_layers=1,
        decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, max_source_positions=50,
        max_target_positions=32, pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
    )
    torch.manual_seed(7)
    WhisperForConditionalGeneration(config).to(torch.float16).save_pretrained(str(target))
    return str(target)


@pytest.fixture
def precision(monkeypatch: pytest.MonkeyPatch, aten_half_matmuls: Any):
    """Pin the ``(bf16, fp16)`` flags the classifier asks for; the run stays on CPU."""
    import soup_cli.trainer.classifier as classifier_mod

    def apply(bf16: bool, fp16: bool) -> None:
        monkeypatch.setattr(
            classifier_mod, "bf16_fp16_flags", lambda device, **_kw: (bf16, fp16)
        )

    apply(False, False)
    return apply


def _classifier(base: str, out: Path, *, task: str = "classifier", device: str = "cuda",
                epochs: int = 1, **training: Any):
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.classifier import ClassifierTrainerWrapper

    block: dict[str, Any] = {
        "num_labels": 2, "epochs": epochs, "lr": _LR, "batch_size": 2,
        "gradient_accumulation_steps": 1, "warmup_ratio": 0.0, "save_steps": 10_000,
        "logging_steps": 1, "seed": 7,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    block.update(training)
    cfg = load_config_from_string(yaml.safe_dump({
        "base": base, "task": task, "backend": "transformers",
        "data": {"train": "train.jsonl", "max_length": 64},
        "training": block, "output": str(out),
    }))
    if task == "cross_encoder":
        rows = [{"text_a": text, "text_b": text[::-1], "label": index % 2}
                for index, text in enumerate(_TEXTS)]
    else:
        rows = [{"text": text, "label": index % 2} for index, text in enumerate(_TEXTS)]
    wrapper = ClassifierTrainerWrapper(cfg, device=device)
    wrapper.setup({"train": rows})
    return wrapper


def _asr(base: str, out: Path, monkeypatch: pytest.MonkeyPatch, *, device: str = "cuda",
         **training: Any):
    import numpy as np
    import yaml

    import soup_cli.utils.tts_codec as tts_codec
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.asr import AsrTrainerWrapper

    def sine(path: str, *, target_sr: int) -> Any:
        seconds = np.arange(target_sr // 2, dtype=np.float32) / target_sr
        return (0.1 * np.sin(2 * np.pi * 220.0 * seconds)).astype(np.float32)

    monkeypatch.setattr(tts_codec, "load_audio_mono", sine)
    block: dict[str, Any] = {
        "quantization": "none", "epochs": 1, "lr": _LR, "batch_size": 2,
        "gradient_accumulation_steps": 1, "warmup_ratio": 0.0, "save_steps": 10_000,
        "seed": 7, "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
    }
    block.update(training)
    cfg = load_config_from_string(yaml.safe_dump({
        "base": base, "task": "asr", "backend": "transformers",
        "data": {"train": "train.jsonl", "format": "asr", "max_length": 64},
        "training": block, "output": str(out),
    }))
    wrapper = AsrTrainerWrapper(cfg, device=device)
    wrapper.setup({"train": [{"audio": f"clip{i}.wav", "text": "hello world"} for i in range(4)]})
    return wrapper


def _group(name: str) -> str:
    if "score" in name:
        return "score"
    if "norm" in name:
        return "norms"
    if "embed_tokens" in name:
        return "embed_tokens"
    return "attn/mlp"


def _unchanged_fraction_per_group(before: dict, model: Any) -> dict[str, float]:
    """Per group of trained parameters, the fraction of elements bit-identical after training.

    Compared in fp32: a parameter upcast inside ``train()`` can move by less than the
    bf16 ulp its snapshot was taken in.
    """
    totals: dict[str, list[int]] = {}
    for name, param in model.named_parameters():
        if name not in before:
            continue
        same = int((before[name].float() == param.detach().float()).sum())
        entry = totals.setdefault(_group(name), [0, 0])
        entry[0] += same
        entry[1] += param.numel()
    return {group: same / total for group, (same, total) in totals.items()}


class TestClassifierLoadDtype:
    """A full fine-tune of the classifier family loads fp32 master weights."""

    @pytest.mark.parametrize("task", ["classifier", "reranker", "cross_encoder"])
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_finetune_loads_every_parameter_as_fp32(
        self, bf16_llama, tmp_path, monkeypatch, task, device
    ):
        """On ``main`` every parameter, the new ``score`` head included, was bf16."""
        import torch

        monkeypatch.chdir(tmp_path)
        wrapper = _classifier(bf16_llama, tmp_path / "out", task=task, device=device)
        params = dict(wrapper.model.named_parameters())
        assert "score.weight" in params
        assert all(p.requires_grad for p in params.values()), "a full fine-tune trains all"
        wrong = {name: str(p.dtype) for name, p in params.items() if p.dtype != torch.float32}
        assert not wrong, f"not fp32 master weights for {task} on {device!r}: {wrong}"

    def test_lora_path_keeps_the_frozen_base_at_its_checkpoint_dtype(
        self, bf16_llama, tmp_path, monkeypatch
    ):
        """Control: a frozen base needs no master copy (#339); only the full fine-tune
        changes. peft's adapters are fp32 as before."""
        import torch

        monkeypatch.chdir(tmp_path)
        wrapper = _classifier(bf16_llama, tmp_path / "out", classifier_lora=True)
        frozen = {p.dtype for p in wrapper.model.parameters() if not p.requires_grad}
        lora = {p.dtype for n, p in wrapper.model.named_parameters() if "lora_" in n}
        assert frozen == {torch.bfloat16}
        assert lora == {torch.float32}


class TestClassifierTrains:
    """The default lr moves every trained group, under bf16 autocast and on plain CPU."""

    @pytest.mark.parametrize("bf16", [True, False], ids=["bf16-autocast", "no-autocast"])
    def test_full_finetune_leaves_no_group_bit_identical(
        self, bf16_llama, tmp_path, monkeypatch, precision, bf16
    ):
        """On ``main``, 4 steps at the schema-default lr left every RMSNorm weight (which
        starts at 1.0, two bf16 ulps from its neighbours) bit-identical."""
        import torch

        monkeypatch.chdir(tmp_path)
        precision(bf16, False)
        wrapper = _classifier(bf16_llama, tmp_path / "out", epochs=2)
        before = {n: p.detach().clone() for n, p in wrapper.model.named_parameters()}

        wrapper.train()

        trainer = wrapper.trainer
        assert trainer.args.use_cpu and trainer.args.bf16 is bf16
        assert trainer.state.global_step == 4
        assert {p.dtype for p in wrapper.model.parameters()} == {torch.float32}
        unchanged = _unchanged_fraction_per_group(before, wrapper.model)
        assert set(unchanged) == {"score", "norms", "embed_tokens", "attn/mlp"}, unchanged
        # The embedding rows of tokens no batch contains get no gradient in any
        # precision, so that group is held to "not every element", the rest to 1%.
        assert unchanged["embed_tokens"] < 1.0, unchanged
        for group in ("score", "norms", "attn/mlp"):
            assert unchanged[group] <= 0.01, (group, unchanged)

    def test_lora_head_gets_an_fp32_master_under_mixed_precision(
        self, bf16_llama, tmp_path, monkeypatch, precision
    ):
        """On ``main`` the head copy peft trains (``modules_to_save``) stayed bf16 and
        most of its weights never moved at the default lr."""
        import torch

        monkeypatch.chdir(tmp_path)
        precision(True, False)
        wrapper = _classifier(bf16_llama, tmp_path / "out", epochs=2, classifier_lora=True)
        head = "base_model.model.score.modules_to_save.default.weight"
        before = {n: p.detach().clone() for n, p in wrapper.model.named_parameters()
                  if p.requires_grad}
        assert head in before, sorted(before)

        wrapper.train()

        params = dict(wrapper.model.named_parameters())
        assert params[head].dtype == torch.float32
        trained = {n: p.dtype for n, p in params.items() if p.requires_grad}
        assert set(trained.values()) == {torch.float32}, trained
        unchanged = _unchanged_fraction_per_group(before, wrapper.model)
        assert unchanged["score"] <= 0.01, unchanged
        frozen = {p.dtype for p in params.values() if not p.requires_grad}
        assert frozen == {torch.bfloat16}, "the frozen base is not upcast"

    def test_lora_head_is_left_alone_without_mixed_precision(
        self, bf16_llama, tmp_path, monkeypatch, precision
    ):
        """Without autocast the bf16 activations of the frozen base would meet an fp32
        head, so nothing is cast and the run trains as before."""
        import torch

        monkeypatch.chdir(tmp_path)
        precision(False, False)
        wrapper = _classifier(bf16_llama, tmp_path / "out", classifier_lora=True)

        wrapper.train()

        head = dict(wrapper.model.named_parameters())[
            "base_model.model.score.modules_to_save.default.weight"
        ]
        assert head.dtype == torch.bfloat16
        assert wrapper.trainer.state.global_step == 2


@pytest.mark.gpu
@pytest.mark.parametrize("classifier_lora", [False, True], ids=["full-finetune", "lora"])
def test_real_card_steps_move_every_trained_group(
    bf16_llama, tmp_path, monkeypatch, classifier_lora
):
    """The same run on a CUDA card, under the mixed precision it picks there.

    On ``main``, on an RTX PRO 5000 Blackwell (bf16 autocast), every trained tensor was
    bf16 and this run failed on the dtype; a 2-layer Qwen2 classifier with the Qwen2.5
    vocabulary left 100% of its norms and embeddings, 69.7% of attention/MLP and 69.3%
    of the head bit-identical after 8 steps at the default lr, and with
    ``classifier_lora`` 69.3% of the head.
    """
    import torch

    monkeypatch.chdir(tmp_path)
    wrapper = _classifier(
        bf16_llama, tmp_path / "out", device="cuda", epochs=2,
        classifier_lora=classifier_lora,
    )
    before = {n: p.detach().float().clone() for n, p in wrapper.model.named_parameters()
              if p.requires_grad}

    wrapper.train()

    args = wrapper.trainer.args
    assert args.bf16 or args.fp16, "a CUDA card trains under autocast"
    params = dict(wrapper.model.named_parameters())
    assert next(iter(params.values())).device.type == "cuda"
    trained = {n: p.dtype for n, p in params.items() if p.requires_grad}
    assert set(trained.values()) == {torch.float32}, trained
    unchanged = _unchanged_fraction_per_group(before, wrapper.model)
    groups = ("score",) if classifier_lora else ("score", "norms", "attn/mlp")
    for group in groups:
        assert unchanged[group] <= 0.01, (group, unchanged)


class TestUpcastHelper:
    def test_casts_only_trainable_half_parameters_under_mixed_precision(self):
        import torch

        from soup_cli.utils.mixed_precision import upcast_trainable_half_to_fp32

        def model() -> Any:
            net = torch.nn.Sequential(
                torch.nn.Linear(4, 4).to(torch.bfloat16),
                torch.nn.Linear(4, 4).to(torch.float16),
                torch.nn.Linear(4, 2).to(torch.bfloat16),
            )
            net[0].requires_grad_(False)
            return net

        for flags in ({"bf16": True, "fp16": False}, {"bf16": False, "fp16": True}):
            net = model()
            assert upcast_trainable_half_to_fp32(net, **flags) == 4
            assert [p.dtype for p in net.parameters()] == [
                torch.bfloat16, torch.bfloat16, torch.float32, torch.float32,
                torch.float32, torch.float32,
            ]
        net = model()
        assert upcast_trainable_half_to_fp32(net, bf16=False, fp16=False) == 0
        assert upcast_trainable_half_to_fp32(None, bf16=True, fp16=False) == 0


class TestAsrLoadDtype:
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_finetune_loads_every_parameter_as_fp32(
        self, fp16_whisper, tmp_path, monkeypatch, device
    ):
        """On ``main`` an fp16 Whisper checkpoint trained in fp16, all of it."""
        import torch

        monkeypatch.chdir(tmp_path)
        wrapper = _asr(fp16_whisper, tmp_path / "out", monkeypatch, device=device)
        params = dict(wrapper.model.named_parameters())
        assert all(p.requires_grad for p in params.values())
        wrong = {n: str(p.dtype) for n, p in params.items() if p.dtype != torch.float32}
        assert not wrong, f"not fp32 master weights on {device!r}: {wrong}"

    def test_full_finetune_step_moves_the_layer_norms(
        self, fp16_whisper, tmp_path, monkeypatch
    ):
        """Whisper's LayerNorm weights start at 1.0. On ``main`` an fp16 AdamW step of
        about ``lr`` (below half an fp16 ulp at 1.0, 2**-11) left all of them where they
        were."""
        monkeypatch.chdir(tmp_path)
        wrapper = _asr(fp16_whisper, tmp_path / "out", monkeypatch, device="cpu")
        norms = {n: p.detach().clone() for n, p in wrapper.model.named_parameters()
                 if "layer_norm" in n and n.endswith("weight")}
        assert norms

        wrapper.trainer.train()

        params = dict(wrapper.model.named_parameters())
        stuck = sorted(n for n, w in norms.items() if bool((params[n].detach() == w).all()))
        assert not stuck, f"bit-identical after a step at lr={_LR}: {stuck}"

    def test_asr_lora_keeps_the_frozen_base_at_its_checkpoint_dtype(
        self, fp16_whisper, tmp_path, monkeypatch
    ):
        import torch

        monkeypatch.chdir(tmp_path)
        wrapper = _asr(fp16_whisper, tmp_path / "out", monkeypatch, device="cuda",
                       asr_lora=True)
        frozen = {p.dtype for p in wrapper.model.parameters() if not p.requires_grad}
        assert frozen == {torch.float16}
