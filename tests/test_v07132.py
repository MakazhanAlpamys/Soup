"""v0.71.32 — ASR (Whisper) fine-tuning.

Covers pure-python WER/CER metrics (``utils/asr_metrics``), the ``task='asr'``
schema surface (task/format Literals + ``asr_language``/``asr_task`` +
cross-validators), the ``AsrTrainerWrapper`` (arch guard + row validation +
``Seq2SeqTrainer`` build), ``soup infer --task asr``, and the whisper/smolvlm
recipes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Task 1 — WER / CER metrics
# ---------------------------------------------------------------------------


class TestAsrMetrics:
    def test_normalize_text_lower_punct_ws(self):
        from soup_cli.utils.asr_metrics import normalize_text

        assert normalize_text("The Cat, sat!") == "the cat sat"
        assert normalize_text("the   cat    sat") == "the cat sat"
        assert normalize_text("HELLO") == "hello"

    def test_normalize_text_opt_out(self):
        from soup_cli.utils.asr_metrics import normalize_text

        assert normalize_text("The Cat!", lower=False, strip_punct=False) == "The Cat!"

    def test_wer_identity_zero(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat sat", "the cat sat") == 0.0

    def test_wer_all_substitutions_one(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat sat", "a dog ran") == 1.0

    def test_wer_single_deletion(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat sat", "the cat") == pytest.approx(1 / 3)

    def test_wer_single_insertion(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat", "the cat sat") == pytest.approx(1 / 2)

    def test_wer_single_substitution(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)

    def test_wer_normalizes_by_default(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("The cat, sat.", "the cat sat") == 0.0

    def test_wer_empty_both_zero(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("", "") == 0.0

    def test_wer_empty_ref_nonempty_hyp_one(self):
        from soup_cli.utils.asr_metrics import wer

        assert wer("", "hello world") == 1.0

    def test_cer_char_level(self):
        from soup_cli.utils.asr_metrics import cer

        assert cer("abc", "abc") == 0.0
        assert cer("cat", "cot") == pytest.approx(1 / 3)

    def test_word_accuracy_is_one_minus_wer(self):
        from soup_cli.utils.asr_metrics import word_accuracy

        assert word_accuracy("the cat sat", "the cat sat") == 1.0
        # all-wrong -> wer 1.0 -> accuracy 0.0 (clamped, never negative)
        assert word_accuracy("the cat sat", "a dog ran") == 0.0

    def test_word_accuracy_clamps_at_zero(self):
        from soup_cli.utils.asr_metrics import word_accuracy

        # many insertions push wer > 1.0; accuracy must clamp to 0.0
        assert word_accuracy("cat", "a b c d e f") == 0.0

    def test_corpus_wer_is_not_mean(self):
        from soup_cli.utils.asr_metrics import corpus_wer

        # per-example wers are 0.0 and 1.0 -> mean would be 0.5.
        # corpus = (0 edits + 1 edit) / (3 + 1 ref words) = 0.25.
        val = corpus_wer(["the cat sat", "hello"], ["the cat sat", "world"])
        assert val == pytest.approx(0.25)

    def test_corpus_wer_length_mismatch_raises(self):
        from soup_cli.utils.asr_metrics import corpus_wer

        with pytest.raises(ValueError, match="same length"):
            corpus_wer(["a"], ["a", "b"])

    def test_seq_cap_raises(self):
        from soup_cli.utils.asr_metrics import _MAX_SEQ, wer

        long_ref = " ".join(["w"] * (_MAX_SEQ + 1))
        with pytest.raises(ValueError, match="too long"):
            wer(long_ref, "short", normalize=False)


class TestWhisperSizeGate:
    # Regression for the live-smoke finding: model_size_from_name returned the
    # 7.0-billion default for whisper checkpoints, making the hardware-fit gate
    # predict ~19 GB and block ASR training on any consumer GPU.
    def test_whisper_sizes_not_default(self):
        from soup_cli.utils.gpu import model_size_from_name

        assert model_size_from_name("openai/whisper-tiny") == pytest.approx(0.039)
        assert model_size_from_name("openai/whisper-base") == pytest.approx(0.074)
        assert model_size_from_name("openai/whisper-small") == pytest.approx(0.244)
        assert model_size_from_name("openai/whisper-large-v3") == pytest.approx(1.55)

    def test_non_whisper_still_defaults(self):
        from soup_cli.utils.gpu import model_size_from_name

        # unchanged behaviour for a nameless model
        assert model_size_from_name("some/unknown-model") == 7.0
        assert model_size_from_name("meta-llama/Llama-3.1-8B") == 8


class TestAsrMetricsBranches:
    def test_normalize_text_type_error(self):
        from soup_cli.utils.asr_metrics import normalize_text

        with pytest.raises(TypeError):
            normalize_text(123)

    def test_wer_empty_hyp_nonempty_ref(self):
        from soup_cli.utils.asr_metrics import wer

        # 3 ref words, empty hyp -> 3 deletions / 3 = 1.0
        assert wer("the cat sat", "") == 1.0

    def test_corpus_wer_mixed_empty_ref_counts_hallucination(self):
        # Regression for the tdd-review correctness gap: an empty-ref row with a
        # hallucinated hypothesis must NOT vanish from the numerator.
        from soup_cli.utils.asr_metrics import corpus_wer

        # row0: ref='' hyp='x' -> 1 insertion; row1: ref='a b' hyp='a b' -> 0.
        # edits=1, ref words=2 -> 0.5 (would be 0.0 if the empty-ref row dropped).
        assert corpus_wer(["", "a b"], ["x", "a b"]) == pytest.approx(0.5)


class TestNoTopLevelTorch:
    def test_asr_metrics_has_no_heavy_top_level_import(self):
        import soup_cli.utils.asr_metrics as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        banned = {"torch", "transformers", "peft", "datasets"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split(".")[0] not in banned


# ---------------------------------------------------------------------------
# Task 2 — schema (task=asr / format=asr / asr fields + validators)
# ---------------------------------------------------------------------------


def _asr_yaml(
    *,
    task="asr",
    backend="transformers",
    fmt="asr",
    asr_language=None,
    asr_task=None,
):
    lines = [
        "base: openai/whisper-tiny",
        f"task: {task}",
        f"backend: {backend}",
        "data:",
        "  train: ./data/train.jsonl",
        f"  format: {fmt}",
        "training:",
        "  epochs: 1",
        "  lr: 1e-4",
        "  batch_size: 4",
    ]
    if asr_language is not None:
        lines.append(f"  asr_language: {asr_language}")
    if asr_task is not None:
        lines.append(f"  asr_task: {asr_task}")
    return "\n".join(lines) + "\n"


class TestAsrSchema:
    def test_happy_parse_defaults(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_asr_yaml())
        assert cfg.task == "asr"
        assert cfg.data.format == "asr"
        assert cfg.training.asr_language is None
        assert cfg.training.asr_task == "transcribe"

    def test_language_and_task_parse(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_asr_yaml(asr_language="en", asr_task="translate"))
        assert cfg.training.asr_language == "en"
        assert cfg.training.asr_task == "translate"

    def test_reject_mlx_backend(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="transformers"):
            load_config_from_string(_asr_yaml(backend="mlx"))

    def test_reject_unsloth_backend(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="transformers"):
            load_config_from_string(_asr_yaml(backend="unsloth"))

    def test_footgun_language_on_non_asr(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="asr"):
            load_config_from_string(_asr_yaml(task="sft", fmt="alpaca", asr_language="en"))

    def test_footgun_translate_on_non_asr(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="asr"):
            load_config_from_string(_asr_yaml(task="sft", fmt="alpaca", asr_task="translate"))

    def test_field_validator_empty_language(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="asr_language"):
            load_config_from_string(_asr_yaml(asr_language='"   "'))

    def test_field_validator_oversize_language(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="asr_language"):
            load_config_from_string(_asr_yaml(asr_language="x" * 40))

    def test_invalid_asr_task_literal(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError):
            load_config_from_string(_asr_yaml(asr_task="frobnicate"))

    def test_asr_rejects_wrong_data_format(self):
        # task=asr must pin data.format to asr/auto (python-review MEDIUM).
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="data.format"):
            load_config_from_string(_asr_yaml(fmt="alpaca"))

    def test_asr_allows_auto_format(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(_asr_yaml(fmt="auto"))
        assert cfg.data.format == "auto"


# ---------------------------------------------------------------------------
# Task 3 — AsrTrainerWrapper + routing + data format
# ---------------------------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


_TORCH = _torch_available()
_TINY_WHISPER = "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"


def _write_sine_wav(path, *, seconds: float = 1.0, sr: int = 16000):
    import numpy as np
    import soundfile as sf

    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    wave = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    sf.write(str(path), wave.astype("float32"), sr)


class TestAsrRow:
    def test_validate_asr_row_ok(self):
        from soup_cli.trainer.asr import _validate_asr_row

        assert _validate_asr_row({"audio": "a.wav", "text": "hello"}) == (
            "a.wav",
            "hello",
        )

    def test_validate_asr_row_missing_audio(self):
        from soup_cli.trainer.asr import _validate_asr_row

        with pytest.raises(ValueError, match="audio"):
            _validate_asr_row({"text": "hi"})

    def test_validate_asr_row_missing_text(self):
        from soup_cli.trainer.asr import _validate_asr_row

        with pytest.raises(ValueError, match="text"):
            _validate_asr_row({"audio": "a.wav"})

    def test_validate_asr_row_non_str_audio(self):
        from soup_cli.trainer.asr import _validate_asr_row

        with pytest.raises((ValueError, TypeError)):
            _validate_asr_row({"audio": 123, "text": "hi"})

    def test_validate_asr_row_empty_audio_string(self):
        from soup_cli.trainer.asr import _validate_asr_row

        with pytest.raises(ValueError, match="audio"):
            _validate_asr_row({"audio": "   ", "text": "hi"})


class TestAsrArchGuard:
    def test_require_whisper_rejects_non_whisper(self, monkeypatch):
        from soup_cli.trainer import asr as asrmod

        class _FakeCfg:
            model_type = "gpt2"

        monkeypatch.setattr(asrmod, "_load_autoconfig", lambda base, trust: _FakeCfg())
        with pytest.raises(ValueError, match="whisper"):
            asrmod._require_whisper_base("some/gpt2-model", False)

    def test_require_whisper_accepts_whisper(self, monkeypatch):
        from soup_cli.trainer import asr as asrmod

        class _FakeCfg:
            model_type = "whisper"

        monkeypatch.setattr(asrmod, "_load_autoconfig", lambda base, trust: _FakeCfg())
        # Returns the config, does not raise.
        assert asrmod._require_whisper_base("openai/whisper-tiny", False).model_type == "whisper"


class TestAsrPrefixCustomized:
    def test_default_is_not_customized(self):
        from soup_cli.trainer.asr import _prefix_customized

        assert _prefix_customized(None, "transcribe") is False

    def test_language_customizes(self):
        from soup_cli.trainer.asr import _prefix_customized

        assert _prefix_customized("en", "transcribe") is True

    def test_bare_translate_customizes(self):
        # python-review HIGH: asr_task='translate' with no language must NOT
        # silently fall back to transcribe.
        from soup_cli.trainer.asr import _prefix_customized

        assert _prefix_customized(None, "translate") is True


class TestAsrCollatorStrip:
    @pytest.mark.skipif(not _TORCH, reason="needs torch")
    def test_strips_leading_decoder_start(self):
        import torch

        from soup_cli.trainer.asr import _strip_decoder_start

        labels = torch.tensor([[50258, 100, 200, 50257], [50258, 300, 400, 50257]])
        out = _strip_decoder_start(labels, 50258)
        assert out.shape[1] == 3
        assert out[0].tolist() == [100, 200, 50257]

    @pytest.mark.skipif(not _TORCH, reason="needs torch")
    def test_does_not_strip_when_not_all_start(self):
        import torch

        from soup_cli.trainer.asr import _strip_decoder_start

        labels = torch.tensor([[50258, 100], [999, 100]])
        out = _strip_decoder_start(labels, 50258)
        assert out.shape[1] == 2  # unchanged

    @pytest.mark.skipif(not _TORCH, reason="needs torch")
    def test_none_decoder_start_is_noop(self):
        import torch

        from soup_cli.trainer.asr import _strip_decoder_start

        labels = torch.tensor([[50258, 100]])
        assert _strip_decoder_start(labels, None).shape[1] == 2


class TestAsrCollatorCall:
    @pytest.mark.skipif(not _TORCH, reason="needs torch")
    def test_masks_pad_and_strips_decoder_start(self):
        import torch

        from soup_cli.trainer.asr import _SpeechSeq2SeqCollator

        class _FE:
            def pad(self, feats, return_tensors="pt"):
                return {"input_features": torch.zeros((2, 4))}

        class _Tok:
            def pad(self, feats, return_tensors="pt"):
                # row0 len 3, row1 len 2 padded to 3 -> attention_mask marks pad.
                ids = torch.tensor([[50258, 7, 9], [50258, 8, 0]])
                mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
                return {"input_ids": ids, "attention_mask": mask}

        class _Proc:
            feature_extractor = _FE()
            tokenizer = _Tok()

        collator = _SpeechSeq2SeqCollator(_Proc(), decoder_start_token_id=50258)
        batch = collator([{"input_features": None, "labels": [50258, 7, 9]},
                          {"input_features": None, "labels": [50258, 8]}])
        labels = batch["labels"]
        # decoder-start column stripped -> width 2
        assert labels.shape[1] == 2
        # pad position masked to -100
        assert labels[1, 1].item() == -100


class TestAsrGenPrefix:
    def test_flag_beats_sidecar(self):
        from soup_cli.commands.infer import _resolve_asr_gen_prefix

        got = _resolve_asr_gen_prefix("en", "translate", {"language": "es", "task": "transcribe"})
        assert got == {"language": "en", "task": "translate"}

    def test_sidecar_when_no_flag(self):
        from soup_cli.commands.infer import _resolve_asr_gen_prefix

        got = _resolve_asr_gen_prefix(None, None, {"language": "es", "task": "translate"})
        assert got == {"language": "es", "task": "translate"}

    def test_empty_when_neither(self):
        from soup_cli.commands.infer import _resolve_asr_gen_prefix

        assert _resolve_asr_gen_prefix(None, None, {}) == {}


class TestAsrBuildTranscriberGuards:
    def test_adapter_without_base_exits(self, tmp_path, monkeypatch):
        import json

        import typer

        import soup_cli.commands.infer as infer_mod

        (tmp_path / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")
        with pytest.raises(typer.Exit) as exc:
            infer_mod._build_asr_transcriber(
                str(tmp_path), None, "cpu", 8, False
            )
        assert exc.value.exit_code == 1


class TestAsrInferExitPaths:
    def test_output_outside_cwd_exits(self, tmp_path, monkeypatch):
        import typer

        import soup_cli.commands.infer as infer_mod

        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.jsonl").write_text('{"audio": "a.wav"}\n', encoding="utf-8")
        with pytest.raises(typer.Exit) as exc:
            infer_mod._infer_asr(
                model="m", base=None, input_file="in.jsonl", device="cpu",
                output_file="../evil.jsonl", max_tokens=8, trust_remote_code=False,
            )
        assert exc.value.exit_code == 1

    def test_audio_dir_outside_cwd_exits(self, tmp_path, monkeypatch):
        import typer

        import soup_cli.commands.infer as infer_mod

        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.jsonl").write_text('{"audio": "a.wav"}\n', encoding="utf-8")
        with pytest.raises(typer.Exit) as exc:
            infer_mod._infer_asr(
                model="m", base=None, input_file="in.jsonl", device="cpu",
                output_file="out.jsonl", max_tokens=8, trust_remote_code=False,
                audio_dir="../elsewhere",
            )
        assert exc.value.exit_code == 1

    def test_zero_rows_exits(self, tmp_path, monkeypatch):
        import typer

        import soup_cli.commands.infer as infer_mod

        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.jsonl").write_text('{"text": "no audio"}\n', encoding="utf-8")
        with pytest.raises(typer.Exit) as exc:
            infer_mod._infer_asr(
                model="m", base=None, input_file="in.jsonl", device="cpu",
                output_file="out.jsonl", max_tokens=8, trust_remote_code=False,
            )
        assert exc.value.exit_code == 1

    def test_build_importerror_exits_1(self, tmp_path, monkeypatch):
        import typer

        import soup_cli.commands.infer as infer_mod

        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.jsonl").write_text('{"audio": "a.wav"}\n', encoding="utf-8")
        monkeypatch.setattr(infer_mod, "_ASR_TRANSCRIBER_OVERRIDE", None)

        def _boom(*a, **k):
            raise ImportError("no soundfile")

        monkeypatch.setattr(infer_mod, "_build_asr_transcriber", _boom)
        with pytest.raises(typer.Exit) as exc:
            infer_mod._infer_asr(
                model="m", base=None, input_file="in.jsonl", device="cpu",
                output_file="out.jsonl", max_tokens=8, trust_remote_code=False,
            )
        assert exc.value.exit_code == 1

    def test_build_valueerror_exits_2(self, tmp_path, monkeypatch):
        import typer

        import soup_cli.commands.infer as infer_mod

        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.jsonl").write_text('{"audio": "a.wav"}\n', encoding="utf-8")
        monkeypatch.setattr(infer_mod, "_ASR_TRANSCRIBER_OVERRIDE", None)

        def _boom(*a, **k):
            raise ValueError("not a whisper base")

        monkeypatch.setattr(infer_mod, "_build_asr_transcriber", _boom)
        with pytest.raises(typer.Exit) as exc:
            infer_mod._infer_asr(
                model="m", base=None, input_file="in.jsonl", device="cpu",
                output_file="out.jsonl", max_tokens=8, trust_remote_code=False,
            )
        assert exc.value.exit_code == 2


class TestAsrSidecar:
    def test_write_read_roundtrip(self, tmp_path):
        from soup_cli.trainer.asr import read_asr_sidecar, write_asr_sidecar

        write_asr_sidecar(str(tmp_path), "spanish", "translate")
        got = read_asr_sidecar(str(tmp_path))
        assert got == {"language": "spanish", "task": "translate"}

    def test_read_absent_is_empty(self, tmp_path):
        from soup_cli.trainer.asr import read_asr_sidecar

        assert read_asr_sidecar(str(tmp_path)) == {}

    def test_read_malformed_json_is_empty(self, tmp_path):
        from soup_cli.trainer.asr import _ASR_SIDECAR, read_asr_sidecar

        (tmp_path / _ASR_SIDECAR).write_text("{not json", encoding="utf-8")
        assert read_asr_sidecar(str(tmp_path)) == {}

    def test_read_non_dict_is_empty(self, tmp_path):
        from soup_cli.trainer.asr import _ASR_SIDECAR, read_asr_sidecar

        (tmp_path / _ASR_SIDECAR).write_text("[1, 2, 3]", encoding="utf-8")
        assert read_asr_sidecar(str(tmp_path)) == {}


class TestAsrUnwrapModel:
    def test_unwraps_peft(self):
        from soup_cli.trainer.asr import AsrTrainerWrapper

        class _Base:
            pass

        base = _Base()

        class _Peft:
            def get_base_model(self):
                return base

        w = object.__new__(AsrTrainerWrapper)
        w.model = _Peft()
        assert w._unwrapped_model() is base

    def test_returns_plain_model(self):
        from soup_cli.trainer.asr import AsrTrainerWrapper

        plain = object()
        w = object.__new__(AsrTrainerWrapper)
        w.model = plain
        assert w._unwrapped_model() is plain


class TestAsrTrainSidecar:
    def _fake_wrapper(self, tmp_path, *, prefix_customized, language, task):
        from types import SimpleNamespace

        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.asr import AsrTrainerWrapper

        yaml = _asr_yaml(asr_language=language, asr_task=task)
        cfg = load_config_from_string(yaml)
        w = object.__new__(AsrTrainerWrapper)
        w.config = cfg
        w._output_dir = str(tmp_path)
        w._prefix_customized = prefix_customized
        saved = {"model": False, "proc": False}

        w.trainer = SimpleNamespace(
            train=lambda resume_from_checkpoint=None: None,
            save_model=lambda d: saved.__setitem__("model", True),
            state=SimpleNamespace(log_history=[{"loss": 1.0}], global_step=1),
        )
        w.processor = SimpleNamespace(
            save_pretrained=lambda d: saved.__setitem__("proc", True)
        )
        return w

    def test_writes_sidecar_when_customized(self, tmp_path):
        from soup_cli.trainer.asr import read_asr_sidecar

        w = self._fake_wrapper(
            tmp_path, prefix_customized=True, language="es", task="translate"
        )
        w.train()
        assert read_asr_sidecar(str(tmp_path)) == {"language": "es", "task": "translate"}

    def test_no_sidecar_when_default(self, tmp_path):
        import os

        from soup_cli.trainer.asr import _ASR_SIDECAR

        w = self._fake_wrapper(
            tmp_path, prefix_customized=False, language=None, task=None
        )
        w.train()
        assert not os.path.exists(os.path.join(str(tmp_path), _ASR_SIDECAR))


class TestAsrLoraGate:
    def test_default_is_full_ft(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.asr import AsrTrainerWrapper

        cfg = load_config_from_string(_asr_yaml())
        # Bare task: asr with the schema-default lora block must NOT enable LoRA.
        assert AsrTrainerWrapper._should_use_lora(None, cfg.training) is False

    def test_opt_in_enables_lora(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.asr import AsrTrainerWrapper

        yaml = _asr_yaml() + "  asr_lora: true\n  lora:\n    r: 8\n"
        cfg = load_config_from_string(yaml)
        assert cfg.training.asr_lora is True
        assert AsrTrainerWrapper._should_use_lora(None, cfg.training) is True

    def test_opt_in_but_zero_rank_stays_off(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.asr import AsrTrainerWrapper

        yaml = _asr_yaml() + "  asr_lora: true\n  lora:\n    r: 0\n"
        cfg = load_config_from_string(yaml)
        assert AsrTrainerWrapper._should_use_lora(None, cfg.training) is False

    def test_asr_lora_footgun_on_non_asr(self):
        from soup_cli.config.loader import load_config_from_string

        yaml = (
            "base: HuggingFaceTB/SmolLM2-135M-Instruct\n"
            "task: sft\n"
            "data:\n  train: ./d.jsonl\n  format: alpaca\n"
            "training:\n  epochs: 1\n  lr: 1e-4\n  batch_size: 4\n  asr_lora: true\n"
        )
        with pytest.raises(ValueError, match="asr_lora"):
            load_config_from_string(yaml)


class TestAsrDataFormat:
    def test_is_audio_format_includes_asr(self):
        from soup_cli.data.formats import is_audio_format

        assert is_audio_format("asr")

    def test_format_to_messages_asr_passthrough(self):
        from soup_cli.data.formats import format_to_messages

        row = format_to_messages({"audio": "a.wav", "text": "hi there"}, "asr")
        assert row == {"audio": "a.wav", "text": "hi there"}

    def test_format_to_messages_asr_missing_text(self):
        from soup_cli.data.formats import format_to_messages

        assert format_to_messages({"audio": "a.wav"}, "asr") is None

    def test_detect_format_asr_not_plaintext(self):
        # {"audio","text"} must auto-detect as asr, NOT plaintext (which would
        # silently drop the audio path). Regression for the python-review HIGH.
        from soup_cli.data.formats import detect_format

        assert detect_format([{"audio": "clip.wav", "text": "hello"}]) == "asr"

    def test_convert_asr_delegates_to_validate(self):
        # _convert_asr reuses the trainer's canonical validator (single source
        # of truth) — a missing text still yields None via the wrapper.
        from soup_cli.data.formats import format_to_messages

        assert format_to_messages({"audio": "a.wav", "text": 5}, "asr") is None


class TestAsrRouting:
    def test_train_routes_asr(self):
        import inspect

        import soup_cli.commands.train as train_mod

        src = inspect.getsource(train_mod)
        assert 'cfg.task == "asr"' in src
        assert "AsrTrainerWrapper" in src


class TestAsrTrainerSetup:
    @pytest.mark.skipif(not _TORCH, reason="needs torch + transformers")
    def test_setup_builds_seq2seq_trainer(self, tmp_path):
        pytest.importorskip("soundfile")
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.trainer.asr import AsrTrainerWrapper

        clip = tmp_path / "clip0.wav"
        _write_sine_wav(clip)
        dataset = {"train": [{"audio": str(clip), "text": "hello world"}]}
        yaml = (
            f"base: {_TINY_WHISPER}\n"
            "task: asr\n"
            "backend: transformers\n"
            f"output: {str(tmp_path / 'out').replace(chr(92), '/')}\n"
            "data:\n"
            "  train: ./data/train.jsonl\n"
            "  format: asr\n"
            "training:\n"
            "  epochs: 1\n"
            "  lr: 1e-4\n"
            "  batch_size: 2\n"
        )
        cfg = load_config_from_string(yaml)
        wrapper = AsrTrainerWrapper(cfg, device="cpu")
        try:
            wrapper.setup(dataset)
        except (OSError, ImportError) as exc:  # network / hub unavailable
            pytest.skip(f"tiny whisper unavailable: {exc}")
        except ValueError as exc:  # torch<2.6 .bin-load restriction (dev env)
            if "torch" in str(exc).lower() or "safetensors" in str(exc).lower():
                pytest.skip(f"tiny whisper not loadable in this env: {exc}")
            raise
        assert wrapper.trainer is not None
        # Seq2SeqTrainer with predict_with_generate wired.
        assert wrapper.trainer.args.predict_with_generate is True


# ---------------------------------------------------------------------------
# Task 4 — soup infer --task asr
# ---------------------------------------------------------------------------


class TestAsrInfer:
    def test_help_shows_task_option(self):
        import re

        from typer.testing import CliRunner

        from soup_cli.cli import app

        # Wide COLUMNS + ANSI-strip: Rich splits flag names with color codes and
        # wraps at a narrow CI terminal, so a raw substring check is flaky
        # (v0.71.26 precedent).
        result = CliRunner().invoke(app, ["infer", "--help"], env={"COLUMNS": "200"})
        assert result.exit_code == 0, result.output
        cleaned = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--task" in cleaned

    def test_infer_asr_writes_transcriptions_and_wer(self, tmp_path, monkeypatch):
        import json

        import soup_cli.commands.infer as infer_mod

        # Test seam: skip the real Whisper load, return a fixed hypothesis.
        monkeypatch.setattr(
            infer_mod, "_ASR_TRANSCRIBER_OVERRIDE", lambda audio_path: "hello world"
        )

        in_path = tmp_path / "in.jsonl"
        in_path.write_text(
            json.dumps({"audio": "clip.wav", "text": "hello world"}) + "\n"
            + json.dumps({"audio": "clip2.wav", "text": "goodbye"}) + "\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "out.jsonl"
        # Run under cwd so the output-containment check passes.
        monkeypatch.chdir(tmp_path)
        infer_mod._infer_asr(
            model="openai/whisper-tiny",
            base=None,
            input_file="in.jsonl",
            device="cpu",
            output_file="out.jsonl",
            max_tokens=64,
            trust_remote_code=False,
        )
        lines = [json.loads(x) for x in out_path.read_text(encoding="utf-8").splitlines()]
        assert lines[0]["transcription"] == "hello world"
        assert lines[0]["wer"] == 0.0
        # ref "goodbye" (1 word) vs hyp "hello world" -> 2 edits / 1 = 2.0
        assert lines[1]["wer"] == 2.0

    def test_read_asr_rows_rejects_missing_audio(self, tmp_path):
        import json

        import soup_cli.commands.infer as infer_mod

        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps({"text": "no audio"}) + "\n", encoding="utf-8")
        rows = infer_mod._read_asr_rows(p)
        # rows with no 'audio' are dropped -> empty
        assert rows == []

    def test_resolve_asr_audio_rejects_traversal(self, tmp_path):
        import soup_cli.commands.infer as infer_mod

        with pytest.raises(ValueError, match="under"):
            infer_mod._resolve_asr_audio("../../etc/passwd", tmp_path)

    def test_resolve_asr_audio_rejects_unc(self, tmp_path):
        import soup_cli.commands.infer as infer_mod

        with pytest.raises(ValueError, match="UNC"):
            infer_mod._resolve_asr_audio("\\\\attacker\\share\\x.wav", tmp_path)

    def test_resolve_asr_audio_allows_contained(self, tmp_path):
        import soup_cli.commands.infer as infer_mod

        got = infer_mod._resolve_asr_audio("clip.wav", tmp_path)
        assert got == str(tmp_path / "clip.wav")

    def test_infer_asr_skips_traversal_row(self, tmp_path, monkeypatch):
        import json

        import soup_cli.commands.infer as infer_mod

        monkeypatch.setattr(
            infer_mod, "_ASR_TRANSCRIBER_OVERRIDE", lambda audio_path: "hi"
        )
        in_path = tmp_path / "in.jsonl"
        in_path.write_text(
            json.dumps({"audio": "../evil.wav", "text": "x"}) + "\n"
            + json.dumps({"audio": "ok.wav", "text": "hi"}) + "\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        infer_mod._infer_asr(
            model="openai/whisper-tiny", base=None, input_file="in.jsonl",
            device="cpu", output_file="out.jsonl", max_tokens=8,
            trust_remote_code=False,
        )
        lines = [json.loads(x) for x in (tmp_path / "out.jsonl").read_text().splitlines()]
        # traversal row dropped, only the contained one survives
        assert len(lines) == 1
        assert lines[0]["audio"] == "ok.wav"


class TestAsrMetricsDoS:
    def test_raw_char_cap(self):
        from soup_cli.utils.asr_metrics import _MAX_RAW_CHARS, wer

        huge = "a" * (_MAX_RAW_CHARS + 1)
        with pytest.raises(ValueError, match="too long"):
            wer(huge, "b")


class TestAsrSidecarValidation:
    def test_read_sidecar_drops_hostile_values(self, tmp_path):
        import json

        from soup_cli.trainer.asr import _ASR_SIDECAR, read_asr_sidecar

        (tmp_path / _ASR_SIDECAR).write_text(
            json.dumps({"language": "x" * 100, "task": "rm -rf"}), encoding="utf-8"
        )
        # oversize language + invalid task both dropped
        assert read_asr_sidecar(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# Task 5 — recipes (138 -> 142)
# ---------------------------------------------------------------------------


class TestAsrRecipes:
    def test_new_recipes_resolve(self):
        from soup_cli.recipes.catalog import get_recipe

        for name in (
            "whisper-tiny-asr",
            "whisper-base-asr",
            "whisper-large-v3-asr",
            "smolvlm-256m-sft",
        ):
            assert get_recipe(name) is not None, name

    def test_whisper_recipes_are_asr(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        for name in ("whisper-tiny-asr", "whisper-base-asr", "whisper-large-v3-asr"):
            recipe = get_recipe(name)
            cfg = load_config_from_string(recipe.yaml_str)
            assert cfg.task == "asr", name
            assert cfg.data.format == "asr", name

    def test_smolvlm_recipe_is_vision_sft(self):
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.recipes.catalog import get_recipe

        recipe = get_recipe("smolvlm-256m-sft")
        cfg = load_config_from_string(recipe.yaml_str)
        assert cfg.task == "sft"
        assert cfg.modality == "vision"

    def test_catalog_size_is_142(self):
        from soup_cli.recipes.catalog import RECIPES

        assert len(RECIPES) == 142
