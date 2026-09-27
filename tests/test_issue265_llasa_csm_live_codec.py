"""#265 slice 1: Llasa/XCodec2 live codec and honest CSM refusal."""

from __future__ import annotations

import math
import struct
import wave

import pytest


def _write_wav(path, *, sr=16_000, seconds=0.03):
    frames = bytearray()
    for i in range(int(sr * seconds)):
        value = int(8000 * math.sin(2 * math.pi * 220.0 * i / sr))
        frames += struct.pack("<h", value)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(bytes(frames))


class _FakeExtractor:
    def __call__(self, *, audio, sampling_rate, return_tensors):
        import torch

        assert sampling_rate == 16_000
        assert return_tensors == "pt"
        return {
            "input_values": torch.as_tensor(audio).reshape(1, 1, -1),
            "input_features": torch.zeros(1, 2, 2),
        }


class _FakeXCodec2:
    def encode(self, **inputs):
        from types import SimpleNamespace

        import torch

        device = inputs["input_values"].device
        return SimpleNamespace(
            audio_codes=torch.tensor([[[7, 42, 65535]]], device=device)
        )


class TestLlasaRendering:
    def test_renders_native_speech_token_envelope(self):
        from soup_cli.utils.tts_codec import llasa_codes_to_string

        assert llasa_codes_to_string([7, 42]) == (
            "<|SPEECH_GENERATION_START|>"
            "<|s_7|><|s_42|>"
            "<|SPEECH_GENERATION_END|>"
        )

    @pytest.mark.parametrize("bad", [[True], [-1], [65536], [1.5], []])
    def test_rejects_invalid_codes(self, bad):
        from soup_cli.utils.tts_codec import llasa_codes_to_string

        with pytest.raises((TypeError, ValueError)):
            llasa_codes_to_string(bad)

    def test_audio_encode_uses_16khz_xcodec2_and_renders(self, tmp_path):
        pytest.importorskip("soundfile")
        pytest.importorskip("torch")
        from soup_cli.utils.tts_codec import encode_audio_llasa

        wav = tmp_path / "a.wav"
        _write_wav(wav, sr=8_000)
        out = encode_audio_llasa(
            str(wav),
            xcodec2_model=_FakeXCodec2(),
            feature_extractor=_FakeExtractor(),
        )
        assert out == (
            "<|SPEECH_GENERATION_START|>"
            "<|s_7|><|s_42|><|s_65535|>"
            "<|SPEECH_GENERATION_END|>"
        )


class TestDispatch:
    def test_llasa_is_live(self):
        from soup_cli.utils.tts_codec import tts_encoder_for_family

        assert callable(tts_encoder_for_family("llasa"))

    def test_csm_refuses_codec_string_path_for_the_right_reason(self):
        from soup_cli.utils.tts_codec import tts_encoder_for_family

        with pytest.raises(RuntimeError, match="32 Mimi codebooks"):
            tts_encoder_for_family("sesame_csm")

    def test_missing_native_xcodec2_names_the_transformers_floor(self, monkeypatch):
        import sys
        from types import ModuleType

        from soup_cli.utils import tts_codec

        fake_transformers = ModuleType("transformers")
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setattr(tts_codec, "_XCODEC2_CACHE", {})
        with pytest.raises(RuntimeError, match="transformers>=5.16.1"):
            tts_codec._get_xcodec2_components()

    def test_component_loader_builds_extractor_before_model(self, monkeypatch):
        import sys
        from types import ModuleType

        from soup_cli.utils import tts_codec

        events = []

        class FakeExtractor:
            @classmethod
            def from_pretrained(cls, _model_id):
                events.append("extractor")
                return cls()

        class FakeModel:
            @classmethod
            def from_pretrained(cls, _model_id):
                assert events == ["extractor"]
                events.append("model")
                return cls()

            def eval(self):
                return self

            def to(self, _device):
                return self

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoFeatureExtractor = FakeExtractor
        fake_transformers.Xcodec2Model = FakeModel
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setattr(tts_codec, "_XCODEC2_CACHE", {})

        tts_codec._get_xcodec2_components(device="cpu")
        assert events == ["extractor", "model"]

    @pytest.mark.parametrize("error", [ImportError("missing"), OSError("libcudart mismatch")])
    def test_llasa_gate_names_audio_extra_when_torchaudio_cannot_load(
        self, monkeypatch, error
    ):
        import builtins

        from soup_cli.trainer.tts import TTSTrainerWrapper

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torchaudio":
                raise error
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        wrapper = object.__new__(TTSTrainerWrapper)
        with pytest.raises(RuntimeError, match=r"soup-cli\[audio\].*torch"):
            wrapper._require_tts_codec("llasa")

    def test_injected_model_and_extractor_must_arrive_together(self, tmp_path):
        pytest.importorskip("soundfile")
        from soup_cli.utils.tts_codec import encode_audio_llasa

        wav = tmp_path / "a.wav"
        _write_wav(wav)
        with pytest.raises(ValueError, match="provided together"):
            encode_audio_llasa(str(wav), xcodec2_model=_FakeXCodec2())


class TestTrainerContract:
    def test_csm_refuses_before_dependency_probe(self):
        from soup_cli.trainer.tts import TTSTrainerWrapper

        wrapper = object.__new__(TTSTrainerWrapper)
        with pytest.raises(RuntimeError, match="parallel multimodal frames"):
            wrapper._require_tts_codec("sesame_csm")


class TestCsmParseTimeRefusal:
    @pytest.mark.parametrize("data_format", ["audio", "chatml"])
    def test_csm_is_refused_at_config_parse_for_both_input_shapes(self, data_format):
        from soup_cli.config.loader import load_config_from_string

        yaml = f"""base: sesame/csm-1b
task: tts
modality: audio_out
data:
  train: ./tts.jsonl
  format: {data_format}
training:
  tts_family: sesame_csm
"""
        with pytest.raises(ValueError, match="dedicated CSM trainer"):
            load_config_from_string(yaml)

    def test_unrunnable_csm_recipe_is_not_advertised(self):
        from soup_cli.recipes.catalog import RECIPES

        assert "sesame-csm-tts" not in RECIPES
