"""Load-bearing contracts for the #1112 Llasa/XCodec2 review fixes."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import get_args

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_transformers_xcodec2_feature_extractor_contract_is_networkless(monkeypatch):
    pytest.importorskip("torchaudio")
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    from transformers import Xcodec2FeatureExtractor, Xcodec2Model

    from soup_cli.utils import tts_codec

    extractor = Xcodec2FeatureExtractor()
    audio = np.zeros(1280, dtype=np.float32)
    seen = {}

    class SpyModel:
        def encode(self, **inputs):
            encode_params = set(inspect.signature(Xcodec2Model.encode).parameters)
            encode_params.discard("self")
            assert set(inputs) <= encode_params
            seen["keys"] = set(inputs)
            return type("Encoded", (), {"audio_codes": torch.tensor([[1, 2]])})()

    monkeypatch.setattr(tts_codec, "load_audio_mono", lambda *_a, **_k: audio)
    rendered = tts_codec.encode_audio_llasa(
        "networkless.wav",
        xcodec2_model=SpyModel(),
        feature_extractor=extractor,
    )

    assert seen["keys"]
    assert rendered.startswith("<|SPEECH_GENERATION_START|>")
    assert rendered.endswith("<|SPEECH_GENERATION_END|>")


def test_tts_docs_preencoded_example_loads_through_real_schema():
    from soup_cli.config.loader import load_config_from_string

    text = (ROOT / "docs" / "training.md").read_text(encoding="utf-8")
    marker = "**Pre-encoded chat (live for codec-string families).**"
    start = text.index(marker)
    fence = text.index("```yaml", start) + len("```yaml")
    end = text.index("```", fence)
    yaml = text[fence:end].strip()

    cfg = load_config_from_string(yaml)
    assert cfg.task == "tts"
    assert cfg.data.format == "chatml"
    assert cfg.training.lora.r == 16
    assert cfg.training.lora.alpha == 32


def test_every_source_data_format_literal_is_schema_valid():
    from soup_cli.config.schema import DataConfig

    allowed = set(get_args(DataConfig.model_fields["format"].annotation))
    pattern = re.compile(
        r"data\.format\s*(?:=|:)\s*[\"\'`]?([a-z][a-z0-9_-]*)",
        re.IGNORECASE,
    )
    found: dict[str, set[str]] = {}

    for path in (ROOT / "src" / "soup_cli").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        values = {match.group(1) for match in pattern.finditer(source)}
        if values:
            found[str(path.relative_to(ROOT))] = values

    assert found, "ratchet scanned no data.format literals"
    invalid = {
        value
        for values in found.values()
        for value in values
        if value not in allowed
    }
    assert not invalid, f"source names refused data.format values: {sorted(invalid)}"


def test_clear_xcodec2_cache_releases_the_requested_device(monkeypatch):
    from soup_cli.utils import tts_codec

    marker = object()
    monkeypatch.setattr(tts_codec, "_XCODEC2_CACHE", {"cpu": marker, "cuda": object()})

    tts_codec.clear_xcodec2_cache("cpu")

    assert "cpu" not in tts_codec._XCODEC2_CACHE
    assert "cuda" in tts_codec._XCODEC2_CACHE


def test_llasa_setup_clears_exact_encode_device_before_base_setup(monkeypatch):
    from types import SimpleNamespace

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.trainer.tts import TTSTrainerWrapper
    from soup_cli.utils import tts_codec

    wrapper = object.__new__(TTSTrainerWrapper)
    wrapper.config = SimpleNamespace(
        training=SimpleNamespace(tts_family="llasa", tts_emotion=None),
        data=SimpleNamespace(format="audio", new_special_tokens=None),
    )
    wrapper.device = "cuda"
    wrapper._tts_family = None

    monkeypatch.setattr(
        TTSTrainerWrapper, "_require_tts_codec", lambda self, family: None
    )

    def fake_encode(dataset, family, *, device=None, console=None):
        assert family == "llasa"
        assert device == "cuda"
        tts_codec._XCODEC2_CACHE["cuda"] = (object(), object())
        return dataset

    seen = {}

    def fake_super_setup(self, dataset):
        seen["cache"] = dict(tts_codec._XCODEC2_CACHE)

    monkeypatch.setattr(tts_codec, "_XCODEC2_CACHE", {})
    monkeypatch.setattr(tts_codec, "encode_tts_dataset", fake_encode)
    monkeypatch.setattr(SFTTrainerWrapper, "setup", fake_super_setup)

    wrapper.setup({"train": []})

    assert seen["cache"] == {}


def test_llasa_setup_clears_exact_encode_device_when_encoding_fails(monkeypatch):
    from types import SimpleNamespace

    from soup_cli.trainer.tts import TTSTrainerWrapper
    from soup_cli.utils import tts_codec

    wrapper = object.__new__(TTSTrainerWrapper)
    wrapper.config = SimpleNamespace(
        training=SimpleNamespace(tts_family="llasa", tts_emotion=None),
        data=SimpleNamespace(format="audio", new_special_tokens=None),
    )
    wrapper.device = "cuda"
    wrapper._tts_family = None

    monkeypatch.setattr(
        TTSTrainerWrapper, "_require_tts_codec", lambda self, family: None
    )

    def fake_encode(dataset, family, *, device=None, console=None):
        assert family == "llasa"
        assert device == "cuda"
        tts_codec._XCODEC2_CACHE["cuda"] = (object(), object())
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(tts_codec, "_XCODEC2_CACHE", {})
    monkeypatch.setattr(tts_codec, "encode_tts_dataset", fake_encode)

    with pytest.raises(RuntimeError, match="encoding failed"):
        wrapper.setup({"train": []})

    assert tts_codec._XCODEC2_CACHE == {}


def test_floor_ci_pins_matching_torch_and_torchaudio():
    constraints = (
        ROOT / ".github" / "constraints" / "transformers-floor.txt"
    ).read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "torch==2.6.0" in constraints
    assert "torchaudio==2.6.0" in constraints
    assert '"torchaudio"' in workflow


def test_tts_docs_never_recommend_refused_chat_format():
    text = (ROOT / "docs" / "training.md").read_text(encoding="utf-8")
    start = text.index("## TTS")
    end = text.index("## Classifier", start)
    section = text[start:end]
    assert not re.search(r"data\.format\s*(?::|=)\s*chat(?!ml)", section)
