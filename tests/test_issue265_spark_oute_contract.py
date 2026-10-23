"""#265 slice 2: fail closed on incompatible Spark/Oute live-codec installs."""

import pytest


@pytest.mark.parametrize(
    "family,needle",
    [
        ("spark", "no installable 'sparktts' PyPI package"),
        ("oute", "transformers==4.52.3"),
    ],
)
def test_encoder_dispatch_names_the_real_upstream_blocker(family, needle):
    from soup_cli.utils.tts_codec import tts_encoder_for_family

    with pytest.raises(RuntimeError, match=needle):
        tts_encoder_for_family(family)


@pytest.mark.parametrize("family", ["spark", "oute"])
def test_trainer_refuses_before_import_probe(family):
    from soup_cli.trainer.tts import TTSTrainerWrapper

    wrapper = object.__new__(TTSTrainerWrapper)
    with pytest.raises(RuntimeError, match="Refusing|refus|raw-audio"):
        wrapper._require_tts_codec(family)


@pytest.mark.parametrize("family", ["spark", "oute"])
def test_raw_audio_is_refused_at_config_parse_time(family):
    from soup_cli.config.loader import load_config_from_string

    yaml = f"""base: test/model
task: tts
modality: audio_out
data:
  train: ./tts.jsonl
  format: audio
  audio_dir: ./audio
training:
  tts_family: {family}
"""
    with pytest.raises(ValueError, match="Spark|Oute|spark|oute"):
        load_config_from_string(yaml)


def test_oute_recipe_uses_the_working_preencoded_path():
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.recipes.catalog import RECIPES

    cfg = load_config_from_string(RECIPES["oute-tts"].yaml_str)
    assert cfg.training.tts_family == "oute"
    assert cfg.data.format == "chatml"


def test_unrunnable_spark_recipe_is_not_advertised():
    from soup_cli.recipes.catalog import RECIPES

    assert "spark-tts" not in RECIPES


def test_spark_message_does_not_recommend_nonexistent_pip_package():
    from soup_cli.utils.tts_codec import incompatible_live_codec_error

    message = str(incompatible_live_codec_error("spark"))
    assert "pip install sparktts" not in message
    assert "torch==2.5.1" in message
    assert "transformers==4.46.2" in message
    assert "data.format=chatml" in message


def test_oute_message_does_not_recommend_resolver_breaking_install():
    from soup_cli.utils.tts_codec import incompatible_live_codec_error

    message = str(incompatible_live_codec_error("oute"))
    assert "pip install outetts" not in message
    assert "transcript/word alignment" in message
    assert "transformers>=5.16.1" in message
    assert "data.format=chatml" in message


def test_incompatibility_helper_always_returns_runtime_error():
    from soup_cli.utils.tts_codec import incompatible_live_codec_error

    assert isinstance(incompatible_live_codec_error("orpheus"), RuntimeError)


def test_tts_docs_list_only_runnable_codec_string_recipes():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "training.md").read_text(encoding="utf-8")
    start = text.index("## TTS")
    end = text.index("## Classifier", start)
    section = text[start:end]

    assert "Three ready-made codec-string recipes ship" in section
    assert "spark-tts" not in section
    for recipe in ("orpheus-tts-sft", "llasa-tts", "oute-tts"):
        assert recipe in section


def test_command_reference_keeps_emotion_templating():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    lines = (root / "docs" / "commands.md").read_text(encoding="utf-8").splitlines()
    row = next(line for line in lines if "task='tts'" in line)
    assert "emotion templating" in row
    assert "LIVE (v0.71.20)" in row
