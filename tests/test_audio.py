"""Tests for audio modality — config, data format, template, routing."""


import pytest
from pydantic import ValidationError

from soup_cli.config.schema import TEMPLATES, SoupConfig

# ─── Config Tests ───────────────────────────────────────────────────────────


class TestAudioConfig:
    """Test audio modality config validation."""

    def test_audio_modality_accepted(self):
        """audio should be a valid modality."""
        cfg = SoupConfig(
            base="Qwen/Qwen2-Audio-7B-Instruct",
            task="sft",
            modality="audio",
            data={"train": "./data.jsonl"},
        )
        assert cfg.modality == "audio"

    def test_audio_modality_with_audio_format(self):
        """audio modality with audio format should validate."""
        cfg = SoupConfig(
            base="Qwen/Qwen2-Audio-7B-Instruct",
            task="sft",
            modality="audio",
            data={"train": "./data.jsonl", "format": "audio"},
        )
        assert cfg.data.format == "audio"

    def test_audio_format_accepted(self):
        """audio should be a valid data format."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl", "format": "audio"},
        )
        assert cfg.data.format == "audio"

    def test_audio_dir_field(self):
        """DataConfig should support audio_dir field."""
        cfg = SoupConfig(
            base="some-model",
            data={
                "train": "./data.jsonl",
                "format": "audio",
                "audio_dir": "./data/audio",
            },
        )
        assert cfg.data.audio_dir == "./data/audio"

    def test_audio_dir_default_none(self):
        """audio_dir should default to None."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl"},
        )
        assert cfg.data.audio_dir is None

    def test_invalid_modality_rejected(self):
        """Invalid modality should be rejected."""
        with pytest.raises(ValidationError):
            SoupConfig(
                base="some-model",
                modality="video",
                data={"train": "./data.jsonl"},
            )

    def test_audio_full_config(self):
        """Full audio config should validate correctly."""
        cfg = SoupConfig(
            base="Qwen/Qwen2-Audio-7B-Instruct",
            task="sft",
            modality="audio",
            data={
                "train": "./data/audio_train.jsonl",
                "format": "audio",
                "audio_dir": "./data/audio",
                "max_length": 2048,
            },
            training={
                "epochs": 3,
                "lr": 1e-5,
                "quantization": "4bit",
            },
        )
        assert cfg.modality == "audio"
        assert cfg.data.format == "audio"
        assert cfg.data.audio_dir == "./data/audio"


# ─── Audio Data Format Tests ────────────────────────────────────────────


class TestAudioDataFormat:
    """Test audio data format detection and conversion."""

    def test_format_signature_exists(self):
        """audio format signature should be registered."""
        from soup_cli.data.formats import FORMAT_SIGNATURES

        assert "audio" in FORMAT_SIGNATURES
        assert FORMAT_SIGNATURES["audio"] == {"audio", "messages"}

    def test_detect_audio_format(self):
        """Should auto-detect audio format from audio+messages keys."""
        from soup_cli.data.formats import detect_format

        data = [{
            "audio": "test.wav",
            "messages": [
                {"role": "user", "content": "Transcribe this."},
                {"role": "assistant", "content": "Hello world."},
            ],
        }]
        assert detect_format(data) == "audio"

    def test_convert_audio_format(self):
        """Should convert audio row correctly."""
        from soup_cli.data.formats import format_to_messages

        row = {
            "audio": "test.wav",
            "messages": [
                {"role": "user", "content": "Transcribe."},
                {"role": "assistant", "content": "Hello."},
            ],
        }
        result = format_to_messages(row, "audio")
        assert result["audio"] == "test.wav"
        assert len(result["messages"]) == 2

    def test_convert_audio_empty_audio_returns_none(self):
        """Empty audio path should cause conversion to return None."""
        from soup_cli.data.formats import format_to_messages

        row = {
            "audio": "",
            "messages": [{"role": "user", "content": "Transcribe."}],
        }
        result = format_to_messages(row, "audio")
        assert result is None

    def test_convert_audio_missing_messages_returns_none(self):
        """Missing messages should cause conversion to return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"audio": "test.wav"}
        result = format_to_messages(row, "audio")
        assert result is None

    def test_convert_audio_empty_messages_returns_none(self):
        """Empty messages list should cause conversion to return None."""
        from soup_cli.data.formats import format_to_messages

        row = {"audio": "test.wav", "messages": []}
        result = format_to_messages(row, "audio")
        assert result is None

    def test_audio_not_confused_with_chatml(self):
        """Audio data (audio+messages) should not be detected as chatml."""
        from soup_cli.data.formats import detect_format

        data = [{
            "audio": "test.wav",
            "messages": [{"role": "user", "content": "test"}],
        }]
        assert detect_format(data) == "audio"

    def test_is_audio_format(self):
        """is_audio_format should correctly identify audio format."""
        from soup_cli.data.formats import is_audio_format

        assert is_audio_format("audio") is True
        assert is_audio_format("chatml") is False
        assert is_audio_format("llava") is False


# ─── Template Tests ──────────────────────────────────────────────────────


class TestAudioTemplate:
    """Test the audio template."""

    def test_audio_template_exists(self):
        assert "audio" in TEMPLATES

    def test_audio_template_valid_yaml(self):
        import yaml

        config = yaml.safe_load(TEMPLATES["audio"])
        assert config["task"] == "sft"
        assert config["modality"] == "audio"
        assert config["data"]["format"] == "audio"

    def test_audio_template_valid_config(self):
        import yaml

        raw = yaml.safe_load(TEMPLATES["audio"])
        cfg = SoupConfig(**raw)
        assert cfg.modality == "audio"
        assert cfg.data.format == "audio"
        assert cfg.data.audio_dir == "./data/audio"


# ─── Audio Loader Tests ─────────────────────────────────────────────────


class TestAudioLoader:
    """Test audio file validation in data loader."""

    def test_validate_audio_files_resolves_paths(self, tmp_path):
        """_validate_audio_files should resolve relative paths."""
        from soup_cli.data.loader import _validate_audio_files

        data = [
            {"audio": "test.wav", "messages": [{"role": "user", "content": "x"}]},
        ]
        result = _validate_audio_files(data, tmp_path)
        assert len(result) == 1
        assert str(tmp_path) in result[0]["audio"]

    def test_validate_audio_files_skips_missing(self):
        """_validate_audio_files should skip rows without audio path."""
        from pathlib import Path

        from soup_cli.data.loader import _validate_audio_files

        data = [
            {"audio": "", "messages": [{"role": "user", "content": "x"}]},
            {"messages": [{"role": "user", "content": "x"}]},
        ]
        result = _validate_audio_files(data, Path("."))
        assert len(result) == 0

    def test_validate_audio_files_keeps_absolute_paths(self, tmp_path):
        """Absolute audio paths should not be modified."""
        from soup_cli.data.loader import _validate_audio_files

        abs_path = str(tmp_path / "test.wav")
        data = [
            {"audio": abs_path, "messages": [{"role": "user", "content": "x"}]},
        ]
        result = _validate_audio_files(data, tmp_path)
        assert len(result) == 1
        assert result[0]["audio"] == abs_path


# ─── SFT Trainer Audio Setup Tests ──────────────────────────────────────


class TestAudioTrainerSetup:
    """Test that SFT trainer handles audio modality."""

    def test_sft_wrapper_accepts_audio_modality(self):
        """SFTTrainerWrapper should accept audio modality config."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="Qwen/Qwen2-Audio-7B-Instruct",
            task="sft",
            modality="audio",
            data={"train": "./data.jsonl", "format": "audio"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.modality == "audio"

    def test_audio_modality_triggers_audio_branch(self):
        """SFTTrainerWrapper with audio modality should have _setup_audio_transformers."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="Qwen/Qwen2-Audio-7B-Instruct",
            task="sft",
            modality="audio",
            data={"train": "./data.jsonl", "format": "audio"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cpu")
        assert hasattr(wrapper, "_setup_audio_transformers")
        assert hasattr(wrapper, "_prepare_audio_dataset")
        assert cfg.modality == "audio"


# ─── CLI Init Template Tests ──────────────────────────────────────────────


class TestAudioInitTemplate:
    """Test that soup init --template audio works."""

    def test_init_audio_template_creates_file(self, tmp_path):
        """soup init --template audio should write a file with audio modality."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        result = runner.invoke(
            app, ["init", "--template", "audio", "--output", str(output)]
        )
        assert result.exit_code == 0
        assert output.exists()
        content = output.read_text()
        assert "modality: audio" in content
        assert "format: audio" in content

    def test_init_audio_template_produces_valid_config(self, tmp_path):
        """The file written by soup init --template audio should parse."""
        from pathlib import Path

        from typer.testing import CliRunner

        from soup_cli.cli import app
        from soup_cli.config.loader import load_config

        runner = CliRunner()
        output = tmp_path / "soup.yaml"
        runner.invoke(
            app, ["init", "--template", "audio", "--output", str(output)]
        )
        cfg = load_config(Path(output))
        assert cfg.modality == "audio"
        assert cfg.data.format == "audio"


# ─── Config Loader Round-trip Tests ──────────────────────────────────────


class TestAudioConfigLoaderRoundTrip:
    """Test audio template YAML survives round-trip."""

    def test_audio_template_round_trip(self):
        """TEMPLATES['audio'] should parse via load_config_from_string."""
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(TEMPLATES["audio"])
        assert cfg.modality == "audio"
        assert cfg.data.format == "audio"

    def test_audio_custom_yaml_round_trip(self):
        """Custom audio YAML string should round-trip correctly."""
        from soup_cli.config.loader import load_config_from_string

        yaml_str = """
base: Qwen/Qwen2-Audio-7B-Instruct
task: sft
modality: audio

data:
  train: ./data/audio.jsonl
  format: audio
  audio_dir: ./data/wav
  max_length: 4096

training:
  epochs: 5
  lr: 1e-5
  quantization: 4bit

output: ./output_audio
"""
        cfg = load_config_from_string(yaml_str)
        assert cfg.modality == "audio"
        assert cfg.data.format == "audio"
        assert cfg.data.audio_dir == "./data/wav"
        assert cfg.output == "./output_audio"
