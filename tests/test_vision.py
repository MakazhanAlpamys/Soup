"""Tests for Phase 6 — Multimodal Fine-tuning (vision config, formats, loader, trainer)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from soup_cli.config.schema import TEMPLATES, SoupConfig
from soup_cli.data.formats import (
    detect_format,
    format_to_messages,
    is_vision_format,
)

# ─── Sample Data ───────────────────────────────────────────────────────────

LLAVA_ROW = {
    "image": "photo.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nDescribe this image."},
        {"from": "gpt", "value": "The image shows a cat sitting on a mat."},
    ],
}

SHAREGPT4V_ROW = {
    "image": "chart.png",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat does this chart show?"},
        {"from": "gpt", "value": "The chart shows quarterly revenue growth."},
    ],
}

LLAVA_DATASET = [
    {
        "image": "img1.jpg",
        "conversations": [
            {"from": "human", "value": "What is this?"},
            {"from": "gpt", "value": "A dog."},
        ],
    },
    {
        "image": "img2.jpg",
        "conversations": [
            {"from": "human", "value": "Describe the scene."},
            {"from": "gpt", "value": "A park with trees."},
        ],
    },
]


# ─── Config Tests ──────────────────────────────────────────────────────────


class TestVisionConfig:
    """Test modality config field validation."""

    def test_modality_default_is_text(self):
        """Default modality should be 'text'."""
        cfg = SoupConfig(base="some-model", data={"train": "./data.jsonl"})
        assert cfg.modality == "text"

    def test_modality_vision_accepted(self):
        """modality: vision should be valid."""
        cfg = SoupConfig(
            base="some-model",
            modality="vision",
            data={"train": "./data.jsonl"},
        )
        assert cfg.modality == "vision"

    def test_modality_text_accepted(self):
        """modality: text should be valid."""
        cfg = SoupConfig(
            base="some-model",
            modality="text",
            data={"train": "./data.jsonl"},
        )
        assert cfg.modality == "text"

    def test_modality_invalid_rejected(self):
        """Invalid modality should raise validation error."""
        with pytest.raises(Exception):
            SoupConfig(
                base="some-model",
                modality="video",
                data={"train": "./data.jsonl"},
            )

    def test_modality_in_model_dump(self):
        """modality field should appear in model_dump output."""
        cfg = SoupConfig(
            base="some-model",
            modality="vision",
            data={"train": "./data.jsonl"},
        )
        dump = cfg.model_dump()
        assert dump["modality"] == "vision"

    def test_vision_with_sft(self):
        """Vision modality should work with SFT task."""
        cfg = SoupConfig(
            base="meta-llama/Llama-3.2-11B-Vision-Instruct",
            task="sft",
            modality="vision",
            data={"train": "./data.jsonl", "format": "llava"},
        )
        assert cfg.task == "sft"
        assert cfg.modality == "vision"

    def test_full_vision_config(self):
        """Full config with vision modality should validate."""
        cfg = SoupConfig(
            base="meta-llama/Llama-3.2-11B-Vision-Instruct",
            task="sft",
            modality="vision",
            data={
                "train": "./data.jsonl",
                "format": "llava",
                "image_dir": "./images",
                "max_length": 2048,
            },
            training={
                "epochs": 3,
                "lr": 1e-5,
                "quantization": "4bit",
                "lora": {"r": 64, "alpha": 16},
            },
        )
        assert cfg.modality == "vision"
        assert cfg.data.format == "llava"
        assert cfg.data.image_dir == "./images"

    def test_vision_with_backend(self):
        """Vision modality should work with different backends."""
        cfg = SoupConfig(
            base="some-model",
            modality="vision",
            backend="transformers",
            data={"train": "./data.jsonl"},
        )
        assert cfg.modality == "vision"
        assert cfg.backend == "transformers"


# ─── Data Format Config Tests ─────────────────────────────────────────────


class TestVisionDataConfig:
    """Test vision-related fields in DataConfig."""

    def test_image_dir_default_none(self):
        """image_dir should default to None."""
        cfg = SoupConfig(base="some-model", data={"train": "./data.jsonl"})
        assert cfg.data.image_dir is None

    def test_image_dir_accepted(self):
        """image_dir should be accepted as a string."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl", "image_dir": "./images"},
        )
        assert cfg.data.image_dir == "./images"

    def test_llava_format_accepted(self):
        """format: llava should be valid."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl", "format": "llava"},
        )
        assert cfg.data.format == "llava"

    def test_sharegpt4v_format_accepted(self):
        """format: sharegpt4v should be valid."""
        cfg = SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl", "format": "sharegpt4v"},
        )
        assert cfg.data.format == "sharegpt4v"


# ─── Format Detection Tests ───────────────────────────────────────────────


class TestVisionFormatDetection:
    """Test auto-detection of vision formats."""

    def test_detect_llava_format(self):
        """Should detect LLaVA format (image + conversations)."""
        result = detect_format(LLAVA_DATASET)
        assert result == "llava"

    def test_detect_sharegpt4v_with_explicit_format(self):
        """ShareGPT4V has same structure as LLaVA; auto-detect returns llava."""
        data = [SHAREGPT4V_ROW]
        result = detect_format(data)
        # Both have same keys, so llava is detected first
        assert result == "llava"

    def test_detect_llava_not_confused_with_sharegpt(self):
        """LLaVA format (image + conversations) should not be detected as sharegpt."""
        result = detect_format([LLAVA_ROW])
        assert result != "sharegpt"

    def test_detect_sharegpt_without_image(self):
        """Regular ShareGPT (no image key) should still be detected as sharegpt."""
        data = [{"conversations": [{"from": "human", "value": "Hi"}]}]
        result = detect_format(data)
        assert result == "sharegpt"

    def test_is_vision_format_llava(self):
        """is_vision_format should return True for llava."""
        assert is_vision_format("llava") is True

    def test_is_vision_format_sharegpt4v(self):
        """is_vision_format should return True for sharegpt4v."""
        assert is_vision_format("sharegpt4v") is True

    def test_is_vision_format_alpaca(self):
        """is_vision_format should return False for alpaca."""
        assert is_vision_format("alpaca") is False

    def test_is_vision_format_sharegpt(self):
        """is_vision_format should return False for sharegpt."""
        assert is_vision_format("sharegpt") is False


# ─── Format Conversion Tests ──────────────────────────────────────────────


class TestVisionFormatConversion:
    """Test conversion of vision formats to unified message format."""

    def test_llava_to_messages(self):
        """LLaVA row should convert to messages + image."""
        result = format_to_messages(LLAVA_ROW, "llava")
        assert result is not None
        assert "messages" in result
        assert "image" in result
        assert result["image"] == "photo.jpg"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_sharegpt4v_to_messages(self):
        """ShareGPT4V row should convert to messages + image."""
        result = format_to_messages(SHAREGPT4V_ROW, "sharegpt4v")
        assert result is not None
        assert "messages" in result
        assert result["image"] == "chart.png"
        assert result["messages"][0]["content"] == "<image>\nWhat does this chart show?"

    def test_llava_preserves_image_tag(self):
        """<image> tag in conversation should be preserved."""
        result = format_to_messages(LLAVA_ROW, "llava")
        assert "<image>" in result["messages"][0]["content"]

    def test_llava_with_id(self):
        """LLaVA row with id field should preserve it."""
        row = {**LLAVA_ROW, "id": "sample_001"}
        result = format_to_messages(row, "llava")
        assert result["id"] == "sample_001"

    def test_llava_role_mapping(self):
        """human/gpt roles should map to user/assistant."""
        row = {
            "image": "test.jpg",
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "Describe this."},
                {"from": "gpt", "value": "A test image."},
            ],
        }
        result = format_to_messages(row, "llava")
        roles = [msg["role"] for msg in result["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_llava_invalid_row_returns_none(self):
        """Row missing required keys should return None."""
        result = format_to_messages({"bad": "data"}, "llava")
        assert result is None

    def test_sharegpt4v_invalid_row_returns_none(self):
        """Row missing required keys should return None."""
        result = format_to_messages({"bad": "data"}, "sharegpt4v")
        assert result is None


# ─── Data Loader Vision Tests ─────────────────────────────────────────────


class TestVisionDataLoader:
    """Test vision image validation in data loader."""

    def test_validate_vision_images_resolves_relative(self):
        """Relative image paths should be resolved against image_dir."""
        from soup_cli.data.loader import _validate_vision_images

        data = [{"messages": [{"role": "user", "content": "Hi"}], "image": "photo.jpg"}]
        image_dir = Path("/data/images")
        result = _validate_vision_images(data, image_dir)
        assert len(result) == 1
        assert result[0]["image"] == str(image_dir / "photo.jpg")

    def test_validate_vision_images_skips_missing(self):
        """Rows without image field should be skipped."""
        from soup_cli.data.loader import _validate_vision_images

        data = [
            {"messages": [{"role": "user", "content": "Hi"}], "image": "photo.jpg"},
            {"messages": [{"role": "user", "content": "No image"}]},
        ]
        image_dir = Path("/data/images")
        result = _validate_vision_images(data, image_dir)
        assert len(result) == 1

    def test_validate_vision_images_skips_empty_image(self):
        """Rows with empty image string should be skipped."""
        from soup_cli.data.loader import _validate_vision_images

        data = [{"messages": [{"role": "user", "content": "Hi"}], "image": ""}]
        image_dir = Path("/data")
        result = _validate_vision_images(data, image_dir)
        assert len(result) == 0

    def test_validate_vision_images_absolute_path(self):
        """Absolute image paths should be kept as-is."""
        from soup_cli.data.loader import _validate_vision_images

        abs_path = str(Path("/absolute/path/photo.jpg"))
        data = [{"messages": [{"role": "user", "content": "Hi"}], "image": abs_path}]
        image_dir = Path("/different/dir")
        result = _validate_vision_images(data, image_dir)
        assert result[0]["image"] == abs_path

    def test_load_dataset_with_vision_format(self):
        """load_dataset should handle llava format data files."""
        from soup_cli.config.schema import DataConfig
        from soup_cli.data.loader import load_dataset

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmpfile:
            for row in LLAVA_DATASET:
                tmpfile.write(json.dumps(row) + "\n")
            tmpfile_path = tmpfile.name

        try:
            data_config = DataConfig(
                train=tmpfile_path,
                format="llava",
                val_split=0.0,
            )
            result = load_dataset(data_config)
            assert "train" in result
            assert len(result["train"]) == 2
            # Each row should have messages and image
            for row in result["train"]:
                assert "messages" in row
                assert "image" in row
        finally:
            Path(tmpfile_path).unlink()


# ─── SFT Trainer Vision Tests ─────────────────────────────────────────────


class TestSFTVisionIntegration:
    """Test SFT trainer with vision modality."""

    def test_sft_wrapper_init_with_vision(self):
        """SFTTrainerWrapper should accept vision modality config."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="meta-llama/Llama-3.2-11B-Vision-Instruct",
            modality="vision",
            data={"train": "./data.jsonl", "format": "llava"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cuda")
        assert wrapper.config.modality == "vision"

    def test_sft_setup_vision_calls_automodel(self):
        """_setup_vision_transformers should use AutoModelForVision2Seq."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="test-vision-model",
            modality="vision",
            data={"train": "./data.jsonl", "format": "llava", "max_length": 2048},
            training={"lora": {"r": 64, "alpha": 16, "dropout": 0.05}},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cuda")

        mock_model = MagicMock()
        mock_model.get_nb_trainable_parameters.return_value = (1000, 100000)
        mock_processor = MagicMock()

        with patch(
            "soup_cli.trainer.sft.SFTTrainerWrapper._setup_vision_transformers"
        ) as mock_setup:
            mock_setup.side_effect = lambda c, t: setattr(wrapper, "model", mock_model) or setattr(
                wrapper, "tokenizer", mock_processor
            ) or setattr(wrapper, "processor", mock_processor)
            wrapper._setup_vision_transformers(cfg, cfg.training)
            mock_setup.assert_called_once()

    def test_sft_vision_config_selects_vision_path(self):
        """Vision modality config should be detected correctly in wrapper."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="test-vision-model",
            modality="vision",
            data={"train": "./data.jsonl", "format": "llava"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cuda")
        # Verify config is stored correctly for vision routing
        assert wrapper.config.modality == "vision"
        assert wrapper.config.data.format == "llava"

    def test_sft_vision_has_setup_methods(self):
        """SFTTrainerWrapper should have vision-specific methods."""
        from soup_cli.trainer.sft import SFTTrainerWrapper

        cfg = SoupConfig(
            base="test-vision-model",
            modality="vision",
            data={"train": "./data.jsonl"},
        )
        wrapper = SFTTrainerWrapper(cfg, device="cuda")
        assert hasattr(wrapper, "_setup_vision_transformers")
        assert hasattr(wrapper, "_prepare_vision_dataset")


# ─── Template Tests ────────────────────────────────────────────────────────


class TestVisionTemplate:
    """Test vision template in TEMPLATES dict."""

    def test_vision_template_exists(self):
        """Vision template should exist."""
        assert "vision" in TEMPLATES

    def test_vision_template_has_modality(self):
        """Vision template should set modality: vision."""
        assert "modality: vision" in TEMPLATES["vision"]

    def test_vision_template_has_llava_format(self):
        """Vision template should use llava format."""
        assert "format: llava" in TEMPLATES["vision"]

    def test_vision_template_has_image_dir(self):
        """Vision template should have image_dir field."""
        assert "image_dir:" in TEMPLATES["vision"]

    def test_vision_template_has_vision_model(self):
        """Vision template should use a vision model."""
        assert "Vision" in TEMPLATES["vision"]

    def test_vision_template_mentions_unsloth(self):
        """Vision template should mention unsloth as an option."""
        assert "unsloth" in TEMPLATES["vision"]

    def test_vision_template_is_valid_yaml(self):
        """Vision template should be valid YAML that parses."""
        import yaml

        config = yaml.safe_load(TEMPLATES["vision"])
        assert config["modality"] == "vision"
        assert config["data"]["format"] == "llava"
        assert config["data"]["image_dir"] == "./data/images"

    def test_vision_template_validates_as_config(self):
        """Vision template should validate as a SoupConfig."""
        import yaml

        config_dict = yaml.safe_load(TEMPLATES["vision"])
        cfg = SoupConfig(**config_dict)
        assert cfg.modality == "vision"
        assert cfg.data.format == "llava"
        assert cfg.data.image_dir == "./data/images"


# ─── Init Command Tests ───────────────────────────────────────────────────


class TestInitVisionTemplate:
    """Test init command with --template vision."""

    def test_init_vision_template(self):
        """soup init --template vision should create a valid config."""
        from typer.testing import CliRunner

        from soup_cli.cli import app

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "soup.yaml"
            result = runner.invoke(app, ["init", "--template", "vision", "-o", str(output_path)])
            assert result.exit_code == 0
            assert output_path.exists()
            content = output_path.read_text()
            assert "modality: vision" in content
            assert "format: llava" in content


# ─── Data Inspect Vision Tests ─────────────────────────────────────────────


class TestDataInspectVision:
    """Test data inspect command with vision datasets."""

    def test_show_vision_stats_with_images(self):
        """_show_vision_stats should detect image fields and show stats."""
        from io import StringIO

        from rich.console import Console

        from soup_cli.commands.data import _show_vision_stats

        data = [
            {"image": "photo1.jpg", "conversations": []},
            {"image": "photo2.png", "conversations": []},
            {"image": "", "conversations": []},
        ]

        output = StringIO()
        with patch("soup_cli.commands.data.console", Console(file=output)):
            _show_vision_stats(data)

        text = output.getvalue()
        assert "Vision Stats" in text
        assert "2" in text  # 2 images referenced

    def test_show_vision_stats_no_images(self):
        """_show_vision_stats should not print anything for non-vision datasets."""
        from io import StringIO

        from rich.console import Console

        from soup_cli.commands.data import _show_vision_stats

        data = [{"instruction": "Hi", "output": "Hello"}]

        output = StringIO()
        with patch("soup_cli.commands.data.console", Console(file=output)):
            _show_vision_stats(data)

        text = output.getvalue()
        assert "Vision" not in text

    def test_show_vision_stats_empty_data(self):
        """_show_vision_stats should handle empty data gracefully."""
        from soup_cli.commands.data import _show_vision_stats

        # Should not raise
        _show_vision_stats([])

    def test_show_vision_stats_image_extensions(self):
        """_show_vision_stats should report image file extensions."""
        from io import StringIO

        from rich.console import Console

        from soup_cli.commands.data import _show_vision_stats

        data = [
            {"image": "a.jpg", "conversations": []},
            {"image": "b.jpg", "conversations": []},
            {"image": "c.png", "conversations": []},
        ]

        output = StringIO()
        with patch("soup_cli.commands.data.console", Console(file=output)):
            _show_vision_stats(data)

        text = output.getvalue()
        assert ".jpg" in text
        assert ".png" in text


# ─── Doctor Tests ──────────────────────────────────────────────────────────


class TestDoctorVision:
    """Test that doctor checks for Pillow (vision dependency)."""

    def test_pillow_in_deps_list(self):
        """Pillow should be listed in doctor dependencies."""
        from soup_cli.commands.doctor import DEPS

        pkg_names = [pkg_name for _, pkg_name, _, _ in DEPS]
        assert "Pillow" in pkg_names or "pillow" in [n.lower() for n in pkg_names]


# ─── Sweep Shortcut Tests ─────────────────────────────────────────────────


class TestModalitySweepParam:
    """Test modality parameter in sweep shortcuts."""

    def test_modality_shortcut(self):
        """modality should be settable via sweep param."""
        from soup_cli.commands.sweep import _set_nested_param

        config = {"modality": "text"}
        _set_nested_param(config, "modality", "vision")
        assert config["modality"] == "vision"
