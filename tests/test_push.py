"""Tests for soup push command."""

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.push import _format_size, _generate_model_card

runner = CliRunner()


# --- _format_size ---


def test_format_size_bytes():
    assert _format_size(512) == "512.0 B"


def test_format_size_kb():
    assert _format_size(2048) == "2.0 KB"


def test_format_size_mb():
    assert _format_size(5 * 1024 * 1024) == "5.0 MB"


def test_format_size_gb():
    assert _format_size(3 * 1024**3) == "3.0 GB"


def test_format_size_tb():
    assert _format_size(2 * 1024**4) == "2.0 TB"


# --- _generate_model_card ---


def test_model_card_adapter(tmp_path: Path):
    """Model card for LoRA adapter should include base model info."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config = adapter_dir / "adapter_config.json"
    config.write_text(
        json.dumps(
            {
                "base_model_name_or_path": "meta-llama/Llama-3.1-8B",
                "r": 64,
                "lora_alpha": 16,
            }
        )
    )

    card = _generate_model_card(adapter_dir, "user/my-model", is_adapter=True)
    assert "my-model" in card
    assert "meta-llama/Llama-3.1-8B" in card
    assert "LoRA rank" in card
    assert "soup-cli" in card


def test_model_card_full_model(tmp_path: Path):
    """Model card for full model should not include LoRA info."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    card = _generate_model_card(model_dir, "user/full-model", is_adapter=False)
    assert "full-model" in card
    assert "fine-tuned language model" in card
    assert "soup-cli" in card


def test_model_card_adapter_bad_config(tmp_path: Path):
    """Model card should handle malformed adapter_config.json gracefully."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config = adapter_dir / "adapter_config.json"
    config.write_text("not valid json")

    card = _generate_model_card(adapter_dir, "user/broken", is_adapter=True)
    assert "broken" in card


def test_model_card_repo_without_slash():
    """Model card should handle repo ID without slash."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        card = _generate_model_card(Path(td), "my-model", is_adapter=False)
        assert "my-model" in card


# --- CLI validation ---


def test_push_missing_model():
    result = runner.invoke(app, ["push", "--model", "/nonexistent", "--repo", "user/model"])
    assert result.exit_code == 1


def test_push_not_a_directory(tmp_path: Path):
    fake_file = tmp_path / "model.bin"
    fake_file.write_text("content")
    result = runner.invoke(app, ["push", "--model", str(fake_file), "--repo", "user/model"])
    assert result.exit_code == 1
    assert "directory" in result.output.lower()


def test_push_invalid_model_dir(tmp_path: Path):
    """Empty directory (no adapter_config.json or config.json) should fail."""
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    result = runner.invoke(app, ["push", "--model", str(model_dir), "--repo", "user/model"])
    assert result.exit_code == 1


def test_push_no_token(tmp_path: Path, monkeypatch):
    """Should fail if no HF token is available."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    # Point HOME at an empty dir so the cached-token fallback also misses.
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no-creds")
    # `soup push` now requires --model under cwd.
    monkeypatch.chdir(tmp_path)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "adapter_config.json").write_text("{}")
    (model_dir / "adapter_model.safetensors").write_text("fake")

    result = runner.invoke(app, ["push", "--model", str(model_dir), "--repo", "user/model"])
    assert result.exit_code == 1
    assert "token" in result.output.lower()
