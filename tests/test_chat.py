"""Tests for soup chat command."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.chat import _detect_base_model

runner = CliRunner()


def test_chat_missing_model_path():
    """Chat with nonexistent path should fail."""
    result = runner.invoke(app, ["chat", "--model", "/nonexistent/path"])
    assert result.exit_code == 1


def test_detect_base_model_valid(tmp_path: Path):
    """Should read base_model_name_or_path from adapter_config.json."""
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(json.dumps({
        "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
        "r": 64,
        "lora_alpha": 16,
    }))
    result = _detect_base_model(config_path)
    assert result == "meta-llama/Llama-3.1-8B-Instruct"


def test_detect_base_model_missing_key(tmp_path: Path):
    """Should return None if base_model_name_or_path is missing."""
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(json.dumps({"r": 64}))
    result = _detect_base_model(config_path)
    assert result is None


def test_detect_base_model_invalid_json(tmp_path: Path):
    """Should return None for malformed JSON."""
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text("not valid json {{{")
    result = _detect_base_model(config_path)
    assert result is None


def test_detect_base_model_missing_file(tmp_path: Path):
    """Should return None for nonexistent file."""
    config_path = tmp_path / "nonexistent.json"
    result = _detect_base_model(config_path)
    assert result is None


def test_chat_adapter_without_base_model(tmp_path: Path):
    """Chat with adapter that has no base model info should fail."""
    # Create fake adapter directory with adapter_config.json but no base model
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config = adapter_dir / "adapter_config.json"
    config.write_text(json.dumps({"r": 64}))

    result = runner.invoke(app, ["chat", "--model", str(adapter_dir)])
    assert result.exit_code == 1
    assert "Cannot detect base model" in result.output


def test_chat_non_adapter_directory(tmp_path: Path):
    """Chat with directory that has no adapter_config.json skips base model detection."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # No adapter_config.json → not an adapter, will try to load directly
    # This will fail because there's no actual model, but it should get past validation
    result = runner.invoke(app, ["chat", "--model", str(model_dir)])
    # Should fail during model loading, not during validation
    assert result.exit_code == 1


# #2003: scoring bundles must stop before the first LM-consumer operation.
@pytest.mark.parametrize("backend", ["transformers", "mlx"])
@pytest.mark.parametrize("artifact_kind", ["root", "payload", "recorded-base"])
@pytest.mark.parametrize(
    "function_name, argument, first_work_line, list_input",
    [
        ("chat", "model", "from soup_cli.utils.mlx_generation import "
         "reject_native_adapter, validate_native_options", False),
        ("chat", "base_model", "from soup_cli.utils.mlx_generation import "
         "reject_native_adapter, validate_native_options", False),
    ],
)
def test_reward_bundle_refused_before_work(
    tmp_path,
    function_name,
    argument,
    first_work_line,
    list_input,
    artifact_kind,
    backend,
):
    from importlib import import_module

    from tests.reward_guard_helpers import assert_reward_guard

    root = tmp_path / "reward"
    root.mkdir()
    if artifact_kind == "payload":
        (root / "reward.safetensors").write_bytes(b"broken")
    else:
        (root / "reward_model.json").write_text("broken")
    if artifact_kind == "recorded-base":
        import json

        adapter = tmp_path / "lm-adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(root)})
        )
        root = adapter
    function = getattr(import_module("soup_cli.commands.chat"), function_name)
    value = [str(root)] if list_input else str(root)
    kwargs = {"backend": backend}
    kwargs[argument] = value
    assert_reward_guard(function, kwargs, first_work_line)


def test_reward_recorded_base_refused_at_resolution(tmp_path):
    import json

    from soup_cli.commands.chat import _detect_base_model
    from tests.reward_guard_helpers import assert_reward_guard

    root = tmp_path / "reward"
    root.mkdir()
    (root / "reward_model.json").write_text("broken")
    config = tmp_path / "adapter_config.json"
    config.write_text(json.dumps({"base_model_name_or_path": str(root)}))
    assert_reward_guard(_detect_base_model, {"adapter_config_path": config}, "return base")
