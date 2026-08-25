"""Regression coverage for streamed adapter base-model provenance (#531)."""

import json

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.chat import _detect_base_model
from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

MODEL_ID = "Qwen/Qwen3.8-27B"


def _blank_tiny_config():
    from transformers import LlamaConfig

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._name_or_path = ""
    return config


def test_streamed_peft_config_saves_exact_configured_base(monkeypatch, tmp_path):
    """Pin the saved PEFT contract, not merely the in-memory symptom."""
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: _blank_tiny_config(),
    )

    model = build_meta_skeleton(MODEL_ID, dtype="float32")
    assert model.__dict__["name_or_path"] == MODEL_ID
    assert model.config._name_or_path == MODEL_ID

    adapter = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    output = tmp_path / "adapter"
    adapter.peft_config["default"].save_pretrained(output)

    saved = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["base_model_name_or_path"] == MODEL_ID
    assert _detect_base_model(output / "adapter_config.json") == MODEL_ID


def test_blank_provenance_control_breaks_auto_detection(tmp_path):
    adapter = tmp_path / "blank-adapter"
    adapter.mkdir()
    config_path = adapter / "adapter_config.json"
    config_path.write_text(
        json.dumps({"base_model_name_or_path": "", "peft_type": "LORA"}),
        encoding="utf-8",
    )

    assert _detect_base_model(config_path) == ""
    assert not _detect_base_model(config_path)


def test_explicit_base_still_bypasses_auto_detection(monkeypatch, tmp_path):
    adapter = tmp_path / "blank-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "", "peft_type": "LORA"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "soup_cli.utils.trust_remote.model_requires_trust_remote_code",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "soup_cli.utils.trust_remote.resolve_trust_remote_code",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "soup_cli.commands.chat._load_model",
        lambda **_kwargs: (object(), object()),
    )

    result = CliRunner().invoke(
        app,
        [
            "chat",
            "--model",
            str(adapter),
            "--base",
            MODEL_ID,
            "--device",
            "cpu",
        ],
        input="/quit\n",
    )

    assert result.exit_code == 0
    assert "Cannot detect base model" not in result.output
    assert MODEL_ID in result.output
