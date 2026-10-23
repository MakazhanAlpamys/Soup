"""Portable LoRA+ config/dry-run and bounded native support contracts."""

import math

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import SoupConfig


def _config(**training):
    return SoupConfig(base="owner/base", backend="mlx", data={"train": "unused.jsonl"},
                      training={"quantization": "none", **training})


@pytest.mark.parametrize("ratio", [1, 2, 16])
@pytest.mark.parametrize("quantization", ["none", "4bit"])
@pytest.mark.parametrize("optimizer", ["adamw_torch", "adamw_hf", "adamw", "adamw_torch_fused"])
@pytest.mark.parametrize("rslora", [False, True])
def test_bounded_plus_config_accepts_ratio(ratio, quantization, optimizer, rslora):
    cfg = _config(loraplus_lr_ratio=ratio, quantization=quantization, optimizer=optimizer,
                  lora={"r": 4, "alpha": 8, "use_rslora": rslora})
    assert cfg.training.loraplus_lr_ratio == ratio


@pytest.mark.parametrize("ratio", [True, False, 0, -1, math.nan, math.inf, -math.inf, "oops"])
def test_invalid_ratios_refuse_with_option_diagnostic(ratio):
    with pytest.raises(ValueError, match="loraplus_lr_ratio"):
        _config(loraplus_lr_ratio=ratio)


@pytest.mark.parametrize("change, diagnostic", [
    ({"optimizer": "sgd"}, "AdamW"), ({"lora": {"use_dora": True}}, "DoRA"),
    ({"lora": {"rank_pattern": {"q_proj": 2}}}, "patterns"),
    ({"lora": {"alpha_pattern": {"q_proj": 4}}}, "patterns"),
    ({"quantization": "8bit"}, "quantization"),
    ({"lora": {"init_strategy": "olora"}}, "variant .olora. requires backend"),
])
def test_supported_control_and_specific_plus_refusal(change, diagnostic):
    assert _config(loraplus_lr_ratio=2).training.loraplus_lr_ratio == 2
    with pytest.raises(ValueError, match=diagnostic):
        _config(loraplus_lr_ratio=2, **change)


def test_public_dry_run_accepts_plus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "rows.jsonl"
    data.write_text('{"text":"alpha beta gamma"}\n' * 4)
    cfg = _config(loraplus_lr_ratio=2)
    cfg.data.train = str(data)
    cfg.data.format = "plaintext"
    cfg.data.train_on_responses_only = False
    cfg.data.val_split = 0
    path = tmp_path / "config.json"
    path.write_text(cfg.model_dump_json())
    result = CliRunner().invoke(app, ["train", "--config", str(path), "--dry-run"])
    assert result.exit_code == 0, (result.output, result.exception)


@pytest.mark.parametrize("ratio", [None])
def test_disabled_plus_and_other_backend_controls(ratio):
    assert _config(loraplus_lr_ratio=ratio, optimizer="sgd").training.loraplus_lr_ratio is None
    for backend in ("transformers", "unsloth"):
        cfg = _config().model_dump()
        cfg["backend"] = backend
        cfg["training"]["loraplus_lr_ratio"] = 2
        assert SoupConfig.model_validate(cfg).training.loraplus_lr_ratio == 2
