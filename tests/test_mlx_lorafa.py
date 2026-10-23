"""Portable native LoRA-FA recipe and public CLI contracts."""

import copy

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.schema import SoupConfig


def recipe():
    return dict(
        base="local-base",
        backend="mlx",
        task="sft",
        data={"train": "rows.jsonl", "format": "plaintext"},
        training={
            "quantization": "none",
            "use_lorafa": True,
            "lora": {"r": 4, "alpha": 8, "dropout": 0},
        },
    )


@pytest.mark.parametrize("quantization", ["none", "4bit"])
@pytest.mark.parametrize("optimizer", ["adamw_torch", "adamw", "adamw_torch_fused"])
def test_public_config_and_dry_run(tmp_path, quantization, optimizer):
    raw = recipe()
    raw["training"].update(quantization=quantization, optimizer=optimizer)
    data = tmp_path / "rows.jsonl"
    data.write_text('{"text":"alpha beta gamma"}\n' * 4)
    raw["data"]["train"] = str(data)
    cfg = SoupConfig.model_validate(raw)
    path = tmp_path / "config.json"
    path.write_text(cfg.model_dump_json())
    result = CliRunner().invoke(app, ["train", "--config", str(path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert cfg.training.use_lorafa


@pytest.mark.parametrize("field", ["r", "alpha"])
@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "4", float("nan"), float("inf")])
def test_requires_explicit_positive_integer_dimensions(field, value):
    raw = recipe()
    assert SoupConfig.model_validate(raw).training.use_lorafa
    if value is None:
        del raw["training"]["lora"][field]
    else:
        raw["training"]["lora"][field] = value
    with pytest.raises(ValueError, match=rf"use_lorafa.*explicit positive integer.*{field}"):
        SoupConfig.model_validate(raw)


@pytest.mark.parametrize(
    "option,value",
    [
        ("use_dora", True),
        ("use_rslora", True),
        ("rank_pattern", {"q_proj": 2}),
        ("alpha_pattern", {"q_proj": 12}),
    ],
)
def test_unsupported_adapter_options(option, value):
    raw = recipe()
    assert SoupConfig.model_validate(raw).training.use_lorafa
    raw["training"]["lora"][option] = value
    with pytest.raises(ValueError, match="use_lorafa.*ordinary linear LoRA"):
        SoupConfig.model_validate(raw)


def test_transformers_scope_and_ordinary_mlx_unchanged():
    for task in ("sft", "pretrain", "embedding"):
        raw = recipe()
        raw.update(backend="transformers", task=task)
        assert SoupConfig.model_validate(raw).training.use_lorafa
    raw = recipe()
    raw["training"]["use_lorafa"] = False
    control = copy.deepcopy(raw)
    del control["training"]["use_lorafa"]
    assert SoupConfig.model_validate(raw) == SoupConfig.model_validate(control)


@pytest.mark.parametrize(
    "section,field,value,diagnostic",
    [
        ("training", "optimizer", "sgd", "AdamW-based"),
        ("training", "quantization", "8bit", "use_lorafa.*ordinary linear LoRA"),
        ("training", "loraplus_lr_ratio", 2, "mutually exclusive"),
        ("training", "use_galore", True, "mutually exclusive"),
        ("lora", "use_vera", True, "mutually exclusive"),
        ("lora", "use_olora", True, "use_olora"),
    ],
)
def test_specific_refusals_with_adjacent_valid_control(section, field, value, diagnostic):
    raw = recipe()
    assert SoupConfig.model_validate(raw).training.use_lorafa
    target = raw["training"] if section == "training" else raw["training"]["lora"]
    target[field] = value
    with pytest.raises(ValueError, match=diagnostic):
        SoupConfig.model_validate(raw)


@pytest.mark.parametrize('disabled', [False, 'false', 'off', '0'])
def test_disabled_flag_keeps_ordinary_default_dimensions(disabled):
    raw = recipe()
    del raw['training']['lora']
    raw['training']['use_lorafa'] = disabled
    assert SoupConfig.model_validate(raw).training.use_lorafa is False


@pytest.mark.parametrize('task', ['dpo', 'orpo'])
def test_other_supported_mlx_tasks_require_fa_disabled(task):
    raw = recipe()
    raw['task'] = task
    raw['training']['use_lorafa'] = False
    assert SoupConfig.model_validate(raw).task == task
    raw['training']['use_lorafa'] = True
    with pytest.raises(ValueError, match='use_lorafa.*ordinary linear LoRA text SFT'):
        SoupConfig.model_validate(raw)
