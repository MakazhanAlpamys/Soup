"""#2228 — ``soup adapters audit`` reads an MLX rsLoRA scale back as alpha.

mlx-lm writes ``scale = alpha / sqrt(rank)`` when ``lora.use_rslora`` is set
and ``scale = alpha / rank`` otherwise. The audit rebuilt alpha as
``scale * rank`` in both cases, so a correct rsLoRA run (rank 8, alpha 16)
reported about 45.25 and DIVERGED.
"""

from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

pytestmark = pytest.mark.unit

# Hand-checked: scale = alpha / sqrt(rank), then scale * rank is not alpha.
# (8, 16) reconstructs as 45.254... under the old formula.
_PAIRS = ((8, 16), (5, 5), (2, 7), (8, 100))


def _writer_record(rank: int, alpha: int, *, use_rslora: bool) -> dict:
    """The ``lora_parameters`` block ``build_mlx_adapter_config`` actually writes."""
    from types import SimpleNamespace

    from soup_cli.trainer.mlx_sft import build_mlx_adapter_config

    lora = SimpleNamespace(
        r=rank,
        alpha=alpha,
        use_rslora=use_rslora,
        use_dora=False,
        dropout=0.0,
        target_modules="auto",
    )
    written = build_mlx_adapter_config(lora, adapter_path="out")
    return {
        "fine_tune_type": "lora",
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "warmup_updates": 3,
        "total_updates": 12,
        "weight_decay": 0.01,
        "peak_lr": 1e-4,
        "max_grad_norm": 1.0,
        "grad_checkpoint": False,
        "grad_accumulation_steps": 4,
        "mask_prompt": False,
        "response_token_mask": True,
        "lora_parameters": written["lora_parameters"],
    }


def _config(rank: int, alpha: int, *, use_rslora: bool) -> dict:
    return {
        "base": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "task": "sft",
        "backend": "mlx",
        "training": {
            "lr": 1e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "optimizer": "adamw_torch",
            "scheduler": "cosine",
            "warmup_ratio": 0.25,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lora": {"r": rank, "alpha": alpha, "use_rslora": use_rslora},
        },
        "data": {
            "train": "./d.jsonl",
            "format": "chatml",
            "train_on_responses_only": True,
        },
    }


def _alpha_row(rank: int, alpha: int, *, config_rslora: bool, record_rslora: bool):
    from soup_cli.utils.adapter_audit import audit_adapter

    result = audit_adapter(
        _config(rank, alpha, use_rslora=config_rslora),
        _writer_record(rank, alpha, use_rslora=record_rslora),
    )
    return next(row for row in result.rows if row.setting == "lora.alpha")


@pytest.mark.parametrize(("rank", "alpha"), _PAIRS)
def test_matching_rslora_alpha_agrees(rank, alpha):
    row = _alpha_row(rank, alpha, config_rslora=True, record_rslora=True)
    assert row.status == "ok", f"ran={row.ran!r}"
    assert row.ran == alpha


def test_matching_plain_lora_alpha_agrees():
    row = _alpha_row(8, 16, config_rslora=False, record_rslora=False)
    assert row.status == "ok", f"ran={row.ran!r}"
    assert row.ran == 16


def test_plain_record_against_rslora_config_diverges():
    """A scale written as alpha/rank must not pass an rsLoRA config."""
    row = _alpha_row(8, 16, config_rslora=True, record_rslora=False)
    assert row.status == "diverged"
    assert row.ran != 16


def test_rslora_record_against_plain_config_diverges():
    """A scale written as alpha/sqrt(rank) must not pass a plain LoRA config."""
    row = _alpha_row(8, 16, config_rslora=False, record_rslora=True)
    assert row.status == "diverged"
    assert "45.25" in str(row.ran)


def test_peft_lora_alpha_is_not_rescaled_when_rslora_is_set():
    from soup_cli.utils.adapter_audit import audit_adapter

    config = _config(8, 16, use_rslora=True)
    config["backend"] = "transformers"
    result = audit_adapter(config, {"peft_type": "LORA", "r": 8, "lora_alpha": 16})
    row = next(item for item in result.rows if item.setting == "lora.alpha")
    assert row.status == "ok"
    assert row.ran == 16


def test_cli_matching_rslora_exits_zero_without_false_alpha(tmp_path, monkeypatch):
    from rich.console import Console

    import soup_cli.commands.adapters as adapters_module
    from soup_cli.commands.adapters import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(adapters_module, "console", Console(width=200, force_terminal=False))
    record = _writer_record(8, 16, use_rslora=True)
    (tmp_path / "adapter_config.json").write_text(json.dumps(record))
    config_path = tmp_path / "soup.yaml"
    config_path.write_text(yaml.safe_dump(_config(8, 16, use_rslora=True)))

    result = CliRunner(env={"COLUMNS": "200"}).invoke(
        app, ["audit", ".", "--config", config_path.name]
    )
    assert result.exit_code == 0, result.output
    assert "45.25" not in result.output
