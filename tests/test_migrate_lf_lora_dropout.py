"""A LLaMA-Factory config without ``lora_dropout`` migrates to LLaMA-Factory's 0.0.

Before: the key was copied only when present, so every config that left it out
(``examples/train_lora/qwen3_lora_sft.yaml`` among them) trained at Soup's default
dropout of 0.05, where LLaMA-Factory's ``FinetuningArguments`` default is 0.0. The
rank and alpha defaults already followed LLaMA-Factory (#1807).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.migrate.common import config_to_yaml
from soup_cli.migrate.llamafactory import migrate_llamafactory

# LLaMA-Factory's own examples/train_lora/qwen3_lora_sft.yaml, trimmed.
_EXAMPLE = (
    "model_name_or_path: Qwen/Qwen3-4B-Instruct-2507\n"
    "stage: sft\nfinetuning_type: lora\nlora_rank: 8\nlora_target: all\n"
    "dataset: identity\ntemplate: qwen3_nothink\ncutoff_len: 2048\n"
    "per_device_train_batch_size: 1\ngradient_accumulation_steps: 8\n"
    "learning_rate: 1.0e-4\nnum_train_epochs: 3.0\nlr_scheduler_type: cosine\n"
    "warmup_ratio: 0.1\n"
)


def _migrate(tmp_path: Path, text: str):
    path = tmp_path / "lf.yaml"
    path.write_text(text, encoding="utf-8")
    result = migrate_llamafactory(path)
    return result, load_config_from_string(config_to_yaml(result))


def test_the_stock_example_trains_without_lora_dropout(tmp_path: Path) -> None:
    result, cfg = _migrate(tmp_path, _EXAMPLE)
    lora = cfg.training.lora
    assert (lora.r, lora.alpha, lora.dropout) == (8, 16, 0.0)
    assert result["training"]["lora"]["dropout"] == 0.0


@pytest.mark.parametrize("finetuning_type", ["freeze", "oft"])
def test_the_lora_fallback_takes_the_same_default(tmp_path: Path, finetuning_type: str) -> None:
    _, cfg = _migrate(tmp_path, _EXAMPLE.replace("finetuning_type: lora",
                                                 f"finetuning_type: {finetuning_type}"))
    assert cfg.training.lora.dropout == 0.0


def test_an_explicit_dropout_is_kept(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _EXAMPLE + "lora_dropout: 0.1\n")
    assert cfg.training.lora.dropout == 0.1


def test_a_full_fine_tune_still_writes_no_adapter(tmp_path: Path) -> None:
    result, _ = _migrate(tmp_path, _EXAMPLE.replace("finetuning_type: lora",
                                                    "finetuning_type: full"))
    assert result["training"]["lora"] == {"r": 0}
