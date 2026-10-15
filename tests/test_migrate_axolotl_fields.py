"""#1807 — the axolotl migrator maps the validation split and every-linear targets.

Before: ``val_set_size`` was never read, so every migrated config got Soup's
default ``val_split: 0.1`` where axolotl's default is 0 (the source trained on
every row), and ``lora_target_linear: true`` (every linear layer) became
``auto``, which on Llama wraps q_proj / v_proj only. Keys with no mapping were
dropped without a word.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string
from soup_cli.migrate.axolotl import migrate_axolotl
from soup_cli.migrate.common import config_to_yaml

_BASE = (
    "base_model: meta-llama/Llama-3.1-8B\nadapter: lora\nlora_r: 16\nlora_alpha: 32\n"
    "datasets:\n  - path: d.jsonl\n    type: alpaca\n"
)


def _migrate(tmp_path: Path, text: str):
    path = tmp_path / "ax.yaml"
    path.write_text(text, encoding="utf-8")
    result = migrate_axolotl(path)
    cfg = load_config_from_string(config_to_yaml(result))
    return result, cfg


def test_the_reported_config_now_trains_what_axolotl_trains(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _BASE + "lora_target_linear: true\nval_set_size: 0\n")
    assert cfg.training.lora.target_modules == "all-linear"
    assert cfg.data.val_split == 0


@pytest.mark.parametrize(
    "line,expected",
    [("val_set_size: 0\n", 0.0), ("val_set_size: 0.0\n", 0.0), ("val_set_size: 0.05\n", 0.05),
     ("", 0.0)],
)
def test_val_set_size_round_trips_and_is_zero_when_absent(tmp_path: Path, line, expected) -> None:
    result, cfg = _migrate(tmp_path, _BASE + line)
    assert result["data"]["val_split"] == expected
    assert cfg.data.val_split == expected


def test_a_row_count_is_named_not_defaulted_silently(tmp_path: Path) -> None:
    # axolotl accepts an absolute number of rows; data.val_split is a ratio <= 0.5.
    result, cfg = _migrate(tmp_path, _BASE + "val_set_size: 500\n")
    assert "val_split" not in result["data"]
    assert any("val_set_size" in w and "500" in w for w in result["_warnings"])
    assert cfg.data.val_split == 0.1


def test_a_fraction_above_one_half_is_named_not_written(tmp_path: Path) -> None:
    # The same shared helper as LLaMA-Factory's val_size: the "above 0.5" branch,
    # pinned apart from the row-count one (#1901 review).
    result, cfg = _migrate(tmp_path, _BASE + "val_set_size: 0.7\n")
    assert "val_split" not in result["data"]
    assert any(w.startswith("val_set_size: 0.7") and "above 0.5" in w and "not written" in w
               for w in result["_warnings"])
    assert cfg.data.val_split == 0.1


def test_lora_target_modules_list_is_still_copied(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _BASE + "lora_target_modules:\n  - q_proj\n  - k_proj\n")
    assert cfg.training.lora.target_modules == ["q_proj", "k_proj"]


def test_without_either_target_key_the_default_stays_auto(tmp_path: Path) -> None:
    result, _ = _migrate(tmp_path, _BASE)
    assert "target_modules" not in result["training"]["lora"]


def test_unmapped_source_keys_are_listed_in_one_warning(tmp_path: Path) -> None:
    result, _ = _migrate(
        tmp_path, _BASE + "logging_steps: 1\nsaves_per_epoch: 1\nspecial_tokens:\n  pad_token: x\n"
    )
    listing = [w for w in result["_warnings"] if "not migrated" in w]
    assert len(listing) == 1, result["_warnings"]
    for key in ("logging_steps", "saves_per_epoch", "special_tokens"):
        assert key in listing[0]
    for key in ("base_model", "adapter", "lora_r", "datasets"):
        assert key not in listing[0]


def test_a_fully_mapped_config_has_no_unmapped_listing(tmp_path: Path) -> None:
    result, _ = _migrate(tmp_path, _BASE + "val_set_size: 0\nsequence_len: 1024\nnum_epochs: 1\n")
    assert not any("not migrated" in w for w in result["_warnings"])


def test_all_linear_wraps_every_linear_projection_of_a_tiny_llama(tmp_path: Path) -> None:
    import torch
    from peft import get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.utils.peft_wiring import build_lora_config

    _, cfg = _migrate(tmp_path, _BASE + "lora_target_linear: true\n")
    torch.manual_seed(0)
    tiny = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    ))
    model = get_peft_model(tiny, build_lora_config(
        cfg.training.lora, target_modules=cfg.training.lora.target_modules, task_type=None))
    adapted = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if hasattr(m, "lora_A")}
    assert adapted == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
