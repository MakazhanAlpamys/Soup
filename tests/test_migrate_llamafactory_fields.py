"""#1807 — the LLaMA-Factory migrator maps what the source trains with, or says what it drops.

Before: ``val_size`` was never read (every migrated config got Soup's default
``val_split: 0.1`` where LLaMA-Factory trains on every row), a missing
``lora_rank`` / ``lora_alpha`` fell back to Soup's ``r: 64, alpha: 16`` where
LLaMA-Factory uses ``r: 8, alpha: 2 * r``, a comma-separated ``lora_target`` was
copied as one string (peft then refused it), ``lora_target: all`` became ``auto``
(two of seven projections on Llama), and ``pref_beta`` / ``simpo_gamma`` under
``pref_loss: simpo`` vanished without a word.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.migrate.common import config_to_yaml
from soup_cli.migrate.llamafactory import migrate_llamafactory

_BASE = "model_name_or_path: meta-llama/Llama-3.1-8B\nfinetuning_type: lora\ndataset: alpaca_en\n"


def _migrate(tmp_path: Path, text: str):
    path = tmp_path / "lf.yaml"
    path.write_text(text, encoding="utf-8")
    result = migrate_llamafactory(path)
    cfg = load_config_from_string(config_to_yaml(result))
    return result, cfg


# ---------------------------------------------------------------------------
# The issue's reproduction
# ---------------------------------------------------------------------------


def test_the_reported_config_now_trains_what_llamafactory_trains(tmp_path: Path) -> None:
    result, cfg = _migrate(
        tmp_path,
        "model_name_or_path: meta-llama/Llama-3.1-8B\nstage: dpo\npref_loss: simpo\n"
        "pref_beta: 2.0\nsimpo_gamma: 1.0\nfinetuning_type: lora\n"
        "lora_target: q_proj,v_proj\nval_size: 0\ndataset: prefs\n",
    )
    lora = cfg.training.lora
    assert (lora.r, lora.alpha) == (8, 16)
    assert lora.target_modules == ["q_proj", "v_proj"]
    assert cfg.data.val_split == 0
    assert cfg.training.simpo_gamma == 1.0
    assert any("pref_beta" in w and "no Soup equivalent" in w for w in result["_warnings"])


# ---------------------------------------------------------------------------
# Validation split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line,expected", [("val_size: 0\n", 0.0), ("val_size: 0.2\n", 0.2),
                                           ("", 0.0)])
def test_val_size_round_trips_and_is_zero_when_absent(tmp_path: Path, line, expected) -> None:
    result, cfg = _migrate(tmp_path, _BASE + line)
    assert result["data"]["val_split"] == expected
    assert cfg.data.val_split == expected


def test_a_val_size_soup_cannot_express_is_named_not_defaulted_silently(tmp_path: Path) -> None:
    # LLaMA-Factory accepts an absolute row count; data.val_split is a ratio <= 0.5.
    result, cfg = _migrate(tmp_path, _BASE + "val_size: 1000\n")
    assert "val_split" not in result["data"]
    assert any("val_size" in w and "1000" in w and "val_split" in w for w in result["_warnings"])
    assert cfg.data.val_split == 0.1  # Soup's default, and the warning says so


def test_a_fraction_above_one_half_is_named_not_written(tmp_path: Path) -> None:
    # A ratio LLaMA-Factory accepts but data.val_split caps at 0.5: the "above 0.5"
    # branch, pinned apart from the row-count one (#1901 review).
    result, cfg = _migrate(tmp_path, _BASE + "val_size: 0.7\n")
    assert "val_split" not in result["data"]
    assert any(w.startswith("val_size: 0.7") and "above 0.5" in w and "not written" in w
               for w in result["_warnings"])
    assert cfg.data.val_split == 0.1


def test_exactly_one_half_is_still_written(tmp_path: Path) -> None:
    result, cfg = _migrate(tmp_path, _BASE + "val_size: 0.5\n")
    assert result["data"]["val_split"] == 0.5
    assert cfg.data.val_split == 0.5
    assert not any(w.startswith("val_size:") for w in result["_warnings"])


# ---------------------------------------------------------------------------
# LoRA defaults and targets
# ---------------------------------------------------------------------------


def test_missing_rank_and_alpha_use_llamafactory_defaults(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _BASE)
    assert (cfg.training.lora.r, cfg.training.lora.alpha) == (8, 16)


def test_rank_alone_gives_alpha_twice_the_rank(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _BASE + "lora_rank: 32\n")
    assert (cfg.training.lora.r, cfg.training.lora.alpha) == (32, 64)


def test_alpha_alone_keeps_the_default_rank(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _BASE + "lora_alpha: 32\n")
    assert (cfg.training.lora.r, cfg.training.lora.alpha) == (8, 32)


def test_both_set_are_copied(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _BASE + "lora_rank: 64\nlora_alpha: 16\n")
    assert (cfg.training.lora.r, cfg.training.lora.alpha) == (64, 16)


def test_a_quoted_rank_is_the_integer_and_still_defaults_alpha(tmp_path: Path) -> None:
    # YAML `lora_rank: "32"` is a str; the author meant 32, and alpha = 2 * rank needs it.
    result, cfg = _migrate(tmp_path, _BASE + 'lora_rank: "32"\n')
    assert result["training"]["lora"]["r"] == 32
    assert (cfg.training.lora.r, cfg.training.lora.alpha) == (32, 64)
    assert not any(w.startswith("lora_rank:") for w in result["_warnings"])


def test_a_quoted_alpha_is_the_integer(tmp_path: Path) -> None:
    result, cfg = _migrate(tmp_path, _BASE + 'lora_rank: 16\nlora_alpha: " 8 "\n')
    assert result["training"]["lora"]["alpha"] == 8
    assert (cfg.training.lora.r, cfg.training.lora.alpha) == (16, 8)


def test_a_rank_that_is_not_an_integer_is_named_not_defaulted_silently(tmp_path: Path) -> None:
    path = tmp_path / "lf.yaml"
    path.write_text(_BASE + "lora_rank: eight\n", encoding="utf-8")
    result = migrate_llamafactory(path)
    assert result["training"]["lora"]["r"] == "eight"
    assert "alpha" not in result["training"]["lora"]
    assert any(w.startswith("lora_rank: 'eight'") and "lora_alpha was not defaulted" in w
               for w in result["_warnings"])


@pytest.mark.parametrize(
    "value,expected",
    [
        ("q_proj,v_proj", ["q_proj", "v_proj"]),
        ("q_proj, v_proj", ["q_proj", "v_proj"]),
        ("q_proj", ["q_proj"]),
        ("[q_proj, k_proj]", ["q_proj", "k_proj"]),
    ],
)
def test_lora_target_becomes_a_list(tmp_path: Path, value, expected) -> None:
    _, cfg = _migrate(tmp_path, _BASE + f"lora_target: {value}\n")
    assert cfg.training.lora.target_modules == expected


@pytest.mark.parametrize("line", ["lora_target: all\n", ""])
def test_all_and_the_absent_key_map_to_every_linear_layer(tmp_path: Path, line) -> None:
    _, cfg = _migrate(tmp_path, _BASE + line)
    assert cfg.training.lora.target_modules == "all-linear"


def test_full_finetuning_writes_no_lora_defaults(tmp_path: Path) -> None:
    result, cfg = _migrate(
        tmp_path, "model_name_or_path: org/m\nfinetuning_type: full\ndataset: d\n"
    )
    assert result["training"]["lora"] == {"r": 0}
    assert cfg.training.lora.r == 0


# ---------------------------------------------------------------------------
# Preference strength
# ---------------------------------------------------------------------------


def test_pref_beta_under_ipo_is_the_ipo_tau(tmp_path: Path) -> None:
    # trainer/ipo.py passes training.ipo_tau as trl's beta, which is what pref_beta sets.
    _, cfg = _migrate(tmp_path, "model_name_or_path: org/m\nstage: dpo\npref_loss: ipo\n"
                                "pref_beta: 0.3\nfinetuning_type: lora\ndataset: prefs\n")
    assert cfg.task == "ipo"
    assert cfg.training.ipo_tau == 0.3


def test_ipo_without_pref_beta_leaves_ipo_tau_at_the_shared_default(tmp_path: Path) -> None:
    # LLaMA-Factory's pref_beta, trl's DPOConfig.beta and Soup's ipo_tau all default to
    # 0.1, so an absent pref_beta is left absent: no ipo_tau key, and the loaded config
    # trains at 0.1 (#1807 review).
    result, cfg = _migrate(tmp_path, "model_name_or_path: org/m\nstage: dpo\npref_loss: ipo\n"
                                     "finetuning_type: lora\ndataset: prefs\n")
    assert cfg.task == "ipo"
    assert "ipo_tau" not in result["training"]
    assert cfg.training.ipo_tau == 0.1
    assert not any("pref_beta" in w for w in result["_warnings"])


def test_the_upstream_default_the_equivalence_rests_on_is_still_0_1() -> None:
    # The mapping above leaves an absent pref_beta absent because trl's beta (what
    # trainer/ipo.py sets from ipo_tau) defaults to the same 0.1 as LLaMA-Factory's
    # pref_beta. If trl moves its default, this names the assumption that broke.
    import dataclasses

    trl = pytest.importorskip("trl")
    beta = next(f for f in dataclasses.fields(trl.DPOConfig) if f.name == "beta")
    assert beta.default == 0.1


def test_simpo_gamma_is_mapped_and_pref_beta_is_named(tmp_path: Path) -> None:
    result, cfg = _migrate(tmp_path, "model_name_or_path: org/m\nstage: dpo\npref_loss: simpo\n"
                                     "pref_beta: 2.0\nsimpo_gamma: 1.5\nfinetuning_type: lora\n"
                                     "dataset: prefs\n")
    assert cfg.task == "simpo"
    assert cfg.training.simpo_gamma == 1.5
    # Mapped, so neither dropped by name nor in the unmapped listing.
    assert not any(w.startswith("simpo_gamma:") for w in result["_warnings"])
    assert not any("not migrated" in w and "simpo_gamma" in w for w in result["_warnings"])
    assert any(w.startswith("pref_beta: 2.0") and "no Soup equivalent" in w
               for w in result["_warnings"])


@pytest.mark.parametrize("stage,field", [("dpo", "dpo_beta"), ("kto", "kto_beta")])
def test_pref_beta_still_reaches_the_existing_knobs(tmp_path: Path, stage, field) -> None:
    _, cfg = _migrate(tmp_path, f"model_name_or_path: org/m\nstage: {stage}\npref_beta: 0.25\n"
                                "finetuning_type: lora\ndataset: prefs\n")
    assert getattr(cfg.training, field) == 0.25


# ---------------------------------------------------------------------------
# Nothing is dropped silently
# ---------------------------------------------------------------------------


def test_unmapped_source_keys_are_listed_in_one_warning(tmp_path: Path) -> None:
    result, _ = _migrate(tmp_path, _BASE + "logging_steps: 10\nsave_steps: 500\nplot_loss: true\n")
    listing = [w for w in result["_warnings"] if "not migrated" in w]
    assert len(listing) == 1, result["_warnings"]
    for key in ("logging_steps", "plot_loss", "save_steps"):
        assert key in listing[0]
    # Keys that were mapped or warned about individually are not in the listing.
    for key in ("model_name_or_path", "finetuning_type", "dataset"):
        assert key not in listing[0]


def test_a_fully_mapped_config_has_no_unmapped_listing(tmp_path: Path) -> None:
    result, _ = _migrate(tmp_path, _BASE + "lora_rank: 8\nval_size: 0\ncutoff_len: 1024\n")
    assert not any("not migrated" in w for w in result["_warnings"])


# ---------------------------------------------------------------------------
# The migrated targets build a real adapter
# ---------------------------------------------------------------------------


def _tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


def _adapted_module_names(cfg) -> set[str]:
    from peft import get_peft_model

    from soup_cli.utils.peft_wiring import build_lora_config

    model = get_peft_model(
        _tiny_llama(),
        build_lora_config(cfg.training.lora, target_modules=cfg.training.lora.target_modules,
                          task_type=None),
    )
    return {name.rsplit(".", 1)[-1] for name, module in model.named_modules()
            if hasattr(module, "lora_A")}


def test_a_comma_target_builds_an_adapter_on_exactly_those_projections(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _BASE + "lora_target: q_proj,v_proj\n")
    assert _adapted_module_names(cfg) == {"q_proj", "v_proj"}


def test_all_wraps_every_linear_projection_of_a_tiny_llama(tmp_path: Path) -> None:
    _, cfg = _migrate(tmp_path, _BASE + "lora_target: all\n")
    assert _adapted_module_names(cfg) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }


def test_the_old_default_target_would_have_wrapped_two(tmp_path: Path) -> None:
    # The control for the test above: ``auto`` on this Llama is q_proj / v_proj only.
    _, cfg = _migrate(tmp_path, _BASE + "lora_target: all\n")
    cfg.training.lora.target_modules = "auto"
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(_tiny_llama(), LoraConfig(r=8, lora_alpha=16, task_type=None))
    adapted = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if hasattr(m, "lora_A")}
    assert adapted == {"q_proj", "v_proj"}


def test_yaml_round_trip_keeps_the_list(tmp_path: Path) -> None:
    result, _ = _migrate(tmp_path, _BASE + "lora_target: q_proj,v_proj\n")
    dumped = yaml.safe_load(config_to_yaml(result))
    assert dumped["training"]["lora"]["target_modules"] == ["q_proj", "v_proj"]
