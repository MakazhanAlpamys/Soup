"""Tests for Issue #1330: soup doctor --config reports settings by spelling, not value.

Ensures that settings written in their inactive state (e.g. false) do not produce
false positive 'ignored' or 'rejected' rows.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from soup_cli.autopilot.generate_config import write_yaml
from soup_cli.config.backend_support import (
    IGNORED,
    REGISTRY,
    REJECTED,
    SupportEntry,
    check_config,
    unsupported_for,
)
from soup_cli.config.loader import load_config, load_config_from_string
from tests.conftest import strip_ansi

_FORMAT = {"text": "alpaca", "vision": "llava", "audio": "chatml"}

# How to switch each declared row on. A row not listed is tried as
# ``<field>: true``; one that needs a sibling setting or a non-boolean value is
# listed. If the loader refuses the setting, the row can never reach doctor and
# has to go (#2238).
_ROW_SETTINGS = {
    "training.neftune_alpha": {"training.neftune_alpha": 5.0},
    "data.mask_history": {
        "data.mask_history": True,
        "data.train_on_responses_only": True,
    },
    "data.train_on_messages_with_train_field": {
        "data.train_on_messages_with_train_field": True,
        "data.train_on_responses_only": False,
    },
    "data.train_on_prompt": {
        "data.train_on_prompt": True,
        "data.train_on_responses_only": False,
    },
}
_ROWS = [
    (key, entry.field) for key, entries in REGISTRY.items() for entry in entries
]


def _config(task: str, backend: str, modality: str) -> dict:
    """The smallest config that loads for a registered (task, backend, modality)."""
    cfg = {
        "base": "test/model",
        "task": task,
        "backend": backend,
        "data": {"train": "./data/train.jsonl", "format": _FORMAT[modality]},
        "output": "./output",
    }
    if modality != "text":
        cfg["modality"] = modality
    return cfg


def _load(cfg: dict):
    return load_config_from_string(yaml.safe_dump(cfg, sort_keys=False))


def test_mlx_config_with_false_settings_reports_zero_gaps(tmp_path, capsys, monkeypatch):
    """AC1: An MLX config with inactive/false settings gets the all-clear (0 rows)."""
    from rich.console import Console

    import soup_cli.commands.doctor as doctor_module
    from soup_cli.commands.doctor import doctor

    monkeypatch.setattr(doctor_module, "console", Console(width=200))
    config_text = textwrap.dedent(
        """\
        base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
        task: sft
        backend: mlx
        data:
          train: ./data/train.jsonl
          format: alpaca
        training:
          epochs: 1
          loss_watchdog: false
          loss_spike_recovery: false
          grad_accum_auto_tune: false
          quantization_aware: false
          use_liger: false
          lora:
            r: 8
            alpha: 16
        output: ./output
        """
    )
    cfg_file = tmp_path / "c7_mlx_false.yaml"
    cfg_file.write_text(config_text, encoding="utf-8")

    cfg = load_config(str(cfg_file))
    assert check_config(cfg) == []

    doctor(nccl=False, disk=False, config=str(cfg_file))
    out = strip_ansi(capsys.readouterr().out)
    n = len(unsupported_for("sft", "mlx", "text"))
    assert (
        f"None of the {n} setting(s) known to be unread on task=sft backend=mlx "
        "is switched on in this config." in out
    )


@pytest.mark.parametrize("key", list(REGISTRY), ids=lambda key: "-".join(key))
def test_a_full_dump_reports_what_the_minimal_config_reports(key, tmp_path):
    """AC2: A config reloaded after write_yaml gets the same doctor report as the minimal config."""
    src = tmp_path / "min.yaml"
    src.write_text(yaml.safe_dump(_config(*key), sort_keys=False), encoding="utf-8")
    minimal = load_config(str(src))
    dump = tmp_path / "dump.yaml"
    write_yaml(minimal, dump)
    full = load_config(str(dump))
    assert [e.field for e in check_config(full)] == [e.field for e in check_config(minimal)]


def test_seed_zero_on_mlx_is_not_reported_now_that_mlx_seeds():
    """AC3 was "seed: 0 on MLX is still reported (0 is an active seed)". Since
    #1850 MLX SFT honours both seeds, so neither value is a gap any more. That
    0 is a real seed rather than "unset" is now pinned where it matters, in the
    run itself: tests/test_mlx_sft_seed.py::test_seed_zero_is_a_seed."""
    yaml_text = textwrap.dedent(
        """\
        base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
        task: sft
        backend: mlx
        data:
          train: ./data/train.jsonl
          format: alpaca
        training:
          epochs: 1
          seed: 0
          data_seed: 0
        output: ./output
        """
    )
    cfg = load_config_from_string(yaml_text)
    reported_fields = {g.field for g in check_config(cfg)}
    assert "training.seed" not in reported_fields
    assert "training.data_seed" not in reported_fields


def test_schema_refused_settings_have_no_doctor_rows():
    assert all(entry.status != REJECTED for entries in REGISTRY.values() for entry in entries)


@pytest.mark.parametrize(
    ("key", "field"),
    _ROWS,
    ids=[f"{'-'.join(key)}:{field}" for key, field in _ROWS],
)
def test_every_row_left_in_a_table_can_reach_doctor(key, field):
    config = _config(*key)
    for dotted, value in _ROW_SETTINGS.get(field, {field: True}).items():
        section, name = dotted.split(".")
        config.setdefault(section, {})[name] = value
    cfg = _load(config)
    assert field in {entry.field for entry in check_config(cfg)}


@pytest.mark.parametrize("task", ["sft", "dpo", "orpo", "simpo", "kto"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("use_lorafa", True),
        ("quantization_aware", True),
        ("use_liger", True),
        ("use_flash_attn", True),
    ],
)
def test_every_mlx_support_row_is_reachable(task, field, value):
    """A refused setting cannot remain in doctor's support registry."""
    config = _config(task, "mlx", "text")
    config["training"] = {field: value}
    dotted = f"training.{field}"

    if task == "sft" and field in {"use_liger", "use_flash_attn"}:
        cfg = _load(config)
        assert dotted in {entry.field for entry in check_config(cfg)}
        assert dotted in {entry.field for entry in unsupported_for(task, "mlx", "text")}
    else:
        with pytest.raises(ValueError, match=field):
            _load(config)
        assert dotted not in {
            entry.field for entry in unsupported_for(task, "mlx", "text")
        }


@pytest.mark.parametrize(
    ("task", "expected_count"),
    [("sft", 10), ("dpo", 5), ("orpo", 5), ("simpo", 5), ("kto", 5)],
)
def test_mlx_doctor_all_clear_counts_only_reachable_settings(
    task, expected_count, tmp_path, capsys, monkeypatch
):
    from rich.console import Console

    import soup_cli.commands.doctor as doctor_module
    from soup_cli.commands.doctor import doctor

    monkeypatch.setattr(doctor_module, "console", Console(width=200))
    config_file = tmp_path / f"{task}.yaml"
    config_file.write_text(
        yaml.safe_dump(_config(task, "mlx", "text"), sort_keys=False),
        encoding="utf-8",
    )

    doctor(nccl=False, disk=False, config=str(config_file))

    output = strip_ansi(capsys.readouterr().out)
    assert len(unsupported_for(task, "mlx", "text")) == expected_count
    assert (
        f"None of the {expected_count} setting(s) known to be unread "
        f"on task={task} backend=mlx is switched on in this config."
    ) in output


@pytest.mark.parametrize("modality", ["vision", "audio"])
@pytest.mark.parametrize("value", [True, False])
def test_multimodal_response_mask_has_no_unreachable_doctor_row(modality, value):
    config = _config("sft", "transformers", modality)
    config["data"]["train_on_responses_only"] = value
    assert check_config(_load(config)) == []
    assert "data.train_on_responses_only" not in {
        entry.field for entry in unsupported_for("sft", "transformers", modality)
    }


def test_written_true_schema_default_counts_as_unwritten(monkeypatch):
    """A future true-default registry entry cannot revive spelling-dependent reports."""
    key = ("sft", "transformers", "vision")
    entry = SupportEntry("data.train_on_responses_only", IGNORED, "test gap")
    monkeypatch.setitem(REGISTRY, key, (entry,))
    minimal = _config(*key)
    explicit = _config(*key)
    explicit["data"]["train_on_responses_only"] = True
    assert check_config(_load(minimal)) == []
    assert check_config(_load(explicit)) == []
