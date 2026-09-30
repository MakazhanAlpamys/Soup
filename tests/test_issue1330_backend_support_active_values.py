"""Tests for Issue #1330: soup doctor --config reports settings by spelling, not value.

Ensures that settings written in their inactive state (e.g. false) do not produce
false positive 'ignored' or 'rejected' rows, and that active values like seed: 0
are properly detected.
"""

from __future__ import annotations

import re
import textwrap

import pytest
import yaml

from soup_cli.autopilot.generate_config import write_yaml
from soup_cli.config.backend_support import (
    REGISTRY,
    REJECTED,
    check_config,
    unsupported_for,
)
from soup_cli.config.loader import load_config, load_config_from_string
from tests.conftest import strip_ansi

_FORMAT = {"text": "alpaca", "vision": "llava", "audio": "chatml"}


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


# data.train_on_responses_only defaults to true, so write_yaml writes `true` and
# the reloaded vision/audio config reports it. Whether the dump or #1156's
# explicit-true test gives way is still open on #1330; flip these once it is ruled.
_ROUND_TRIP_PENDING_1330 = {
    ("sft", "transformers", "vision"),
    ("sft", "transformers", "audio"),
}


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(
            key,
            marks=pytest.mark.xfail(
                strict=True,
                reason="data.train_on_responses_only defaults to true; pending #1330",
            ),
        )
        if key in _ROUND_TRIP_PENDING_1330
        else key
        for key in REGISTRY
    ],
    ids=lambda k: "-".join(k),
)
def test_a_full_dump_reports_what_the_minimal_config_reports(key, tmp_path):
    """AC2: A config reloaded after write_yaml gets the same doctor report as the minimal config."""
    src = tmp_path / "min.yaml"
    src.write_text(yaml.safe_dump(_config(*key), sort_keys=False), encoding="utf-8")
    minimal = load_config(str(src))
    dump = tmp_path / "dump.yaml"
    write_yaml(minimal, dump)
    full = load_config(str(dump))
    assert [e.field for e in check_config(full)] == [e.field for e in check_config(minimal)]


def test_seed_zero_on_mlx_is_still_reported():
    """AC3: training.seed: 0 and data_seed: 0 on MLX are still reported (0 is an active seed)."""
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
    assert "training.seed" in reported_fields
    assert "training.data_seed" in reported_fields


_REJECTED = [
    pytest.param(key, entry, id=f"{'-'.join(key)}-{entry.field}")
    for key, entries in REGISTRY.items()
    for entry in entries
    if entry.status == REJECTED
]


@pytest.mark.parametrize(("key", "entry"), _REJECTED)
def test_a_rejected_rows_active_value_is_refused_at_load(key, entry):
    """AC4: Each REJECTED field's active value is refused at load on its own, and its
    off value reports nothing, so the row cannot fire on a loadable config."""
    section, name = entry.field.split(".", 1)
    _load(_config(*key))  # control: the base config itself loads

    active = _config(*key)
    active.setdefault(section, {})[name] = True
    with pytest.raises(ValueError, match=re.escape(name)):
        _load(active)

    off = _config(*key)
    off.setdefault(section, {})[name] = False
    assert check_config(_load(off)) == []
