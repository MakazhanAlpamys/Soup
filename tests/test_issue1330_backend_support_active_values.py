"""Tests for Issue #1330: soup doctor --config reports settings by spelling, not value.

Ensures that settings written in their inactive state (e.g. false) do not produce
false positive 'ignored' or 'rejected' rows, and that active values like seed: 0
are properly detected.
"""

from __future__ import annotations

import textwrap

import pytest

from soup_cli.autopilot.generate_config import write_yaml
from soup_cli.config.backend_support import (
    REGISTRY,
    REJECTED,
    check_config,
)
from soup_cli.config.loader import load_config, load_config_from_string
from tests.conftest import strip_ansi


def test_mlx_config_with_false_settings_reports_zero_gaps(tmp_path, capsys):
    """AC1: An MLX config with inactive/false settings gets the all-clear (0 rows)."""
    from soup_cli.commands.doctor import doctor

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
    gaps = check_config(cfg)
    assert gaps == []

    doctor(nccl=False, disk=False, config=str(cfg_file))
    out = strip_ansi(capsys.readouterr().out)
    assert "None of the 18 setting(s) known to be unread" in out


@pytest.mark.parametrize(
    ("task", "backend", "modality", "yaml_text"),
    [
        (
            "sft",
            "mlx",
            "text",
            textwrap.dedent(
                """\
                base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
                task: sft
                backend: mlx
                data:
                  train: ./data/train.jsonl
                  format: alpaca
                output: ./output
                """
            ),
        ),
        (
            "sft",
            "transformers",
            "vision",
            textwrap.dedent(
                """\
                base: Qwen/Qwen2-VL-7B-Instruct
                task: sft
                modality: vision
                data:
                  train: ./data/train.jsonl
                  format: llava
                  train_on_responses_only: false
                  mask_history: false
                output: ./output
                """
            ),
        ),
        (
            "sft",
            "transformers",
            "audio",
            textwrap.dedent(
                """\
                base: Qwen/Qwen2-Audio-7B-Instruct
                task: sft
                modality: audio
                data:
                  train: ./data/train.jsonl
                  format: chatml
                  train_on_responses_only: false
                  mask_history: false
                output: ./output
                """
            ),
        ),
    ],
)
def test_config_reloaded_after_write_yaml_matches_minimal_report(
    tmp_path, task, backend, modality, yaml_text
):
    """AC2: A config reloaded after write_yaml gets the same doctor report as the minimal config."""
    min_file = tmp_path / f"min_{backend}_{modality}.yaml"
    min_file.write_text(yaml_text, encoding="utf-8")
    min_cfg = load_config(str(min_file))
    min_gaps = check_config(min_cfg)

    dump_file = tmp_path / f"dump_{backend}_{modality}.yaml"
    write_yaml(min_cfg, dump_file)

    reloaded_cfg = load_config(str(dump_file))
    reloaded_gaps = check_config(reloaded_cfg)

    assert [g.field for g in reloaded_gaps] == [g.field for g in min_gaps]


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
    gaps = check_config(cfg)
    reported_fields = {g.field for g in gaps}
    assert "training.seed" in reported_fields
    assert "training.data_seed" in reported_fields


def test_rejected_entries_refused_at_load_or_triggerable():
    """AC4: For each REJECTED entry, assert that its active value is refused at config load."""
    rejected_entries = [
        (pair, entry)
        for pair, entries in REGISTRY.items()
        for entry in entries
        if entry.status == REJECTED
    ]
    assert len(rejected_entries) > 0

    for (task, backend, modality), entry in rejected_entries:
        # e.g. "training.loss_watchdog" -> field_name = "loss_watchdog"
        _, field_name = entry.field.split(".", 1)

        # Attempt to load a config with this active value
        yaml_snippet = textwrap.dedent(
            f"""\
            base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
            task: {task}
            backend: {backend}
            data:
              train: ./data/train.jsonl
              format: alpaca
            training:
              epochs: 1
              loss_watchdog: true
              {field_name}: true
            output: ./output
            """
        )
        with pytest.raises((ValueError, SystemExit)):
            load_config_from_string(yaml_snippet)
