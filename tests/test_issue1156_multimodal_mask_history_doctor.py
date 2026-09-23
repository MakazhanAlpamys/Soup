"""#1156 — train_on_responses_only and mask_history on vision and audio modalities.

data.mask_history and train_on_responses_only are honoured by the transformers
text path. On modality: vision and modality: audio they are accepted and then
silently ignored because multimodal collators do not implement assistant-turn
masking.

backend_support.REGISTRY carries the modality axis (task, backend, modality),
and soup doctor --config reports them as ignored rather than printing an
affirmative all-clear.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from soup_cli.config.backend_support import (
    IGNORED,
    check_config,
    unsupported_for,
)
from soup_cli.config.loader import load_config_from_string
from tests.conftest import strip_ansi


@pytest.fixture
def temp_config_file(tmp_path: pathlib.Path):
    def _create(yaml_text: str) -> str:
        cfg_file = tmp_path / "soup.yaml"
        cfg_file.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
        return str(cfg_file)

    return _create


def test_vision_sft_reports_mask_history_as_ignored():
    """AC3: A test that FAILS on earlier tree for modality: vision with mask_history: true."""
    yaml_text = """\
        base: Qwen/Qwen2-VL-7B-Instruct
        task: sft
        modality: vision
        data:
          train: ./data.jsonl
          format: llava
          mask_history: true
    """
    cfg = load_config_from_string(textwrap.dedent(yaml_text))
    gaps = {e.field: e for e in check_config(cfg)}
    assert "data.mask_history" in gaps
    entry = gaps["data.mask_history"]
    assert entry.status == IGNORED
    assert "multimodal vision collator" in entry.reason
    assert entry.issue == 1156


def test_vision_sft_reports_train_on_responses_only_as_ignored():
    yaml_text = """\
        base: Qwen/Qwen2-VL-7B-Instruct
        task: sft
        modality: vision
        data:
          train: ./data.jsonl
          format: llava
          train_on_responses_only: true
    """
    cfg = load_config_from_string(textwrap.dedent(yaml_text))
    gaps = {e.field: e for e in check_config(cfg)}
    assert "data.train_on_responses_only" in gaps
    entry = gaps["data.train_on_responses_only"]
    assert entry.status == IGNORED
    assert "multimodal vision collator" in entry.reason
    assert entry.issue == 1156


def test_audio_sft_reports_mask_history_as_ignored():
    """AC3: A test that FAILS on earlier tree for modality: audio with mask_history: true."""
    yaml_text = """\
        base: Qwen/Qwen2-Audio-7B-Instruct
        task: sft
        modality: audio
        data:
          train: ./data.jsonl
          format: chatml
          mask_history: true
    """
    cfg = load_config_from_string(textwrap.dedent(yaml_text))
    gaps = {e.field: e for e in check_config(cfg)}
    assert "data.mask_history" in gaps
    entry = gaps["data.mask_history"]
    assert entry.status == IGNORED
    assert "audio training" in entry.reason
    assert entry.issue == 1156


def test_audio_sft_reports_train_on_responses_only_as_ignored():
    yaml_text = """\
        base: Qwen/Qwen2-Audio-7B-Instruct
        task: sft
        modality: audio
        data:
          train: ./data.jsonl
          format: chatml
          train_on_responses_only: true
    """
    cfg = load_config_from_string(textwrap.dedent(yaml_text))
    gaps = {e.field: e for e in check_config(cfg)}
    assert "data.train_on_responses_only" in gaps
    entry = gaps["data.train_on_responses_only"]
    assert entry.status == IGNORED
    assert "audio training" in entry.reason
    assert entry.issue == 1156


def test_multimodal_clean_config_reports_no_gaps():
    yaml_text = """\
        base: Qwen/Qwen2-VL-7B-Instruct
        task: sft
        modality: vision
        data:
          train: ./data.jsonl
          format: llava
    """
    cfg = load_config_from_string(textwrap.dedent(yaml_text))
    gaps = check_config(cfg)
    assert gaps == []


def test_text_modality_does_not_report_mask_history_on_transformers():
    yaml_text = """\
        base: Qwen/Qwen2.5-7B-Instruct
        task: sft
        modality: text
        backend: transformers
        data:
          train: ./data.jsonl
          format: chatml
          train_on_responses_only: true
          mask_history: true
    """
    cfg = load_config_from_string(textwrap.dedent(yaml_text))
    gaps = check_config(cfg)
    assert gaps == []


def test_doctor_config_prints_gaps_for_vision_with_mask_history(
    temp_config_file, capsys, monkeypatch
):
    """AC2: soup doctor --config prints a row for a vision config that sets mask_history."""
    from rich.console import Console

    import soup_cli.commands.doctor as doctor_module
    from soup_cli.commands.doctor import doctor

    monkeypatch.setattr(doctor_module, "console", Console(width=200))
    cfg_path = temp_config_file("""\
        base: Qwen/Qwen2-VL-7B-Instruct
        task: sft
        modality: vision
        data:
          train: ./data.jsonl
          format: llava
          train_on_responses_only: true
          mask_history: true
    """)

    doctor(nccl=False, disk=False, config=cfg_path)
    out = strip_ansi(capsys.readouterr().out)
    assert "Config check - task=sft backend=transformers modality=vision" in out
    assert "data.mask_history" in out
    assert "data.train_on_responses_only" in out
    assert "multimodal vision collator" in out
    assert (
        "2 setting(s) written here are not read on backend=transformers (modality=vision)"
        in out
    )


def test_doctor_config_prints_all_clear_when_no_unread_flags_set(
    temp_config_file, capsys, monkeypatch
):
    from rich.console import Console

    import soup_cli.commands.doctor as doctor_module
    from soup_cli.commands.doctor import doctor

    monkeypatch.setattr(doctor_module, "console", Console(width=200))
    cfg_path = temp_config_file("""\
        base: Qwen/Qwen2-VL-7B-Instruct
        task: sft
        modality: vision
        data:
          train: ./data.jsonl
          format: llava
    """)

    doctor(nccl=False, disk=False, config=cfg_path)
    out = strip_ansi(capsys.readouterr().out)
    n = len(unsupported_for("sft", "transformers", "vision"))
    expected_msg = (
        f"None of the {n} setting(s) known to be unread on task=sft "
        f"backend=transformers modality=vision is set"
    )
    assert expected_msg in out


def test_mlx_reports_unread_flags_regardless_of_modality(
    temp_config_file, capsys, monkeypatch
):
    """MLX has no multimodal vision collator and ignores modality; unread flags must be reported."""
    from rich.console import Console

    import soup_cli.commands.doctor as doctor_module
    from soup_cli.commands.doctor import doctor

    monkeypatch.setattr(doctor_module, "console", Console(width=200))
    cfg_path = temp_config_file("""\
        base: meta-llama/Llama-3.1-8B-Instruct
        task: sft
        backend: mlx
        modality: vision
        training:
          seed: 42
        data:
          train: ./data.jsonl
    """)

    doctor(nccl=False, disk=False, config=cfg_path)
    out = strip_ansi(capsys.readouterr().out)
    assert "Config check - task=sft backend=mlx modality=vision" in out
    assert "training.seed" in out
    assert (
        "1 setting(s) written here are not read on backend=mlx (modality=vision)"
        in out
    )
