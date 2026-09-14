"""Regression coverage for Autopilot's unwired training-intelligence gates."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app


def _write_data(path: Path) -> Path:
    path.write_text(
        "\n".join(
            json.dumps({"instruction": f"q{index}", "output": f"a{index}"})
            for index in range(20)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_autopilot_enables_no_unwired_training_intelligence(tmp_path: Path) -> None:
    from soup_cli.autopilot.generate_config import build_soup_config

    config = build_soup_config(
        model="meta-llama/Llama-3.1-8B-Instruct",
        data_path=str(_write_data(tmp_path / "data.jsonl")),
        goal="chat",
        vram_gb=24.0,
    )

    assert config.training.forgetting_detection is False
    assert config.training.checkpoint_intelligence is False
    assert config.training.early_stop_on_regression is False


def test_autopilot_panel_does_not_advertise_unwired_protections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = _write_data(tmp_path / "data.jsonl")

    result = CliRunner().invoke(
        app,
        [
            "autopilot",
            "--model",
            "meta-llama/Llama-3.1-8B-Instruct",
            "--data",
            data_path.name,
            "--goal",
            "chat",
            "--gpu-budget",
            "24GB",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Forgetting detection" not in result.output
    assert "Checkpoint intelligence" not in result.output


def test_train_warns_for_nondefault_unwired_sibling_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import soup_cli.commands.train as train_mod

    data_path = _write_data(tmp_path / "data.jsonl")
    config_path = tmp_path / "soup.yaml"
    config_path.write_text(
        "base: sshleifer/tiny-gpt2\n"
        "task: sft\n"
        f"output: {tmp_path / 'out'}\n"
        "data:\n"
        f"  train: {data_path}\n"
        "training:\n"
        "  forgetting_threshold: 0.20\n"
        "  checkpoint_keep_top: 5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_mod, "detect_device", lambda backend=None: ("cpu", "CPU"))
    monkeypatch.setattr(
        train_mod,
        "get_gpu_info",
        lambda backend=None: {"memory_total": "N/A"},
    )
    monkeypatch.setattr(
        train_mod,
        "load_dataset",
        lambda *args, **kwargs: {"train": [{"text": "hello"}]},
    )

    result = CliRunner().invoke(
        app,
        ["train", "--config", str(config_path), "--dry-run", "--yes"],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    output = " ".join(result.output.split())
    assert "forgetting_threshold" in output
    assert "checkpoint_keep_top" in output
    assert "not enforced during training" in output


def test_forgetting_threshold_is_declared_unconsumed() -> None:
    from tests.test_issue748_config_fields_reach_a_consumer import (
        KNOWN_UNCONSUMED,
        _consumed_in_src,
        field_reaches_a_consumer,
        training_receiver_reads,
    )

    assert "training.forgetting_threshold" in KNOWN_UNCONSUMED
    consumed = _consumed_in_src()
    assert "forgetting_threshold" in consumed  # ship.py's unrelated CLI argument
    assert not training_receiver_reads(
        Path(__file__).resolve().parents[1].joinpath("src", "soup_cli").rglob("*.py"),
        "forgetting_threshold",
    )
    assert not field_reaches_a_consumer(
        "training.forgetting_threshold",
        "forgetting_threshold",
        consumed,
    )


def test_training_intelligence_documented_config_parses() -> None:
    from soup_cli.config.loader import load_config_from_string

    docs = (
        Path(__file__).resolve().parents[1] / "docs" / "peft-and-efficiency.md"
    ).read_text(encoding="utf-8")
    section = docs.split("## Training Intelligence", maxsplit=1)[1].split(
        "## GaLore", maxsplit=1
    )[0]
    match = re.search(r"```yaml\n(.*?)```", section, flags=re.DOTALL)
    assert match is not None

    config = load_config_from_string(match.group(1))
    assert config.training.eval_gate is not None
    assert config.training.eval_gate.enabled is True
