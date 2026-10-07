"""The final trainer owns migration settings, not the last config constructor."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from soup_cli.migrate.common import config_to_yaml
from soup_cli.migrate.unsloth import migrate_unsloth

from .conftest import strip_ansi


def _notebook(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "stage.ipynb"
    model = "model = FastLanguageModel.from_pretrained(model_name='test/model')\n"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": model + source}]}),
                    encoding="utf-8")
    return path


def _stage(task: str = "GRPO", warmup: str = "SFTConfig", first: bool = True) -> str:
    learning_rate = "5e-6" if task == "GRPO" else "5e-7"
    cfg = (f"cfg = {task}Config(learning_rate={learning_rate}, "
           "per_device_train_batch_size=4, output_dir='stage_out')\n")
    warm = (f"warmup = SFTTrainer(model, args={warmup}(learning_rate=2e-4, "
            "per_device_train_batch_size=1, output_dir='warmup_out'))\n")
    return (cfg + warm if first else warm + cfg) + f"trainer = {task}Trainer(model, args=cfg)\n"


def _migrate(tmp_path: Path, source: str) -> dict:
    result = migrate_unsloth(_notebook(tmp_path, source))
    load_config_from_string(config_to_yaml(result))
    return result


@pytest.mark.parametrize("task,lr", [("GRPO", 5e-6), ("DPO", 5e-7)])
@pytest.mark.parametrize("warmup", ["SFTConfig", "TrainingArguments"])
def test_grpo_first_uses_stage_hyperparameters(tmp_path: Path, task: str, lr: float,
                                              warmup: str) -> None:
    result = _migrate(tmp_path, _stage(task, warmup))
    assert result["task"] == task.lower()
    assert result["training"]["lr"] == lr
    assert result["training"]["batch_size"] == 4


@pytest.mark.parametrize("warmup", ["SFTConfig", "TrainingArguments"])
def test_grpo_first_uses_stage_output(tmp_path: Path, warmup: str) -> None:
    assert _migrate(tmp_path, _stage(warmup=warmup))["output"] == "stage_out"


def test_unused_config_after_final_trainer_does_not_change_task(tmp_path: Path) -> None:
    baseline = _migrate(tmp_path, _stage())
    unused = ("unused = SFTConfig(learning_rate=0.1, per_device_train_batch_size=8, "
              "output_dir='unused', max_steps=999)\n")
    assert _migrate(tmp_path, _stage() + unused) == baseline


def test_warmup_first_control_is_unchanged(tmp_path: Path) -> None:
    assert _migrate(tmp_path, _stage(first=False)) == _migrate(tmp_path, _stage())


@pytest.mark.parametrize("trainer,arguments", [
    ("GRPO", "args=cfg"),
    ("GRPO", "args=GRPOConfig(learning_rate=5e-6)"),
    ("GRPO", "reward_fn, cfg"),
    ("GRPO", "[a, b], cfg"),
    ("DPO", "None, cfg"),
    ("DPO", "ref_model, cfg"),
    ("SFT", "cfg"),
    ("Reward", "cfg"),
])
def test_inline_named_and_positional_configs(tmp_path: Path, trainer: str,
                                             arguments: str) -> None:
    source = (f"cfg = {trainer}Config(learning_rate=5e-6)\n"
              "unused = SFTConfig(learning_rate=0.1)\n"
              f"trainer = {trainer}Trainer(model, {arguments})\n")
    assert _migrate(tmp_path, source)["training"]["lr"] == 5e-6


@pytest.mark.parametrize("task", ["DPO", "KTO"])
def test_beta_mapping_uses_final_trainer_task_before_config(tmp_path: Path, task: str) -> None:
    source = f"cfg = {task}Config(beta=0.3)\ntrainer = {task}Trainer(model, args=cfg)"
    assert _migrate(tmp_path, source)["training"][f"{task.lower()}_beta"] == 0.3


def test_final_trainer_task_wins_over_config_class(tmp_path: Path) -> None:
    source = "trainer = GRPOTrainer(model, args=SFTConfig(learning_rate=1e-4))"
    result = _migrate(tmp_path, source)
    assert result["task"] == "grpo"
    assert result["training"]["lr"] == 1e-4


def test_annotated_config_binding_is_resolved(tmp_path: Path) -> None:
    source = "cfg: GRPOConfig = GRPOConfig(learning_rate=5e-6)\nGRPOTrainer(model, args=cfg)"
    assert _migrate(tmp_path, source)["training"]["lr"] == 5e-6


@pytest.mark.parametrize("statement", [
    "if flag:\n    cfg = GRPOConfig(learning_rate=0.1)",
    "try:\n    cfg = GRPOConfig(learning_rate=0.1)\nexcept Exception:\n    pass",
    "with context:\n    cfg = GRPOConfig(learning_rate=0.1)",
])
def test_conditional_rebinding_is_not_silently_reused(tmp_path: Path, statement: str) -> None:
    source = f"cfg = GRPOConfig(learning_rate=5e-6)\n{statement}\nGRPOTrainer(model, args=cfg)"
    result = _migrate(tmp_path, source)
    assert "lr" not in result["training"]
    assert any("Could not read" in note for note in result["_warnings"])


def test_expanded_trainer_keywords_report_unreadable_args(tmp_path: Path) -> None:
    result = _migrate(tmp_path, "GRPOTrainer(model, **opts)")
    assert any("Could not read" in note for note in result["_warnings"])


def test_binding_at_trainer_and_no_settings_leak_from_warmup(tmp_path: Path) -> None:
    source = ("cfg = GRPOConfig(learning_rate=5e-6)\n"
              "warmup = SFTTrainer(model, args=SFTConfig(num_train_epochs=9))\n"
              "trainer = GRPOTrainer(model, args=cfg)\n"
              "cfg = SFTConfig(learning_rate=0.1)\n")
    result = _migrate(tmp_path, source)
    assert result["training"]["lr"] == 5e-6
    assert "epochs" not in result["training"]


@pytest.mark.parametrize("binding", ["cfg = None", "cfg = other", "cfg = make_config()"])
def test_unreadable_config_does_not_reuse_a_shadowed_binding(tmp_path: Path, binding: str) -> None:
    source = (f"cfg = GRPOConfig(learning_rate=5e-6)\n{binding}\n"
              "trainer = GRPOTrainer(model, args=cfg)")
    result = _migrate(tmp_path, source)
    assert result["task"] == "grpo"
    assert "lr" not in result["training"]
    assert any("Could not read" in warning for warning in result["_warnings"])


def test_later_trainer_without_args_uses_defaults(tmp_path: Path) -> None:
    result = _migrate(tmp_path, _stage() + "last = SFTTrainer(model)\n")
    assert result["task"] == "sft"
    assert result["output"] == "./output"
    assert "lr" not in result["training"]


@pytest.mark.parametrize("force_color", [False, True])
def test_cli_dry_run_emits_the_final_stage_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_color: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    _notebook(tmp_path, _stage())
    runner = CliRunner(env={"FORCE_COLOR": str(int(force_color)), "COLUMNS": "300"})
    result = runner.invoke(
        app, ["migrate", "--from", "unsloth", "stage.ipynb", "--dry-run"], color=force_color,
    )
    assert result.exit_code == 0, result.output
    # The CLI preview includes notes around the YAML; pin the emitted values.
    assert "task: grpo" in strip_ansi(result.output)
    assert "output: stage_out" in strip_ansi(result.output)
    assert "lr: 5.0e-06" in strip_ansi(result.output)
    assert "batch_size: 4" in strip_ansi(result.output)
    assert not (tmp_path / "soup.yaml").exists()


def test_real_warmup_warnings_are_preserved_without_borrowing_settings(tmp_path: Path) -> None:
    source = ("warmup = SFTTrainer(model, packing=True, args=SFTConfig(max_steps=100))\n"
              "trainer = GRPOTrainer(model, args=GRPOConfig(max_steps=10, learning_rate=5e-6))")
    result = _migrate(tmp_path, source)
    assert any("max_steps=100" in note for note in result["_warnings"])
    assert any("packing=True" in note for note in result["_warnings"])
    assert result["training"]["lr"] == 5e-6
