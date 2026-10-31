"""The preference config alias selects an existing native MLX objective."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.loader import load_config_from_string
from tests.conftest import strip_ansi


def _config(loss: str, *, top: str = "", training: str = "", data: str = "") -> str:
    return (
        "base: local-test-model\n"
        "task: preference\n"
        "backend: mlx\n"
        f"{top}"
        "data:\n"
        "  train: train.jsonl\n"
        "  format: dpo\n"
        f"{data}"
        "training:\n"
        "  epochs: 1\n"
        f"  preference_loss: {loss}\n"
        f"{training}"
    )


@pytest.mark.parametrize("loss", ["dpo", "orpo", "simpo"])
def test_native_preference_alias_loads_without_rewriting_the_config(loss: str) -> None:
    cfg = load_config_from_string(_config(loss))

    assert cfg.task == "preference"
    assert cfg.training.preference_loss == loss
    assert cfg.model_dump()["task"] == "preference"


@pytest.mark.parametrize("loss, class_name", [
    ("dpo", "MLXDPOTrainerWrapper"),
    ("orpo", "MLXORPOTrainerWrapper"),
    ("simpo", "MLXSimPOTrainerWrapper"),
])
def test_preference_builds_the_selected_native_trainer(
    loss: str, class_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.trainer.dispatch import build_trainer

    cfg = load_config_from_string(_config(loss))

    class NativeTrainer:
        def __init__(self, config: Any, **kwargs: Any) -> None:
            self.config = config
            self.kwargs = kwargs

    module = importlib.import_module(f"soup_cli.trainer.mlx_{loss}")
    monkeypatch.setattr(module, class_name, NativeTrainer)
    trainer = build_trainer(cfg, device="cpu", trust_remote_code=False)

    assert isinstance(trainer, NativeTrainer)
    assert trainer.config is cfg
    assert trainer.kwargs == {"device": "cpu", "trust_remote_code": False}


@pytest.mark.parametrize("loss", ["dpo", "orpo", "simpo"])
@pytest.mark.parametrize("enabled", [False, True])
def test_doctor_uses_the_native_support_table(
    loss: str, enabled: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import soup_cli.commands.doctor as doctor_module
    from soup_cli.config.backend_support import unsupported_for

    monkeypatch.setattr(doctor_module, "console", Console(width=120))
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "soup.yaml"
    path.write_text(
        _config(loss, training="  use_galore: true\n" if enabled else ""), encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["doctor", "--config", str(path)])
    output = " ".join(strip_ansi(result.output).split())

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "task=preference" in output
    if enabled:
        assert "training.use_galore" in output
        assert "GaLore has no MLX implementation" in output
        assert "1 setting(s) switched on here are not read" in output
    else:
        count = len(unsupported_for(loss, "mlx", "text"))
        assert count > 0
        assert f"None of the {count} setting(s)" in output


@pytest.mark.parametrize("loss", ["ipo", "bco"])
def test_unimplemented_preference_objectives_keep_the_refusal(loss: str) -> None:
    with pytest.raises(ValueError) as info:
        load_config_from_string(_config(loss))
    message = str(info.value)
    assert "task='preference' has no trainer on backend='mlx'" in message
    assert "Use backend=transformers for task=preference." in message


def test_preference_blends_are_still_refused() -> None:
    text = _config("dpo").replace(
        "  preference_loss: dpo\n", "  preference_loss_weights: {dpo: 0.5, orpo: 0.5}\n",
    )
    with pytest.raises(ValueError, match="preference_loss_weights is not supported on the mlx"):
        load_config_from_string(text)


@pytest.mark.parametrize("loss", ["dpo", "orpo", "simpo"])
def test_scalar_and_blended_losses_remain_mutually_exclusive(loss: str) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_config_from_string(_config(
            loss, training="  preference_loss_weights: {dpo: 0.5, orpo: 0.5}\n",
        ))


def test_preference_still_requires_a_loss() -> None:
    with pytest.raises(ValueError, match="requires either training.preference_loss"):
        load_config_from_string(_config("dpo").replace("  preference_loss: dpo\n", "  epochs: 1\n"))


@pytest.mark.parametrize("loss", ["dpo", "orpo", "simpo"])
@pytest.mark.parametrize("top, training, data, message", [
    ("modality: vision\n", "", "", "trains on text and chat rows only"),
    ("", "  lora: {r: 0}\n", "", "training.lora.r must be at least 1"),
    ("", "  lora: {rank_pattern: {q_proj: 4}}\n", "", "lora.rank_pattern"),
    ("", "  loraplus_lr_ratio: 2\n", "", "loraplus_lr_ratio is not implemented"),
    ("", "", "  add_new_tokens: ['<extra>']\n", "not applied on backend='mlx'"),
    ("", "", "  chat_template: chatml\n", "data.chat_template is not applied"),
    ("", "  eval_steps: 2\n", "", "training.eval_steps is not supported"),
])
def test_alias_keeps_native_scope_refusals(
    loss: str, top: str, training: str, data: str, message: str,
) -> None:
    alias = _config(loss, top=top, training=training, data=data)
    native = alias.replace("task: preference\n", f"task: {loss}\n").replace(
        f"  preference_loss: {loss}\n", "",
    )
    for is_alias, text in ((False, native), (True, alias)):
        with pytest.raises(ValueError, match=message) as info:
            load_config_from_string(text)
        task_label = (
            f"task='preference' with preference_loss='{loss}'"
            if is_alias else f"task='{loss}'"
        )
        if training == "  lora: {r: 0}\n":
            reasons = {
                "dpo": "the reference model is the untrained LoRA policy, which needs "
                "a LoRA adapter to exist",
                "orpo": "MLX ORPO trains a LoRA adapter; full fine-tuning is not "
                "implemented on backend='mlx'",
                "simpo": "MLX SimPO trains a LoRA adapter; full fine-tuning is not "
                "implemented on backend='mlx'",
            }
            assert (
                f"training.lora.r must be at least 1 for {task_label} on "
                f"backend='mlx': {reasons[loss]}."
            ) in str(info.value)
        if top == "modality: vision\n":
            objective = {"dpo": "DPO", "orpo": "ORPO", "simpo": "SimPO"}[loss]
            assert (
                f"modality='vision' is not supported by {task_label} on "
                f"backend='mlx': MLX {objective} trains on text and chat rows only. "
                "Use backend=transformers, or modality: text."
            ) in str(info.value)


@pytest.mark.parametrize("loss", ["orpo", "simpo"])
@pytest.mark.parametrize("is_alias", [False, True], ids=["native", "alias"])
def test_alias_does_not_admit_unsupported_eight_bit_training(loss: str, is_alias: bool) -> None:
    text = _config(loss, training="  quantization: 8bit\n")
    task_label = f"task='preference' with preference_loss='{loss}'"
    if not is_alias:
        text = text.replace("task: preference\n", f"task: {loss}\n").replace(
            f"  preference_loss: {loss}\n", "",
        )
        task_label = f"task='{loss}'"
    with pytest.raises(ValueError) as info:
        load_config_from_string(text)
    assert (
        f"training.quantization='8bit' is not supported by {task_label} on "
        "backend='mlx': use quantization: none, or a pre-quantized "
        "mlx-community 4-bit base with quantization: 4bit."
    ) in str(info.value)


@pytest.mark.parametrize("loss", ["ipo", "bco", "dpo", "orpo", "simpo"])
def test_transformers_keeps_the_preference_dispatcher(loss: str) -> None:
    from soup_cli.trainer.dispatch import build_trainer
    from soup_cli.trainer.preference import PreferenceTrainerWrapper

    cfg = load_config_from_string(_config(loss).replace("backend: mlx", "backend: transformers"))
    assert isinstance(build_trainer(cfg), PreferenceTrainerWrapper)


@pytest.mark.parametrize("loss", ["ipo", "bco"])
def test_doctor_refuses_unimplemented_objectives(loss: str, tmp_path: Path) -> None:
    path = tmp_path / "soup.yaml"
    path.write_text(_config(loss), encoding="utf-8")
    result = CliRunner().invoke(app, ["doctor", "--config", str(path)])
    output = " ".join(strip_ansi(result.output).split())

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert "task='preference' has no trainer on backend='mlx'" in output
    assert "All checks passed" not in output
