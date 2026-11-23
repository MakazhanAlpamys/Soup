"""Commands offered by real training output obey the real consumers' CLI contract."""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

import pytest
import yaml
from typer.main import get_command
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi


def _plain(text):
    return " ".join(strip_ansi(text).split())


def _offered_params(output, label, following, command):
    offered = output.split(label, 1)[1].split(following, 1)[0].strip(" │")
    argv = shlex.split(offered)
    assert argv[:2] == ["soup", command]
    # Use the registered consumers, including their defaults and option names.
    consumer = get_command(app).commands[command]
    with consumer.make_context(command, argv[2:]) as ctx:
        return ctx.params


@pytest.fixture
def complete_training(tmp_path, monkeypatch):
    import soup_cli.commands.train as train_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("COLUMNS", "240")
    monkeypatch.setattr(train_mod.console, "width", 240)
    # Hardware and expensive training are the boundaries; config, local data,
    # run tracking, the completion panel and consumer argument parsing are real.
    monkeypatch.setattr(train_mod, "detect_device", lambda **kw: ("cpu", "test CPU"))
    monkeypatch.setattr(train_mod, "get_gpu_info", lambda **kw: {"memory_total": "0 GB"})
    (tmp_path / "base").mkdir()

    def complete(*, backend="mlx", task="sft", stop_reason=None, output_dir="./saved", **training):
        row = {"instruction": "Question", "output": "Answer"}
        fmt = "alpaca"
        if task in {"dpo", "orpo", "simpo", "preference"}:
            row = {"prompt": "Question", "chosen": "Answer", "rejected": "Wrong"}
            fmt = "dpo"
        elif task == "kto":
            row = {"prompt": "Question", "completion": "Answer", "label": True}
            fmt = "kto"
        (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        config = {
            "base": "./base", "backend": backend, "task": task, "output": "./configured",
            "data": {"train": "./train.jsonl", "format": fmt, "val_split": 0},
            "training": {"quantization": "none", "lora": {"r": 4}, **training},
        }
        if stop_reason == "loss_watchdog":
            config["training"]["loss_watchdog"] = True
        path = tmp_path / "soup.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        result = {
            # Consumers must target the artifact actually returned by the trainer.
            "output_dir": output_dir, "initial_loss": 2.0, "final_loss": 1.0,
            "total_steps": 1, "duration_secs": 1.0, "stop_reason": stop_reason,
        }
        trainer = SimpleNamespace(setup=lambda dataset: None, train=lambda **kw: result)
        monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", lambda *a, **kw: trainer)
        run = CliRunner().invoke(app, ["train", "--config", str(path), "--yes"])
        assert run.exit_code == 0, (run.output, repr(run.exception))
        output = _plain(run.output)
        title = "Training Stopped — Loss watchdog" if stop_reason else "Training Complete!"
        assert title in output
        return output

    return complete


@pytest.mark.parametrize("quantization,dequantize", [("none", False), ("4bit", True)])
@pytest.mark.parametrize("task,training", [
    ("sft", {}), ("dpo", {}), ("orpo", {}), ("simpo", {}), ("kto", {}), ("grpo", {}),
    ("preference", {"preference_loss": "dpo"}),
    ("preference", {"preference_loss": "orpo"}),
    ("preference", {"preference_loss": "simpo"}),
    ("sft", {"stop_reason": "loss_watchdog"}),
])
def test_mlx_lora_commands_select_native_consumers(
    complete_training, task, training, quantization, dequantize,
):
    output = complete_training(task=task, quantization=quantization, **training)
    chat = _offered_params(output, "Quick test:", "Push to HF:", "chat")
    merge = _offered_params(output, "Merge LoRA:", "Export GGUF:", "merge")
    assert (chat["backend"], chat["model"]) == ("mlx", "./saved")
    assert (merge["backend"], merge["adapter"]) == ("mlx", "./saved")
    assert merge["dequantize"] is dequantize


@pytest.mark.parametrize("stop_reason", [None, "loss_watchdog"])
def test_mlx_full_fine_tune_offers_native_chat_and_no_merge(complete_training, stop_reason):
    output = complete_training(lora={"r": 0}, stop_reason=stop_reason)
    chat = _offered_params(output, "Quick test:", "Push to HF:", "chat")
    assert (chat["backend"], chat["model"]) == ("mlx", "./saved")
    assert "Full fine-tuning output is a standalone model; no soup merge is needed" in output
    assert "Merge LoRA:" not in output


def test_transformers_suggestions_keep_the_default_consumers(complete_training):
    output = complete_training(backend="transformers")
    chat = _offered_params(output, "Quick test:", "Push to HF:", "chat")
    merge = _offered_params(output, "Merge LoRA:", "Export GGUF:", "merge")
    assert (chat["backend"], chat["model"]) == ("transformers", "./saved")
    assert (merge["backend"], merge["adapter"]) == ("transformers", "./saved")
    assert merge["dequantize"] is False
    assert "--backend" not in output


def test_relora_keeps_dense_guidance_and_default_chat(complete_training):
    output = complete_training(backend="transformers", relora_steps=2)
    chat = _offered_params(output, "Quick test:", "Push to HF:", "chat")
    assert (chat["backend"], chat["model"]) == ("transformers", "./saved")
    assert "ReLoRA output is already dense; no soup merge is needed" in output
    assert "Merge LoRA:" not in output
    assert "--backend" not in output


def test_completion_panel_shows_output_on_one_line(complete_training, monkeypatch):
    # Windows cannot create a directory whose name contains these characters.
    monkeypatch.setattr("soup_cli.utils.run_provenance.write_run_provenance", lambda *_: None)
    output = complete_training(output_dir="./runs\nnew\rline\tend")

    assert r"Output: ./runs\nnew\rline\tend" in output


@pytest.mark.parametrize("path, argument", [
    pytest.param("./owner's $adapter", "'./owner''s $adapter'", id="dollar"),
    pytest.param("./saved`adapter", "'./saved`adapter'", id="backtick"),
    pytest.param("./saved%TEMP%", "'./saved%TEMP%'", id="percent"),
    pytest.param("./saved!TEMP!", "'./saved!TEMP!'", id="delayed-expansion"),
    pytest.param("./owner’s“adapter”", "'./owner’’s“adapter”'", id="smart-quotes"),
])
def test_panel_labels_powershell_spelling_when_no_portable_form(
    complete_training, monkeypatch, path, argument,
):
    import soup_cli.commands.train as train_mod

    monkeypatch.setattr(train_mod, "_WINDOWS", True, raising=False)
    output = complete_training(output_dir=path)
    assert "Use PowerShell for the commands below." in output
    assert f"Quick test: soup chat --backend mlx --model {argument}" in output
    assert f"Merge LoRA: soup merge --backend mlx --adapter {argument}" in output
    assert "Use your shell's quoting rules" not in output


def test_panel_keeps_generic_prose_without_windows(complete_training, monkeypatch):
    import soup_cli.commands.train as train_mod

    monkeypatch.setattr(train_mod, "_WINDOWS", False, raising=False)
    output = complete_training(output_dir="./owner's $adapter")
    assert "Use PowerShell for the commands below." not in output
    assert "Use your shell's quoting rules for the Output path." in output
    assert "'./owner''s $adapter'" not in output


def test_panel_keeps_generic_prose_for_a_control_character(complete_training, monkeypatch):
    import soup_cli.commands.train as train_mod

    # paste_arg refuses control characters on every host; the panel falls back to prose.
    monkeypatch.setattr("soup_cli.utils.run_provenance.write_run_provenance", lambda *_: None)
    monkeypatch.setattr(train_mod, "_WINDOWS", True, raising=False)
    output = complete_training(output_dir="./runs\nnew\rline")
    assert "Use PowerShell for the commands below." not in output
    assert "Use your shell's quoting rules for the Output path." in output


def test_panel_fallback_full_fine_tune_keeps_the_no_merge_note(complete_training, monkeypatch):
    import soup_cli.commands.train as train_mod

    monkeypatch.setattr(train_mod, "_WINDOWS", True, raising=False)
    output = complete_training(output_dir="./owner's $adapter", lora={"r": 0})
    assert "Use PowerShell for the commands below." in output
    assert "Full fine-tuning output is a standalone model; no soup merge is needed" in output
    assert "Merge LoRA:" not in output


def test_panel_fallback_follows_the_dequantize_hint(complete_training, monkeypatch):
    import soup_cli.commands.train as train_mod

    monkeypatch.setattr(train_mod, "_WINDOWS", True, raising=False)
    output = complete_training(output_dir="./owner's $adapter", quantization="4bit")
    assert (
        "Merge LoRA: soup merge --backend mlx --adapter './owner''s $adapter' --dequantize"
        in output
    )
