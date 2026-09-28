"""#1324 -- ``training.loraplus_lr_ratio`` is refused at config load on ``backend: mlx``.

The MLX trainer builds its adapter with mlx-lm's ``linear_to_lora_layers`` and its
optimizer from one scalar learning rate (``plan_optimizer`` in
``trainer/mlx_optim.py``), and ``loraplus`` occurs nowhere in the MLX trainers. So a
``backend: mlx`` config with ``loraplus_lr_ratio`` loaded, passed ``soup train
--dry-run`` and ``soup doctor --config``, and then trained plain LoRA with every
tensor at one learning rate. ``training.use_lorafa``, the sibling optimizer knob, is
already refused there at load.

This file pins:

* the refusal, through the loader and through direct construction, with a message
  that names the setting, the backend, why, and how to make the config load;
* ``soup train --dry-run`` and ``soup doctor --config`` stop at config load;
* the controls that must NOT change: ``backend: transformers`` and ``backend:
  unsloth`` keep the ratio (#1080 wired both), and an MLX config without it loads;
* ``docs/peft-and-efficiency.md`` states the MLX behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.config.schema import SoupConfig
from tests.conftest import strip_ansi

REPO_ROOT = Path(__file__).resolve().parents[1]

_MLX_BASE = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def _plain(text: "str | None") -> str:
    """Rich colours per character and wraps, so compare stripped + collapsed."""
    return " ".join(strip_ansi(text).split())


def _yaml(*, backend: str = "mlx", task: str = "sft", ratio: "str | None" = "16") -> str:
    loraplus = f"  loraplus_lr_ratio: {ratio}\n" if ratio is not None else ""
    return (
        f"base: {_MLX_BASE}\ntask: {task}\nbackend: {backend}\n"
        "data:\n  train: ./data/train.jsonl\n  format: alpaca\n"
        "training:\n  epochs: 1\n  lr: 1.0e-4\n"
        f"{loraplus}"
        "  lora:\n    r: 8\n    alpha: 16\n"
        "output: ./output\n"
    )


def _assert_is_the_1324_refusal(message: str) -> None:
    flat = " ".join(message.split())
    assert "training.loraplus_lr_ratio is not implemented on backend='mlx'" in flat, flat


# --------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------


def test_the_message_names_the_setting_the_backend_the_reason_and_the_remedy():
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml())
    message = " ".join(str(excinfo.value).split())

    _assert_is_the_1324_refusal(message)
    # The reason: what the MLX optimizer does with the ratio.
    assert "one learning rate" in message
    assert "silently ignored" in message
    # How to make the config load again.
    assert "Remove training.loraplus_lr_ratio" in message
    assert "backend: transformers" in message


@pytest.mark.parametrize("ratio", ["16", "1.0", "0.5", "2"])
def test_every_ratio_is_refused_on_mlx(ratio):
    """The refusal is about the backend, not the size of the ratio."""
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_yaml(ratio=ratio))
    _assert_is_the_1324_refusal(str(excinfo.value))


def test_direct_construction_is_refused_too():
    """``soup sweep``, ``soup rewind`` and autopilot build the model directly."""
    with pytest.raises(ValueError) as excinfo:
        SoupConfig(
            base=_MLX_BASE,
            task="sft",
            backend="mlx",
            data={"train": "t.jsonl"},
            training={"loraplus_lr_ratio": 16.0},
        )
    _assert_is_the_1324_refusal(str(excinfo.value))


def _write_config(tmp_path: Path) -> Path:
    train = tmp_path / "train.jsonl"
    train.write_text(
        '{"instruction": "q", "output": "a"}\n' * 4,
        encoding="utf-8",
    )
    path = tmp_path / "soup.yaml"
    path.write_text(
        f"base: {_MLX_BASE}\ntask: sft\nbackend: mlx\n"
        f"data:\n  train: {train.as_posix()}\n  format: alpaca\n  val_split: 0\n"
        "training:\n  epochs: 1\n  lr: 1.0e-4\n  loraplus_lr_ratio: 16\n"
        "  lora: {r: 8, alpha: 16}\n"
        f"output: {(tmp_path / 'out').as_posix()}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("extra_args", [[], ["--dry-run"]], ids=["train", "dry-run"])
def test_soup_train_stops_at_config_load(tmp_path, monkeypatch, extra_args):
    """Before the fix ``--dry-run`` printed "Config valid. Ready to train!"."""
    from soup_cli.cli import app

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path)

    result = CliRunner().invoke(app, ["train", "--config", str(config), *extra_args])

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "loraplus_lr_ratio is not implemented on backend='mlx'" in output, output
    for later_line in ("Config valid.", "Ready to train", "Data OK", "Training Setup"):
        assert later_line not in output, (later_line, output)


def test_soup_doctor_config_reports_the_config_as_unloadable(tmp_path, capsys):
    """Before the fix doctor printed its all-clear for this config (exit 0); now it
    cannot load, and an unloadable config is exit 2."""
    from soup_cli.commands.doctor import doctor

    config = _write_config(tmp_path)

    with pytest.raises(typer.Exit) as excinfo:
        doctor(nccl=False, disk=False, config=str(config))

    assert excinfo.value.exit_code == 2
    output = _plain(capsys.readouterr().out)
    assert "loraplus_lr_ratio is not implemented on backend='mlx'" in output, output


# --------------------------------------------------------------------------
# controls -- these pass with and without the fix, by design
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["transformers", "unsloth"])
@pytest.mark.parametrize("task", ["sft", "pretrain", "dpo"])
def test_the_wired_backends_keep_the_ratio(backend, task):
    """#1080 wired LoRA+ after the backend branch of these trainers, so both
    backends still load the same field, and keep its value."""
    cfg = load_config_from_string(_yaml(backend=backend, task=task))
    assert cfg.backend == backend
    assert cfg.training.loraplus_lr_ratio == 16.0


def test_an_mlx_config_without_the_ratio_still_loads():
    """The refusal is the ratio's, not the rest of the config's."""
    cfg = load_config_from_string(_yaml(ratio=None))
    assert cfg.backend == "mlx"
    assert cfg.training.loraplus_lr_ratio is None


def test_an_explicit_null_ratio_still_loads_on_mlx():
    cfg = load_config_from_string(_yaml(ratio="null"))
    assert cfg.training.loraplus_lr_ratio is None


# --------------------------------------------------------------------------
# docs
# --------------------------------------------------------------------------


def test_the_loraplus_docs_state_the_mlx_refusal():
    text = (REPO_ROOT / "docs" / "peft-and-efficiency.md").read_text(encoding="utf-8")
    section = text.split("## LoRA+ (Differentiated Learning Rates)", 1)[1]
    section = section.split("\n## ", 1)[0]
    flat = " ".join(section.split())
    assert "Refused on `backend: mlx`" in flat, flat
    assert "`backend: unsloth` gets the same LoRA+ optimizer" in flat, flat
