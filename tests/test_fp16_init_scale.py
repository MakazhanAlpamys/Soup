"""``training.fp16_init_scale`` sets the fp16 GradScaler's starting scale (rnd PR #67, ported).

torch starts the scaler at 2**16 and halves it on every step whose gradients overflow;
the step is skipped. A model that overflows well below that start skips its first
optimizer steps before it trains at all. The setting starts the scaler lower.

The schema tests load real configs. The runtime tests use a real CPU ``GradScaler``
(torch's scaler code is device-generic; the CUDA start is the same call with
``"cuda"``), so the replacement and its policy are checked without a GPU.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string


def _cfg(training_extra: dict | None = None, *, task: str = "sft", data_format: str = "alpaca"):
    raw = {
        "base": "org/m",
        "task": task,
        "data": {"train": "./x.jsonl", "format": data_format},
        "training": {"epochs": 1, **(training_extra or {})},
    }
    return load_config_from_string(yaml.safe_dump(raw))


def _msg(text: str) -> str:
    return re.escape(text)


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def test_default_is_unset():
    assert _cfg().training.fp16_init_scale is None


@pytest.mark.parametrize("value", [1, 256, 65536])
def test_accepts_a_power_of_two_in_range(value):
    assert _cfg({"fp16_init_scale": value}).training.fp16_init_scale == value


@pytest.mark.parametrize("value", [0, -4, 3, 300, 131072])
def test_refuses_a_value_that_is_not_a_power_of_two_in_range(value):
    with pytest.raises(ValueError, match=_msg(
        f"training.fp16_init_scale must be a power of two from 1 to 65536, got {value}"
    )):
        _cfg({"fp16_init_scale": value})


@pytest.mark.parametrize("value", [True, False])
def test_refuses_a_bool(value):
    # The TrainingConfig wildcard bool guard runs before the range check.
    with pytest.raises(ValueError, match=_msg("fp16_init_scale must be an int, not a bool")):
        _cfg({"fp16_init_scale": value})


@pytest.mark.parametrize("task, data_format", [("dpo", "dpo"), ("grpo", "alpaca")])
def test_refused_outside_transformers_sft(task, data_format):
    with pytest.raises(ValueError, match=_msg(
        "training.fp16_init_scale is applied only with task='sft' and "
        f"backend='transformers' (got task='{task}', backend='transformers'); "
        "remove it here."
    )):
        _cfg({"fp16_init_scale": 256}, task=task, data_format=data_format)


def test_sft_on_transformers_accepts_it():
    cfg = _cfg({"fp16_init_scale": 256})
    assert (cfg.task, cfg.backend, cfg.training.fp16_init_scale) == ("sft", "transformers", 256)


# --------------------------------------------------------------------------
# runtime: apply_fp16_init_scale on a real CPU GradScaler
# --------------------------------------------------------------------------


def _trainer(scaler):
    return SimpleNamespace(accelerator=SimpleNamespace(scaler=scaler))


def test_replaces_the_scaler_with_the_start_scale_and_keeps_its_device_and_policy():
    torch = pytest.importorskip("torch")
    from soup_cli.utils.mixed_precision import apply_fp16_init_scale

    old = torch.amp.GradScaler(
        "cpu", init_scale=65536.0, growth_factor=3.0, backoff_factor=0.25, growth_interval=7,
    )
    trainer = _trainer(old)

    assert apply_fp16_init_scale(trainer, 256) is True

    new = trainer.accelerator.scaler
    assert new is not old
    assert new.get_scale() == 256.0
    assert (new._device, new._growth_factor, new._backoff_factor, new._growth_interval) == (
        "cpu", 3.0, 0.25, 7,
    )
    assert new._enabled is True


def test_a_disabled_scaler_is_left_alone():
    torch = pytest.importorskip("torch")
    from soup_cli.utils.mixed_precision import apply_fp16_init_scale

    old = torch.amp.GradScaler("cpu", enabled=False)
    trainer = _trainer(old)

    assert apply_fp16_init_scale(trainer, 256) is False
    assert trainer.accelerator.scaler is old


def test_a_scaler_subclass_is_left_alone():
    """FSDP's ShardedGradScaler (and the XLA / NPU ones) are subclasses; a plain scaler
    rebuilt in their place would not reduce the overflow check the same way."""
    torch = pytest.importorskip("torch")
    from soup_cli.utils.mixed_precision import apply_fp16_init_scale

    class _ShardedLike(torch.amp.GradScaler):
        pass

    old = _ShardedLike("cpu", init_scale=65536.0)
    trainer = _trainer(old)

    assert apply_fp16_init_scale(trainer, 256) is False
    assert trainer.accelerator.scaler is old
    assert old.get_scale() == 65536.0


def test_without_a_scaler_it_changes_nothing():
    from soup_cli.utils.mixed_precision import apply_fp16_init_scale

    assert apply_fp16_init_scale(_trainer(None), 256) is False
    assert apply_fp16_init_scale(SimpleNamespace(), 256) is False


# --------------------------------------------------------------------------
# runtime: the SFT wrapper's one line of output per outcome
# --------------------------------------------------------------------------


@pytest.fixture()
def sft_console(monkeypatch):
    from rich.console import Console

    from soup_cli.trainer import sft

    console = Console(record=True, force_terminal=False, width=10_000, soft_wrap=True)
    monkeypatch.setattr(sft, "console", console)
    return console


def _wrapper(trainer):
    from soup_cli.trainer.sft import SFTTrainerWrapper

    wrapper = SFTTrainerWrapper.__new__(SFTTrainerWrapper)
    wrapper.trainer = trainer
    return wrapper


def test_wrapper_names_the_start_scale_when_it_applies(sft_console):
    torch = pytest.importorskip("torch")

    wrapper = _wrapper(_trainer(torch.amp.GradScaler("cpu", init_scale=65536.0)))
    wrapper._apply_fp16_init_scale(SimpleNamespace(fp16_init_scale=256))

    assert "fp16 GradScaler: starts at 256" in sft_console.export_text()
    assert wrapper.trainer.accelerator.scaler.get_scale() == 256.0


def test_wrapper_says_the_key_was_ignored_when_there_is_no_plain_scaler(sft_console):
    wrapper = _wrapper(_trainer(None))
    wrapper._apply_fp16_init_scale(SimpleNamespace(fp16_init_scale=256))

    out = sft_console.export_text()
    assert "training.fp16_init_scale ignored" in out
    assert "a bf16 run, or fp16 under DeepSpeed or FSDP" in out
    assert "starts at" not in out


def test_wrapper_is_silent_when_the_key_is_unset(sft_console):
    torch = pytest.importorskip("torch")

    old = torch.amp.GradScaler("cpu", init_scale=65536.0)
    wrapper = _wrapper(_trainer(old))
    wrapper._apply_fp16_init_scale(SimpleNamespace(fp16_init_scale=None))

    assert sft_console.export_text() == ""
    assert wrapper.trainer.accelerator.scaler is old
