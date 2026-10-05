"""#1631 — the ``soup train`` VRAM pre-flight checks ``quantization: mxfp4`` runs.

``mxfp4`` is loaded with ``Mxfp4Config(dequantize=True)`` and trains in bf16, so it
needs the memory of the ``none`` run (#1466). ``_build_hardware_fit_input`` mapped
only ``none`` / ``4bit`` / ``8bit``, returned ``None`` for ``mxfp4`` and the gate was
skipped: a GPT-OSS-20B run was let through on a 24 GB card that refuses the same
config with ``quantization: none``.

No GPU here: the gate is called with a card size.
"""

from __future__ import annotations

import pytest
import typer

from soup_cli.commands import train as train_mod
from soup_cli.commands.train import _build_hardware_fit_input, _hardware_fit_preflight
from soup_cli.config.loader import load_config_from_string

_YAML = """
base: openai/gpt-oss-20b
task: sft
data:
  train: train.jsonl
  max_length: 2048
training:
  batch_size: 1
  quantization: {quant}
"""

SMALL_CARD = {"memory_total_bytes": int(24e9)}
LARGE_CARD = {"memory_total_bytes": int(96e9)}


def _cfg(quant: str):
    return load_config_from_string(_YAML.format(quant=quant))


class _Console:
    def __init__(self):
        self.printed = []

    def print(self, renderable):
        self.printed.append(renderable)


@pytest.fixture
def console(monkeypatch):
    fake = _Console()
    monkeypatch.setattr(train_mod, "console", fake)
    return fake


class TestTheInput:
    def test_mxfp4_gets_an_input(self):
        inp = _build_hardware_fit_input(_cfg("mxfp4"))
        assert inp is not None
        assert inp.quant == "mxfp4"

    def test_mxfp4_is_budgeted_as_lora_on_a_frozen_base(self):
        # Not qlora: nothing stays quantized once the base is dequantized.
        assert _build_hardware_fit_input(_cfg("mxfp4")).peft == "lora"

    def test_mxfp4_differs_from_none_only_in_the_quant_name(self):
        mxfp4 = _build_hardware_fit_input(_cfg("mxfp4"))
        none = _build_hardware_fit_input(_cfg("none"))
        assert (mxfp4.params_b, mxfp4.seq_len, mxfp4.batch_size, mxfp4.peft) == (
            none.params_b,
            none.seq_len,
            none.batch_size,
            none.peft,
        )

    @pytest.mark.parametrize("quant", ["gptq", "awq", "aqlm", "eetq", "fp8", "hqq:4bit"])
    def test_the_formats_that_stay_packed_are_still_skipped(self, quant):
        assert _build_hardware_fit_input(_cfg(quant)) is None


@pytest.mark.parametrize("quant", ["mxfp4", "none"])
class TestTheGate:
    """``none`` runs beside ``mxfp4`` as the control: the two must agree."""

    def test_a_card_too_small_for_the_bf16_model_is_refused(self, quant, console):
        with pytest.raises(typer.Exit) as exit_info:
            _hardware_fit_preflight(_cfg(quant), SMALL_CARD, allow_oom_attempt=False)
        assert exit_info.value.exit_code == 1
        assert len(console.printed) == 1

    def test_allow_oom_attempt_warns_and_launches(self, quant, console):
        _hardware_fit_preflight(_cfg(quant), SMALL_CARD, allow_oom_attempt=True)
        assert len(console.printed) == 1

    def test_a_card_that_fits_is_let_through_without_a_word(self, quant, console):
        _hardware_fit_preflight(_cfg(quant), LARGE_CARD, allow_oom_attempt=False)
        assert console.printed == []


def test_mxfp4_and_none_are_refused_at_the_same_prediction(console):
    panels = []
    for quant in ("mxfp4", "none"):
        with pytest.raises(typer.Exit):
            _hardware_fit_preflight(_cfg(quant), SMALL_CARD, allow_oom_attempt=False)
        panels.append(console.printed[-1].renderable)
    assert panels[0] == panels[1]
    assert "Predicted peak VRAM" in panels[0]
