"""#1152 — an explicitly requested FP8 conversion that fails stops the run.

#1154 made a card that cannot run FP8, and a missing torchao, stop at setup. Three
failures were still turned into a yellow line while training went on, under a
config that said FP8 (ruled on #1152, 2026-09-22 and 2026-09-24):

- row 3: ``quantization_aware: fp8``, the float8 conversion raises -> trained bf16;
- row 4: ``fp8_attention``, the conversion fails partway -> trained the
  half-converted model that ``apply_fp8_attention``'s own error says not to;
- row 5: ``fp8_attention`` on a model with no attention projections -> trained
  on as the no-op its error says it refuses.

Every trainer's FP8 goes through ``apply_v028_speed_memory``, so that is where
these are exercised, plus SFT's call into it. The card is patched and torchao is
the #835 file's stub; nothing here needs a GPU.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tests import test_issue835_fp8_gate as _gate  # noqa: E402

_Attn, _card = _gate._Attn, _gate._card
converts = _gate.converts  # the #835 torchao stub, as a fixture here too


def _console():
    from io import StringIO

    from rich.console import Console

    out = StringIO()
    return out, Console(file=out, width=400, force_terminal=False, color_system=None)


def _apply(tcfg, console):
    from soup_cli.utils.v028_features import apply_v028_speed_memory

    return apply_v028_speed_memory(
        model=_Attn(), tcfg=tcfg, base_model="m", console=console, device="cuda"
    )


def _break_the_converter(monkeypatch, error):
    def _raise(*_a, **_k):
        raise error

    monkeypatch.setattr(sys.modules["torchao.float8"], "convert_to_float8_training", _raise)


class _NoAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Linear(4, 4)


class TestAFailedConversionStops:
    def test_row3_quantization_aware_stops_naming_torchao_s_error(self, monkeypatch, converts):
        from soup_cli.config.schema import TrainingConfig

        _card(monkeypatch, (9, 0))
        _break_the_converter(monkeypatch, RuntimeError("scaled_mm needs K % 16 == 0"))
        out, console = _console()

        with pytest.raises(RuntimeError, match="float8 conversion failed") as info:
            _apply(TrainingConfig(quantization_aware="fp8"), console)

        assert "scaled_mm needs K % 16 == 0" in str(info.value)
        assert "torchao.float8 missing" not in str(info.value)
        assert "enabled" not in out.getvalue()

    def test_row4_a_partial_fp8_attention_conversion_stops(self, monkeypatch, converts):
        from soup_cli.config.schema import TrainingConfig

        _card(monkeypatch, (9, 0))
        monkeypatch.setattr("soup_cli.utils.fp8.apply_fp8_training", lambda *_a, **_k: True)
        _break_the_converter(monkeypatch, RuntimeError("out of memory"))
        out, console = _console()

        with pytest.raises(RuntimeError, match="PARTIALLY converted"):
            _apply(TrainingConfig(quantization_aware="fp8", fp8_attention=True), console)

        assert "FP8 attention" not in out.getvalue()

    def test_row5_no_attention_projections_stops(self, monkeypatch, converts):
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        _card(monkeypatch, (9, 0))
        monkeypatch.setattr("soup_cli.utils.fp8.apply_fp8_training", lambda *_a, **_k: True)
        out, console = _console()

        with pytest.raises(ValueError, match="no attention projections"):
            apply_v028_speed_memory(
                model=_NoAttention(),
                tcfg=TrainingConfig(quantization_aware="fp8", fp8_attention=True),
                base_model="m",
                console=console,
                device="cuda",
            )

        assert "FP8 attention" not in out.getvalue()

    def test_sft_stops_at_setup(self, monkeypatch, converts):
        """Through ``SFTTrainerWrapper._apply_quantization_aware``, as #1154's stop."""
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.trainer.sft import SFTTrainerWrapper

        _card(monkeypatch, (9, 0))
        _break_the_converter(monkeypatch, RuntimeError("boom"))
        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.model = _Attn()
        wrapper.config = SimpleNamespace(base="org/model", backend="transformers")
        wrapper.device = "cuda:0"

        with pytest.raises(RuntimeError, match="float8 conversion failed"):
            wrapper._apply_quantization_aware(TrainingConfig(quantization_aware="fp8"))


class TestTheSuccessfulPathIsUnchanged:
    """Controls: a card that runs FP8 with torchao present still converts."""

    def test_quantization_aware_converts(self, monkeypatch, converts):
        from soup_cli.config.schema import TrainingConfig

        _card(monkeypatch, (9, 0))
        out, console = _console()

        applied = _apply(TrainingConfig(quantization_aware="fp8"), console)

        assert applied["fp8"] is True and converts == ["tensorwise"]
        assert "FP8 training enabled" in out.getvalue()

    def test_fp8_attention_converts(self, monkeypatch, converts):
        from soup_cli.config.schema import TrainingConfig

        _card(monkeypatch, (9, 0))
        out, console = _console()

        applied = _apply(TrainingConfig(quantization_aware="fp8", fp8_attention=True), console)

        assert applied["fp8"] is True and applied["fp8_attention"] is True
        assert "FP8 attention enabled (4 projections)" in out.getvalue()


class TestEveryConversionErrorNamesFp8:
    """torchao 0.18's own conversion path raises AssertionError (swap_linear_layers,
    a root nn.Linear with children) and TypeError (Float8Linear.from_float, partway
    through the swap), not only RuntimeError/ValueError."""

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("scaled_mm needs K % 16 == 0"),
            ValueError("bad shape"),
            AssertionError("Does not support a root nn.Linear with children"),
            TypeError("cannot assign 'torch.FloatTensor' as parameter 'weight'"),
        ],
        ids=lambda e: type(e).__name__,
    )
    def test_row3_every_conversion_error_names_fp8(self, monkeypatch, converts, error):
        from soup_cli.config.schema import TrainingConfig

        _card(monkeypatch, (9, 0))
        _break_the_converter(monkeypatch, error)
        out, console = _console()

        with pytest.raises(RuntimeError, match="float8 conversion failed") as info:
            _apply(TrainingConfig(quantization_aware="fp8"), console)

        assert f"{type(error).__name__}: {error}" in str(info.value)
        assert info.value.__cause__ is error
        assert "enabled" not in out.getvalue()
