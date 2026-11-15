"""#2867: ``training.nvfp4`` is gated on the cards torchao's NVFP4 training compiles for.

torchao's ``NVFP4TrainingConfig`` quantises the backward's gradients with stochastic
rounding, ``cvt.rs.satfinite.e2m1x4.f32``. ptxas assembles that for sm_100a and
sm_103a and refuses it for sm_101a, sm_120a and sm_121a, so on an RTX 50-series or
RTX PRO Blackwell card (SM 12.0) the old ``SM >= 10`` gate let the model convert and
the first backward then failed to compile. ``soup doctor`` said ``NVFP4:
hardware=yes`` on the same card.

The CPU tests stub the capability and torchao; the ``gpu`` test asks the real card
whether torchao's NVFP4 backward runs, and checks that the gate agrees with it.
"""

from __future__ import annotations

import types

import pytest

from tests.test_v07121 import _install_fake_torchao_quant, _tiny_attn_model

#: SM 10.0 (B200/GB200) and SM 10.3 (B300/GB300) assemble the instruction.
_TRAINS = [(10, 0), (10, 3)]
#: Blackwell parts that are refused: 10.1 (Thor's pre-CUDA-13 number), 12.0 and 12.1 refuse
#: the instruction in ptxas; 11.0 is refused by the allow-list, not by a measurement (the
#: ptxas triton 3.6.0 ships does not know sm_110a).
_REFUSED_BLACKWELL = [(10, 1), (11, 0), (12, 0), (12, 1)]


def _card(monkeypatch, capability):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: capability)
    return torch


class TestTheGate:
    @pytest.mark.parametrize("capability", _TRAINS)
    def test_sm_100_and_103_pass(self, monkeypatch, capability):
        from soup_cli.utils.advanced_precision import (
            is_nvfp4_training_gpu,
            nvfp4_training_refusal,
        )

        _card(monkeypatch, capability)
        assert nvfp4_training_refusal() is None
        assert is_nvfp4_training_gpu() is True

    @pytest.mark.parametrize("capability", _REFUSED_BLACKWELL)
    def test_other_blackwell_cards_are_refused_by_capability(self, monkeypatch, capability):
        from soup_cli.utils.advanced_precision import (
            is_blackwell_gpu,
            is_nvfp4_training_gpu,
            nvfp4_training_refusal,
        )

        _card(monkeypatch, capability)
        assert is_blackwell_gpu() is True  # still Blackwell: only NVFP4 training is refused
        reason = nvfp4_training_refusal()
        assert reason is not None
        assert f"SM {capability[0]}.{capability[1]}" in reason
        assert "cvt.rs" in reason
        assert is_nvfp4_training_gpu() is False

    def test_hopper_keeps_the_no_blackwell_message(self, monkeypatch):
        from soup_cli.utils.advanced_precision import nvfp4_training_refusal

        _card(monkeypatch, (9, 0))
        assert "no Blackwell device detected" in nvfp4_training_refusal()


class TestApplyNvfp4:
    @pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
    def test_sm_12x_is_refused_before_anything_converts(self, monkeypatch, capability):
        """On ``main`` this converted the model, and the run died at its first backward."""
        from soup_cli.utils.advanced_precision import apply_nvfp4

        record: dict = {}
        _install_fake_torchao_quant(monkeypatch, record)
        _card(monkeypatch, capability)
        with pytest.raises(RuntimeError, match=rf"SM {capability[0]}\.{capability[1]}"):
            apply_nvfp4(_tiny_attn_model())
        assert record == {}, "quantize_ ran on a card that cannot train NVFP4"

    @pytest.mark.parametrize("capability", _TRAINS)
    def test_sm_100_and_103_still_convert(self, monkeypatch, capability):
        from soup_cli.utils.advanced_precision import apply_nvfp4

        record: dict = {}
        _install_fake_torchao_quant(monkeypatch, record)
        _card(monkeypatch, capability)
        model = _tiny_attn_model()
        assert apply_nvfp4(model) == 5
        assert record["model"] is model

    def test_the_run_gets_the_documented_advisory_on_sm_120(self, monkeypatch):
        """The refusal takes the existing NVFP4 path: a yellow line, nvfp4 not applied."""
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        record: dict = {}
        _install_fake_torchao_quant(monkeypatch, record)
        _card(monkeypatch, (12, 0))
        printed: list[str] = []
        console = types.SimpleNamespace(print=lambda text, **_k: printed.append(str(text)))
        tcfg = types.SimpleNamespace(nvfp4=True)
        applied = apply_v028_speed_memory(
            model=_tiny_attn_model(), tcfg=tcfg, base_model="x", console=console,
        )
        assert applied["nvfp4"] is False
        assert record == {}
        assert any("NVFP4" in line and "SM 12.0" in line for line in printed), printed


class TestDoctor:
    @pytest.mark.parametrize(
        ("capability", "expected"),
        [((12, 0), False), ((12, 1), False), ((10, 0), True), ((10, 3), True), ((9, 0), False)],
    )
    def test_precision_panel_reports_the_nvfp4_training_cards(
        self, monkeypatch, capability, expected,
    ):
        from soup_cli.commands.doctor import _get_precision_capabilities

        torch = _card(monkeypatch, capability)
        _install_fake_torchao_quant(monkeypatch, {})
        hardware, _software = _get_precision_capabilities(torch)["NVFP4"]
        assert hardware is expected


@pytest.mark.gpu
def test_the_gate_agrees_with_torchao_nvfp4_training_on_this_card():
    """Run one NVFP4 forward+backward with the real torchao and compare with the gate.

    On an SM 12.0 card ``main`` reported ``NVFP4: hardware=yes`` and ``apply_nvfp4``
    converted the model, while the backward failed to compile (PTXASError).
    """
    torch = pytest.importorskip("torch")
    training = pytest.importorskip("torchao.prototype.moe_training.nvfp4_training.nvfp4_training")
    from torchao.quantization import quantize_

    from soup_cli.commands.doctor import _get_precision_capabilities
    from soup_cli.utils.advanced_precision import apply_nvfp4

    probe = torch.nn.Sequential(torch.nn.Linear(256, 256, bias=False)).cuda().bfloat16()
    quantize_(probe, training.NVFP4TrainingConfig())
    x = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    try:
        probe(x).float().sum().backward()
        torch.cuda.synchronize()
        trains = True
    except Exception:  # noqa: BLE001 - PTXASError on SM 12.x; any failure means "cannot train"
        trains = False

    hardware, _software = _get_precision_capabilities(torch)["NVFP4"]
    assert hardware is trains, f"doctor says hardware={hardware}, torchao trains={trains}"

    model = torch.nn.Sequential(torch.nn.Linear(256, 256, bias=False)).cuda().bfloat16()
    if trains:
        assert apply_nvfp4(model) == 1
    else:
        with pytest.raises(RuntimeError, match="NVFP4 training does not run"):
            apply_nvfp4(model)
        assert type(model[0]).__name__ == "Linear"
