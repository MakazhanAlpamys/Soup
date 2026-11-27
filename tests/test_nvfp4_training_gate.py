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

import sys
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

    @pytest.mark.parametrize(
        ("capability", "measured"),
        [((10, 1), False), ((11, 0), False), ((12, 0), True), ((12, 1), True)],
    )
    def test_the_message_says_which_refusals_were_measured(
        self, monkeypatch, capability, measured
    ):
        """SM 12.x was measured to fail its first backward; SM 10.1 and 11.0 are refused
        by the table without a measurement, and the message says so."""
        from soup_cli.utils.advanced_precision import nvfp4_training_refusal

        _card(monkeypatch, capability)
        reason = nvfp4_training_refusal()
        assert ("was not measured" in reason) is not measured
        assert ("fails to compile" in reason) is measured

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

    @pytest.mark.parametrize("converted", [0, 4])
    def test_a_quantize_that_converts_only_some_linears_is_refused(
        self, monkeypatch, converted
    ):
        """#3249: ``quantize_`` returned without raising, but left linears as
        ``nn.Linear``. Training on that model is training partly in bf16."""
        from soup_cli.utils.advanced_precision import (
            NVFP4PartialConversionError,
            apply_nvfp4,
        )

        _install_fake_torchao_quant(monkeypatch, {}, convert=converted)
        _card(monkeypatch, (10, 0))
        with pytest.raises(
            NVFP4PartialConversionError, match=rf"converted {converted} of 5 linears"
        ):
            apply_nvfp4(_tiny_attn_model())

    def test_the_linear_torchao_skips_on_purpose_is_not_counted(self, monkeypatch):
        """torchao's default filter leaves out ``NonDynamicallyQuantizableLinear``
        (``nn.MultiheadAttention.out_proj``), so a model that has one, with every
        other linear converted, is fully converted."""
        torch = _card(monkeypatch, (10, 0))
        from soup_cli.utils.advanced_precision import apply_nvfp4

        _install_fake_torchao_quant(monkeypatch, {})
        model = torch.nn.Sequential(
            torch.nn.MultiheadAttention(8, 2), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
        )
        assert apply_nvfp4(model) == 2
        assert type(model[0].out_proj).__name__ == "NonDynamicallyQuantizableLinear"

    def test_the_run_stops_on_sm_120(self, monkeypatch):
        """#3249: an explicit ``nvfp4: true`` the card cannot run stops the run, as FP8's
        refusals do (#835, #1152). It used to print a yellow line and train in bf16."""
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        record: dict = {}
        _install_fake_torchao_quant(monkeypatch, record)
        _card(monkeypatch, (12, 0))
        printed: list[str] = []
        console = types.SimpleNamespace(print=lambda text, **_k: printed.append(str(text)))
        tcfg = types.SimpleNamespace(nvfp4=True)
        with pytest.raises(RuntimeError, match=r"SM 12\.0"):
            apply_v028_speed_memory(
                model=_tiny_attn_model(), tcfg=tcfg, base_model="x", console=console,
            )
        assert record == {}
        assert printed == []

    def test_a_partial_conversion_stops_the_run(self, monkeypatch):
        """#3249: ``quantize_`` failing after it started leaves the model PARTIALLY
        quantised. The error says to restart; the run must not train on that model."""
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        record: dict = {}
        _install_fake_torchao_quant(monkeypatch, record)
        _card(monkeypatch, (10, 0))

        def _fails_partway(model, config):
            record["model"] = model
            raise ValueError("requires M, K, N all divisible by 128")

        monkeypatch.setattr(sys.modules["torchao.quantization"], "quantize_", _fails_partway)
        from soup_cli.utils.advanced_precision import NVFP4PartialConversionError

        with pytest.raises(NVFP4PartialConversionError, match="PARTIALLY quantised"):
            apply_v028_speed_memory(
                model=_tiny_attn_model(), tcfg=types.SimpleNamespace(nvfp4=True),
                base_model="x",
            )
        assert "model" in record, "the control: quantize_ really started"

    def test_a_torchao_without_the_training_config_stops_the_run(self, monkeypatch):
        """#3249: a supported card, but no torchao ``NVFP4TrainingConfig`` to train with."""
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        record: dict = {}
        _install_fake_torchao_quant(monkeypatch, record, with_nvfp4=False)
        _card(monkeypatch, (10, 0))
        with pytest.raises(RuntimeError, match="NVFP4Training"):
            apply_v028_speed_memory(
                model=_tiny_attn_model(), tcfg=types.SimpleNamespace(nvfp4=True),
                base_model="x",
            )
        assert record == {}


#: The clause every refusal ends with, as the user reads it (#3249).
_FIX = (
    "Remove training.nvfp4 to train in bf16, or run on SM 10.0 (B200/GB200) "
    "or SM 10.3 (B300/GB300)."
)


class _ReachedTrainError(Exception):
    pass


class TestSoupTrainStops:
    """#3249: ``soup train`` through ``run()``, on a tiny local llama, with the card
    and torchao stubbed. On ``main`` every case below printed a yellow line and
    reached the trainer's ``train``."""

    @staticmethod
    def _torchao(monkeypatch, *, missing=False, convert=None, fail=False):
        """Stub the resolver, not ``sys.modules``: a fake ``torchao`` module breaks
        transformers' own import (``torchao.__spec__ is None``)."""
        import torch

        from soup_cli.utils import torchao_compat

        class NVFP4TrainingConfig:
            pass

        class NVFP4Linear(torch.nn.Linear):
            pass

        def quantize_(model, config):
            linears = [m for m in model.modules() if type(m) is torch.nn.Linear]
            for module in linears[:convert]:
                module.__class__ = NVFP4Linear
            if fail:
                raise ValueError("requires M, K, N all divisible by 128")

        classes = {
            "NVFP4Training": NVFP4TrainingConfig,
            "NVFP4Linear": NVFP4Linear,
            "quantize_": quantize_,
        }

        def resolve(key):
            if missing:
                raise RuntimeError(
                    f"{torchao_compat.torchao_install_hint(key)} Could not import torchao "
                    "(ModuleNotFoundError: No module named 'torchao')."
                )
            return classes[key]

        monkeypatch.setattr(torchao_compat, "resolve_torchao_class", resolve)

    def _train(self, tmp_path, monkeypatch, capsys, *, capability, blackwell=True, **torchao):
        for module in ("torch", "transformers", "peft", "trl", "datasets"):
            pytest.importorskip(module, reason=f"{module} is only in the [train] extra")
        import transformers

        from soup_cli.cli import run
        from soup_cli.utils import advanced_precision
        from tests.conftest import strip_ansi
        from tests.test_issue804_chat_template_trainers import _tiny_llama_dir

        monkeypatch.chdir(tmp_path)
        weights = _tiny_llama_dir(tmp_path)
        (tmp_path / "train.jsonl").write_text(
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": "hello"}]}\n' * 4,
            encoding="utf-8",
        )
        (tmp_path / "soup.yaml").write_text(
            f"base: {weights.as_posix()}\ntask: sft\ndata:\n  train: train.jsonl\n"
            "training:\n  epochs: 1\n  quantization: none\n  nvfp4: true\n"
            "  lora: {r: 4, alpha: 8, target_modules: [q_proj, v_proj]}\noutput: ./out\n",
            encoding="utf-8",
        )
        # The card is stubbed at the gate, not on torch.cuda: a CUDA-looking torch
        # would send the tiny model to a device this host does not have.
        monkeypatch.setattr(advanced_precision, "is_blackwell_gpu", lambda: blackwell)
        monkeypatch.setattr(advanced_precision, "_cuda_capability", lambda: capability)
        self._torchao(monkeypatch, **torchao)
        trained: list[object] = []

        def spy(trainer, *_a, **_k):
            trained.append(trainer)
            raise _ReachedTrainError

        monkeypatch.setattr(transformers.Trainer, "train", spy)
        monkeypatch.setattr(sys, "argv", [
            "soup", "--no-audit-log", "--no-telemetry", "train", "--config", "soup.yaml",
            "--yes", "--allow-oom-attempt",
        ])
        with pytest.raises(SystemExit) as exit_info:
            run()
        captured = capsys.readouterr()
        out = " ".join(strip_ansi(captured.out + captured.err).split())
        return exit_info.value.code, out, trained

    @pytest.mark.parametrize(
        ("capability", "blackwell", "torchao", "reason"),
        [
            pytest.param(
                (12, 0), True, {}, "does not run on this SM 12.0 card", id="sm-12.0"
            ),
            pytest.param(None, False, {}, "no Blackwell device detected", id="no-blackwell"),
            pytest.param(
                (10, 0), True, {"missing": True},
                "NVFP4Training requires torchao (pip install", id="no-torchao",
            ),
            pytest.param(
                (10, 0), True, {"convert": 1, "fail": True},
                "PARTIALLY quantised (ValueError: requires M, K, N", id="quantize-raises",
            ),
            pytest.param(
                (10, 0), True, {"convert": 1}, "converted 1 of ", id="quantize-converts-some",
            ),
        ],
    )
    def test_the_run_stops_before_training_and_says_how_to_go_on(
        self, tmp_path, monkeypatch, capsys, capability, blackwell, torchao, reason
    ):
        code, out, trained = self._train(
            tmp_path, monkeypatch, capsys, capability=capability, blackwell=blackwell,
            **torchao,
        )
        assert code == 1, out
        assert reason in out, out
        # The fix clause ends the message: the next words are run()'s own footer.
        assert f"{_FIX} Run with --verbose for the full traceback." in out, out
        assert trained == [], "the trainer's train() was reached after the refusal"
        assert "NVFP4:" not in out, "the old yellow advisory came back"

    def test_a_full_conversion_reaches_training(self, tmp_path, monkeypatch, capsys):
        """Control: the same row on SM 10.0 with every linear converted reaches the
        trainer, so the stops above are not this row refusing everything."""
        _code, out, trained = self._train(
            tmp_path, monkeypatch, capsys, capability=(10, 0),
        )
        assert len(trained) == 1, out
        assert "NVFP4 quantisation applied" in out, out
        assert _FIX not in out


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
