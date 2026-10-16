"""Drive ``pincost.main()`` through its real CPU-testable decisions."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.conftest import strip_ansi
from tests.test_pincost import module as pincost

HIDDEN, VOCAB, LAYERS = 8, 32, 2
FAST, SLOW = 1.0, 4.0

def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())

class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.factor, None

class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(HIDDEN, HIDDEN, bias=False).requires_grad_(False)
        self.lora_A = nn.Linear(HIDDEN, 2, bias=False)
        self.lora_B = nn.Linear(2, HIDDEN, bias=False)
    def forward(self, x, factor=None):
        delta = self.lora_B(self.lora_A(x))
        if hasattr(self, "lora_extra"):
            delta = delta + self.lora_extra(x).sum() * 0.0
        if factor is not None:
            delta = _GradScale.apply(delta, factor)
        return torch.tanh(self.base(x) + delta)

class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(LAYERS))

class _Toy(nn.Module):
    def __init__(self, scenario: str, clock: SimpleNamespace, *, pin=None) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.config = SimpleNamespace(vocab_size=VOCAB, model_type="llama")
        self.embed = nn.Embedding(VOCAB, HIDDEN).requires_grad_(False)
        self.model = _Inner()
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False).requires_grad_(False)
        self.scenario, self.clock, self.pin = scenario, clock, pin
        self.streamed_calls = 0
        streamed = pin is not None
        if scenario == "extra_lora" and streamed:
            self.model.layers[0].lora_extra = nn.Linear(HIDDEN, 2, bias=False)
        if scenario == "inf_grads" or (
            scenario == "inf_grads_streamed" and streamed
        ) or (scenario == "inf_grads_reference" and not streamed):
            for name, parameter in self.named_parameters():
                if "lora_" in name:
                    parameter.register_hook(lambda g: torch.full_like(g, float("inf")))
    def forward(self, input_ids, labels):
        streamed = self.pin is not None
        factor = None
        if streamed:
            self.streamed_calls += 1
            slow = not self.pin
            if self.scenario == "inverted":
                slow = self.pin
            elif self.scenario == "no_cost":
                slow = False
            step = SLOW if slow else FAST
            self.clock.now += -step if self.scenario == "clock_backwards" else step
            if self.scenario == "wrong_first":
                factor = 3.0
            elif self.scenario == "pinned_wrong_all" and self.pin:
                factor = 3.0
            elif (
                self.scenario in {"wrong_after_first", "pinned_wrong_after_first"}
                and self.streamed_calls > 1
            ):
                factor = 3.0
            if self.scenario == "pinned_wrong_after_first" and not self.pin:
                factor = None
        x = self.embed(input_ids)
        for layer in self.model.layers:
            x = layer(x, factor)
        loss = nn.functional.cross_entropy(self.head(x).view(-1, VOCAB), labels.view(-1))
        if streamed and self.scenario == "loss_offset":
            loss = loss + 1e-3
        if self.scenario == "inf_loss" or (
            self.scenario == "inf_loss_streamed" and streamed
        ) or (self.scenario == "inf_loss_reference" and not streamed):
            loss = loss + torch.tensor(float("inf"))
        return SimpleNamespace(loss=loss)

class _Runtime:
    def __init__(self, pinned: bool) -> None:
        self.pinned = pinned
        self.source = SimpleNamespace(nbytes=1)
    def close(self) -> None:
        pass

def _run(monkeypatch, capsys, tmp_path, scenario: str) -> tuple[int, str]:
    import soup_cli.utils.layer_shard as layer_shard
    import soup_cli.utils.layer_stream_runtime as layer_runtime
    import soup_cli.utils.spectrum_scan as spectrum_scan
    clock = SimpleNamespace(now=0.0)
    def build_streamed(*args, **kwargs):
        pin = kwargs["pin"]
        reported = True if scenario == "lying_pinned" else pin
        runtime = _Runtime(reported)
        if scenario == "empty_store":
            runtime.source.nbytes = 0
        return _Toy(scenario, clock, pin=pin), runtime
    def cpu_non_vacuous(model) -> None:
        generator = torch.Generator(device="cpu").manual_seed(23)
        with torch.no_grad():
            for name, parameter in layer_runtime.canonical_named_parameters(model):
                if "lora_B" in name:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    monkeypatch.setattr(pincost, "DEVICE", "cpu")
    monkeypatch.setattr(pincost, "cuda_available", lambda: True)
    monkeypatch.setattr(pincost, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "cpu-stub")
    monkeypatch.setattr(spectrum_scan, "resolve_model_weights", lambda weights: str(weights))
    monkeypatch.setattr(layer_runtime, "build_meta_skeleton", lambda *a, **k: _Toy(scenario, clock))
    monkeypatch.setattr(layer_runtime, "quantised_layer_suffixes", lambda probe: ())
    monkeypatch.setattr(layer_runtime, "build_streamed_model", build_streamed)
    monkeypatch.setattr(layer_shard, "shard_checkpoint", lambda *a, **k: {})
    monkeypatch.setattr(
        pincost.bitexact,
        "load_resident_reference",
        lambda *a, **k: _Toy(scenario, clock),
    )
    monkeypatch.setattr(pincost.bitexact, "make_non_vacuous_lora", cpu_non_vacuous)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pincost.py", "--weights", str(tmp_path), "--shards",
            str(tmp_path / "shards"), "--seq", "4", "--warmup", "1",
            "--steps", "3", "--repeats", "2",
        ],
    )
    return pincost.main(), _plain(capsys.readouterr().out)

def test_a_real_cost_passes_with_the_arms_the_right_way_round(monkeypatch, capsys, tmp_path):
    code, out = _run(monkeypatch, capsys, tmp_path, "cost")
    assert code == 0, out
    assert "median pin=True 4.00 tok/s" in out
    assert "median pin=False 1.00 tok/s" in out
    assert "pin=True/pin=False ratio 4.00x" in out

@pytest.mark.parametrize("scenario, ratio", [("no_cost", "1.00x"), ("inverted", "0.25x")])
def test_no_cost_or_an_inverted_cost_fails(monkeypatch, capsys, tmp_path, scenario, ratio):
    code, out = _run(monkeypatch, capsys, tmp_path, scenario)
    assert code == 1, out
    assert f"pin=True/pin=False ratio {ratio}" in out
    assert "measured pinning advantage is absent or too small" in out

@pytest.mark.parametrize(
    "scenario, message",
    [
        ("loss_offset", "loss is not bit-exact before timing"),
        ("inf_loss", "produced a non-finite loss before timing"),
        ("inf_grads", "gradient is non-finite"),
        ("lying_pinned", "requested pin=False, but runtime reports pinned=True"),
    ],
)
def test_an_arm_that_fails_its_gate_is_never_reported(
    monkeypatch, capsys, tmp_path, scenario, message
):
    code, out = _run(monkeypatch, capsys, tmp_path, scenario)
    assert code == 1, out
    assert message in out
    assert "RESULT:" not in out

def test_wrong_pinned_arm_is_reported_but_can_still_be_timed(
    monkeypatch, capsys, tmp_path
):
    code, out = _run(monkeypatch, capsys, tmp_path, "pinned_wrong_after_first")
    assert code == 0, out
    assert "gradients pin=True 4/4,0/4,0/4 WRONG" in out
    assert "RESULT: historical-control pinning cost relationship reproduced" in out

def test_an_invalid_timing_exits_3(monkeypatch, capsys, tmp_path):
    code, out = _run(monkeypatch, capsys, tmp_path, "clock_backwards")
    assert code == 3, out
    assert "invalid measurement" in out

def test_verdict_refuses_a_ratio_inside_the_arms_own_spread():
    reproduced, _, _, ratio, spread = pincost.pinning_verdict([400.0, 100.0, 200.0], [100.0] * 3)
    assert ratio == pytest.approx(2.0)
    assert spread == pytest.approx(1.5)
    assert reproduced is False

@pytest.mark.parametrize("scenario", ["cost", "loss_offset"])
def test_historical_control_is_restored_after_measurement(monkeypatch, capsys, tmp_path, scenario):
    import soup_cli.utils.layer_stream_runtime as layer_runtime
    original = layer_runtime.install_dequant_forward
    _run(monkeypatch, capsys, tmp_path, scenario)
    assert layer_runtime.install_dequant_forward is original


@pytest.mark.parametrize("scenario", ["wrong_first", "wrong_after_first"])
def test_wrong_pageable_arm_refuses_timing_result(monkeypatch, capsys, tmp_path, scenario):
    code, out = _run(monkeypatch, capsys, tmp_path, scenario)
    assert code == 1, out
    assert "pin=False failed gradient correctness across all repetitions:" in out
    assert "median pin=True" not in out
    assert "RESULT:" not in out


def test_historical_pinned_arm_wrong_on_every_backward_can_be_timed(monkeypatch, capsys, tmp_path):
    code, out = _run(monkeypatch, capsys, tmp_path, "pinned_wrong_all")
    assert code == 0, out
    assert "gradients pin=True 0/4,0/4,0/4 WRONG" in out
    assert "gradients pin=False 4/4,4/4,4/4 OK" in out
    assert "median pin=True 4.00 tok/s" in out
    assert "pin=True/pin=False ratio 4.00x" in out
    assert "timing pin=True 4.00 tok/s, median_step=1.0000 s, gradients=WRONG" in out


@pytest.mark.parametrize(
    "scenario, message",
    [
        ("inf_loss_streamed", "produced a non-finite loss before timing"),
        ("inf_loss_reference", "produced a non-finite loss before timing"),
        ("inf_grads_streamed", "streamed gradient is non-finite"),
        ("inf_grads_reference", "reference gradient is non-finite"),
        ("empty_store", "reported an empty streaming store"),
        ("extra_lora", "gradient sets differ"),
    ],
)
def test_independent_measurement_guards(monkeypatch, capsys, tmp_path, scenario, message):
    code, out = _run(monkeypatch, capsys, tmp_path, scenario)
    assert code == 1, out
    assert message in out
    assert "RESULT:" not in out


def test_historical_control_is_installed_for_every_arm(monkeypatch, capsys, tmp_path):
    import soup_cli.utils.layer_stream_runtime as layer_runtime

    installed = []
    monkeypatch.setattr(pincost, "install_historical_control", installed.append)
    code, out = _run(monkeypatch, capsys, tmp_path, "cost")
    assert code == 0, out
    assert installed == [layer_runtime] * 4


def test_self_test_runs():
    assert pincost.run_self_test() == 0


def test_correctness_rejects_empty_canonical_intersection():
    streamed, reference = nn.Module(), nn.Module()
    streamed.register_parameter("streamed_only", nn.Parameter(torch.ones(1)))
    reference.register_parameter("reference_only", nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="no canonical parameter name is shared"):
        pincost.run_correctness(streamed, reference, torch.ones(1), pin=True)
