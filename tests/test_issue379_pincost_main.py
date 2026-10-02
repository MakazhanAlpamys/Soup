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
        if scenario == "inf_grads":
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
        if self.scenario == "inf_loss":
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
        return _Toy(scenario, clock, pin=pin), _Runtime(reported)
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
        ("wrong_first", "pin=False failed gradient correctness across all repetitions"),
        ("lying_pinned", "requested pin=False, but runtime reports pinned=True"),
        ("wrong_after_first", "pin=False failed gradient correctness across all repetitions"),
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
