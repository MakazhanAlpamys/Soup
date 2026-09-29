"""CPU tests of repeat_backward.main() with toy LoRA models."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness"
sys.path.insert(0, str(HARNESS))
import mechanism_cost  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "repeat_backward_main", HARNESS / "repeat_backward.py"
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

LAYERS = 8
HIDDEN = 16
VOCAB = 64


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.factor, None


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(HIDDEN, HIDDEN, bias=False).requires_grad_(False)
        self.lora_A = nn.Linear(HIDDEN, 4, bias=False)
        self.lora_B = nn.Linear(4, HIDDEN, bias=False)

    def forward(self, x, factor=None):
        delta = self.lora_B(self.lora_A(x))
        if factor is not None:
            delta = _GradScale.apply(delta, factor)
        return torch.tanh(self.base(x) + delta)


class _Out:
    def __init__(self, loss):
        self.loss = loss


class _Config:
    vocab_size = VOCAB


class _Toy(nn.Module):
    def __init__(self, wrong_from=None, loss_shift=0.0, grad_factor=3.0):
        super().__init__()
        torch.manual_seed(0)
        self.config = _Config()
        self.embed = nn.Embedding(VOCAB, HIDDEN).requires_grad_(False)
        self.layers = nn.ModuleList(_Layer() for _ in range(LAYERS))
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False).requires_grad_(False)
        self.wrong_from = wrong_from
        self.loss_shift = loss_shift
        self.grad_factor = grad_factor
        self.calls = 0

    def forward(self, input_ids, labels):
        self.calls += 1
        wrong = self.wrong_from is not None and self.calls >= self.wrong_from
        x = self.embed(input_ids)
        for index, layer in enumerate(self.layers):
            factor = self.grad_factor if wrong and index < LAYERS - 2 else None
            x = layer(x, factor)
        logits = self.head(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, VOCAB), labels.view(-1)
        )
        return _Out(loss + self.loss_shift)


class _Runtime:
    def close(self):
        pass


def _run(monkeypatch, streamed, reference=None, sync=True):
    monkeypatch.setattr(mechanism_cost, "DEVICE", "cpu")
    monkeypatch.setattr(module, "cuda_available", lambda: True)
    monkeypatch.setattr(
        module,
        "prepare_shards",
        lambda weights, shards: ({}, "llama"),
    )
    monkeypatch.setattr(
        module,
        "build_streamed_arm",
        lambda weights, shards, index, arch, arm, buffers: (
            streamed,
            _Runtime(),
            lambda: None,
        ),
    )
    monkeypatch.setattr(
        module, "load_resident_reference", lambda weights: reference or _Toy()
    )
    if not sync:
        monkeypatch.setattr(module, "copy_lora", lambda source, target: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repeat_backward.py", "--weights", "w", "--shards", "s",
            "--seq", "8",
        ],
    )
    return module.main()


def test_defect_from_the_second_backward_is_reproduced(monkeypatch):
    assert _run(monkeypatch, _Toy(wrong_from=2)) == 0


def test_defect_from_the_first_backward_is_still_the_defect(monkeypatch):
    assert _run(monkeypatch, _Toy(wrong_from=1)) == 0


def test_healthy_streaming_is_not_reproduced(monkeypatch):
    assert _run(monkeypatch, _Toy()) == 1


def test_changed_forward_is_not_the_defect(monkeypatch):
    assert _run(monkeypatch, _Toy(wrong_from=2, loss_shift=1.0e-3)) == 1


def test_non_finite_loss_is_an_invalid_measurement(monkeypatch):
    assert _run(monkeypatch, _Toy(wrong_from=2, loss_shift=float("nan"))) == 3


def test_non_finite_gradients_are_an_invalid_measurement(monkeypatch):
    assert _run(monkeypatch, _Toy(wrong_from=2, grad_factor=float("inf"))) == 3


def test_unsynced_adapters_are_not_the_defect(monkeypatch):
    assert _run(monkeypatch, _Toy(wrong_from=2), sync=False) == 1


def test_empty_and_mismatched_gradient_maps_are_invalid():
    with pytest.raises(module.MeasurementInvalidError, match="empty"):
        module.compare_gradient_maps(
            {}, {"layers.0.lora_A.weight": torch.ones(1)}
        )
    with pytest.raises(module.MeasurementInvalidError, match="sets differ"):
        module.compare_gradient_maps(
            {"layers.0.lora_A.weight": torch.ones(1)},
            {"layers.1.lora_A.weight": torch.ones(1)},
        )
