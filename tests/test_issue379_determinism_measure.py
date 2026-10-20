"""CPU tests of determinism.measure_models() with toy LoRA models."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "harness" / "determinism.py"
SPEC = importlib.util.spec_from_file_location("determinism_measure", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

VOCAB = 64


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, factor):
        ctx.factor = factor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.factor, None


class _Out:
    def __init__(self, logits, loss):
        self.logits = logits
        self.loss = loss


class _Toy(nn.Module):
    def __init__(self, defect, layers=6, hidden=16):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(VOCAB, hidden).requires_grad_(False)
        self.base = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False).requires_grad_(False)
            for _ in range(layers)
        )
        self.lora_A = nn.ModuleList(
            nn.Linear(hidden, 4, bias=False) for _ in range(layers)
        )
        self.lora_B = nn.ModuleList(
            nn.Linear(4, hidden, bias=False) for _ in range(layers)
        )
        for linear in self.lora_B:
            nn.init.normal_(linear.weight, std=0.02)
        self.head = nn.Linear(hidden, VOCAB, bias=False).requires_grad_(False)
        self.defect = defect
        self.backwards = 0

    def _count(self, grad):
        self.backwards += 1
        return grad

    def forward(self, input_ids, labels):
        x = self.embed(input_ids)
        layers = len(self.base)
        for index in range(layers):
            delta = self.lora_B[index](self.lora_A[index](x))
            if self.defect and self.backwards >= 1 and index < layers - 2:
                delta = _GradScale.apply(delta, 1.0 + 0.5 * self.backwards)
            x = torch.tanh(self.base[index](x) + delta)
        logits = self.head(x)
        if logits.requires_grad:
            logits.register_hook(self._count)
        loss = nn.functional.cross_entropy(
            logits.view(-1, VOCAB), labels.view(-1)
        )
        return _Out(logits, loss)


class _Renamed(nn.Module):
    def __init__(self, toy):
        super().__init__()
        self.wrapped = toy

    def forward(self, input_ids, labels):
        return self.wrapped(input_ids=input_ids, labels=labels)


def _inputs():
    generator = torch.Generator().manual_seed(17)
    ids = torch.randint(0, VOCAB, (1, 12), generator=generator)
    return ids, [ids.clone() for _ in range(5)]


def test_measure_models_reports_the_record_pattern():
    ids, batches = _inputs()
    record = module.measure_models(_Toy(defect=True), _Toy(defect=False), ids, batches)
    assert record["streamed_forward_equal"] is True
    assert record["resident_forward_equal"] is True
    assert record["streamed_gradient_equal"] is False
    assert record["resident_gradient_equal"] is True
    assert record["streamed_curve_equal"] is False
    assert record["resident_curve_equal"] is True
    assert module.determinism_verdict(record)


def test_measure_models_healthy_pair_is_not_the_pattern():
    ids, batches = _inputs()
    record = module.measure_models(_Toy(defect=False), _Toy(defect=False), ids, batches)
    assert record["streamed_gradient_equal"] is True
    assert record["streamed_curve_equal"] is True
    assert not module.determinism_verdict(record)


def test_measure_models_refuses_an_empty_intersection():
    ids, batches = _inputs()
    with pytest.raises(ValueError, match="no canonical parameter name"):
        module.measure_models(
            _Toy(defect=True), _Renamed(_Toy(defect=False)), ids, batches
        )
