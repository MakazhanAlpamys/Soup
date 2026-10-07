"""CPU tests of layercount._measure_checkpoint() with toy LoRA models.

Only the CUDA and checkpoint boundary is replaced. The production copy,
backward, gradient snapshot, and measurement assertions run on CPU.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness"
sys.path.insert(0, str(HARNESS))
import bitexact  # noqa: E402
import mechanism_cost  # noqa: E402

from benchmarks.harness import layercount as module  # noqa: E402

LAYERS = 8
HIDDEN = 16
VOCAB = 64
STORE_BYTES = LAYERS * 123_456


class _GradScale(torch.autograd.Function):
    """Identity forward, scaled backward: same loss, wrong gradients."""

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


class _Toy(nn.Module):
    """wrong_from is the first 1-based backward with corrupted gradients."""

    def __init__(self, wrong_from=None, loss_shift=0.0, grad_factor=3.0):
        super().__init__()
        torch.manual_seed(0)
        self.config = SimpleNamespace(vocab_size=VOCAB)
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
        loss = nn.functional.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
        return _Out(loss + self.loss_shift)


class _Runtime:
    def __init__(self, nbytes):
        self.source = SimpleNamespace(nbytes=nbytes)
        self.closed = 0

    def close(self):
        self.closed += 1


def _measure(monkeypatch, streamed, store_bytes=STORE_BYTES):
    runtime = _Runtime(store_bytes)
    restored = []
    arms = []
    monkeypatch.setattr(mechanism_cost, "DEVICE", "cpu")
    monkeypatch.setattr(
        bitexact,
        "make_non_vacuous_lora",
        mechanism_cost.make_non_vacuous_lora,
    )
    monkeypatch.setattr(
        mechanism_cost,
        "prepare_shards",
        lambda weights, shards: (SimpleNamespace(n_layers=LAYERS), "llama"),
    )
    monkeypatch.setattr(
        mechanism_cost,
        "build_streamed_arm",
        lambda weights, shards, index, arch, arm, buffers: (
            arms.append(arm),
            streamed,
            runtime,
            lambda: restored.append(True),
        )[1:],
    )
    monkeypatch.setattr(bitexact, "load_resident_reference", lambda weights, quant: _Toy())
    row = module._measure_checkpoint("weights", "shards", "nf4")
    assert runtime.closed == 1 and restored == [True]
    assert arms == ["control"]
    return row


def test_later_corruption_is_counted_per_read(monkeypatch):
    row = _measure(monkeypatch, _Toy(wrong_from=2))
    total = row["total"]
    assert total > 0
    assert row["exact_counts"][0] == total
    assert all(count < total for count in row["exact_counts"][1:])


def test_healthy_streaming_reads_exact_three_times(monkeypatch):
    row = _measure(monkeypatch, _Toy())
    assert row["exact_counts"] == [row["total"]] * 3


def test_layer_mib_is_store_bytes_per_layer(monkeypatch):
    row = _measure(monkeypatch, _Toy())
    assert row["layer_mib"] == pytest.approx(STORE_BYTES / LAYERS / (1024 * 1024))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"wrong_from": 2, "loss_shift": 1.0e-3}, "loss differs"),
        ({"wrong_from": 2, "loss_shift": float("nan")}, "loss differs"),
        ({"wrong_from": 2, "grad_factor": float("inf")}, "non-finite"),
    ],
)
def test_changed_or_nonfinite_measurements_are_invalid(monkeypatch, kwargs, message):
    with pytest.raises(module.MeasurementInvalidError, match=message):
        _measure(monkeypatch, _Toy(**kwargs))


def test_missing_store_size_is_invalid(monkeypatch):
    with pytest.raises(module.MeasurementInvalidError, match="store size"):
        _measure(monkeypatch, _Toy(wrong_from=2), store_bytes=0)


def test_bf16_control_only_asks_the_runtime_for_supported_quants(monkeypatch):
    from soup_cli.utils import layer_shard, layer_stream_runtime

    seen = {}

    class _StopError(Exception):
        pass

    def skeleton(model_id, *, dtype, quant="none", **kwargs):
        seen["build_meta_skeleton"] = quant
        return SimpleNamespace(config=SimpleNamespace(model_type="llama"))

    def shard(weights, shards, *, quant="none", **kwargs):
        seen["shard_checkpoint"] = quant
        return SimpleNamespace(n_layers=LAYERS)

    def build(*, quant="none", **kwargs):
        seen["build_streamed_model"] = quant
        raise _StopError

    monkeypatch.setattr(layer_stream_runtime, "build_meta_skeleton", skeleton)
    monkeypatch.setattr(layer_shard, "shard_checkpoint", shard)
    monkeypatch.setattr(layer_stream_runtime, "build_streamed_model", build)
    with pytest.raises(_StopError):
        module._measure_checkpoint("weights", "shards", "bf16")
    assert set(seen) == {
        "build_meta_skeleton",
        "shard_checkpoint",
        "build_streamed_model",
    }
    for call, quant in seen.items():
        assert quant in layer_shard.SUPPORTED_STREAM_QUANTS, (call, quant)
