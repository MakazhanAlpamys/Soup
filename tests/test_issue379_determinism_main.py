"""CPU tests of determinism.main() through the real measurement path."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness"
sys.path.insert(0, str(HARNESS))
import bitexact  # noqa: E402
import mechanism_cost  # noqa: E402

from tests.test_issue379_determinism_measure import VOCAB, _Out, _Toy, module  # noqa: E402

_REAL_GENERATOR = torch.Generator
_REAL_RANDINT = torch.randint


class _CpuGenerator(_REAL_GENERATOR):
    def __new__(cls, device="cpu"):
        return _REAL_GENERATOR.__new__(cls)


class _NanLoss(_Toy):
    def forward(self, input_ids, labels):
        out = super().forward(input_ids, labels)
        return _Out(out.logits, out.loss * float("nan"))


class _Runtime:
    def close(self):
        pass


def _run(monkeypatch, streamed, resident=None):
    resident = resident if resident is not None else _Toy(defect=False)
    streamed.config = resident.config = SimpleNamespace(vocab_size=VOCAB)
    seen = {"arms": [], "restored": 0}
    monkeypatch.setattr(torch, "Generator", _CpuGenerator)
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: _REAL_RANDINT(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        ),
    )
    monkeypatch.setattr(
        mechanism_cost, "prepare_shards", lambda weights, shards: ({}, "llama")
    )

    def build(weights, shards, index, arch, arm, buffers):
        seen["arms"].append(arm)
        return streamed, _Runtime(), lambda: seen.__setitem__(
            "restored", seen["restored"] + 1
        )

    monkeypatch.setattr(mechanism_cost, "build_streamed_arm", build)
    monkeypatch.setattr(bitexact, "load_resident_reference", lambda weights, quant: resident)
    import soup_cli.utils.spectrum_scan as scan

    monkeypatch.setattr(scan, "resolve_model_weights", lambda value: value)
    monkeypatch.setattr(
        sys,
        "argv",
        ["determinism.py", "--weights", "w", "--shards", "s", "--seq", "8"],
    )
    calls = {"n": 0}

    def available_once():
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr(torch.cuda, "is_available", available_once)
    return module.main(), seen


def test_record_pattern_is_reproduced_with_the_historical_control(monkeypatch):
    code, seen = _run(monkeypatch, _Toy(defect=True))
    assert code == 0
    assert seen["arms"] == ["control"]
    assert seen["restored"] == 1


def test_healthy_pair_is_not_reproduced(monkeypatch):
    code, seen = _run(monkeypatch, _Toy(defect=False))
    assert code == 1
    assert seen["restored"] == 1


def test_nan_loss_is_an_invalid_measurement(monkeypatch):
    code, seen = _run(monkeypatch, _NanLoss(defect=True))
    assert code == 3
    assert seen["restored"] == 1
