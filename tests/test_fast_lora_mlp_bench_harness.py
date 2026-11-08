"""CPU-only protocol guards for the #837 benchmark harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_HARNESS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "harness"
    / "fast_lora_mlp_bench.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("fast_lora_mlp_bench", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def test_cli_self_test_checks_balanced_schedule(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(_HARNESS), "--self-test"])

    assert harness.main() == 0
    assert "all six arm orders occur five times" in capsys.readouterr().out


def test_schedule_covers_all_six_orders_equally():
    schedule = harness._schedule(30)
    groups = [tuple(schedule[index : index + 3]) for index in range(0, len(schedule), 3)]
    expected = {
        ("fast", "plain", "liger"),
        ("fast", "liger", "plain"),
        ("liger", "fast", "plain"),
        ("liger", "plain", "fast"),
        ("plain", "liger", "fast"),
        ("plain", "fast", "liger"),
    }

    assert set(groups) == expected
    assert all(groups.count(order) == 5 for order in expected)


class PlainBackward:
    next_functions = ()


class FastLoraSwiGLUBackward:
    next_functions = ()


class LigerSiLUMulBackward:
    next_functions = ()


class Output:
    def __init__(self, node):
        self.grad_fn = node


@pytest.mark.parametrize(
    ("arm", "node", "expected"),
    [
        ("fast", FastLoraSwiGLUBackward(), True),
        ("plain", PlainBackward(), True),
        ("liger", LigerSiLUMulBackward(), True),
        ("plain", FastLoraSwiGLUBackward(), False),
        ("fast", LigerSiLUMulBackward(), False),
        ("liger", PlainBackward(), False),
    ],
)
def test_path_proof_accepts_only_the_requested_arm(arm, node, expected):
    assert harness._path_ok(arm, Output(node)) is expected


@pytest.mark.parametrize("args", [["--iters", "4"], ["--tokens", "0"]])
def test_invalid_measurement_arguments_exit_two(args, monkeypatch):
    import torch

    monkeypatch.setattr(sys, "argv", [str(_HARNESS), *args])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(SystemExit) as error:
        harness.main()

    assert error.value.code == 2


def test_missing_cuda_exits_two_without_producing_a_result(monkeypatch):
    import torch

    monkeypatch.setattr(sys, "argv", [str(_HARNESS)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit) as error:
        harness.main()

    assert error.value.code == 2
