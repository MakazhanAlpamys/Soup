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


class Output:
    def __init__(self, node):
        self.grad_fn = node


_FAST = "FastLoraSwiGLUBackward"
_LIGER = "LigerSiLUMulBackward"


def _node(name, *children):
    node = type(name, (), {})()
    node.next_functions = tuple((child, 0) for child in children)
    return node


def _graph(*markers):
    return _node("AddBackward0", _node("MulBackward0", *[_node(name) for name in markers]))


@pytest.mark.parametrize(
    ("arm", "markers", "expected"),
    [
        ("fast", (_FAST,), True),
        ("liger", (_LIGER,), True),
        ("plain", (), True),
        ("plain", (_FAST,), False),
        ("plain", (_LIGER,), False),
        ("fast", (_FAST, _LIGER), False),
        ("liger", (_FAST, _LIGER), False),
    ],
)
def test_path_proof_walks_the_whole_graph(arm, markers, expected):
    assert harness._path_ok(arm, Output(_graph(*markers))) is expected


@pytest.mark.parametrize("args", [["--iters", "4"], ["--tokens", "0"]])
def test_invalid_measurement_arguments_exit_two(args, monkeypatch, capsys):
    import torch

    monkeypatch.setattr(sys, "argv", [str(_HARNESS), *args])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(SystemExit) as error:
        harness.main()

    assert error.value.code == 2
    assert "iters must be at least 5" in capsys.readouterr().err


def test_missing_cuda_exits_two_without_producing_a_result(monkeypatch, capsys):
    import torch

    monkeypatch.setattr(sys, "argv", [str(_HARNESS)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit) as error:
        harness.main()

    assert error.value.code == 2
    assert "a CUDA GPU is required" in capsys.readouterr().err
