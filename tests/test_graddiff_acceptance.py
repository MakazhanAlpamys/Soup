from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.conftest import strip_ansi

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "harness"
    / "graddiff.py"
)
HARNESS_DIR = MODULE_PATH.parent
sys.path.insert(0, str(HARNESS_DIR))

SPEC = importlib.util.spec_from_file_location(
    "graddiff_acceptance",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None

module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output for assertions."""
    return " ".join(strip_ansi(text).split())


class FakeRuntime:
    pinned = True
    source = SimpleNamespace(nbytes=1)

    def close(self) -> None:
        pass


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(1))
        self.meta_parameter = SimpleNamespace(is_meta=True)

    def parameters(self, recurse: bool = True):
        yield self.lora_A
        yield self.meta_parameter

    def forward(self, input_ids, labels):
        loss = self.lora_A.sum() * 0.0 + 1.0
        return SimpleNamespace(loss=loss)


def make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        weights=str(tmp_path / "weights"),
        shards=str(tmp_path / "shards"),
        seq=2,
        batch=1,
        buffers=2,
        curve_steps=2,
        self_test=False,
    )


def install_measurement_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    streamed_curves: tuple[list[float], list[float]],
    resident_curves: tuple[list[float], list[float]],
    exact: int = 1,
    total: int = 1,
    streamed_loss: float = 1.0,
    reference_loss: float = 1.0,
) -> None:
    weights = Path("weights")
    weights.mkdir(exist_ok=True)

    monkeypatch.setattr(module, "cuda_available", lambda: True)
    monkeypatch.setattr(module, "DEVICE", "cpu")

    import soup_cli.utils.layer_shard as layer_shard
    import soup_cli.utils.layer_stream_runtime as layer_runtime

    monkeypatch.setattr(
        layer_runtime,
        "build_meta_skeleton",
        lambda *args, **kwargs: SimpleNamespace(
            config=SimpleNamespace(vocab_size=16),
        ),
    )
    monkeypatch.setattr(
        module.bitexact,
        "model_arch_name",
        lambda model: "fake",
    )
    monkeypatch.setattr(
        layer_runtime,
        "quantised_layer_suffixes",
        lambda model: (),
    )
    monkeypatch.setattr(
        layer_shard,
        "shard_checkpoint",
        lambda *args, **kwargs: {},
    )

    streamed = FakeModel()
    reference = FakeModel()

    monkeypatch.setattr(
        layer_runtime,
        "build_streamed_model",
        lambda *args, **kwargs: (streamed, FakeRuntime()),
    )
    monkeypatch.setattr(
        module.bitexact,
        "load_resident_reference",
        lambda *args, **kwargs: reference,
    )
    monkeypatch.setattr(
        module.bitexact,
        "make_non_vacuous_lora",
        lambda model: None,
    )
    monkeypatch.setattr(
        module.bitexact,
        "copy_lora",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        module.bitexact,
        "lora_config",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        module,
        "capture_lora_state",
        lambda model: {"lora_A": torch.ones(1)},
    )

    backward_calls = iter(
        [
            (
                torch.tensor(streamed_loss),
                {"lora_A": torch.ones(1)},
            ),
            (
                torch.tensor(reference_loss),
                {"lora_A": torch.ones(1)},
            ),
        ]
    )
    monkeypatch.setattr(
        module,
        "backward_once",
        lambda *args, **kwargs: next(backward_calls),
    )

    monkeypatch.setattr(
        module,
        "compare_gradients",
        lambda *args, **kwargs: (
            exact,
            total,
            0.0,
            0.0,
        ),
    )

    curve_calls = iter(
        [
            streamed_curves[0],
            streamed_curves[1],
            resident_curves[0],
            resident_curves[1],
        ]
    )
    monkeypatch.setattr(
        module,
        "train_curve_once",
        lambda *args, **kwargs: next(curve_calls),
    )

    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda index: "CPU-stub",
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: None,
    )

    monkeypatch.setattr(
        module,
        "install_historical_control",
        lambda runtime: None,
    )
    monkeypatch.setattr(
        module,
        "runtime_versions",
        lambda: ("stub-bnb", "stub-peft", "stub-transformers"),
    )

    generator_type = torch.Generator
    monkeypatch.setattr(
        torch,
        "Generator",
        lambda device=None: generator_type(device="cpu"),
    )


def test_run_measurement_reaches_abc_and_returns_historical_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()

    install_measurement_stubs(
        monkeypatch,
        streamed_curves=([1.0, 2.0], [1.0, 2.1]),
        resident_curves=([1.0, 2.0], [1.0, 2.0]),
    )

    args = make_args(tmp_path)
    args.weights = str(weights)

    result = module.run_measurement(args)

    output = capsys.readouterr().out
    plain = _plain(output)

    assert "A grads: 1/1 bit-exact" in plain
    assert "B streamed self-identical: False" in plain
    assert "C resident self-identical: True" in plain
    assert "RESULT: historical GRADDIFF pattern reproduced" in plain
    assert result == 0


def test_run_measurement_rejects_non_historical_verdict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()

    install_measurement_stubs(
        monkeypatch,
        streamed_curves=([1.0, 2.0], [1.0, 2.0]),
        resident_curves=([1.0, 2.0], [1.0, 2.0]),
    )

    args = make_args(tmp_path)
    args.weights = str(weights)

    result = module.run_measurement(args)

    output = capsys.readouterr().out
    plain = _plain(output)

    assert "A grads: 1/1 bit-exact" in plain
    assert "B streamed self-identical: True" in plain
    assert "C resident self-identical: True" in plain
    assert "historical GRADDIFF pattern was not reproduced" in plain
    assert result == 1


def test_run_measurement_rejects_non_matching_gradients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()

    install_measurement_stubs(
        monkeypatch,
        streamed_curves=([1.0, 2.0], [1.0, 2.1]),
        resident_curves=([1.0, 2.0], [1.0, 2.0]),
        exact=0,
        total=1,
    )

    args = make_args(tmp_path)
    args.weights = str(weights)

    result = module.run_measurement(args)

    output = capsys.readouterr().out
    plain = _plain(output)

    assert "A grads: 0/1 bit-exact" in plain
    assert "B streamed self-identical: False" in plain
    assert "C resident self-identical: True" in plain
    assert result == 1


def test_run_measurement_rejects_mismatched_initial_losses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()

    install_measurement_stubs(
        monkeypatch,
        streamed_curves=([1.0, 2.0], [1.0, 2.1]),
        resident_curves=([1.0, 2.0], [1.0, 2.0]),
        streamed_loss=1.0,
        reference_loss=2.0,
    )

    args = make_args(tmp_path)
    args.weights = str(weights)

    with pytest.raises(RuntimeError, match="initial streamed/reference losses differ"):
        module.run_measurement(args)


def test_main_executes_real_measurement_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    install_measurement_stubs(
        monkeypatch,
        streamed_curves=([1.0, 2.0], [1.0, 2.1]),
        resident_curves=([1.0, 2.0], [1.0, 2.0]),
    )
    args = make_args(tmp_path)
    args.weights = str(weights)
    monkeypatch.setattr(module, "parse_args", lambda: args)

    assert module.main() == 0


def test_run_measurement_restores_historical_runtime_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    install_measurement_stubs(
        monkeypatch,
        streamed_curves=([1.0, 2.0], [1.0, 2.1]),
        resident_curves=([1.0, 2.0], [1.0, 2.0]),
    )

    import soup_cli.utils.layer_stream_runtime as layer_runtime

    original = object()
    installed: list[object] = []
    monkeypatch.setattr(layer_runtime, "install_dequant_forward", original)

    def install(runtime: object) -> None:
        installed.append(runtime)
        runtime.install_dequant_forward = object()

    monkeypatch.setattr(module, "install_historical_control", install)
    args = make_args(tmp_path)
    args.weights = str(weights)

    assert module.run_measurement(args) == 0
    assert installed == [layer_runtime]
    assert layer_runtime.install_dequant_forward is original
