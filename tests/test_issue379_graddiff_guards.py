"""Pin the GRADDIFF pieces the acceptance stubs replace (#379)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.test_graddiff_acceptance import (
    FakeModel,
    FakeRuntime,
    install_measurement_stubs,
    make_args,
)
from tests.test_graddiff_acceptance import module as graddiff


class Tiny(torch.nn.Module):
    def __init__(self, name: str = "lora_A") -> None:
        super().__init__()
        self.register_parameter(name, torch.nn.Parameter(torch.ones(2)))

    def forward(self, input_ids, labels):
        weight = next(iter(self.parameters()))
        return SimpleNamespace(loss=(weight * input_ids.float().mean()).sum())


def test_self_test_runs_the_empty_intersection_negative_in_ci() -> None:
    assert graddiff.run_self_test() == 0


def test_compare_gradients_counts_a_mismatch() -> None:
    same = {"lora_A": torch.ones(2)}
    other = {"lora_A": torch.tensor([1.0, 2.0])}
    assert graddiff.compare_gradients(Tiny(), Tiny(), same, dict(same))[:2] == (1, 1)
    assert graddiff.compare_gradients(Tiny(), Tiny(), same, other)[:2] == (0, 1)


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
@pytest.mark.parametrize("side", ["streamed", "reference"])
def test_compare_gradients_refuses_non_finite_on_either_side(
    bad: float, side: str
) -> None:
    good = {"lora_A": torch.ones(2)}
    broken = {"lora_A": torch.tensor([bad, 1.0])}
    streamed, reference = (broken, good) if side == "streamed" else (good, broken)
    with pytest.raises(graddiff.MeasurementInvalidError, match="non-finite"):
        graddiff.compare_gradients(Tiny(), Tiny(), streamed, reference)


def test_compare_gradients_refuses_different_gradient_sets() -> None:
    with pytest.raises(RuntimeError, match="sets differ"):
        graddiff.compare_gradients(
            Tiny(), Tiny(), {"lora_A": torch.ones(2)}, {"lora_B": torch.ones(2)}
        )


def test_train_curve_once_restarts_from_the_same_adapter() -> None:
    model = Tiny()
    initial = graddiff.capture_lora_state(model)
    batches = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]]), torch.tensor([[5, 6]])]
    first = graddiff.train_curve_once(model, batches, initial)
    second = graddiff.train_curve_once(model, batches, initial)
    assert first == second


class _NoMetaModel(FakeModel):
    def parameters(self, recurse: bool = True):
        yield self.lora_A


@pytest.mark.parametrize(
    ("runtime_pinned", "model_cls", "message"),
    [
        (False, FakeModel, "pinned=False"),
        (True, _NoMetaModel, "no remaining meta parameters"),
    ],
)
def test_run_measurement_refuses_an_unexercised_streaming_path(
    monkeypatch, tmp_path, runtime_pinned, model_cls, message
) -> None:
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "weights"
    weights.mkdir()
    install_measurement_stubs(
        monkeypatch,
        streamed_curves=([1.0, 2.0], [1.0, 2.1]),
        resident_curves=([1.0, 2.0], [1.0, 2.0]),
    )
    import soup_cli.utils.layer_stream_runtime as layer_runtime

    runtime = FakeRuntime()
    runtime.pinned = runtime_pinned
    monkeypatch.setattr(
        layer_runtime, "build_streamed_model", lambda *a, **k: (model_cls(), runtime)
    )
    args = make_args(tmp_path)
    args.weights = str(weights)
    with pytest.raises(RuntimeError, match=message):
        graddiff.run_measurement(args)
