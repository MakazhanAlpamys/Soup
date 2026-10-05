"""CPU checks for the verdict of benchmarks/harness/mechanism.py (#379)."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("bitsandbytes")

HARNESS = Path(__file__).parents[1] / "benchmarks" / "harness" / "mechanism.py"
KEYS = [
    f"{layer}.{name}"
    for layer in range(4)
    for name in ("q.A", "q.B", "v.A", "v.B")
]


def _truth():
    generator = torch.Generator().manual_seed(0)
    return {key: torch.randn(8, 8, generator=generator) for key in KEYS}


def _stale(grads):
    """Two tensors exact, the rest off by 1e-3."""
    return {
        key: value + (1e-3 if index >= 2 else 0.0)
        for index, (key, value) in enumerate(grads.items())
    }


def _with_inf(grads):
    planted = {key: value.clone() for key, value in grads.items()}
    planted["0.q.B"][0, 0] = float("inf")
    return planted


def _main(monkeypatch, capsys, argv=(), clone=None, bypassed_control=None):
    """Run the real main() with planted gradients."""
    spec = importlib.util.spec_from_file_location("_mechanism_harness_rv", HARNESS)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    truth = _truth()
    calls = {}

    def run_once(sources, state, arm, *, bypass_pool):
        if bypass_pool and arm != "control":
            return dict(truth)
        repetition = calls[arm] = calls.get(arm, -1) + 1
        if arm == "clone":
            return clone(truth, repetition) if clone else dict(truth)
        if arm == "control" and bypass_pool:
            return bypassed_control(truth) if bypassed_control else dict(truth)
        return _stale(truth)

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda, "get_device_name", lambda index=0: "planted"
    )
    monkeypatch.setattr(module, "load_sources", lambda: None)
    monkeypatch.setattr(module, "make_lora_parameters", lambda: None)
    monkeypatch.setattr(module, "make_arm_state", lambda sources, initial: None)
    monkeypatch.setattr(module, "run_once", run_once)
    monkeypatch.setattr(sys, "argv", ["mechanism.py", *argv])
    code = module.main()
    return code, capsys.readouterr().out


def test_reproduced_mechanism_exits_0(monkeypatch, capsys):
    code, out = _main(monkeypatch, capsys)
    assert code == 0, out
    assert "RESULT: mechanism reproduced" in out


@pytest.mark.parametrize(
    "clone",
    [
        pytest.param(
            lambda truth, rep: {key: value * 1.5 for key, value in truth.items()},
            id="wrong",
        ),
        pytest.param(
            lambda truth, rep: {
                key: value + 0.5 * rep for key, value in truth.items()
            },
            id="drifts",
        ),
    ],
)
def test_clone_that_disagrees_with_the_resident_reference_is_refused(
    monkeypatch, capsys, clone
):
    code, out = _main(monkeypatch, capsys, clone=clone)
    assert code == 1, out
    assert "RESULT: mechanism reproduced" not in out


@pytest.mark.parametrize(
    "argv", [(), ("--bypass-pool",)], ids=["default", "bypass-pool"]
)
def test_non_finite_gradients_give_no_verdict_and_exit_3(
    monkeypatch, capsys, argv
):
    code, out = _main(
        monkeypatch,
        capsys,
        argv,
        clone=lambda truth, rep: _with_inf(truth),
    )
    assert code == 3, out
    assert "no verdict" in out


def test_bypass_pool_exit_codes_stay_distinct(monkeypatch, capsys):
    caught, _ = _main(monkeypatch, capsys, ("--bypass-pool",))
    not_caught, _ = _main(
        monkeypatch,
        capsys,
        ("--bypass-pool",),
        bypassed_control=_stale,
    )
    assert (caught, not_caught) == (1, 2)
