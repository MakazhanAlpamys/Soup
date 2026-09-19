"""#379: publish the STEP 14 control-versus-repair measurement harness.

The historical gate compared a streamed NF4 control and the repaired streamed
path against one resident NF4 reference. Its first run reported ``0/0`` exact
gradients because the parameter-name intersection was empty. These tests pin
the non-vacuous contract on CPU; real 32B/72B numbers still require CUDA.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_DIR = _REPO_ROOT / "benchmarks" / "harness"
_HARNESS = _HARNESS_DIR / "variant2_gate.py"


def _load_harness():
    sys.path.insert(0, str(_HARNESS_DIR))
    try:
        spec = importlib.util.spec_from_file_location("variant2_gate", _HARNESS)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_HARNESS_DIR))


_harness = _load_harness()


class _FakeParameter:
    def __init__(self, grad=object()) -> None:
        self.grad = grad


class _FakeModel:
    def __init__(self, names: list[str], *, grad=object()) -> None:
        self._parameters = [(name, _FakeParameter(grad)) for name in names]

    def named_parameters(self):
        return iter(self._parameters)


class TestArgumentValidation:
    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--seq", "0"), ("--repeats", "0"), ("--buffers", "1")],
    )
    def test_bad_protocol_values_fail_before_cuda_skip(
        self, monkeypatch, capsys, flag: str, value: str
    ) -> None:
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["variant2_gate.py", "--weights", "x", "--shards", "y", flag, value],
        )

        assert _harness.main() == 2
        assert flag in capsys.readouterr().out

    def test_valid_protocol_is_an_intentional_no_cuda_skip(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(_harness, "cuda_available", lambda: False)
        monkeypatch.setattr(
            sys,
            "argv",
            ["variant2_gate.py", "--weights", "x", "--shards", "y"],
        )

        assert _harness.main() == 0
        assert "intentional skip" in capsys.readouterr().out.lower()


class TestCanonicalGradientIntersection:
    def test_empty_model_intersection_is_a_hard_failure(self) -> None:
        left = _FakeModel(["model.layers.0.q_proj.lora_A.default.weight"])
        right = _FakeModel(["model.layers.1.q_proj.lora_A.default.weight"])

        with pytest.raises(ValueError, match="no canonical parameter name"):
            _harness.shared_lora_gradient_names(left, right)

    def test_shared_parameters_without_shared_lora_gradients_fail(self) -> None:
        left = _FakeModel(["model.embed_tokens.weight"])
        right = _FakeModel(["model.embed_tokens.weight"])

        with pytest.raises(ValueError, match="no shared LoRA gradient"):
            _harness.shared_lora_gradient_names(left, right)

    def test_shared_lora_gradient_names_are_returned(self) -> None:
        name = "model.layers.0.q_proj.lora_A.default.weight"
        assert _harness.shared_lora_gradient_names(
            _FakeModel([name]), _FakeModel([name])
        ) == frozenset({name})


class TestGradientVerdict:
    def test_gate_requires_the_control_to_break_and_repair_to_be_exact(self) -> None:
        broken = _harness.GradientSummary(compared=4, exact=1, wrong_layers=2, worst_abs=0.5)
        exact = _harness.GradientSummary(compared=4, exact=4, wrong_layers=0, worst_abs=0.0)

        assert _harness.gate_passes(control=broken, repaired=exact)
        assert not _harness.gate_passes(control=exact, repaired=exact)
        assert not _harness.gate_passes(control=broken, repaired=broken)

    def test_every_repetition_must_have_a_matching_control_and_repair(self) -> None:
        broken = _harness.GradientSummary(compared=4, exact=1, wrong_layers=2, worst_abs=0.5)
        exact = _harness.GradientSummary(compared=4, exact=4, wrong_layers=0, worst_abs=0.0)

        assert _harness._paired_gradient_gate(
            {"gradients": [vars(broken), vars(broken)]},
            {"gradients": [vars(exact), vars(exact)]},
        )
        assert not _harness._paired_gradient_gate(
            {"gradients": [vars(broken), vars(broken)]},
            {"gradients": [vars(exact)]},
        )

    def test_real_tensors_distinguish_exact_and_broken_gradients(self) -> None:
        torch = pytest.importorskip("torch")
        name = "model.layers.3.q_proj.lora_A.default.weight"
        left = _FakeModel([name], grad=torch.tensor([1.0, 2.0]))
        same = _FakeModel([name], grad=torch.tensor([1.0, 2.0]))
        different = _FakeModel([name], grad=torch.tensor([1.0, 4.0]))

        exact = _harness.compare_lora_gradients(left, same)
        broken = _harness.compare_lora_gradients(left, different)

        assert exact == _harness.GradientSummary(1, 1, 0, 0.0)
        assert broken == _harness.GradientSummary(1, 0, 1, 2.0)


class TestRepairToggle:
    def test_control_disables_repair_only_while_building(self) -> None:
        calls: list[str] = []

        def install(_module) -> int:
            calls.append("repair")
            return 7

        runtime_module = SimpleNamespace(install_dequant_forward=install)

        def build_model():
            return runtime_module.install_dequant_forward(object())

        assert (
            _harness.build_streamed_arm(
                repair_enabled=False,
                runtime_module=runtime_module,
                build_model=build_model,
            )
            == 0
        )
        assert calls == []
        assert runtime_module.install_dequant_forward is install

        assert (
            _harness.build_streamed_arm(
                repair_enabled=True,
                runtime_module=runtime_module,
                build_model=build_model,
            )
            == 7
        )
        assert calls == ["repair"]


def test_harness_is_indexed() -> None:
    readme = (_REPO_ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    assert "harness/variant2_gate.py" in readme
