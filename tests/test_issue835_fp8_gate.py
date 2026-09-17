"""#835 — both FP8 training paths ask one hardware gate, with the floor torch uses.

``training.fp8_attention`` refused Ada (sm_89), which ``torch._scaled_mm`` accepts,
while ``training.quantization_aware: fp8`` converted the model on any card,
Ampere included. Both now ask ``fp8_training_supported(recipe)``.

The floors are torch's own (source, not hardware):

- every recipe: ``_scaled_mm_allowed_device()`` is ``major >= 9 or (8, 9)``
  (identical at v2.4.0, v2.6.0, v2.7.0, v2.13.0);
- rowwise at (8, 9): torch >= 2.7, the first release with a CUTLASS sm89 rowwise
  kernel (v2.6.0 ``RowwiseScaledMM.cu`` builds only ``cutlass::arch::Sm90``).

torchao is replaced by a stub that records conversions; the capability is patched.
Nothing here needs a GPU.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn


class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


@pytest.fixture
def converts(monkeypatch):
    """Stub torchao.float8; returns the list of recorded conversions."""
    calls: list[str] = []

    float8 = ModuleType("torchao.float8")

    def _convert(model, config=None, module_filter_fn=None):
        calls.append(config)

    float8.convert_to_float8_training = _convert

    config_mod = ModuleType("torchao.float8.config")

    class Float8LinearConfig:
        @staticmethod
        def from_recipe_name(name):
            return name

    config_mod.Float8LinearConfig = Float8LinearConfig

    monkeypatch.setitem(sys.modules, "torchao", ModuleType("torchao"))
    monkeypatch.setitem(sys.modules, "torchao.float8", float8)
    monkeypatch.setitem(sys.modules, "torchao.float8.config", config_mod)
    return calls


def _card(monkeypatch, capability, version="2.13.0"):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: capability)
    monkeypatch.setattr(torch, "__version__", version)


RECIPES = ("tensorwise", "rowwise", "rowwise_with_gw_hp")


class TestTheFloor:
    @pytest.mark.parametrize("recipe", RECIPES)
    @pytest.mark.parametrize("capability", [(9, 0), (10, 0), (12, 0)])
    def test_hopper_and_newer_pass_every_recipe_on_old_torch(
        self, monkeypatch, capability, recipe
    ):
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, capability, version="2.6.0")
        assert fp8_training_supported(recipe) == (True, "")

    @pytest.mark.parametrize("version", ["2.6.0", "2.6.0+cu124", "2.13.0"])
    def test_ada_passes_tensorwise_on_every_supported_torch(self, monkeypatch, version):
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, (8, 9), version=version)
        assert fp8_training_supported("tensorwise") == (True, "")

    @pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
    @pytest.mark.parametrize("version", ["2.7.0", "2.7.1+cu126", "2.13.0", "3.0.0"])
    def test_ada_passes_rowwise_from_torch_2_7(self, monkeypatch, recipe, version):
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, (8, 9), version=version)
        assert fp8_training_supported(recipe) == (True, "")

    @pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
    @pytest.mark.parametrize("version", ["2.6.0", "2.6.0+cu124", "2.6.1"])
    def test_ada_refuses_rowwise_before_torch_2_7(self, monkeypatch, recipe, version):
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, (8, 9), version=version)
        ok, reason = fp8_training_supported(recipe)
        assert ok is False
        assert recipe in reason
        assert "2.7" in reason
        assert "tensorwise" in reason

    def test_ada_refuses_rowwise_when_the_torch_version_is_unreadable(self, monkeypatch):
        """Fail closed: a version the gate cannot parse is not assumed to be >= 2.7."""
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, (8, 9), version="unknown")
        assert fp8_training_supported("rowwise")[0] is False
        assert fp8_training_supported("tensorwise") == (True, "")

    @pytest.mark.parametrize("recipe", RECIPES)
    @pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 6), (8, 7)])
    def test_below_ada_refuses_every_recipe(self, monkeypatch, capability, recipe):
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, capability)
        ok, reason = fp8_training_supported(recipe)
        assert ok is False
        assert "8.9" in reason
        assert "Ada" in reason

    def test_no_cuda_refuses(self, monkeypatch):
        from soup_cli.utils.fp8 import fp8_training_supported

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        ok, reason = fp8_training_supported("tensorwise")
        assert ok is False
        assert reason

    def test_is_fp8_gpu_supported_admits_ada(self, monkeypatch):
        from soup_cli.utils.fp8 import is_fp8_gpu_supported

        _card(monkeypatch, (8, 9))
        assert is_fp8_gpu_supported() is True
        _card(monkeypatch, (8, 6))
        assert is_fp8_gpu_supported() is False


class TestFp8Attention:
    """Acceptance 1: (8, 9) does not raise the Hopper error."""

    def test_ada_converts_attention(self, monkeypatch, converts):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        _card(monkeypatch, (8, 9))
        assert apply_fp8_attention(_Attn()) == 4
        assert converts == ["tensorwise"]

    def test_ampere_is_refused_without_converting(self, monkeypatch, converts):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        _card(monkeypatch, (8, 6))
        with pytest.raises(RuntimeError, match="8.9"):
            apply_fp8_attention(_Attn())
        assert converts == []

    def test_rowwise_on_ada_with_old_torch_is_refused(self, monkeypatch, converts):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        _card(monkeypatch, (8, 9), version="2.6.0")
        with pytest.raises(RuntimeError, match="2.7"):
            apply_fp8_attention(_Attn(), recipe="rowwise")
        assert converts == []


class TestQuantizationAwareFp8:
    """Acceptance 2: (8, 6) is refused before convert_to_float8_training runs."""

    def test_ampere_is_refused_before_conversion(self, monkeypatch, converts):
        from soup_cli.utils.fp8 import apply_fp8_training

        _card(monkeypatch, (8, 6))
        with pytest.raises(RuntimeError, match="8.9"):
            apply_fp8_training(_Attn())
        assert converts == []

    def test_ada_converts(self, monkeypatch, converts):
        from soup_cli.utils.fp8 import apply_fp8_training

        _card(monkeypatch, (8, 9))
        assert apply_fp8_training(_Attn()) is True
        assert converts == ["tensorwise"]

    def test_rowwise_on_ada_with_old_torch_is_refused(self, monkeypatch, converts):
        from soup_cli.utils.fp8 import apply_fp8_training

        _card(monkeypatch, (8, 9), version="2.6.0")
        with pytest.raises(RuntimeError, match="2.7"):
            apply_fp8_training(_Attn(), recipe="rowwise")
        assert converts == []

    def test_missing_torchao_still_returns_false(self, monkeypatch):
        """The dependency answer is unchanged: False, not a refusal."""
        from soup_cli.utils.fp8 import apply_fp8_training

        _card(monkeypatch, (8, 6))
        monkeypatch.setattr("soup_cli.utils.fp8.is_fp8_available", lambda: False)
        assert apply_fp8_training(_Attn()) is False


class TestOneGate:
    """Acceptance 3: both paths call the same gate, with their recipe."""

    @pytest.fixture
    def gate(self, monkeypatch):
        seen: list[str] = []
        verdict = {"value": (False, "SENTINEL-REASON")}

        def _gate(recipe="tensorwise"):
            seen.append(recipe)
            return verdict["value"]

        monkeypatch.setattr("soup_cli.utils.fp8.fp8_training_supported", _gate)
        return seen, verdict

    def test_both_paths_refuse_with_the_gates_reason(self, monkeypatch, converts, gate):
        from soup_cli.utils.advanced_precision import apply_fp8_attention
        from soup_cli.utils.fp8 import apply_fp8_training

        seen, _ = gate
        # A card every old check would admit: only the gate can refuse here.
        _card(monkeypatch, (12, 0))
        with pytest.raises(RuntimeError, match="SENTINEL-REASON"):
            apply_fp8_attention(_Attn(), recipe="rowwise")
        with pytest.raises(RuntimeError, match="SENTINEL-REASON"):
            apply_fp8_training(_Attn(), recipe="rowwise_with_gw_hp")
        assert seen == ["rowwise", "rowwise_with_gw_hp"]
        assert converts == []

    def test_both_paths_convert_when_the_gate_admits(self, monkeypatch, converts, gate):
        """Control: a card every old check would refuse, admitted by the gate."""
        from soup_cli.utils.advanced_precision import apply_fp8_attention
        from soup_cli.utils.fp8 import apply_fp8_training

        seen, verdict = gate
        verdict["value"] = (True, "")
        _card(monkeypatch, (7, 5))
        assert apply_fp8_attention(_Attn()) == 4
        assert apply_fp8_training(_Attn()) is True
        assert seen == ["tensorwise", "tensorwise"]
        assert converts == ["tensorwise", "tensorwise"]

    def test_validate_fp8_config_reports_the_gates_reason(self, gate):
        from soup_cli.utils.fp8 import validate_fp8_config

        seen, _ = gate
        errors = validate_fp8_config("fp8", "transformers", "cuda", recipe="rowwise")
        assert "SENTINEL-REASON" in errors
        assert seen == ["rowwise"]


class TestValidateFp8Config:
    def test_ada_has_no_hardware_error(self, monkeypatch, converts):
        from soup_cli.utils.fp8 import validate_fp8_config

        _card(monkeypatch, (8, 9))
        assert validate_fp8_config("fp8", "transformers", "cuda") == []

    def test_ampere_has_the_ada_error(self, monkeypatch, converts):
        from soup_cli.utils.fp8 import validate_fp8_config

        _card(monkeypatch, (8, 6))
        errors = validate_fp8_config("fp8", "transformers", "cuda")
        assert any("8.9" in e for e in errors)


class TestTheRunSaysWhy:
    """The 11-trainer helper prints the refusal, not "torchao unavailable"."""

    def test_v028_helper_prints_the_gate_reason(self, monkeypatch, converts):
        from io import StringIO

        from rich.console import Console

        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        _card(monkeypatch, (8, 6))
        out = StringIO()
        console = Console(file=out, width=400, force_terminal=False, color_system=None)
        applied = apply_v028_speed_memory(
            model=_Attn(),
            tcfg=TrainingConfig(quantization_aware="fp8"),
            base_model="m",
            console=console,
            device="cuda",
        )
        text = out.getvalue()
        assert applied["fp8"] is False
        assert converts == []
        assert "8.9" in text
        assert "unavailable" not in text
