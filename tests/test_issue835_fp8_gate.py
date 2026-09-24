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

import importlib.machinery
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

    torchao = ModuleType("torchao")
    # A spec, because setup code asks importlib.util.find_spec("torchao"), which
    # raises on a module whose __spec__ is None.
    for module in (torchao, float8, config_mod):
        module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, None)
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.float8", float8)
    monkeypatch.setitem(sys.modules, "torchao.float8.config", config_mod)
    return calls


def _card(monkeypatch, capability, version="2.13.0", *, platform="linux", cuda="12.4"):
    """A patched card. The OS and torch's CUDA build are pinned too: rowwise
    depends on both, and the suite also runs on Windows and macOS."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: capability)
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(torch.version, "cuda", cuda)


RECIPES = ("tensorwise", "rowwise", "rowwise_with_gw_hp")


class TestTheFloor:
    @pytest.mark.parametrize("recipe", RECIPES)
    @pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
    def test_hopper_and_blackwell_datacenter_pass_every_recipe_on_old_torch(
        self, monkeypatch, capability, recipe
    ):
        """v2.6.0 has no rowwise device dispatch; v2.7.0 lists sm9x and sm10x."""
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, capability, version="2.6.0")
        assert fp8_training_supported(recipe) == (True, "")

    @pytest.mark.parametrize("capability", [(11, 0), (12, 0), (13, 0)])
    def test_every_major_from_ada_passes_tensorwise_on_old_torch(self, monkeypatch, capability):
        """``_scaled_mm_allowed_device()`` is ``major >= 9``: tensorwise has no
        per-major dispatch."""
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, capability, version="2.6.0")
        assert fp8_training_supported("tensorwise") == (True, "")

    @pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
    @pytest.mark.parametrize(
        "capability, refused_on, accepted_on",
        [
            # sm12x joins the dispatch in v2.8.0 (RowwiseScaledMM.cu:950).
            ((12, 0), ["2.6.0", "2.7.0", "2.7.1+cu128"], ["2.8.0", "2.14.0"]),
            # sm11x joins it in v2.10.0 (:961).
            ((11, 0), ["2.6.0", "2.8.0", "2.9.1"], ["2.10.0", "2.14.0"]),
        ],
        ids=["rtx50-sm120", "sm110"],
    )
    def test_newer_majors_need_the_torch_that_added_them(
        self, monkeypatch, recipe, capability, refused_on, accepted_on
    ):
        from soup_cli.utils.fp8 import fp8_training_supported

        for version in refused_on:
            _card(monkeypatch, capability, version=version)
            ok, reason = fp8_training_supported(recipe)
            assert ok is False, (capability, version)
            assert recipe in reason and "tensorwise" in reason, reason
            assert f"{capability[0]}.{capability[1]}" in reason, reason
        for version in accepted_on:
            _card(monkeypatch, capability, version=version)
            assert fp8_training_supported(recipe) == (True, ""), (capability, version)

    @pytest.mark.parametrize("version", ["2.8.0", "2.14.0", "3.0.0"])
    def test_a_major_no_release_dispatches_refuses_rowwise(self, monkeypatch, version):
        """No release through v2.14.0 lists sm13x: refused, not assumed to work."""
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, (13, 0), version=version)
        ok, reason = fp8_training_supported("rowwise")
        assert ok is False
        assert "13.0" in reason

    @pytest.mark.parametrize("capability", [(8, 9), (9, 0), (12, 0)])
    @pytest.mark.parametrize("version", ["2.8.0", "2.14.0"])
    def test_windows_refuses_rowwise_at_every_torch(self, monkeypatch, capability, version):
        """``BUILD_ROWWISE_FP8_KERNEL`` requires ``!_WIN32`` through v2.14.0, so the
        kernel does not exist there whatever the card; tensorwise is unaffected."""
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, capability, version=version, platform="win32")
        ok, reason = fp8_training_supported("rowwise")
        assert ok is False
        assert "Windows" in reason
        assert fp8_training_supported("tensorwise") == (True, "")

    @pytest.mark.parametrize(
        "version, cuda, ok",
        [
            ("2.8.0", "11.8", False),
            ("2.10.0", "11.8", False),
            ("2.10.0", "12.1", True),
            ("2.11.0", "11.8", True),
            ("2.10.0", None, False),
        ],
    )
    def test_a_cuda_11_build_has_no_rowwise_kernel_before_torch_2_11(
        self, monkeypatch, version, cuda, ok
    ):
        """Before v2.11.0 the build gate also requires ``CUDA_VERSION >= 12000``."""
        from soup_cli.utils.fp8 import fp8_training_supported

        _card(monkeypatch, (9, 0), version=version, cuda=cuda)
        assert fp8_training_supported("rowwise")[0] is ok

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

    def test_the_card_is_refused_even_when_torchao_is_missing(self, monkeypatch):
        """#1044 review: the gate must run BEFORE the dependency probe.

        torchao is not a default dependency, so absent is the common case. With
        the checks in the other order an Ampere user who asked for FP8 got
        "torchao.float8 unavailable", a yellow line and a bf16 run -- the silent
        no-op this issue exists to remove. What they asked for cannot run on this
        card either way."""
        from soup_cli.utils.fp8 import FP8HardwareUnsupportedError, apply_fp8_training

        _card(monkeypatch, (8, 6))
        monkeypatch.setattr("soup_cli.utils.fp8.is_fp8_available", lambda: False)
        with pytest.raises(FP8HardwareUnsupportedError, match="8.9"):
            apply_fp8_training(_Attn())

    def test_fp8_attention_refuses_the_card_before_naming_torchao(self, monkeypatch):
        """Same ordering in the attention path: the message must name the card,
        not send the user to install a package that would not help."""
        from soup_cli.utils.advanced_precision import apply_fp8_attention
        from soup_cli.utils.fp8 import FP8HardwareUnsupportedError

        _card(monkeypatch, (8, 6))
        monkeypatch.setitem(sys.modules, "torchao", None)
        with pytest.raises(FP8HardwareUnsupportedError, match="8.9"):
            apply_fp8_attention(_Attn())

    def test_missing_torchao_now_stops_the_run(self, monkeypatch):
        """INVERTED by the #835 ruling (2026-09-19); this test used to pin
        ``False``. On a card that CAN run FP8, a missing torchao no longer lets
        the run go on in bf16: FP8 was asked for, and whether the reason is the
        card or a package does not change what the user asked for. The message
        names the fix."""
        from soup_cli.utils.fp8 import FP8DependencyMissingError, apply_fp8_training

        _card(monkeypatch, (9, 0))
        monkeypatch.setattr("soup_cli.utils.fp8.is_fp8_available", lambda: False)
        with pytest.raises(FP8DependencyMissingError, match=r"soup-cli\[qat\]"):
            apply_fp8_training(_Attn())

    def test_transformer_engine_alone_is_still_missing_torchao(self, monkeypatch):
        """``is_fp8_available()`` also accepts transformer-engine, but this
        converter is torchao's, so a box with TE and no torchao stops too."""
        from soup_cli.utils.fp8 import FP8DependencyMissingError, apply_fp8_training

        _card(monkeypatch, (9, 0))
        monkeypatch.setattr("soup_cli.utils.fp8.is_fp8_available", lambda: True)
        monkeypatch.setitem(sys.modules, "torchao", None)
        with pytest.raises(FP8DependencyMissingError):
            apply_fp8_training(_Attn())


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


class TestTheRunStops:
    """#835 ruling: an explicitly requested FP8 that the card cannot run stops the
    run, on every trainer. Warning and training on without it is the defect class
    v0.75.0 removed: a setting accepted, then silently not applied."""

    def _console(self):
        from io import StringIO

        from rich.console import Console

        out = StringIO()
        return out, Console(file=out, width=400, force_terminal=False, color_system=None)

    def test_the_shared_helper_raises_for_quantization_aware(self, monkeypatch, converts):
        """The path 11 trainers take, and SFT's once #1022 routes it here."""
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.fp8 import FP8HardwareUnsupportedError
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        _card(monkeypatch, (8, 6))
        _out, console = self._console()
        with pytest.raises(FP8HardwareUnsupportedError, match="8.9"):
            apply_v028_speed_memory(
                model=_Attn(),
                tcfg=TrainingConfig(quantization_aware="fp8"),
                base_model="m",
                console=console,
                device="cuda",
            )
        assert converts == []

    def test_the_shared_helper_raises_for_fp8_attention(self, monkeypatch, converts):
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.fp8 import FP8HardwareUnsupportedError
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        _card(monkeypatch, (12, 0), version="2.7.0")
        # quantization_aware passes (tensorwise); fp8_attention asks for rowwise,
        # which sm120 cannot run before torch 2.8. Neither may convert.
        tcfg = TrainingConfig(quantization_aware="fp8", fp8_attention=True)
        monkeypatch.setattr(tcfg, "fp8_recipe", "rowwise")
        monkeypatch.setattr(
            "soup_cli.utils.fp8.apply_fp8_training", lambda *_a, **_k: True
        )
        _out, console = self._console()
        with pytest.raises(FP8HardwareUnsupportedError, match="fp8_attention.*2.8"):
            apply_v028_speed_memory(
                model=_Attn(), tcfg=tcfg, base_model="m", console=console, device="cuda"
            )
        assert converts == []

    def test_missing_torchao_now_stops_instead_of_warning(self, monkeypatch):
        """INVERTED by the #835 ruling (2026-09-19); this test used to pin the
        yellow advisory. On a Hopper card with no torchao, both explicitly
        requested FP8 settings now end the run at setup, and neither is reported
        as applied."""
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.fp8 import FP8DependencyMissingError
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        _card(monkeypatch, (9, 0))
        monkeypatch.setattr("soup_cli.utils.fp8.is_fp8_available", lambda: False)
        monkeypatch.setitem(sys.modules, "torchao", None)
        for settings in ({"quantization_aware": "fp8"}, {"fp8_attention": True}):
            out, console = self._console()
            with pytest.raises(FP8DependencyMissingError, match=r"soup-cli\[qat\]"):
                apply_v028_speed_memory(
                    model=_Attn(),
                    tcfg=TrainingConfig(**settings),
                    base_model="m",
                    console=console,
                    device="cuda",
                )
            assert "unavailable" not in out.getvalue(), settings

    def test_a_config_that_asked_for_no_fp8_is_untouched(self, monkeypatch):
        """Control: the stop is scoped to an explicit request. The same box, no
        torchao, a config without FP8 -- nothing raises, nothing is printed."""
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        _card(monkeypatch, (9, 0))
        monkeypatch.setattr("soup_cli.utils.fp8.is_fp8_available", lambda: False)
        monkeypatch.setitem(sys.modules, "torchao", None)
        out, console = self._console()
        applied = apply_v028_speed_memory(
            model=_Attn(), tcfg=TrainingConfig(), base_model="m",
            console=console, device="cuda",
        )
        assert applied["fp8"] is False and "fp8_attention" not in applied
        assert out.getvalue() == ""

    def test_sft_stops_at_setup(self, monkeypatch, converts):
        """``SFTTrainerWrapper._apply_quantization_aware`` on an 8.6 card. Wrapping
        its call in ``try/except RuntimeError`` used to leave every test green."""
        from types import SimpleNamespace

        from soup_cli.config.schema import TrainingConfig
        from soup_cli.trainer.sft import SFTTrainerWrapper
        from soup_cli.utils.fp8 import FP8HardwareUnsupportedError

        _card(monkeypatch, (8, 6))
        wrapper = object.__new__(SFTTrainerWrapper)
        wrapper.model = _Attn()
        wrapper.config = SimpleNamespace(base="org/model", backend="transformers")
        wrapper.device = "cuda:0"
        with pytest.raises(FP8HardwareUnsupportedError, match="8.9"):
            wrapper._apply_quantization_aware(TrainingConfig(quantization_aware="fp8"))
        assert converts == []

    @pytest.mark.parametrize(
        "capability, stops", [((8, 6), True), ((12, 0), False)], ids=["sm86", "sm120"]
    )
    def test_a_non_sft_trainer_stops_at_setup(self, monkeypatch, converts, capability, stops):
        """Through KTO's real ``_setup_transformers``: the loaders and PEFT wrapping
        are stubbed, the v028 call and the FP8 gate are not. The sm120 control
        completes setup and converts, so the stop is the card, not the stubs."""
        from types import SimpleNamespace

        import peft
        import transformers

        from soup_cli.config.schema import SoupConfig
        from soup_cli.trainer.kto import KTOTrainerWrapper
        from soup_cli.utils import peft_wiring
        from soup_cli.utils.fp8 import FP8HardwareUnsupportedError

        _card(monkeypatch, capability)
        cfg = SoupConfig(
            base="org/model",
            task="kto",
            data={"train": "x.jsonl", "format": "kto"},
            training={"quantization": "none", "quantization_aware": "fp8"},
        )
        tokenizer = SimpleNamespace(pad_token=None, eos_token="</s>")
        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer
        )
        monkeypatch.setattr(
            transformers.AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: _Attn()
        )
        monkeypatch.setattr(peft, "get_peft_model", lambda model, _config: model)
        monkeypatch.setattr(peft_wiring, "resolve_lora_target_modules", lambda *_a: ["q_proj"])
        monkeypatch.setattr(peft_wiring, "build_lora_config", lambda *_a, **_k: object())
        monkeypatch.setattr(peft_wiring, "apply_pre_lora_patches", lambda *_a: None)
        monkeypatch.setattr(peft_wiring, "apply_post_lora_patches", lambda *_a: None)

        wrapper = object.__new__(KTOTrainerWrapper)
        wrapper.config = cfg
        wrapper.device = "cuda"
        wrapper._trust_remote_code = False
        if stops:
            with pytest.raises(FP8HardwareUnsupportedError, match="8.9"):
                wrapper._setup_transformers(cfg, cfg.training)
            assert converts == []
        else:
            wrapper._setup_transformers(cfg, cfg.training)
            assert converts == ["tensorwise"]


class TestSoupTrainReachesTheStop:
    """#1154 review: ``soup train`` ran the int8 QAT validator for ``quantization_aware:
    fp8`` too, so the run stopped on "pip install torchao" before the FP8 path could
    name ``soup-cli[qat]``. These drive the real command; the run stops in pre-flight,
    before any model is fetched, so the base need not exist."""

    def _train(
        self, tmp_path, monkeypatch, *, quantization_aware, card_ok, torchao,
        gate=None, recipe=None, backend=None, dry_run=False, te=False,
    ):
        import yaml
        from typer.testing import CliRunner

        import soup_cli.utils.fp8 as fp8
        import soup_cli.utils.qat as qat
        from soup_cli.cli import app
        from tests.conftest import strip_ansi

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        (tmp_path / "train.jsonl").write_text(
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": "hello"}]}\n',
            encoding="utf-8",
        )
        (tmp_path / "soup.yaml").write_text(
            yaml.safe_dump({
                "base": "nobody/not-a-real-model",
                "task": "sft",
                "data": {"train": "train.jsonl", "format": "chatml", "max_length": 64},
                **({"backend": backend} if backend else {}),
                "training": {
                    "epochs": 1,
                    "quantization": "none",
                    "quantization_aware": quantization_aware,
                    **({"fp8_recipe": recipe} if recipe else {}),
                },
                "output": "./out",
            }),
            encoding="utf-8",
        )
        reason = "FP8 training requires an Ada or newer GPU (compute capability >= 8.9)"
        monkeypatch.setattr(
            fp8, "fp8_training_supported",
            gate or (lambda recipe="tensorwise": (True, "") if card_ok else (False, reason)),
        )
        # transformer-engine alone satisfies is_fp8_available(), not the torchao probe.
        monkeypatch.setattr(fp8, "is_fp8_available", lambda: torchao or te)
        monkeypatch.setattr(fp8, "is_torchao_float8_available", lambda: torchao)
        monkeypatch.setattr(qat, "is_qat_available", lambda: torchao)
        args = ["train", "--config", "soup.yaml", "--allow-oom-attempt", "--yes"]
        result = CliRunner().invoke(app, args + (["--dry-run"] if dry_run else []))
        # Rich wraps long lines; compare on single-spaced plain text.
        return result, " ".join(strip_ansi(result.output).split())

    def test_a_supported_card_without_torchao_names_the_extra(self, tmp_path, monkeypatch):
        result, out = self._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=True, torchao=False
        )
        assert result.exit_code == 1
        # The brackets survive: Rich would otherwise read [qat] as a markup tag.
        assert 'pip install "soup-cli[qat]"' in out
        assert "pip install torchao" not in out

    def test_an_unsupported_card_is_named_before_torchao(self, tmp_path, monkeypatch):
        result, out = self._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=False, torchao=False
        )
        assert result.exit_code == 1
        assert "compute capability >= 8.9" in out
        assert "soup-cli[qat]" not in out
        assert "pip install torchao" not in out

    def test_int8_qat_still_asks_for_torchao(self, tmp_path, monkeypatch):
        """Control: ``quantization_aware: true`` is the int8 path and keeps its hint."""
        result, out = self._train(
            tmp_path, monkeypatch, quantization_aware=True, card_ok=True, torchao=False
        )
        assert result.exit_code == 1
        assert "pip install torchao" in out
        assert "soup-cli[qat]" not in out

    def test_fp8_with_card_and_torchao_passes_the_preflight(self, tmp_path, monkeypatch):
        """Control: the pre-flight is not refusing every FP8 config. A dry run, so
        the command ends right after validation and must end there successfully --
        an absent "QAT error" alone would also pass on a crash elsewhere."""
        result, out = self._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=True, torchao=True,
            dry_run=True,
        )
        assert result.exit_code == 0, out
        assert "Config valid" in out
        assert "QAT error" not in out and "Note:" not in out


class TestARealRunGetsThroughThePreflight:
    """#1154 review, round 4: 320e8c20 turned the only real-run pass-through control
    into a dry run (a CodeRabbit suggestion), so a pre-flight that refused FP8 on a
    real run with a working card and torchao -- or exited 1 silently -- stayed
    green. The dry-run control stays; this real-run one sits next to it. The base
    does not exist, so reaching model setup proves the pre-flight let the run go."""

    def test_a_real_fp8_run_with_card_and_torchao_gets_past_the_preflight(
        self, tmp_path, monkeypatch
    ):
        _result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=True, torchao=True
        )
        assert "QAT error" not in out
        assert "Setting up model + trainer" in out


class TestTheRecipeReachesTheGate:
    """#1154 review, round 2: every stub above ignored its ``recipe`` argument, so
    dropping ``fp8_recipe=`` at the call site left the suite green -- and then
    ``fp8_recipe: rowwise`` on Windows told the user to install soup-cli[qat]
    instead of printing the Windows refusal. This gate refuses only non-tensorwise
    recipes, so it can see which recipe it was asked about."""

    @staticmethod
    def _gate(recipe="tensorwise"):
        return (recipe == "tensorwise", "ROWWISE REFUSED")

    def test_a_rowwise_refusal_is_printed_not_the_install_hint(self, tmp_path, monkeypatch):
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=True, torchao=False,
            gate=self._gate, recipe="rowwise",
        )

        assert result.exit_code == 1
        assert "ROWWISE REFUSED" in out
        assert "soup-cli[qat]" not in out

    def test_tensorwise_passes_the_same_gate(self, tmp_path, monkeypatch):
        """Control: the gate is not refusing everything."""
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=True, torchao=False,
            gate=self._gate, recipe="tensorwise",
        )

        assert "ROWWISE REFUSED" not in out
        assert 'pip install "soup-cli[qat]"' in out


class TestADryRunDoesNotFailOnTheLocalCard:
    """#1154 review ruling: a dry run validates a config, and FP8 configs are
    written on a laptop for a remote card. The card is a note there; the
    dependency and schema checks stay errors; the real run still stops."""

    def test_an_unsupported_card_is_a_note_and_exits_0(self, tmp_path, monkeypatch):
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=False, torchao=True,
            dry_run=True,
        )

        assert result.exit_code == 0, out
        assert "Note: this machine could not run it:" in out
        assert "compute capability >= 8.9" in out
        assert "Config valid" in out

    def test_a_missing_torchao_still_fails_the_dry_run(self, tmp_path, monkeypatch):
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=False, torchao=False,
            dry_run=True,
        )

        assert result.exit_code == 1
        assert 'pip install "soup-cli[qat]"' in out

    def test_the_real_run_still_stops_on_the_card(self, tmp_path, monkeypatch):
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=False, torchao=True,
        )

        assert result.exit_code == 1
        assert "compute capability >= 8.9" in out
        assert "Note:" not in out


class TestThePreflightAsksForTorchaoItself:
    def test_transformer_engine_alone_stops_before_the_model_loads(self, tmp_path, monkeypatch):
        """``is_fp8_available()`` accepts transformer-engine, but the converter is
        torchao's: that box used to pass pre-flight, load the model, then stop."""
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=True, torchao=False,
            te=True,
        )

        assert result.exit_code == 1
        assert 'pip install "soup-cli[qat]"' in out

    def test_unsloth_gets_its_refusal_without_an_install_hint(self, tmp_path, monkeypatch):
        """The unsloth refusal already stops the run; a torchao hint under it would
        point at a package that cannot help."""
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=True, torchao=False,
            backend="unsloth",
        )

        assert result.exit_code == 1
        assert "not compatible with the unsloth backend" in out
        # Broad on purpose (round 3): the narrow check passed while the int8
        # "pip install torchao" was printed instead.
        assert "pip install" not in out
        assert "torchao" not in out.lower().replace("not compatible", "")


class TestTheSecondAttentionRaiseSite:
    def test_torchao_present_without_float8_raises_the_dependency_error(self, monkeypatch):
        """``advanced_precision``'s second raise site: torchao importable, its
        float8 module not. Unreachable through the trainers (the fp8 block stops
        first), so it is pinned here directly."""
        import sys
        import types

        import torch.nn as nn

        import soup_cli.utils.advanced_precision as ap
        import soup_cli.utils.fp8 as fp8

        monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
        monkeypatch.setitem(sys.modules, "torchao.float8", None)
        monkeypatch.setattr(fp8, "fp8_training_supported", lambda recipe="tensorwise": (True, ""))

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4)

        with pytest.raises(fp8.FP8DependencyMissingError, match="soup-cli"):
            ap.apply_fp8_attention(_Attn())


class TestTheDryRunNoteRound3:
    """#1154 review, round 3: the note's own call site dropped the recipe, its FP8
    condition was unpinned, and a dry run lacking torchao hid the card verdict."""

    def test_a_rowwise_dry_run_notes_the_recipe_refusal(self, tmp_path, monkeypatch):
        """The note must ask about the configured recipe, not the default."""
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=True, torchao=True,
            gate=TestTheRecipeReachesTheGate._gate, recipe="rowwise", dry_run=True,
        )

        assert result.exit_code == 0, out
        assert "Note: this machine could not run it: ROWWISE REFUSED" in out

    def test_an_int8_dry_run_prints_no_fp8_note(self, tmp_path, monkeypatch):
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware=True, card_ok=False, torchao=True,
            dry_run=True,
        )

        assert result.exit_code == 0, out
        assert "Note:" not in out

    def test_an_unsloth_dry_run_prints_only_the_refusal(self, tmp_path, monkeypatch):
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=False, torchao=False,
            backend="unsloth", dry_run=True,
        )

        assert result.exit_code == 1
        assert "not compatible with the unsloth backend" in out
        assert "Note:" not in out and "pip install" not in out

    def test_no_torchao_and_no_card_shows_both(self, tmp_path, monkeypatch):
        """Nit taken: the card verdict is printed before the dependency error exits."""
        result, out = TestSoupTrainReachesTheStop()._train(
            tmp_path, monkeypatch, quantization_aware="fp8", card_ok=False, torchao=False,
            dry_run=True,
        )

        assert result.exit_code == 1
        assert "Note: this machine could not run it:" in out
        assert 'pip install "soup-cli[qat]"' in out


class TestTheTorchaoProbeItself:
    """Every other test stubs ``is_torchao_float8_available``; these run the real one
    against import-machinery stubs, so a probe that answered True on ImportError
    (which would let a torchao-less run load the model, then stop) fails here."""

    def test_torchao_without_float8_is_not_available(self, monkeypatch):
        import types

        from soup_cli.utils.fp8 import is_torchao_float8_available

        monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
        monkeypatch.setitem(sys.modules, "torchao.float8", None)

        assert is_torchao_float8_available() is False

    def test_no_torchao_at_all_is_not_available(self, monkeypatch):
        from soup_cli.utils.fp8 import is_torchao_float8_available

        monkeypatch.setitem(sys.modules, "torchao", None)

        assert is_torchao_float8_available() is False

    def test_a_float8_module_with_the_converter_is_available(self, monkeypatch):
        import types

        from soup_cli.utils.fp8 import is_torchao_float8_available

        pkg = types.ModuleType("torchao")
        pkg.__path__ = []
        float8 = types.ModuleType("torchao.float8")
        float8.convert_to_float8_training = lambda *a, **k: None
        pkg.float8 = float8
        monkeypatch.setitem(sys.modules, "torchao", pkg)
        monkeypatch.setitem(sys.modules, "torchao.float8", float8)

        assert is_torchao_float8_available() is True


class TestTheConversionFailedWording:
    def test_a_failed_conversion_says_so(self, monkeypatch):
        """Round 1 renamed the yellow line; nothing held it (non-blocking, taken)."""
        from soup_cli.config.schema import TrainingConfig
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        monkeypatch.setattr("soup_cli.utils.fp8.apply_fp8_training", lambda *a, **k: False)
        out, console = TestTheRunStops()._console()
        applied = apply_v028_speed_memory(
            model=_Attn(), tcfg=TrainingConfig(quantization_aware="fp8"), base_model="m",
            console=console, device="cuda",
        )

        assert applied["fp8"] is False
        assert "the float8 conversion failed" in out.getvalue()
        assert "torchao.float8 missing" not in out.getvalue()
