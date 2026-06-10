"""v0.71.21 "Precision & rollout lift" tests.

Closes #141 (live fp8_attention + NVFP4), #124 (vLLM sleep mode),
#125 (multi-turn agent rollout launchers), #228 (apple-adapter converter),
#97 (delinearize-llama4 runtime).

Hardware notes: fp8_attention / nvfp4 / vllm-sleep are BETA hw-gated — the
gates + dispatch are exercised here via monkeypatched capability probes and
fake torchao / vllm modules (sys.modules swap via ``monkeypatch.setitem``,
the established test_v0531_142 pattern). The apple-adapter conversion and
the delinearize runtime are fully CPU-validatable and run on real arrays.
"""

from __future__ import annotations

import itertools
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

_SRC = Path(__file__).resolve().parent.parent / "src" / "soup_cli"

runner = CliRunner()


def _torch_or_skip():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# #141 — attention-projection detection
# ---------------------------------------------------------------------------


class TestIsAttentionProjection:
    def test_canonical_projections(self):
        from soup_cli.utils.advanced_precision import is_attention_projection

        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert is_attention_projection(f"model.layers.0.self_attn.{name}")

    def test_fused_variants(self):
        from soup_cli.utils.advanced_precision import is_attention_projection

        assert is_attention_projection("transformer.h.0.attn.c_attn")
        assert is_attention_projection("model.layers.3.self_attn.qkv_proj")
        assert is_attention_projection(
            "transformer.layers.1.attention.query_key_value"
        )

    def test_mlp_projections_rejected(self):
        from soup_cli.utils.advanced_precision import is_attention_projection

        assert not is_attention_projection("model.layers.0.mlp.gate_proj")
        assert not is_attention_projection("model.layers.0.mlp.down_proj")
        assert not is_attention_projection("lm_head")

    def test_defensive_inputs(self):
        from soup_cli.utils.advanced_precision import is_attention_projection

        assert not is_attention_projection("")
        assert not is_attention_projection(None)  # type: ignore[arg-type]
        assert not is_attention_projection(123)  # type: ignore[arg-type]
        assert not is_attention_projection("a\x00q_proj")

    def test_substring_not_enough(self):
        """Last-component match only — 'my_q_proj_extra' is not q_proj."""
        from soup_cli.utils.advanced_precision import is_attention_projection

        assert not is_attention_projection("model.layers.0.my_q_proj_extra")


class TestIsBlackwellGpu:
    def test_false_without_cuda(self, monkeypatch):
        torch = _torch_or_skip()
        from soup_cli.utils.advanced_precision import is_blackwell_gpu

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert is_blackwell_gpu() is False

    @pytest.mark.parametrize(
        ("capability", "expected"),
        [((8, 6), False), ((9, 0), False), ((10, 0), True), ((12, 0), True)],
    )
    def test_capability_matrix(self, monkeypatch, capability, expected):
        torch = _torch_or_skip()
        from soup_cli.utils.advanced_precision import is_blackwell_gpu

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda _i=0: capability
        )
        assert is_blackwell_gpu() is expected


# ---------------------------------------------------------------------------
# #141 — apply_fp8_attention (live)
# ---------------------------------------------------------------------------


def _tiny_attn_model(with_attention: bool = True):
    """A tiny real-torch model with q/k/v/o projections + an MLP linear."""
    torch = _torch_or_skip()
    nn = torch.nn

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.o_proj = nn.Linear(8, 8)

    class _Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 8)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            if with_attention:
                self.self_attn = _Attn()
            self.mlp = _Mlp()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Block()])

    return _Model()


def _install_fake_torchao_float8(monkeypatch, record):
    """Inject a fake ``torchao.float8`` that performs a real module swap."""
    torch = _torch_or_skip()
    nn = torch.nn

    class _F8(nn.Linear):
        pass

    _F8.__name__ = "Float8Linear"

    class _Cfg:
        @classmethod
        def from_recipe_name(cls, name):
            record["recipe"] = name
            return cls()

    def _convert(model, config=None, module_filter_fn=None):
        record["config"] = config
        swapped = []
        modules = dict(model.named_modules())
        for fqn, mod in list(modules.items()):
            if not isinstance(mod, nn.Linear):
                continue
            if type(mod).__name__ == "Float8Linear":
                continue
            if module_filter_fn is not None and not module_filter_fn(mod, fqn):
                continue
            parent_fqn, _, child = fqn.rpartition(".")
            parent = modules[parent_fqn] if parent_fqn else model
            new = _F8(mod.in_features, mod.out_features, bias=mod.bias is not None)
            setattr(parent, child, new)
            swapped.append(fqn)
        record["swapped"] = swapped
        return model

    fake_cfg_mod = types.ModuleType("torchao.float8.config")
    fake_cfg_mod.Float8LinearConfig = _Cfg
    fake_f8 = types.ModuleType("torchao.float8")
    fake_f8.convert_to_float8_training = _convert
    fake_f8.config = fake_cfg_mod
    fake_root = types.ModuleType("torchao")
    fake_root.float8 = fake_f8
    monkeypatch.setitem(sys.modules, "torchao", fake_root)
    monkeypatch.setitem(sys.modules, "torchao.float8", fake_f8)
    monkeypatch.setitem(sys.modules, "torchao.float8.config", fake_cfg_mod)
    return _F8


def _enable_fp8_gates(monkeypatch):
    # The torchao probe checks sys.modules first (v0.27.0 None-stub
    # idiom), so a bare fake root is enough for the gate; conversion
    # tests install the full fake via _install_fake_torchao_float8.
    if "torchao" not in sys.modules:
        monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setattr("soup_cli.utils.fp8.is_fp8_gpu_supported", lambda: True)


class TestApplyFp8Attention:
    def test_none_model_rejected(self):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        with pytest.raises(TypeError, match="model"):
            apply_fp8_attention(None)

    def test_no_torchao_friendly_gate(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        # v0.27.0 None-stub idiom — the probe must treat this as absent.
        monkeypatch.setitem(sys.modules, "torchao", None)
        with pytest.raises(RuntimeError, match="torchao"):
            apply_fp8_attention(_tiny_attn_model())

    def test_non_hopper_friendly_gate(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        monkeypatch.setitem(
            sys.modules, "torchao", types.ModuleType("torchao")
        )
        monkeypatch.setattr(
            "soup_cli.utils.fp8.is_fp8_gpu_supported", lambda: False
        )
        with pytest.raises(RuntimeError, match="Hopper"):
            apply_fp8_attention(_tiny_attn_model())

    @pytest.mark.parametrize("recipe", [True, "", None, 123, "row\x00wise"])
    def test_bad_recipe_rejected(self, monkeypatch, recipe):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        _enable_fp8_gates(monkeypatch)
        with pytest.raises((TypeError, ValueError), match="recipe"):
            apply_fp8_attention(_tiny_attn_model(), recipe=recipe)

    def test_converts_only_attention(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        record: dict = {}
        _install_fake_torchao_float8(monkeypatch, record)
        _enable_fp8_gates(monkeypatch)
        model = _tiny_attn_model()
        count = apply_fp8_attention(model, recipe="rowwise")
        assert count == 4
        assert record["recipe"] == "rowwise"
        block = model.layers[0]
        assert type(block.self_attn.q_proj).__name__ == "Float8Linear"
        assert type(block.self_attn.o_proj).__name__ == "Float8Linear"
        # MLP linears untouched.
        assert type(block.mlp.gate_proj).__name__ == "Linear"

    def test_already_converted_counted_not_reswapped(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        record: dict = {}
        f8_cls = _install_fake_torchao_float8(monkeypatch, record)
        _enable_fp8_gates(monkeypatch)
        model = _tiny_attn_model()
        # Pre-convert q_proj manually.
        model.layers[0].self_attn.q_proj = f8_cls(8, 8)
        count = apply_fp8_attention(model)
        assert count == 4  # all four attention projections are now Float8
        assert "self_attn.q_proj" not in " ".join(record["swapped"])
        assert len(record["swapped"]) == 3

    def test_no_attention_projections_value_error(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        record: dict = {}
        _install_fake_torchao_float8(monkeypatch, record)
        _enable_fp8_gates(monkeypatch)
        with pytest.raises(ValueError, match="no attention projections"):
            apply_fp8_attention(_tiny_attn_model(with_attention=False))


# ---------------------------------------------------------------------------
# #141 — apply_nvfp4 (live, Blackwell-gated)
# ---------------------------------------------------------------------------


def _install_fake_torchao_quant(monkeypatch, record, *, with_nvfp4=True):
    fake_q = types.ModuleType("torchao.quantization")

    class _NVFP4Config:
        pass

    def _quantize(model, config):
        record["model"] = model
        record["config"] = config

    fake_q.quantize_ = _quantize
    if with_nvfp4:
        fake_q.NVFP4Config = _NVFP4Config
    fake_root = types.ModuleType("torchao")
    fake_root.quantization = fake_q
    monkeypatch.setitem(sys.modules, "torchao", fake_root)
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_q)


class TestApplyNvfp4:
    def test_none_model_rejected(self):
        from soup_cli.utils.advanced_precision import apply_nvfp4

        with pytest.raises(TypeError, match="model"):
            apply_nvfp4(None)

    def test_non_blackwell_friendly_gate(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_nvfp4

        monkeypatch.setattr(
            "soup_cli.utils.advanced_precision.is_blackwell_gpu", lambda: False
        )
        with pytest.raises(RuntimeError, match="Blackwell"):
            apply_nvfp4(_tiny_attn_model())

    def test_missing_torchao_friendly(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_nvfp4

        monkeypatch.setattr(
            "soup_cli.utils.advanced_precision.is_blackwell_gpu", lambda: True
        )
        monkeypatch.setitem(sys.modules, "torchao", None)
        with pytest.raises(RuntimeError, match="torchao"):
            apply_nvfp4(_tiny_attn_model())

    def test_old_torchao_missing_nvfp4config(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_nvfp4

        record: dict = {}
        _install_fake_torchao_quant(monkeypatch, record, with_nvfp4=False)
        monkeypatch.setattr(
            "soup_cli.utils.advanced_precision.is_blackwell_gpu", lambda: True
        )
        with pytest.raises(RuntimeError, match="NVFP4Config"):
            apply_nvfp4(_tiny_attn_model())

    def test_happy_path_quantizes(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_nvfp4

        record: dict = {}
        _install_fake_torchao_quant(monkeypatch, record)
        monkeypatch.setattr(
            "soup_cli.utils.advanced_precision.is_blackwell_gpu", lambda: True
        )
        model = _tiny_attn_model()
        count = apply_nvfp4(model)
        assert count == 5  # 4 attention + 1 mlp linear
        assert record["model"] is model
        assert type(record["config"]).__name__ == "_NVFP4Config"


# ---------------------------------------------------------------------------
# #141 — apply_v028_speed_memory wiring
# ---------------------------------------------------------------------------


class TestV028PrecisionWiring:
    def _tcfg(self, **kwargs):
        base = {
            "use_cut_ce": False,
            "quantization_aware": False,
            "kernel_auto_compose": False,
        }
        base.update(kwargs)
        return SimpleNamespace(**base)

    def test_no_features_dict_unchanged(self):
        """Back-compat regression: 3-key exact dict on the no-features path."""
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        result = apply_v028_speed_memory(
            model=object(), tcfg=self._tcfg(), base_model="x/y",
        )
        assert result == {
            "cut_ce": False, "fp8": False, "kernel_auto_compose": False,
        }

    def test_fp8_attention_gate_failure_degrades(self, monkeypatch):
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        def _gate(model, **kwargs):
            raise RuntimeError("no Hopper")

        # Hermetic: force the gate (review fix — the bare call passed only
        # because the host lacks torchao/Hopper).
        monkeypatch.setattr(
            "soup_cli.utils.advanced_precision.apply_fp8_attention", _gate
        )
        result = apply_v028_speed_memory(
            model=object(),
            tcfg=self._tcfg(quantization_aware="fp8", fp8_attention=True),
            base_model="x/y",
        )
        assert result["fp8_attention"] is False

    def test_fp8_attention_applied(self, monkeypatch):
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        monkeypatch.setattr(
            "soup_cli.utils.advanced_precision.apply_fp8_attention",
            lambda model, recipe="tensorwise": 4,
        )
        result = apply_v028_speed_memory(
            model=object(),
            tcfg=self._tcfg(quantization_aware="fp8", fp8_attention=True),
            base_model="x/y",
        )
        assert result["fp8_attention"] is True

    def test_nvfp4_gate_failure_degrades(self, monkeypatch):
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        def _gate(model, **kwargs):
            raise RuntimeError("no Blackwell")

        monkeypatch.setattr(
            "soup_cli.utils.advanced_precision.apply_nvfp4", _gate
        )
        result = apply_v028_speed_memory(
            model=object(), tcfg=self._tcfg(nvfp4=True), base_model="x/y",
        )
        assert result["nvfp4"] is False

    def test_nvfp4_applied(self, monkeypatch):
        from soup_cli.utils.v028_features import apply_v028_speed_memory

        monkeypatch.setattr(
            "soup_cli.utils.advanced_precision.apply_nvfp4", lambda model: 2,
        )
        result = apply_v028_speed_memory(
            model=object(), tcfg=self._tcfg(nvfp4=True), base_model="x/y",
        )
        assert result["nvfp4"] is True


# ---------------------------------------------------------------------------
# #124 — vLLM sleep mode (live)
# ---------------------------------------------------------------------------


def _install_fake_vllm(monkeypatch, version="0.8.1"):
    fake = types.ModuleType("vllm")
    fake.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", fake)
    return fake


class TestParseVersionTuple:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0.8.1", (0, 8, 1)),
            ("0.7.0.dev0", (0, 7, 0)),
            ("0.6", (0, 6)),
            ("garbage", ()),
            ("", ()),
        ],
    )
    def test_matrix(self, raw, expected):
        from soup_cli.utils.grpo_long_context import _parse_version_tuple

        assert _parse_version_tuple(raw) == expected


class TestVllmSupportsSleepMode:
    def test_false_when_vllm_missing(self, monkeypatch):
        from soup_cli.utils.grpo_long_context import vllm_supports_sleep_mode

        monkeypatch.setitem(sys.modules, "vllm", None)
        assert vllm_supports_sleep_mode() is False

    def test_true_on_modern_vllm(self, monkeypatch):
        from soup_cli.utils.grpo_long_context import vllm_supports_sleep_mode

        _install_fake_vllm(monkeypatch, "0.8.1")
        assert vllm_supports_sleep_mode() is True

    def test_false_on_old_vllm(self, monkeypatch):
        from soup_cli.utils.grpo_long_context import vllm_supports_sleep_mode

        _install_fake_vllm(monkeypatch, "0.6.2")
        assert vllm_supports_sleep_mode() is False


class TestApplyVllmSleepMode:
    def test_none_rejected(self):
        from soup_cli.utils.grpo_long_context import apply_vllm_sleep_mode

        with pytest.raises(TypeError, match="engine_args"):
            apply_vllm_sleep_mode(None)

    def test_missing_vllm_friendly(self, monkeypatch):
        from soup_cli.utils.grpo_long_context import apply_vllm_sleep_mode

        monkeypatch.setitem(sys.modules, "vllm", None)
        with pytest.raises(RuntimeError, match=r"vLLM >= 0\.7"):
            apply_vllm_sleep_mode(SimpleNamespace())

    def test_old_vllm_friendly(self, monkeypatch):
        from soup_cli.utils.grpo_long_context import apply_vllm_sleep_mode

        _install_fake_vllm(monkeypatch, "0.6.2")
        with pytest.raises(RuntimeError, match=r"0\.6\.2"):
            apply_vllm_sleep_mode(SimpleNamespace())

    def test_happy_sets_enable_sleep_mode(self, monkeypatch):
        from soup_cli.utils.grpo_long_context import apply_vllm_sleep_mode

        _install_fake_vllm(monkeypatch, "0.8.1")
        args = SimpleNamespace()
        out = apply_vllm_sleep_mode(args)
        assert out is args
        assert args.enable_sleep_mode is True


class TestVllmSleepCycle:
    def test_sleep_then_wake(self):
        from soup_cli.utils.grpo_long_context import vllm_sleep_cycle

        calls: list = []

        class _Engine:
            def sleep(self, level=1):
                calls.append(("sleep", level))

            def wake_up(self):
                calls.append(("wake_up",))

        with vllm_sleep_cycle(_Engine()):
            calls.append(("body",))
        assert calls == [("sleep", 1), ("body",), ("wake_up",)]

    def test_wakes_even_on_body_exception(self):
        from soup_cli.utils.grpo_long_context import vllm_sleep_cycle

        calls: list = []

        class _Engine:
            def sleep(self, level=1):
                calls.append("sleep")

            def wake_up(self):
                calls.append("wake_up")

        with pytest.raises(RuntimeError, match="boom"):
            with vllm_sleep_cycle(_Engine()):
                raise RuntimeError("boom")
        assert calls == ["sleep", "wake_up"]

    def test_engine_without_sleep_warns_not_crashes(self, caplog):
        import logging

        from soup_cli.utils.grpo_long_context import vllm_sleep_cycle

        with caplog.at_level(logging.WARNING, logger="soup_cli.utils.grpo_long_context"):
            with vllm_sleep_cycle(object()):
                pass
        assert any("sleep" in rec.message for rec in caplog.records)


class TestMaybeEnableTrlSleepMode:
    def test_sets_kwarg_when_trl_exposes_it(self):
        from soup_cli.utils.grpo_long_context import maybe_enable_trl_sleep_mode

        kwargs: dict = {}
        ok = maybe_enable_trl_sleep_mode(
            kwargs, ("output_dir", "vllm_enable_sleep_mode"), None,
        )
        assert ok is True
        assert kwargs["vllm_enable_sleep_mode"] is True

    def test_advisory_when_trl_lacks_hook(self):
        import io

        from rich.console import Console

        from soup_cli.utils.grpo_long_context import maybe_enable_trl_sleep_mode

        buf = io.StringIO()
        kwargs: dict = {}
        ok = maybe_enable_trl_sleep_mode(
            kwargs, ("output_dir",), Console(file=buf),
        )
        assert ok is False
        assert kwargs == {}
        assert "sleep" in buf.getvalue().lower()


class TestSleepModeSourceWiring:
    def test_create_vllm_engine_accepts_sleep_mode(self):
        source = (_SRC / "utils" / "vllm.py").read_text(encoding="utf-8")
        assert "sleep_mode" in source
        assert "apply_vllm_sleep_mode" in source

    def test_grpo_trainer_wires_sleep_mode(self):
        source = (_SRC / "trainer" / "grpo.py").read_text(encoding="utf-8")
        assert "maybe_enable_trl_sleep_mode" in source


# ---------------------------------------------------------------------------
# #125 — rollout launchers
# ---------------------------------------------------------------------------

_MODULE_COUNTER = itertools.count()


def _write_rollout_module(tmp_path, monkeypatch, body: str) -> str:
    """Write an importable rollout module; return the 'module:fn' spec."""
    name = f"soup_test_rollout_{next(_MODULE_COUNTER)}"
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return f"{name}:rollout"


class TestValidateRolloutFunc:
    def test_none_passthrough(self):
        from soup_cli.utils.agent_rollout import validate_rollout_func

        assert validate_rollout_func(None) is None

    def test_happy(self):
        from soup_cli.utils.agent_rollout import validate_rollout_func

        assert validate_rollout_func("my_mod.sub:my_fn") == "my_mod.sub:my_fn"

    @pytest.mark.parametrize(
        "bad",
        ["", "no-colon", "mod:", ":fn", "mod:fn:extra", "a b:c", "mod\x00:fn", 7],
    )
    def test_rejection_matrix(self, bad):
        from soup_cli.utils.agent_rollout import validate_rollout_func

        with pytest.raises(ValueError, match="rollout_func"):
            validate_rollout_func(bad)

    def test_oversize_rejected(self):
        from soup_cli.utils.agent_rollout import validate_rollout_func

        with pytest.raises(ValueError, match="rollout_func"):
            validate_rollout_func("a" * 300 + ":fn")


class TestResolveRolloutFunc:
    def test_resolves_callable(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import resolve_rollout_func

        spec = _write_rollout_module(
            tmp_path, monkeypatch, "def rollout(prompts):\n    return []\n"
        )
        fn = resolve_rollout_func(spec)
        assert callable(fn)

    def test_missing_module_friendly(self):
        from soup_cli.utils.agent_rollout import resolve_rollout_func

        with pytest.raises(ValueError, match="could not be imported"):
            resolve_rollout_func("definitely_not_a_module_xyz:fn")

    def test_missing_attr_friendly(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import resolve_rollout_func

        spec = _write_rollout_module(
            tmp_path, monkeypatch, "def rollout(prompts):\n    return []\n"
        )
        module_name = spec.split(":")[0]
        with pytest.raises(ValueError, match="no attribute"):
            resolve_rollout_func(f"{module_name}:nope")

    def test_non_callable_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import resolve_rollout_func

        spec = _write_rollout_module(tmp_path, monkeypatch, "rollout = 42\n")
        with pytest.raises(ValueError, match="not callable"):
            resolve_rollout_func(spec)


class TestRolloutResult:
    def test_frozen(self):
        import dataclasses

        from soup_cli.utils.agent_rollout import RolloutResult

        result = RolloutResult(backend="openenv", rows=({"prompt": "x"},))
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.backend = "art"  # type: ignore[misc]

    def test_rows_must_be_tuple(self):
        from soup_cli.utils.agent_rollout import RolloutResult

        with pytest.raises(TypeError, match="tuple"):
            RolloutResult(backend="openenv", rows=[{"prompt": "x"}])  # type: ignore[arg-type]

    def test_unknown_backend_rejected(self):
        from soup_cli.utils.agent_rollout import RolloutResult

        with pytest.raises(ValueError, match="not supported"):
            RolloutResult(backend="trlx", rows=({"prompt": "x"},))

    def test_row_without_prompt_rejected(self):
        from soup_cli.utils.agent_rollout import RolloutResult

        with pytest.raises(ValueError, match="prompt"):
            RolloutResult(backend="openenv", rows=({"answer": "y"},))

    def test_row_cap(self):
        from soup_cli.utils.agent_rollout import (
            _MAX_ROLLOUT_ROWS,
            RolloutResult,
        )

        rows = tuple({"prompt": "x"} for _ in range(_MAX_ROLLOUT_ROWS + 1))
        with pytest.raises(ValueError, match="rows"):
            RolloutResult(backend="openenv", rows=rows)


class TestLaunchRolloutOpenenv:
    def test_happy_path(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path,
            monkeypatch,
            "def rollout(prompts):\n"
            "    return [\n"
            "        {'prompt': p, 'answer': 'a'} for p in prompts\n"
            "    ] + [{'prompt': 'extra'}]\n",
        )
        result = launch_rollout(
            "openenv", prompts=["hello", "world"], rollout_func=spec,
        )
        assert result.backend == "openenv"
        assert len(result.rows) == 3
        assert result.rows[0]["prompt"] == "hello"
        assert result.rows[0]["answer"] == "a"
        assert "answer" not in result.rows[2]

    def test_message_list_prompts_preserved(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path,
            monkeypatch,
            "def rollout(prompts):\n"
            "    return [{'prompt': [{'role': 'user', 'content': 'hi'}]}]\n",
        )
        result = launch_rollout("openenv", prompts=[], rollout_func=spec)
        assert result.rows[0]["prompt"] == [{"role": "user", "content": "hi"}]

    def test_openenv_requires_rollout_func(self):
        from soup_cli.utils.agent_rollout import launch_rollout

        with pytest.raises(ValueError, match="rollout_func"):
            launch_rollout("openenv", prompts=["x"])

    def test_empty_rows_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path, monkeypatch, "def rollout(prompts):\n    return []\n"
        )
        with pytest.raises(ValueError, match="no rows"):
            launch_rollout("openenv", prompts=["x"], rollout_func=spec)

    def test_non_iterable_rows_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path, monkeypatch, "def rollout(prompts):\n    return 42\n"
        )
        with pytest.raises(ValueError, match="iterable"):
            launch_rollout("openenv", prompts=["x"], rollout_func=spec)

    def test_non_mapping_row_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path, monkeypatch, "def rollout(prompts):\n    return ['x']\n"
        )
        with pytest.raises(ValueError, match="mapping"):
            launch_rollout("openenv", prompts=["x"], rollout_func=spec)

    @pytest.mark.parametrize("bad_steps", [True, 0, -3, "many"])
    def test_max_steps_validation(self, tmp_path, monkeypatch, bad_steps):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path,
            monkeypatch,
            "def rollout(prompts):\n    return [{'prompt': 'x'}]\n",
        )
        with pytest.raises((TypeError, ValueError), match="max_steps"):
            launch_rollout(
                "openenv", prompts=["x"], rollout_func=spec,
                max_steps=bad_steps,
            )

    def test_prompts_string_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path,
            monkeypatch,
            "def rollout(prompts):\n    return [{'prompt': 'x'}]\n",
        )
        with pytest.raises(TypeError, match="prompts"):
            launch_rollout("openenv", prompts="hi", rollout_func=spec)


class TestLaunchRolloutExternal:
    @pytest.mark.parametrize(
        ("name", "pkg"),
        [("art", "openpipe-art"), ("ruler", "ruler-eval"), ("nemo_gym", "nemo-gym")],
    )
    def test_missing_package_friendly(self, monkeypatch, name, pkg):
        from soup_cli.utils import agent_rollout

        monkeypatch.setattr(agent_rollout, "_spec_exists", lambda _n: False)
        with pytest.raises(ImportError, match=pkg):
            agent_rollout.launch_rollout(name, prompts=["x"])

    def test_present_package_honest_gate(self, monkeypatch):
        from soup_cli.utils import agent_rollout

        monkeypatch.setattr(agent_rollout, "_spec_exists", lambda _n: True)
        with pytest.raises(RuntimeError, match="openenv"):
            agent_rollout.launch_rollout("art", prompts=["x"])

    def test_runner_override_seam(self, monkeypatch):
        from soup_cli.utils import agent_rollout

        def _fake_runner(**kwargs):
            return [{"prompt": "from-art", "answer": "ok"}]

        monkeypatch.setitem(
            agent_rollout._EXTERNAL_ROLLOUT_RUNNERS, "art", _fake_runner
        )
        result = agent_rollout.launch_rollout("art", prompts=["x"])
        assert result.backend == "art"
        assert result.rows[0]["prompt"] == "from-art"

    def test_unknown_backend_still_validated_first(self):
        from soup_cli.utils.agent_rollout import launch_rollout

        with pytest.raises(ValueError, match="not supported"):
            launch_rollout("trlx", prompts=["x"])


class TestRolloutLiveWiredFlags:
    def test_openenv_live_wired(self):
        from soup_cli.utils.agent_rollout import get_rollout_backend_spec

        assert get_rollout_backend_spec("openenv").live_wired is True

    @pytest.mark.parametrize("name", ["art", "ruler", "nemo_gym"])
    def test_external_backends_stay_gated(self, name):
        from soup_cli.utils.agent_rollout import get_rollout_backend_spec

        assert get_rollout_backend_spec(name).live_wired is False


class TestRolloutSchema:
    def test_openenv_with_func_happy(self):
        from soup_cli.config.loader import load_config_from_string

        cfg = load_config_from_string(
            "base: a/b\n"
            "task: grpo\n"
            "data: {train: x.jsonl}\n"
            "training: {rollout_backend: openenv, rollout_func: 'my_mod:my_fn'}\n"
        )
        assert cfg.training.rollout_func == "my_mod:my_fn"

    def test_func_without_backend_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="openenv"):
            load_config_from_string(
                "base: a/b\n"
                "task: grpo\n"
                "data: {train: x.jsonl}\n"
                "training: {rollout_func: 'my_mod:my_fn'}\n"
            )

    def test_openenv_without_func_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="rollout_func"):
            load_config_from_string(
                "base: a/b\n"
                "task: grpo\n"
                "data: {train: x.jsonl}\n"
                "training: {rollout_backend: openenv}\n"
            )

    def test_func_with_art_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="openenv"):
            load_config_from_string(
                "base: a/b\n"
                "task: grpo\n"
                "data: {train: x.jsonl}\n"
                "training: {rollout_backend: art, rollout_func: 'my_mod:my_fn'}\n"
            )

    def test_bad_func_shape_rejected(self):
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError, match="rollout_func"):
            load_config_from_string(
                "base: a/b\n"
                "task: grpo\n"
                "data: {train: x.jsonl}\n"
                "training: {rollout_backend: openenv, rollout_func: 'no-colon'}\n"
            )

    def test_default_none(self):
        from soup_cli.config.schema import TrainingConfig

        assert TrainingConfig().rollout_func is None


class TestRolloutGrpoWiring:
    def test_grpo_trainer_launches_rollout(self):
        source = (_SRC / "trainer" / "grpo.py").read_text(encoding="utf-8")
        assert "launch_rollout" in source
        assert "rollout_backend" in source


# ---------------------------------------------------------------------------
# #228 — apple-adapter live converter
# ---------------------------------------------------------------------------


class TestLoraKeyMapping:
    def test_hf_to_mlx_strips_prefix_and_lowers(self):
        from soup_cli.utils.apple_adapter import hf_key_to_mlx

        key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        assert hf_key_to_mlx(key) == "model.layers.0.self_attn.q_proj.lora_a"

    def test_hf_to_mlx_lora_b(self):
        from soup_cli.utils.apple_adapter import hf_key_to_mlx

        key = "base_model.model.model.layers.2.mlp.gate_proj.lora_B.weight"
        assert hf_key_to_mlx(key) == "model.layers.2.mlp.gate_proj.lora_b"

    def test_non_lora_key_returns_none(self):
        from soup_cli.utils.apple_adapter import hf_key_to_mlx

        assert hf_key_to_mlx("base_model.model.model.embed_tokens.weight") is None

    def test_mlx_to_hf_round_trip(self):
        from soup_cli.utils.apple_adapter import hf_key_to_mlx, mlx_key_to_hf

        original = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        assert mlx_key_to_hf(hf_key_to_mlx(original)) == original

    def test_mlx_non_lora_returns_none(self):
        from soup_cli.utils.apple_adapter import mlx_key_to_hf

        assert mlx_key_to_hf("model.layers.0.self_attn.q_proj.weight") is None


class TestConvertArrays:
    def _hf_arrays(self):
        rng = np.random.default_rng(0)
        return {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
                rng.standard_normal((4, 16)).astype(np.float32),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
                rng.standard_normal((16, 4)).astype(np.float32),
        }

    def test_hf_to_mlx_transposes(self):
        from soup_cli.utils.apple_adapter import convert_hf_to_mlx_arrays

        converted, skipped = convert_hf_to_mlx_arrays(self._hf_arrays())
        assert skipped == ()
        lora_a = converted["model.layers.0.self_attn.q_proj.lora_a"]
        lora_b = converted["model.layers.0.self_attn.q_proj.lora_b"]
        assert lora_a.shape == (16, 4)  # [in, r]
        assert lora_b.shape == (4, 16)  # [r, out]

    def test_non_lora_keys_skipped(self):
        from soup_cli.utils.apple_adapter import convert_hf_to_mlx_arrays

        arrays = self._hf_arrays()
        arrays["base_model.model.model.embed_tokens.weight"] = np.zeros(
            (4, 4), dtype=np.float32
        )
        converted, skipped = convert_hf_to_mlx_arrays(arrays)
        assert len(converted) == 2
        assert skipped == ("base_model.model.model.embed_tokens.weight",)

    def test_zero_lora_keys_rejected(self):
        from soup_cli.utils.apple_adapter import convert_hf_to_mlx_arrays

        with pytest.raises(ValueError, match="no LoRA"):
            convert_hf_to_mlx_arrays(
                {"embed_tokens.weight": np.zeros((2, 2), dtype=np.float32)}
            )

    def test_mlx_to_hf_round_trip_values(self):
        from soup_cli.utils.apple_adapter import (
            convert_hf_to_mlx_arrays,
            convert_mlx_to_hf_arrays,
        )

        original = self._hf_arrays()
        mlx_arrays, _ = convert_hf_to_mlx_arrays(original)
        back, skipped = convert_mlx_to_hf_arrays(mlx_arrays)
        assert skipped == ()
        for key, value in original.items():
            np.testing.assert_allclose(back[key], value)


def _write_peft_adapter(adapter_dir: Path) -> dict:
    """Write a synthetic PEFT LoRA adapter; return its arrays."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(7)
    arrays = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            rng.standard_normal((4, 16)).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
            rng.standard_normal((16, 4)).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight":
            rng.standard_normal((4, 16)).astype(np.float32),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight":
            rng.standard_normal((16, 4)).astype(np.float32),
    }
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(arrays, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({
            "r": 4,
            "lora_alpha": 8,
            "peft_type": "LORA",
            "base_model_name_or_path": "tiny/base",
        }),
        encoding="utf-8",
    )
    return arrays


class TestConvertAppleAdapterLive:
    def test_hf_to_mlx_writes_safetensors(self, tmp_path, monkeypatch):
        """mlx-lm's load_adapters reads adapters.safetensors + num_layers
        (review fix — the legacy npz artifact is unloadable by current
        mlx-lm)."""
        from safetensors.numpy import load_file

        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        arrays = _write_peft_adapter(tmp_path / "adapter")
        plan = build_apple_adapter_plan(
            source_dir="adapter", output_dir="out", direction="hf-to-mlx",
        )
        report = convert_apple_adapter(plan)
        assert report.converted_keys == 4
        st_path = tmp_path / "out" / "adapters.safetensors"
        assert st_path.is_file()
        loaded = load_file(str(st_path))
        key = "model.layers.0.self_attn.q_proj.lora_a"
        assert key in loaded
        np.testing.assert_allclose(
            loaded[key],
            arrays[
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            ].T,
        )
        config = json.loads(
            (tmp_path / "out" / "adapter_config.json").read_text(encoding="utf-8")
        )
        assert config["fine_tune_type"] == "lora"
        assert config["num_layers"] == 1  # max layer index + 1
        assert config["lora_parameters"]["rank"] == 4
        assert config["lora_parameters"]["scale"] == 2.0  # alpha 8 / r 4

    def test_full_round_trip(self, tmp_path, monkeypatch):
        from safetensors.numpy import load_file

        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        arrays = _write_peft_adapter(tmp_path / "adapter")
        convert_apple_adapter(
            build_apple_adapter_plan(
                source_dir="adapter", output_dir="mlx", direction="hf-to-mlx",
            )
        )
        report = convert_apple_adapter(
            build_apple_adapter_plan(
                source_dir="mlx", output_dir="hf2", direction="mlx-to-hf",
            )
        )
        assert report.converted_keys == 4
        back = load_file(str(tmp_path / "hf2" / "adapter_model.safetensors"))
        for key, value in arrays.items():
            np.testing.assert_allclose(back[key], value)
        config = json.loads(
            (tmp_path / "hf2" / "adapter_config.json").read_text(encoding="utf-8")
        )
        assert config["peft_type"] == "LORA"
        assert config["r"] == 4
        # scale (2.0) * rank (4) reconstructs the source lora_alpha — and
        # the round trip exercises the adapters.safetensors input branch
        # (hf-to-mlx now emits safetensors).
        assert config["lora_alpha"] == 8

    def test_sign_writes_signature(self, tmp_path, monkeypatch):
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        _write_peft_adapter(tmp_path / "adapter")
        report = convert_apple_adapter(
            build_apple_adapter_plan(
                source_dir="adapter", output_dir="out",
                direction="hf-to-mlx", sign=True,
            )
        )
        assert report.signed is True
        assert (tmp_path / "out" / ".soup-signature.json").is_file()

    @pytest.mark.parametrize("direction", ["hf-to-apple", "mlx-to-apple"])
    def test_apple_directions_upstream_gated(self, tmp_path, monkeypatch, direction):
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        _write_peft_adapter(tmp_path / "adapter")
        plan = build_apple_adapter_plan(
            source_dir="adapter", output_dir="out", direction=direction,
        )
        with pytest.raises(RuntimeError, match="FoundationModels"):
            convert_apple_adapter(plan)

    def test_missing_safetensors_friendly(self, tmp_path, monkeypatch):
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        (tmp_path / "adapter").mkdir()
        plan = build_apple_adapter_plan(
            source_dir="adapter", output_dir="out", direction="hf-to-mlx",
        )
        with pytest.raises(FileNotFoundError, match="adapter_model"):
            convert_apple_adapter(plan)

    def test_bin_adapter_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.bin").write_bytes(b"\x80\x02")
        plan = build_apple_adapter_plan(
            source_dir="adapter", output_dir="out", direction="hf-to-mlx",
        )
        with pytest.raises(ValueError, match="safetensors"):
            convert_apple_adapter(plan)

    def test_output_outside_cwd_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        _write_peft_adapter(workdir / "adapter")
        plan = build_apple_adapter_plan(
            source_dir="adapter",
            output_dir=str(tmp_path / "outside"),
            direction="hf-to-mlx",
        )
        with pytest.raises(ValueError, match="cwd"):
            convert_apple_adapter(plan)

    def test_mlx_source_missing_files_friendly(self, tmp_path, monkeypatch):
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        (tmp_path / "mlx").mkdir()
        plan = build_apple_adapter_plan(
            source_dir="mlx", output_dir="out", direction="mlx-to-hf",
        )
        with pytest.raises(FileNotFoundError, match="adapters"):
            convert_apple_adapter(plan)

    def test_non_plan_rejected(self):
        from soup_cli.utils.apple_adapter import convert_apple_adapter

        with pytest.raises(TypeError, match="AppleAdapterPlan"):
            convert_apple_adapter({})  # type: ignore[arg-type]

    def test_report_frozen(self):
        import dataclasses

        from soup_cli.utils.apple_adapter import ConversionReport

        report = ConversionReport(
            direction="hf-to-mlx", output_dir="out",
            converted_keys=2, skipped_keys=(), signed=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.direction = "mlx-to-hf"  # type: ignore[misc]


class TestAppleAdapterCli:
    def test_live_conversion_exit_0(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_peft_adapter(tmp_path / "adapter")
        result = runner.invoke(
            app,
            [
                "apple-adapter", "adapter",
                "--direction", "hf-to-mlx",
                "--output", "out",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "out" / "adapters.safetensors").is_file()

    def test_apple_direction_exit_3(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_peft_adapter(tmp_path / "adapter")
        result = runner.invoke(
            app,
            [
                "apple-adapter", "adapter",
                "--direction", "hf-to-apple",
                "--output", "out",
            ],
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))

    def test_missing_weights_exit_2(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "adapter").mkdir()
        result = runner.invoke(
            app,
            [
                "apple-adapter", "adapter",
                "--direction", "hf-to-mlx",
                "--output", "out",
            ],
        )
        assert result.exit_code == 2, (result.output, repr(result.exception))


# ---------------------------------------------------------------------------
# #97 — delinearize-llama4 runtime
# ---------------------------------------------------------------------------


class TestIsExpertWeightKey:
    @pytest.mark.parametrize(
        "key",
        [
            "language_model.model.layers.0.feed_forward.experts.gate_up_proj",
            "language_model.model.layers.3.feed_forward.experts.down_proj",
            "model.layers.0.feed_forward.experts.gate_up_proj.weight",
        ],
    )
    def test_fused_expert_keys(self, key):
        from soup_cli.utils.delinearize_llama4 import is_expert_weight_key

        assert is_expert_weight_key(key)

    @pytest.mark.parametrize(
        "key",
        [
            # Per-expert numbered keys are already unfused (Mixtral-style).
            "model.layers.0.feed_forward.experts.0.gate_proj.weight",
            "model.layers.0.feed_forward.router.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "",
            None,
            123,
        ],
    )
    def test_non_fused_keys(self, key):
        from soup_cli.utils.delinearize_llama4 import is_expert_weight_key

        assert not is_expert_weight_key(key)


class TestDelinearizeTensor:
    def test_reshapes_2d_to_3d(self):
        torch = _torch_or_skip()
        from soup_cli.utils.delinearize_llama4 import delinearize_tensor

        tensor = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        out, status = delinearize_tensor(tensor, num_experts=3)
        assert status == "reshaped"
        assert tuple(out.shape) == (3, 2, 4)
        assert torch.equal(out[0], tensor[0:2])

    def test_3d_passthrough(self):
        torch = _torch_or_skip()
        from soup_cli.utils.delinearize_llama4 import delinearize_tensor

        tensor = torch.zeros(3, 2, 4)
        out, status = delinearize_tensor(tensor, num_experts=3)
        assert status == "already_3d"
        assert out is tensor

    def test_not_divisible_rejected(self):
        torch = _torch_or_skip()
        from soup_cli.utils.delinearize_llama4 import delinearize_tensor

        tensor = torch.zeros(7, 4)
        with pytest.raises(ValueError, match="divisible"):
            delinearize_tensor(tensor, num_experts=3)

    def test_1d_rejected(self):
        torch = _torch_or_skip()
        from soup_cli.utils.delinearize_llama4 import delinearize_tensor

        with pytest.raises(ValueError, match="2-D"):
            delinearize_tensor(torch.zeros(8), num_experts=2)


class TestReadNumExperts:
    def test_text_config_nested(self, tmp_path):
        from soup_cli.utils.delinearize_llama4 import read_num_experts

        (tmp_path / "config.json").write_text(
            json.dumps({"text_config": {"num_local_experts": 16}}),
            encoding="utf-8",
        )
        assert read_num_experts(str(tmp_path)) == 16

    def test_top_level(self, tmp_path):
        from soup_cli.utils.delinearize_llama4 import read_num_experts

        (tmp_path / "config.json").write_text(
            json.dumps({"num_local_experts": 8}), encoding="utf-8"
        )
        assert read_num_experts(str(tmp_path)) == 8

    def test_missing_returns_none(self, tmp_path):
        from soup_cli.utils.delinearize_llama4 import read_num_experts

        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert read_num_experts(str(tmp_path)) is None

    def test_no_config_returns_none(self, tmp_path):
        from soup_cli.utils.delinearize_llama4 import read_num_experts

        assert read_num_experts(str(tmp_path)) is None

    def test_bool_value_rejected(self, tmp_path):
        from soup_cli.utils.delinearize_llama4 import read_num_experts

        (tmp_path / "config.json").write_text(
            json.dumps({"num_local_experts": True}), encoding="utf-8"
        )
        assert read_num_experts(str(tmp_path)) is None


def _write_llama4_stub(source: Path, *, num_experts: int = 4) -> dict:
    """Write a stub Llama-4-shaped checkpoint; return the tensor dict."""
    torch = _torch_or_skip()
    from safetensors.torch import save_file

    tensors = {
        "language_model.model.layers.0.feed_forward.experts.gate_up_proj":
            torch.arange(48, dtype=torch.float32).reshape(8, 6),
        "language_model.model.layers.0.feed_forward.experts.down_proj":
            torch.arange(48, dtype=torch.float32).reshape(12, 4),
        "language_model.model.layers.0.self_attn.q_proj.weight":
            torch.ones(4, 4),
        "language_model.model.layers.1.feed_forward.experts.gate_up_proj":
            torch.zeros(num_experts, 2, 6),
    }
    source.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(source / "model.safetensors"))
    (source / "config.json").write_text(
        json.dumps({"text_config": {"num_local_experts": num_experts}}),
        encoding="utf-8",
    )
    return tensors


class TestRunDelinearize:
    def test_happy_path(self, tmp_path, monkeypatch):
        torch = _torch_or_skip()
        from safetensors.torch import load_file

        from soup_cli.utils.delinearize_llama4 import (
            plan_delinearize,
            run_delinearize,
        )

        monkeypatch.chdir(tmp_path)
        tensors = _write_llama4_stub(tmp_path / "src")
        (tmp_path / "out").mkdir()
        plan = plan_delinearize("src", "out")
        result = run_delinearize(plan)
        assert result.reshaped_keys == 2
        assert result.passthrough_keys == 1
        assert result.already_3d_keys == 1
        out = load_file(str(tmp_path / "out" / "model.safetensors"))
        gate_up = out[
            "language_model.model.layers.0.feed_forward.experts.gate_up_proj"
        ]
        assert tuple(gate_up.shape) == (4, 2, 6)
        original = tensors[
            "language_model.model.layers.0.feed_forward.experts.gate_up_proj"
        ]
        assert torch.equal(gate_up, original.reshape(4, 2, 6))
        # Sidecar config copied for a loadable target checkpoint.
        assert (tmp_path / "out" / "config.json").is_file()

    def test_explicit_num_experts_overrides(self, tmp_path, monkeypatch):
        from safetensors.torch import load_file

        from soup_cli.utils.delinearize_llama4 import (
            plan_delinearize,
            run_delinearize,
        )

        monkeypatch.chdir(tmp_path)
        _write_llama4_stub(tmp_path / "src", num_experts=4)
        (tmp_path / "out").mkdir()
        plan = plan_delinearize("src", "out")
        result = run_delinearize(plan, num_experts=2)
        assert result.reshaped_keys == 2
        out = load_file(str(tmp_path / "out" / "model.safetensors"))
        gate_up = out[
            "language_model.model.layers.0.feed_forward.experts.gate_up_proj"
        ]
        assert tuple(gate_up.shape) == (2, 4, 6)

    def test_missing_num_experts_friendly(self, tmp_path, monkeypatch):
        torch = _torch_or_skip()
        from safetensors.torch import save_file

        from soup_cli.utils.delinearize_llama4 import (
            plan_delinearize,
            run_delinearize,
        )

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        save_file({"x": torch.zeros(2, 2)}, str(src / "model.safetensors"))
        (tmp_path / "out").mkdir()
        plan = plan_delinearize("src", "out")
        with pytest.raises(ValueError, match="--num-experts"):
            run_delinearize(plan)

    def test_non_divisible_names_key(self, tmp_path, monkeypatch):
        torch = _torch_or_skip()
        from safetensors.torch import save_file

        from soup_cli.utils.delinearize_llama4 import (
            plan_delinearize,
            run_delinearize,
        )

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        save_file(
            {
                "model.layers.0.feed_forward.experts.gate_up_proj":
                    torch.zeros(7, 4),
            },
            str(src / "model.safetensors"),
        )
        (tmp_path / "out").mkdir()
        plan = plan_delinearize("src", "out")
        with pytest.raises(ValueError, match="gate_up_proj"):
            run_delinearize(plan, num_experts=4)

    @pytest.mark.parametrize("bad", [True, 0, -2, "four", 1_000_000])
    def test_num_experts_bounds(self, tmp_path, monkeypatch, bad):
        from soup_cli.utils.delinearize_llama4 import (
            plan_delinearize,
            run_delinearize,
        )

        monkeypatch.chdir(tmp_path)
        _write_llama4_stub(tmp_path / "src")
        (tmp_path / "out").mkdir()
        plan = plan_delinearize("src", "out")
        with pytest.raises((TypeError, ValueError), match="num_experts"):
            run_delinearize(plan, num_experts=bad)

    def test_non_plan_rejected(self):
        from soup_cli.utils.delinearize_llama4 import run_delinearize

        with pytest.raises(TypeError, match="DelinearizePlan"):
            run_delinearize({})  # type: ignore[arg-type]

    def test_result_frozen(self):
        import dataclasses

        from soup_cli.utils.delinearize_llama4 import DelinearizeResult

        result = DelinearizeResult(
            source_dir="a", target_dir="b", files_written=("x",),
            reshaped_keys=1, passthrough_keys=0, already_3d_keys=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.reshaped_keys = 9  # type: ignore[misc]


class TestDelinearizeCli:
    def test_live_run_exit_0(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_llama4_stub(tmp_path / "src")
        (tmp_path / "out").mkdir()
        result = runner.invoke(
            app, ["delinearize-llama4", "src", "--target", "out"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "out" / "model.safetensors").is_file()

    def test_plan_only_writes_nothing(self, tmp_path, monkeypatch):
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        _write_llama4_stub(tmp_path / "src")
        (tmp_path / "out").mkdir()
        result = runner.invoke(
            app,
            ["delinearize-llama4", "src", "--target", "out", "--plan-only"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert not (tmp_path / "out" / "model.safetensors").exists()

    def test_missing_num_experts_exit_2(self, tmp_path, monkeypatch):
        torch = _torch_or_skip()
        from safetensors.torch import save_file

        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        save_file({"x": torch.zeros(2, 2)}, str(src / "model.safetensors"))
        (tmp_path / "out").mkdir()
        result = runner.invoke(
            app, ["delinearize-llama4", "src", "--target", "out"],
        )
        assert result.exit_code == 2, (result.output, repr(result.exception))
        assert "--num-experts" in result.output


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


class TestPatchInvariants:
    def test_version_bumped(self):
        import soup_cli

        parts = tuple(int(p) for p in soup_cli.__version__.split(".")[:3])
        assert parts >= (0, 71, 21)

    @pytest.mark.parametrize(
        "rel",
        [
            "utils/advanced_precision.py",
            "utils/grpo_long_context.py",
            "utils/agent_rollout.py",
            "utils/apple_adapter.py",
            "utils/delinearize_llama4.py",
        ],
    )
    def test_no_heavy_top_level_imports(self, rel):
        # Full-source scan — TYPE_CHECKING-block imports are indented so
        # the column-0 patterns can never match them; splitting the source
        # there would blind the guard for everything below the block
        # (review fix).
        source = (_SRC / rel).read_text(encoding="utf-8")
        for heavy in (
            "torch",
            "transformers",
            "vllm",
            "torchao",
            "numpy",
            "safetensors",
            "mlx",
            "peft",
        ):
            assert f"\nimport {heavy}\n" not in source
            assert f"\nimport {heavy} " not in source
            assert f"\nfrom {heavy}" not in source


# ---------------------------------------------------------------------------
# Review-fix follow-ups (v0.71.21 review wave)
# ---------------------------------------------------------------------------


class TestReviewFollowupsPrecision:
    """#141 review fixes — torchao probe, recipe NUL, partial-conversion."""

    def test_out_proj_and_wqkv_in_allowlist(self):
        from soup_cli.utils.advanced_precision import is_attention_projection

        assert is_attention_projection("encoder.layers.0.self_attn.out_proj")
        assert is_attention_projection("transformer.blocks.0.attn.Wqkv")

    def test_is_blackwell_gpu_runtime_error_false(self, monkeypatch):
        torch = _torch_or_skip()
        from soup_cli.utils.advanced_precision import is_blackwell_gpu

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def _boom(_i=0):
            raise RuntimeError("driver mismatch")

        monkeypatch.setattr(torch.cuda, "get_device_capability", _boom)
        assert is_blackwell_gpu() is False

    def test_already_converted_fast_path_no_float8_import(self, monkeypatch):
        """All projections pre-converted -> count returned WITHOUT importing
        torchao.float8 (the bare fake root has no float8 attr)."""
        torch = _torch_or_skip()
        nn = torch.nn
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        class _F8(nn.Linear):
            pass

        _F8.__name__ = "Float8Linear"
        _enable_fp8_gates(monkeypatch)
        model = _tiny_attn_model()
        attn = model.layers[0].self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(attn, name, _F8(8, 8))
        assert apply_fp8_attention(model) == 4

    def test_partial_conversion_failure_is_honest(self, monkeypatch):
        """torchao crashing mid-conversion surfaces 'PARTIALLY converted'."""
        from soup_cli.utils.advanced_precision import apply_fp8_attention

        record: dict = {}
        _install_fake_torchao_float8(monkeypatch, record)
        _enable_fp8_gates(monkeypatch)

        def _boom(model, config=None, module_filter_fn=None):
            raise ValueError("unsupported dim")

        sys.modules["torchao.float8"].convert_to_float8_training = _boom
        with pytest.raises(RuntimeError, match="PARTIALLY"):
            apply_fp8_attention(_tiny_attn_model())

    def test_nvfp4_missing_quantize_friendly(self, monkeypatch):
        from soup_cli.utils.advanced_precision import apply_nvfp4

        fake_q = types.ModuleType("torchao.quantization")

        class _NVFP4Config:
            pass

        fake_q.NVFP4Config = _NVFP4Config  # no quantize_
        fake_root = types.ModuleType("torchao")
        fake_root.quantization = fake_q
        monkeypatch.setitem(sys.modules, "torchao", fake_root)
        monkeypatch.setitem(sys.modules, "torchao.quantization", fake_q)
        monkeypatch.setattr(
            "soup_cli.utils.advanced_precision.is_blackwell_gpu", lambda: True
        )
        with pytest.raises(RuntimeError, match="quantize_"):
            apply_nvfp4(_tiny_attn_model())

    def test_v028_degrade_is_hermetic(self, monkeypatch):
        """Force the degrade path by patching the converters directly —
        environment-independent (review fix: the original test relied on
        the host lacking torchao/Hopper)."""
        from soup_cli.config.loader import load_config_from_string
        from soup_cli.utils import advanced_precision, v028_features

        def _gate(model, **kwargs):
            raise RuntimeError("gate fired")

        # v028_features lazy-imports the converters at call time, so the
        # patch targets the defining module.
        monkeypatch.setattr(advanced_precision, "apply_fp8_attention", _gate)
        monkeypatch.setattr(advanced_precision, "apply_nvfp4", _gate)
        cfg = load_config_from_string(
            "base: test-llama\n"
            "task: sft\n"
            "training:\n"
            "  quantization_aware: fp8\n"
            "  fp8_attention: true\n"
            "  nvfp4: true\n"
            "data:\n"
            "  train: data.jsonl\n"
            "output: ./out\n"
        )
        applied = v028_features.apply_v028_speed_memory(
            model=object(), tcfg=cfg.training, base_model=cfg.base,
            console=None, device="cpu", backend="transformers",
        )
        assert applied.get("fp8_attention") is False
        assert applied.get("nvfp4") is False


class TestReviewFollowupsSleepMode:
    """#124 review fixes — level validation + version edge cases."""

    def test_parse_version_tuple_none(self):
        from soup_cli.utils.grpo_long_context import _parse_version_tuple

        assert _parse_version_tuple(None) == ()  # type: ignore[arg-type]

    def test_vllm_without_version_attr_false(self, monkeypatch):
        from soup_cli.utils.grpo_long_context import vllm_supports_sleep_mode

        fake = types.ModuleType("vllm")  # no __version__
        monkeypatch.setitem(sys.modules, "vllm", fake)
        assert vllm_supports_sleep_mode() is False

    def test_exact_floor_0_7_0_supported(self, monkeypatch):
        from soup_cli.utils.grpo_long_context import vllm_supports_sleep_mode

        _install_fake_vllm(monkeypatch, "0.7.0")
        assert vllm_supports_sleep_mode() is True

    def test_sleep_cycle_level_2_forwarded(self):
        from soup_cli.utils.grpo_long_context import vllm_sleep_cycle

        calls: list = []
        engine = SimpleNamespace(
            sleep=lambda level: calls.append(("sleep", level)),
            wake_up=lambda: calls.append(("wake", None)),
        )
        with vllm_sleep_cycle(engine, level=2):
            pass
        assert calls == [("sleep", 2), ("wake", None)]

    @pytest.mark.parametrize("level", [True, 0, 3, "1"])
    def test_sleep_cycle_bad_level_rejected(self, level):
        from soup_cli.utils.grpo_long_context import vllm_sleep_cycle

        engine = SimpleNamespace(sleep=lambda level: None, wake_up=lambda: None)
        with pytest.raises((TypeError, ValueError), match="level"):
            with vllm_sleep_cycle(engine, level=level):
                pass

    def test_sleep_without_wake_up_warns_and_runs(self, caplog):
        import logging

        from soup_cli.utils.grpo_long_context import vllm_sleep_cycle

        engine = SimpleNamespace(sleep=lambda level: None)  # no wake_up
        ran = []
        with caplog.at_level(
            logging.WARNING, logger="soup_cli.utils.grpo_long_context"
        ):
            with vllm_sleep_cycle(engine):
                ran.append(True)
        assert ran == [True]
        assert any("sleep" in rec.message for rec in caplog.records)


class TestReviewFollowupsRollout:
    """#125 review fixes — smuggle-strip, loud answers, boundaries."""

    def test_extra_keys_stripped(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path,
            monkeypatch,
            "def rollout(prompts):\n"
            "    return [{'prompt': 'x', 'evil': 'payload', 'answer': 'a'}]\n",
        )
        result = launch_rollout("openenv", prompts=["p"], rollout_func=spec)
        assert set(result.rows[0]) == {"prompt", "answer"}

    def test_non_str_answer_rejected_loudly(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path,
            monkeypatch,
            "def rollout(prompts):\n"
            "    return [{'prompt': 'x', 'answer': 42}]\n",
        )
        with pytest.raises(ValueError, match="answer"):
            launch_rollout("openenv", prompts=["p"], rollout_func=spec)

    def test_empty_prompt_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path,
            monkeypatch,
            "def rollout(prompts):\n    return [{'prompt': ''}]\n",
        )
        with pytest.raises(ValueError, match="empty prompt"):
            launch_rollout("openenv", prompts=["p"], rollout_func=spec)

    def test_empty_rows_result_rejected(self):
        from soup_cli.utils.agent_rollout import RolloutResult

        with pytest.raises(ValueError, match="no rows"):
            RolloutResult(backend="openenv", rows=())

    def test_result_stores_canonical_backend(self):
        from soup_cli.utils.agent_rollout import RolloutResult

        result = RolloutResult(backend="ART", rows=({"prompt": "x"},))
        assert result.backend == "art"

    def test_message_list_prompts_alias_broken(self):
        """Mutating the rollout callable's retained message dict must not
        leak into the normalised training rows (immutability policy)."""
        from soup_cli.utils.agent_rollout import _normalise_rollout_rows

        message = {"role": "user", "content": "hi"}
        rows = _normalise_rollout_rows([{"prompt": [message]}], "openenv")
        message["content"] = "MUTATED"
        assert rows[0]["prompt"][0]["content"] == "hi"

    def test_validate_rollout_func_bool_rejected(self):
        from soup_cli.utils.agent_rollout import validate_rollout_func

        # ValueError (not TypeError) so the Pydantic field_validator wraps
        # it into a ValidationError (mode="before" validators re-raise
        # TypeError raw — v2 convention).
        with pytest.raises(ValueError, match="rollout_func"):
            validate_rollout_func(True)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("max_steps", "ok"), [(100_000, True), (100_001, False)]
    )
    def test_max_steps_exact_boundary(
        self, tmp_path, monkeypatch, max_steps, ok
    ):
        from soup_cli.utils.agent_rollout import launch_rollout

        spec = _write_rollout_module(
            tmp_path,
            monkeypatch,
            "def rollout(prompts):\n    return [{'prompt': 'x'}]\n",
        )
        if ok:
            result = launch_rollout(
                "openenv", prompts=["p"], rollout_func=spec,
                max_steps=max_steps,
            )
            assert len(result.rows) == 1
        else:
            with pytest.raises(ValueError, match="max_steps"):
                launch_rollout(
                    "openenv", prompts=["p"], rollout_func=spec,
                    max_steps=max_steps,
                )

    def test_external_runner_receives_all_kwargs(self, monkeypatch):
        from soup_cli.utils import agent_rollout

        seen: dict = {}

        def _runner(**kwargs):
            seen.update(kwargs)
            return [{"prompt": "from-runner"}]

        monkeypatch.setitem(
            agent_rollout._EXTERNAL_ROLLOUT_RUNNERS, "art", _runner
        )
        model, tokenizer, reward = object(), object(), object()
        agent_rollout.launch_rollout(
            "art", prompts=["p"], model=model, tokenizer=tokenizer,
            reward_fn=reward, max_steps=7,
        )
        assert seen["prompts"] == ["p"]
        assert seen["model"] is model
        assert seen["tokenizer"] is tokenizer
        assert seen["reward_fn"] is reward
        assert seen["max_steps"] == 7


class TestReviewFollowupsAppleAdapter:
    """#228 review fixes — symlink TOCTOU, corrupt files, config carry."""

    def test_infer_num_layers_gpt2_style(self):
        """GPT-2-style ``transformer.h.N`` paths also derive num_layers
        (caught by the real bf16 PEFT adapter smoke)."""
        from soup_cli.utils.apple_adapter import _infer_num_layers

        assert _infer_num_layers({
            "transformer.h.4.attn.c_attn.lora_a": None,
            "transformer.h.0.attn.c_attn.lora_b": None,
        }) == 5
        assert _infer_num_layers({"model.layers.2.q_proj.lora_a": None}) == 3
        assert _infer_num_layers({"no_layer_key.lora_a": None}) is None

    def test_prefixless_hf_key_mapped(self):
        from soup_cli.utils.apple_adapter import hf_key_to_mlx, mlx_key_to_hf

        assert (
            hf_key_to_mlx("model.layers.0.self_attn.q_proj.lora_A.weight")
            == "model.layers.0.self_attn.q_proj.lora_a"
        )
        # The reverse direction re-adds the canonical prefix (documented
        # asymmetry — prefix-less inputs do not byte-round-trip).
        assert mlx_key_to_hf("model.layers.0.self_attn.q_proj.lora_a") == (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )

    def test_lora_dropout_carried_through(self, tmp_path, monkeypatch):
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        _write_peft_adapter(tmp_path / "adapter")
        config_path = tmp_path / "adapter" / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["lora_dropout"] = 0.05
        config_path.write_text(json.dumps(config), encoding="utf-8")
        convert_apple_adapter(
            build_apple_adapter_plan(
                source_dir="adapter", output_dir="out", direction="hf-to-mlx",
            )
        )
        out_config = json.loads(
            (tmp_path / "out" / "adapter_config.json").read_text(
                encoding="utf-8"
            )
        )
        assert out_config["lora_parameters"]["dropout"] == 0.05

    def test_legacy_npz_input_still_loads(self, tmp_path, monkeypatch):
        from safetensors.numpy import load_file

        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        mlx_dir = tmp_path / "mlx"
        mlx_dir.mkdir()
        rng = np.random.default_rng(3)
        np.savez(
            str(mlx_dir / "adapters.npz"),
            **{
                "model.layers.0.self_attn.q_proj.lora_a":
                    rng.standard_normal((16, 4)).astype(np.float32),
                "model.layers.0.self_attn.q_proj.lora_b":
                    rng.standard_normal((4, 16)).astype(np.float32),
            },
        )
        report = convert_apple_adapter(
            build_apple_adapter_plan(
                source_dir="mlx", output_dir="hf", direction="mlx-to-hf",
            )
        )
        assert report.converted_keys == 2
        back = load_file(str(tmp_path / "hf" / "adapter_model.safetensors"))
        assert (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            in back
        )

    def test_corrupt_safetensors_value_error(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"not-safetensors")
        plan = build_apple_adapter_plan(
            source_dir="adapter", output_dir="out", direction="hf-to-mlx",
        )
        with pytest.raises(ValueError, match="not a valid safetensors"):
            convert_apple_adapter(plan)

    def test_corrupt_npz_value_error(self, tmp_path, monkeypatch):
        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        mlx_dir = tmp_path / "mlx"
        mlx_dir.mkdir()
        (mlx_dir / "adapters.npz").write_bytes(b"not-a-zip")
        plan = build_apple_adapter_plan(
            source_dir="mlx", output_dir="out", direction="mlx-to-hf",
        )
        with pytest.raises(ValueError, match="not a valid npz"):
            convert_apple_adapter(plan)

    def test_npz_decompression_cap(self, tmp_path, monkeypatch):
        """The 4 GiB cap re-applies to DECOMPRESSED arrays (zip bomb)."""
        from soup_cli.utils import apple_adapter

        monkeypatch.chdir(tmp_path)
        mlx_dir = tmp_path / "mlx"
        mlx_dir.mkdir()
        zeros = np.zeros((128, 128), dtype=np.float32)  # 64 KiB, compresses tiny
        with open(mlx_dir / "adapters.npz", "wb") as handle:
            np.savez_compressed(
                handle, **{"model.layers.0.self_attn.q_proj.lora_a": zeros}
            )
        on_disk = (mlx_dir / "adapters.npz").stat().st_size
        monkeypatch.setattr(
            apple_adapter, "_MAX_ADAPTER_FILE_BYTES", on_disk + 1024
        )
        plan = apple_adapter.build_apple_adapter_plan(
            source_dir="mlx", output_dir="out", direction="mlx-to-hf",
        )
        with pytest.raises(ValueError, match="decompresses past"):
            apple_adapter.convert_apple_adapter(plan)

    def test_adapter_size_cap_branch(self, tmp_path, monkeypatch):
        from soup_cli.utils import apple_adapter

        monkeypatch.chdir(tmp_path)
        _write_peft_adapter(tmp_path / "adapter")
        monkeypatch.setattr(apple_adapter, "_MAX_ADAPTER_FILE_BYTES", 16)
        plan = apple_adapter.build_apple_adapter_plan(
            source_dir="adapter", output_dir="out", direction="hf-to-mlx",
        )
        with pytest.raises(ValueError, match="cap"):
            apple_adapter.convert_apple_adapter(plan)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_symlinked_weights_rejected(self, tmp_path, monkeypatch):
        import os

        from soup_cli.utils.apple_adapter import (
            build_apple_adapter_plan,
            convert_apple_adapter,
        )

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real"
        _write_peft_adapter(real)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        os.symlink(
            str(real / "adapter_model.safetensors"),
            str(adapter / "adapter_model.safetensors"),
        )
        plan = build_apple_adapter_plan(
            source_dir="adapter", output_dir="out", direction="hf-to-mlx",
        )
        with pytest.raises(ValueError, match="symlink"):
            convert_apple_adapter(plan)

    @pytest.mark.parametrize(
        ("field", "value", "exc"),
        [
            ("converted_keys", -1, ValueError),
            ("converted_keys", True, TypeError),
            ("skipped_keys", ["a"], TypeError),
            ("signed", "yes", TypeError),
        ],
    )
    def test_conversion_report_post_init(self, field, value, exc):
        from soup_cli.utils.apple_adapter import ConversionReport

        kwargs = {
            "direction": "hf-to-mlx",
            "output_dir": "out",
            "converted_keys": 2,
            "skipped_keys": (),
            "signed": False,
        }
        kwargs[field] = value
        with pytest.raises(exc):
            ConversionReport(**kwargs)


class TestReviewFollowupsDelinearize:
    """#97 review fixes — plan containment, corrupt shards, boundaries."""

    @pytest.mark.parametrize(
        ("value", "ok"), [(4096, True), (4097, False), (1, True), (0, False)]
    )
    def test_num_experts_exact_boundary(self, value, ok):
        from soup_cli.utils.delinearize_llama4 import _validate_num_experts

        if ok:
            assert _validate_num_experts(value) == value
        else:
            with pytest.raises(ValueError, match="num_experts"):
                _validate_num_experts(value)

    def test_plan_direct_construction_outside_cwd_rejected(
        self, tmp_path, monkeypatch
    ):
        from soup_cli.utils.delinearize_llama4 import DelinearizePlan

        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        with pytest.raises(ValueError, match="outside cwd"):
            DelinearizePlan(
                source_dir=str(workdir / "src"),
                target_dir=str(tmp_path / "outside"),
                weight_files=("model.safetensors",),
            )

    def test_corrupt_shard_value_error(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        from soup_cli.utils.delinearize_llama4 import (
            plan_delinearize,
            run_delinearize,
        )

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        (src / "model.safetensors").write_bytes(b"garbage")
        (tmp_path / "out").mkdir()
        plan = plan_delinearize("src", "out")
        with pytest.raises(ValueError, match="not a valid safetensors"):
            run_delinearize(plan, num_experts=4)

    def test_weight_file_size_cap_branch(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        from soup_cli.utils import delinearize_llama4

        monkeypatch.chdir(tmp_path)
        _write_llama4_stub(tmp_path / "src")
        (tmp_path / "out").mkdir()
        plan = delinearize_llama4.plan_delinearize("src", "out")
        monkeypatch.setattr(delinearize_llama4, "_MAX_WEIGHT_FILE_BYTES", 16)
        with pytest.raises(ValueError, match="cap"):
            delinearize_llama4.run_delinearize(plan)

    def test_multi_shard_run(self, tmp_path, monkeypatch):
        torch = _torch_or_skip()
        from safetensors.torch import save_file

        from soup_cli.utils.delinearize_llama4 import (
            plan_delinearize,
            run_delinearize,
        )

        monkeypatch.chdir(tmp_path)
        _write_llama4_stub(tmp_path / "src")
        save_file(
            {
                "language_model.model.layers.2.feed_forward.experts.down_proj":
                    torch.zeros(8, 4),
            },
            str(tmp_path / "src" / "model-00002.safetensors"),
        )
        (tmp_path / "out").mkdir()
        result = run_delinearize(plan_delinearize("src", "out"))
        assert len(result.files_written) == 2
        assert result.sidecars_copied >= 1  # config.json

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            ({"num_experts": 16}, 16),  # top-level fallback key
            ({"text_config": {"num_local_experts": 5000}}, None),  # OOB
        ],
    )
    def test_read_num_experts_fallbacks(
        self, tmp_path, monkeypatch, config, expected
    ):
        from soup_cli.utils.delinearize_llama4 import read_num_experts

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
        assert read_num_experts(str(src)) == expected

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_symlinked_shard_rejected(self, tmp_path, monkeypatch):
        import os

        pytest.importorskip("torch")
        from soup_cli.utils.delinearize_llama4 import (
            plan_delinearize,
            run_delinearize,
        )

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real"
        _write_llama4_stub(real)
        src = tmp_path / "src"
        src.mkdir()
        os.symlink(
            str(real / "model.safetensors"), str(src / "model.safetensors")
        )
        (tmp_path / "out").mkdir()
        plan = plan_delinearize("src", "out")
        with pytest.raises(ValueError, match="symlink"):
            run_delinearize(plan, num_experts=4)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink")
    def test_symlinked_config_json_skipped(self, tmp_path, monkeypatch):
        import os

        from soup_cli.utils.delinearize_llama4 import read_num_experts

        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real"
        real.mkdir()
        (real / "config.json").write_text(
            json.dumps({"num_local_experts": 4}), encoding="utf-8"
        )
        src = tmp_path / "src"
        src.mkdir()
        os.symlink(str(real / "config.json"), str(src / "config.json"))
        assert read_num_experts(str(src)) is None

    def test_cli_num_experts_flag_passthrough(self, tmp_path, monkeypatch):
        pytest.importorskip("torch")
        from soup_cli.cli import app

        monkeypatch.chdir(tmp_path)
        src = tmp_path / "src"
        _write_llama4_stub(src)
        (src / "config.json").unlink()  # force the flag to matter
        (tmp_path / "out").mkdir()
        result = runner.invoke(
            app,
            [
                "delinearize-llama4", "src",
                "--target", "out",
                "--num-experts", "4",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert (tmp_path / "out" / "model.safetensors").is_file()
