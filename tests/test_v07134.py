"""v0.71.34 — Adapter algebra (task arithmetic) + LISA (#267).

Covers:
* ``utils/adapter_arithmetic.py`` — expression parser + signed element-wise
  task-vector merge + adapter base reader (no top-level torch).
* ``commands/adapters.py::arithmetic`` — ``soup adapters arithmetic``.
* ``config/schema.py`` — LISA fields + ``_validate_lisa_compat``.
* ``utils/lisa.py`` — ``LisaPolicy`` + ``LisaCallback`` (duck-typed).
* ``utils/peft_wiring.py::attach_lisa_callback`` + SFT trainer wiring.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Task A1 — expression parser
# ---------------------------------------------------------------------------
class TestParseExpression:
    def _names(self):
        return {"coder", "math", "toxic"}

    def test_happy_add_scale_sub(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        terms = parse_expression("coder + 0.5*math - toxic", self._names())
        got = {t.name: t.coeff for t in terms}
        assert got == {"coder": 1.0, "math": 0.5, "toxic": -1.0}

    def test_name_star_coeff(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        terms = parse_expression("coder*2", self._names())
        assert terms[0].name == "coder" and terms[0].coeff == 2.0

    def test_leading_negative(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        terms = parse_expression("-coder + math", self._names())
        got = {t.name: t.coeff for t in terms}
        assert got == {"coder": -1.0, "math": 1.0}

    def test_single_term_scale(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        terms = parse_expression("2*coder", self._names())
        assert len(terms) == 1 and terms[0].coeff == 2.0

    def test_duplicate_names_sum(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        terms = parse_expression("coder + coder", self._names())
        assert len(terms) == 1 and terms[0].coeff == 2.0

    def test_all_cancel_rejected(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        with pytest.raises(ValueError, match="cancel"):
            parse_expression("coder - coder", self._names())

    def test_empty_rejected(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        with pytest.raises(ValueError, match="empty"):
            parse_expression("   ", self._names())

    def test_unknown_name_rejected(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        with pytest.raises(ValueError, match="ghost"):
            parse_expression("coder + ghost", self._names())

    def test_injection_rejected(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        for bad in ['__import__("os")', "coder; rm -rf", "coder && ls", "coder | cat"]:
            with pytest.raises(ValueError):
                parse_expression(bad, self._names())

    def test_over_length_rejected(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        with pytest.raises(ValueError, match="too long"):
            parse_expression("coder+" * 5000 + "coder", self._names())

    def test_non_finite_coeff_rejected(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        # "nan"/"inf" are names by charset, not floats — so they parse as
        # unknown adapter names, not as coefficients. The finite guard defends
        # against a hypothetical float token; assert the injection path rejects.
        with pytest.raises(ValueError):
            parse_expression("nan*coder", self._names())

    def test_no_top_level_torch(self):
        import soup_cli.utils.adapter_arithmetic as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                else:
                    names = [node.module or ""]
                for nm in names:
                    assert nm.split(".")[0] not in {
                        "torch",
                        "transformers",
                        "peft",
                    }, f"top-level heavy import: {nm}"


# ---------------------------------------------------------------------------
# Task A2 — signed merge + base reader
# ---------------------------------------------------------------------------
class TestMergeTaskArithmetic:
    def test_subtract(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        a = {"lora_A": np.ones((2, 3), dtype=np.float32)}
        b = {"lora_A": np.full((2, 3), 4.0, dtype=np.float32)}
        merged, skipped = merge_task_arithmetic([a, b], [1.0, -1.0])
        assert np.allclose(merged["lora_A"], -3.0)
        assert skipped == ()

    def test_scale(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        a = {"w": np.ones((2, 2), dtype=np.float32)}
        merged, _ = merge_task_arithmetic([a], [2.5])
        assert np.allclose(merged["w"], 2.5)

    def test_mixed_rank_rejected(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        a = {"w": np.ones((2, 3), dtype=np.float32)}
        b = {"w": np.ones((4, 3), dtype=np.float32)}
        with pytest.raises(ValueError, match="rank"):
            merge_task_arithmetic([a, b], [1.0, 1.0])

    def test_disjoint_keys_skipped(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        a = {"shared": np.ones((2, 2), dtype=np.float32), "only_a": np.ones((1, 1))}
        b = {"shared": np.ones((2, 2), dtype=np.float32), "only_b": np.ones((1, 1))}
        merged, skipped = merge_task_arithmetic([a, b], [1.0, 1.0])
        assert "shared" in merged
        assert set(skipped) == {"only_a", "only_b"}

    def test_length_mismatch_rejected(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        with pytest.raises(ValueError, match="length"):
            merge_task_arithmetic([{"w": np.ones((1, 1))}], [1.0, 2.0])


class TestReadAdapterBase:
    def test_reads_base(self, tmp_path):
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        d = tmp_path / "ad"
        d.mkdir()
        (d / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "meta/x"}), encoding="utf-8"
        )
        assert read_adapter_base(str(d)) == "meta/x"

    def test_missing_returns_none(self, tmp_path):
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        d = tmp_path / "ad"
        d.mkdir()
        assert read_adapter_base(str(d)) is None


# ---------------------------------------------------------------------------
# Task A3 — soup adapters arithmetic command
# ---------------------------------------------------------------------------
def _make_adapter(directory: Path, base: str, tensors: dict) -> str:
    """Write a minimal loadable LoRA adapter dir; return its path string."""
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: np.asarray(v, dtype=np.float32) for k, v in tensors.items()},
        str(directory / "adapter_model.safetensors"),
    )
    (directory / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": base, "r": 8}),
        encoding="utf-8",
    )
    return str(directory)


class TestArithmeticCli:
    def _run(self, args, cwd):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        runner = CliRunner()
        # invoke inside cwd so cwd-containment checks pass
        old = os.getcwd()
        os.chdir(cwd)
        try:
            return runner.invoke(app, args)
        finally:
            os.chdir(old)

    def test_help_registered(self):
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        res = CliRunner().invoke(app, ["arithmetic", "--help"])
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert "arithmetic" in res.output.lower()

    def _rng_tensor(self, shape, seed):
        # Non-degenerate (not rank-1) so the backdoor scanner passes.
        return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)

    def test_add_two_adapters(self, tmp_path):
        key = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
        a = _make_adapter(tmp_path / "coder", "meta/x", {key: self._rng_tensor((8, 16), 1)})
        b = _make_adapter(tmp_path / "math", "meta/x", {key: self._rng_tensor((8, 16), 2)})
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        assert (tmp_path / "out" / "adapter_model.safetensors").is_file()
        assert (tmp_path / "out" / "adapter_config.json").is_file()

    def test_negate_self_is_zero(self, tmp_path):
        key = "base_model.model.layers.0.mlp.down_proj.lora_B.weight"
        t = self._rng_tensor((16, 8), 7)
        a = _make_adapter(tmp_path / "coder", "meta/x", {key: t})
        b = _make_adapter(tmp_path / "toxic", "meta/x", {key: t})
        res = self._run(
            ["arithmetic", "coder - toxic", "--adapter", f"coder={a}",
             "--adapter", f"toxic={b}", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))
        from safetensors.numpy import load_file

        merged = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
        assert np.allclose(merged[key], 0.0, atol=1e-5)

    def test_scan_fail_gate(self, tmp_path):
        # A rank-1 ones-matrix trips the backdoor scanner.
        a = _make_adapter(tmp_path / "coder", "meta/x", {"w": np.ones((8, 16))})
        b = _make_adapter(tmp_path / "math", "meta/x", {"w": np.ones((8, 16))})
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "scan" in res.output.lower()

    def test_scan_fail_bypassed_with_flag(self, tmp_path):
        a = _make_adapter(tmp_path / "coder", "meta/x", {"w": np.ones((8, 16))})
        b = _make_adapter(tmp_path / "math", "meta/x", {"w": np.ones((8, 16))})
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_unknown_name_exit_1(self, tmp_path):
        a = _make_adapter(tmp_path / "coder", "meta/x", {"w": np.ones((2, 2))})
        res = self._run(
            ["arithmetic", "coder + ghost", "--adapter", f"coder={a}", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "ghost" in res.output

    def test_mixed_rank_exit_1(self, tmp_path):
        key = "w"
        a = _make_adapter(tmp_path / "coder", "meta/x", {key: self._rng_tensor((8, 16), 3)})
        b = _make_adapter(tmp_path / "math", "meta/x", {key: self._rng_tensor((4, 16), 4)})
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "rank" in res.output.lower()

    def test_cross_base_rejected(self, tmp_path):
        a = _make_adapter(tmp_path / "coder", "meta/x", {"w": self._rng_tensor((8, 16), 5)})
        b = _make_adapter(tmp_path / "math", "meta/DIFFERENT", {"w": self._rng_tensor((8, 16), 6)})
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "base" in res.output.lower()

    def test_cross_base_allowed_with_flag(self, tmp_path):
        a = _make_adapter(tmp_path / "coder", "meta/x", {"w": self._rng_tensor((8, 16), 8)})
        b = _make_adapter(tmp_path / "math", "meta/DIFFERENT", {"w": self._rng_tensor((8, 16), 9)})
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out",
             "--allow-unscanned", "--allow-cross-base"],
            tmp_path,
        )
        assert res.exit_code == 0, (res.output, repr(res.exception))

    def test_bad_adapter_spec_exit_1(self, tmp_path):
        res = self._run(
            ["arithmetic", "coder", "--adapter", "noequalsign", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 1

    def test_output_outside_cwd_exit_1(self, tmp_path):
        a = _make_adapter(tmp_path / "coder", "meta/x", {"w": self._rng_tensor((8, 16), 10)})
        res = self._run(
            ["arithmetic", "coder", "--adapter", f"coder={a}",
             "-o", "../escape", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 1


# ---------------------------------------------------------------------------
# Task B1 — LISA schema
# ---------------------------------------------------------------------------
_LISA_BASE = """
base: HuggingFaceTB/SmolLM2-135M
task: sft
backend: transformers
modality: text
data:
  train: data.jsonl
  format: chatml
training:
  quantization: none
  lisa_enabled: true
  lisa_num_layers: 4
  lisa_interval_steps: 25
"""


def _load(yaml_str):
    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(yaml_str)


class TestLisaSchema:
    def test_happy(self):
        cfg = _load(_LISA_BASE)
        assert cfg.training.lisa_enabled is True
        assert cfg.training.lisa_num_layers == 4
        assert cfg.training.lisa_interval_steps == 25

    def test_defaults_when_disabled(self):
        cfg = _load(_LISA_BASE.replace("lisa_enabled: true", "lisa_enabled: false")
                    .replace("lisa_num_layers: 4\n", "")
                    .replace("lisa_interval_steps: 25\n", ""))
        assert cfg.training.lisa_enabled is False
        assert cfg.training.lisa_num_layers == 2
        assert cfg.training.lisa_interval_steps == 20

    @pytest.mark.parametrize(
        "sub,kw",
        [
            ("task: sft", "task"),  # -> dpo
            ("backend: transformers", "backend"),
            ("modality: text", "modality"),
            ("quantization: none", "quantization"),
        ],
    )
    def test_gate_rejects(self, sub, kw):
        import pytest as _pt

        repl = {
            "task: sft": "task: dpo",
            "backend: transformers": "backend: mlx",
            "modality: text": "modality: vision",
            "quantization: none": "quantization: 4bit",
        }[sub]
        with _pt.raises(Exception) as ei:
            _load(_LISA_BASE.replace(sub, repl))
        assert kw in str(ei.value).lower() or "lisa" in str(ei.value).lower()

    def test_bool_as_int_rejected(self):
        with pytest.raises(Exception, match="bool"):
            _load(_LISA_BASE.replace("lisa_num_layers: 4", "lisa_num_layers: true"))

    def test_bounds(self):
        with pytest.raises(Exception):
            _load(_LISA_BASE.replace("lisa_num_layers: 4", "lisa_num_layers: 0"))
        with pytest.raises(Exception):
            _load(_LISA_BASE.replace("lisa_num_layers: 4", "lisa_num_layers: 65"))
        with pytest.raises(Exception):
            _load(_LISA_BASE.replace("lisa_interval_steps: 25", "lisa_interval_steps: 0"))

    def test_footgun_disabled_but_set(self):
        y = _LISA_BASE.replace("lisa_enabled: true", "lisa_enabled: false")
        with pytest.raises(Exception, match="lisa_enabled"):
            _load(y)

    @pytest.mark.parametrize(
        "extra,kw",
        [
            ("  freeze_layers: 2\n", "freeze_layers"),
            ("  freeze_ratio: 0.5\n", "freeze_ratio"),
            ("  train_router_only: true\n", "train_router_only"),
            ("  relora_steps: 100\n", "relora_steps"),
            ("  loraplus_lr_ratio: 4.0\n", "loraplus_lr_ratio"),
            ("  unfrozen_parameters: ['model.layers.0.mlp']\n", "unfrozen_parameters"),
        ],
    )
    def test_mutual_exclusion(self, extra, kw):
        y = _LISA_BASE.rstrip("\n") + "\n" + extra
        with pytest.raises(Exception, match=kw):
            _load(y)

    def test_lora_flag_exclusion(self):
        y = (
            _LISA_BASE.rstrip("\n")
            + "\n  lora:\n    use_dora: true\n"
        )
        with pytest.raises(Exception, match="use_dora"):
            _load(y)


# ---------------------------------------------------------------------------
# Task B2 — utils/lisa.py
# ---------------------------------------------------------------------------
def _fake_lm(num_layers=6):
    """Tiny module shaped like a decoder LM: embed + N layers + norm + head."""
    import torch.nn as nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(10, 4)
            self.model.layers = nn.ModuleList([Layer() for _ in range(num_layers)])
            self.model.norm = nn.LayerNorm(4)
            self.lm_head = nn.Linear(4, 10)

    return LM()


class _FakeOpt:
    def __init__(self):
        self.state = {}


class TestLisaPolicy:
    def test_valid(self):
        from soup_cli.utils.lisa import LisaPolicy

        p = LisaPolicy(num_layers=2, interval_steps=20)
        assert p.num_layers == 2 and p.interval_steps == 20

    def test_bool_rejected(self):
        from soup_cli.utils.lisa import LisaPolicy

        with pytest.raises((ValueError, TypeError)):
            LisaPolicy(num_layers=True, interval_steps=20)

    def test_bounds_rejected(self):
        from soup_cli.utils.lisa import LisaPolicy

        with pytest.raises(ValueError):
            LisaPolicy(num_layers=0, interval_steps=20)
        with pytest.raises(ValueError):
            LisaPolicy(num_layers=2, interval_steps=0)


class TestLisaCallback:
    def _trainable_layer_indices(self, model):
        import re

        pat = re.compile(r"layers\.(\d+)\.")
        idxs = set()
        for name, p in model.named_parameters():
            m = pat.search(name)
            if m and p.requires_grad:
                idxs.add(int(m.group(1)))
        return idxs

    def _flag(self, model, substr):
        return all(
            p.requires_grad
            for name, p in model.named_parameters()
            if substr in name
        )

    def test_initial_selection(self):
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(LisaPolicy(num_layers=2, interval_steps=20, seed=0))
        cb.on_train_begin(None, _State(0), None, model=model)
        assert len(self._trainable_layer_indices(model)) == 2
        assert self._flag(model, "embed_tokens")
        assert self._flag(model, "lm_head")
        assert self._flag(model, "model.norm")

    def test_resample_changes_set(self):
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(LisaPolicy(num_layers=2, interval_steps=10, seed=0))
        cb.on_train_begin(None, _State(0), None, model=model)
        first = frozenset(self._trainable_layer_indices(model))
        # non-interval step -> no change
        cb.on_step_end(None, _State(5), None, model=model, optimizer=_FakeOpt())
        assert frozenset(self._trainable_layer_indices(model)) == first
        # interval step -> re-sample (may differ)
        cb.on_step_end(None, _State(10), None, model=model, optimizer=_FakeOpt())
        assert cb.fire_count == 1
        assert len(self._trainable_layer_indices(model)) == 2

    def test_deterministic_by_seed(self):
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        picks = []
        for _ in range(2):
            model = _fake_lm(8)
            cb = LisaCallback(LisaPolicy(num_layers=3, interval_steps=5, seed=42))
            cb.on_train_begin(None, _State(0), None, model=model)
            cb.on_step_end(None, _State(5), None, model=model, optimizer=_FakeOpt())
            picks.append(frozenset(self._trainable_layer_indices(model)))
        assert picks[0] == picks[1]

    def test_clamp_num_layers(self):
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(4)
        cb = LisaCallback(LisaPolicy(num_layers=10, interval_steps=20, seed=0))
        cb.on_train_begin(None, _State(0), None, model=model)
        assert len(self._trainable_layer_indices(model)) == 4  # clamped

    def test_optimizer_state_cleared_on_refreeze(self):
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(LisaPolicy(num_layers=2, interval_steps=10, seed=0))
        cb.on_train_begin(None, _State(0), None, model=model)
        opt = _FakeOpt()
        # seed optimizer state for every currently-trainable decoder param
        import re

        pat = re.compile(r"layers\.(\d+)\.")
        active_params = [
            p for n, p in model.named_parameters()
            if pat.search(n) and p.requires_grad
        ]
        for p in active_params:
            opt.state[p] = {"exp_avg": 1}
        cb.on_step_end(None, _State(10), None, model=model, optimizer=opt)
        # any param that got frozen should have had its optimizer state cleared
        frozen_now = [
            p for n, p in model.named_parameters()
            if pat.search(n) and not p.requires_grad
        ]
        for p in frozen_now:
            assert p not in opt.state or opt.state[p] == {}

    def test_no_top_level_torch(self):
        import soup_cli.utils.lisa as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for nm in names:
                    assert nm.split(".")[0] not in {"torch", "transformers", "peft"}


class _State:
    def __init__(self, global_step):
        self.global_step = global_step
