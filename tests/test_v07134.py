"""v0.71.34 — Adapter algebra (task arithmetic) + LISA (#267).

Covers:
* ``utils/adapter_arithmetic.py`` — expression parser + signed element-wise
  task-vector merge + adapter base reader (no top-level torch).
* ``commands/adapters.py::arithmetic`` — ``soup adapters arithmetic``.
* ``config/schema.py`` — LISA fields + ``_validate_lisa_compat``.
* ``utils/lisa.py`` — ``LisaPolicy`` + ``LisaCallback`` (duck-typed).
* ``utils/peft_wiring.py::attach_lisa_callback`` / ``apply_lisa_setup``
  + the SFT and (#307) pretrain trainer wiring.
"""

from __future__ import annotations

import ast
import json
import os
import re
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

    def test_double_negative_folds_positive(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        terms = parse_expression("- -coder", self._names())
        assert terms[0].name == "coder" and terms[0].coeff == 1.0

    def test_mixed_signs_fold(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        assert parse_expression("+ -coder", self._names())[0].coeff == -1.0
        assert parse_expression("coder - + math", self._names())[1].coeff == -1.0

    def test_spaced_coeff_forms(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        assert parse_expression("coder * 2", self._names())[0].coeff == 2.0
        assert parse_expression("2 * coder", self._names())[0].coeff == 2.0

    def test_overflow_coeff_rejected_as_non_finite(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        # 1e400 overflows Python float -> inf -> caught by the isfinite guard
        with pytest.raises(ValueError, match="finite"):
            parse_expression("1e400*coder", self._names())

    @pytest.mark.parametrize(
        "expr,kw",
        [
            ("coder math", "between terms"),
            ("coder +", "dangling"),
            ("2*", "expected adapter name"),
            ("coder*", "expected coefficient"),
        ],
    )
    def test_malformed_grammar(self, expr, kw):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        with pytest.raises(ValueError, match=kw):
            parse_expression(expr, self._names())

    def test_too_many_terms_rejected(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        names = {f"a{i}" for i in range(70)}
        expr = " + ".join(sorted(names))
        with pytest.raises(ValueError, match="too many"):
            parse_expression(expr, names)

    def test_non_str_input_rejected(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        with pytest.raises(TypeError):
            parse_expression(123, self._names())

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
    def test_linear_on_non_lora_tensor(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        # A tensor that is neither lora_A nor lora_B combines linearly by c.
        a = {"modules_to_save.weight": np.ones((2, 3), dtype=np.float32)}
        b = {"modules_to_save.weight": np.full((2, 3), 4.0, dtype=np.float32)}
        merged, skipped = merge_task_arithmetic([a, b], [1.0, -1.0])
        assert np.allclose(merged["modules_to_save.weight"], -3.0)
        assert skipped == ()

    def test_scale_non_lora(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        a = {"w": np.ones((2, 2), dtype=np.float32)}
        merged, _ = merge_task_arithmetic([a], [2.5])
        assert np.allclose(merged["w"], 2.5)

    def test_reconstructed_delta_negates(self):
        # For a real LoRA, negating the task vector must negate ΔW = B @ A.
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        rng = np.random.default_rng(0)
        a_mat = rng.standard_normal((4, 8)).astype(np.float32)
        b_mat = rng.standard_normal((8, 4)).astype(np.float32)
        ak = "base_model.model.layers.0.mlp.down_proj.lora_A.weight"
        bk = "base_model.model.layers.0.mlp.down_proj.lora_B.weight"
        merged, _ = merge_task_arithmetic([{ak: a_mat, bk: b_mat}], [-1.0])
        delta_orig = b_mat @ a_mat
        delta_neg = merged[bk] @ merged[ak]
        assert np.allclose(delta_neg, -delta_orig, atol=1e-4)

    def test_reconstructed_delta_scales_linearly(self):
        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        rng = np.random.default_rng(1)
        a_mat = rng.standard_normal((4, 8)).astype(np.float32)
        b_mat = rng.standard_normal((8, 4)).astype(np.float32)
        ak = "x.lora_A.weight"
        bk = "x.lora_B.weight"
        merged, _ = merge_task_arithmetic([{ak: a_mat, bk: b_mat}], [0.5])
        delta = merged[bk] @ merged[ak]
        assert np.allclose(delta, 0.5 * (b_mat @ a_mat), atol=1e-4)

    def test_two_adapter_lora_ab_hand_computed(self):
        import math

        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        rng = np.random.default_rng(2)
        a1 = rng.standard_normal((3, 5)).astype(np.float32)
        a2 = rng.standard_normal((3, 5)).astype(np.float32)
        b1 = rng.standard_normal((5, 3)).astype(np.float32)
        b2 = rng.standard_normal((5, 3)).astype(np.float32)
        ak = "m.lora_A.weight"
        bk = "m.lora_B.weight"
        merged, _ = merge_task_arithmetic(
            [{ak: a1, bk: b1}, {ak: a2, bk: b2}], [0.5, -2.0]
        )
        # A-factor coeff = sqrt(|c|); B-factor = sign(c)*sqrt(|c|)
        exp_a = math.sqrt(0.5) * a1 + math.sqrt(2.0) * a2
        exp_b = math.sqrt(0.5) * b1 + (-math.sqrt(2.0)) * b2
        assert np.allclose(merged[ak], exp_a, atol=1e-4)
        assert np.allclose(merged[bk], exp_b, atol=1e-4)

    def test_lora_embedding_factor_branch(self):
        import math

        from soup_cli.utils.adapter_arithmetic import merge_task_arithmetic

        a = np.ones((2, 2), dtype=np.float32)
        merged, _ = merge_task_arithmetic(
            [{"x.lora_embedding_A": a, "x.lora_embedding_B": a}], [4.0]
        )
        assert np.allclose(merged["x.lora_embedding_A"], math.sqrt(4.0))
        assert np.allclose(merged["x.lora_embedding_B"], math.sqrt(4.0))

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
    def test_reads_base(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "ad"
        d.mkdir()
        (d / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "meta/x"}), encoding="utf-8"
        )
        assert read_adapter_base("ad") == "meta/x"

    def test_missing_returns_none(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        monkeypatch.chdir(tmp_path)
        d = tmp_path / "ad"
        d.mkdir()
        assert read_adapter_base("ad") is None

    @pytest.mark.skipif(os.name == "nt", reason="symlink needs admin on Windows")
    def test_symlinked_config_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        monkeypatch.chdir(tmp_path)
        secret = tmp_path / "secret.json"
        secret.write_text(json.dumps({"base_model_name_or_path": "leak"}), encoding="utf-8")
        d = tmp_path / "ad"
        d.mkdir()
        (d / "adapter_config.json").symlink_to(secret)
        with pytest.raises(ValueError, match="symlink"):
            read_adapter_base("ad")

    def _write_cfg(self, tmp_path, monkeypatch, text):
        monkeypatch.chdir(tmp_path)
        d = tmp_path / "ad"
        d.mkdir()
        (d / "adapter_config.json").write_text(text, encoding="utf-8")
        return "ad"

    def test_oversize_config_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        big = '{"base_model_name_or_path": "' + "x" * (300 * 1024) + '"}'
        ad = self._write_cfg(tmp_path, monkeypatch, big)
        with pytest.raises(ValueError, match="cap"):
            read_adapter_base(ad)

    def test_malformed_json_rejected(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        ad = self._write_cfg(tmp_path, monkeypatch, "{not json")
        with pytest.raises(ValueError, match="valid JSON"):
            read_adapter_base(ad)

    def test_non_dict_returns_none(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        ad = self._write_cfg(tmp_path, monkeypatch, "[1, 2, 3]")
        assert read_adapter_base(ad) is None

    def test_non_string_base_returns_none(self, tmp_path, monkeypatch):
        from soup_cli.utils.adapter_arithmetic import read_adapter_base

        ad = self._write_cfg(tmp_path, monkeypatch, '{"base_model_name_or_path": 42}')
        assert read_adapter_base(ad) is None


class TestCoeffCap:
    def test_over_cap_rejected(self):
        from soup_cli.utils.adapter_arithmetic import parse_expression

        with pytest.raises(ValueError, match="cap"):
            parse_expression("1e300*coder", {"coder"})


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
        assert (tmp_path / "out" / "adapter_config.json").is_file()
        # verify the merged VALUE, not just that a file exists (non-lora key
        # -> linear combine)
        from safetensors.numpy import load_file

        merged = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
        assert np.allclose(merged[key], self._rng_tensor((8, 16), 1)
                           + self._rng_tensor((8, 16), 2), atol=1e-4)

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
        assert "name=path" in res.output

    def test_output_outside_cwd_exit_1(self, tmp_path):
        a = _make_adapter(tmp_path / "coder", "meta/x", {"w": self._rng_tensor((8, 16), 10)})
        res = self._run(
            ["arithmetic", "coder", "--adapter", f"coder={a}",
             "-o", "../escape", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "refused" in res.output.lower()

    def test_duplicate_adapter_name_exit_1(self, tmp_path):
        a = _make_adapter(tmp_path / "coder", "meta/x", {"w": self._rng_tensor((8, 16), 11)})
        res = self._run(
            ["arithmetic", "coder", "--adapter", f"coder={a}",
             "--adapter", f"coder={a}", "-o", "out", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "duplicate" in res.output.lower()

    def test_empty_adapter_path_exit_1(self, tmp_path):
        res = self._run(
            ["arithmetic", "coder", "--adapter", "coder=", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "empty path" in res.output.lower()

    def test_invalid_adapter_name_exit_1(self, tmp_path):
        a = _make_adapter(tmp_path / "coder", "meta/x", {"w": self._rng_tensor((8, 16), 12)})
        res = self._run(
            ["arithmetic", "bad", "--adapter", f"bad name!={a}", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "invalid adapter name" in res.output.lower()

    def test_adapter_path_outside_cwd_exit_1(self, tmp_path):
        res = self._run(
            ["arithmetic", "coder", "--adapter", "coder=../elsewhere", "-o", "out"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "refused" in res.output.lower()

    def test_no_shared_tensors_exit_1(self, tmp_path):
        a = _make_adapter(tmp_path / "coder", "meta/x", {"a_only": self._rng_tensor((8, 16), 13)})
        b = _make_adapter(tmp_path / "math", "meta/x", {"b_only": self._rng_tensor((8, 16), 14)})
        res = self._run(
            ["arithmetic", "coder + math", "--adapter", f"coder={a}",
             "--adapter", f"math={b}", "-o", "out", "--allow-unscanned"],
            tmp_path,
        )
        assert res.exit_code == 1
        assert "no shared" in res.output.lower()


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
        # require the SPECIFIC keyword — not just "lisa" (which every message has)
        assert kw in str(ei.value).lower()

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

    def test_reset_optimizer_default_and_footgun(self):
        cfg = _load(_LISA_BASE)
        assert cfg.training.lisa_reset_optimizer is True
        # setting it non-default while LISA is off is a footgun
        y = """
base: HuggingFaceTB/SmolLM2-135M
task: sft
backend: transformers
modality: text
data:
  train: data.jsonl
  format: chatml
training:
  quantization: none
  lisa_enabled: false
  lisa_reset_optimizer: false
"""
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
            ("  expand_layers: 2\n", "expand_layers"),
            ("  freeze_trainable_layers: 3\n", "freeze_trainable_layers"),
        ],
    )
    def test_mutual_exclusion(self, extra, kw):
        y = _LISA_BASE.rstrip("\n") + "\n" + extra
        with pytest.raises(Exception, match=kw):
            _load(y)

    @pytest.mark.parametrize(
        "flag,kw",
        [
            ("use_dora: true", "use_dora"),
            ("use_vera: true", "use_vera"),
            ("use_olora: true", "use_olora"),
            ("use_rslora: true", "use_rslora"),
        ],
    )
    def test_lora_flag_exclusion(self, flag, kw):
        y = _LISA_BASE.rstrip("\n") + f"\n  lora:\n    {flag}\n"
        with pytest.raises(Exception, match=kw):
            _load(y)

    def test_moe_lora_exclusion(self):
        y = _LISA_BASE.rstrip("\n") + "\n  moe_lora: true\n"
        with pytest.raises(Exception, match="moe_lora"):
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

    def test_negative_seed_rejected(self):
        from soup_cli.utils.lisa import LisaPolicy

        with pytest.raises(ValueError, match="seed"):
            LisaPolicy(num_layers=2, interval_steps=20, seed=-1)

    def test_non_bool_reset_optimizer_rejected(self):
        from soup_cli.utils.lisa import LisaPolicy

        with pytest.raises(TypeError, match="reset_optimizer"):
            LisaPolicy(num_layers=2, interval_steps=20, reset_optimizer=1)


class TestLisaCallback:
    def _trainable_layer_indices(self, model):
        import re

        pat = re.compile(r"(?:layers|h)\.(\d+)\.")
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

    def test_optimizer_state_preserved_when_reset_disabled(self):
        import re

        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(
            LisaPolicy(num_layers=2, interval_steps=10, seed=0, reset_optimizer=False)
        )
        cb.on_train_begin(None, _State(0), None, model=model)
        opt = _FakeOpt()
        pat = re.compile(r"layers\.(\d+)\.")
        for n, p in model.named_parameters():
            if pat.search(n) and p.requires_grad:
                opt.state[p] = {"exp_avg": 1}
        cb.on_step_end(None, _State(10), None, model=model, optimizer=opt)
        # reset disabled -> state stays populated even for re-frozen params
        assert all(v == {"exp_avg": 1} for v in opt.state.values())

    def test_non_float_param_in_chosen_layer_skipped(self):
        import torch

        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(4)
        # force every decoder param non-float so any chosen one is skipped
        orig = torch.Tensor.is_floating_point
        cb = LisaCallback(LisaPolicy(num_layers=2, interval_steps=5, seed=0))
        try:
            torch.Tensor.is_floating_point = lambda self: False  # type: ignore
            cb.on_train_begin(None, _State(0), None, model=model)
        finally:
            torch.Tensor.is_floating_point = orig  # type: ignore
        assert len(self._trainable_layer_indices(model)) == 0
        assert cb._active_decoder_params == []

    def test_always_on_persist_after_resample(self):
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        model = _fake_lm(6)
        cb = LisaCallback(LisaPolicy(num_layers=2, interval_steps=10, seed=1))
        cb.on_train_begin(None, _State(0), None, model=model)
        cb.on_step_end(None, _State(10), None, model=model, optimizer=_FakeOpt())
        assert self._flag(model, "embed_tokens")
        assert self._flag(model, "lm_head")
        assert self._flag(model, "model.norm")

    def test_gpt2_style_naming(self):
        import torch.nn as nn

        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        class GPT2ish(nn.Module):
            def __init__(self):
                super().__init__()
                self.wte = nn.Embedding(8, 4)
                self.wpe = nn.Embedding(8, 4)
                self.h = nn.ModuleList([nn.Linear(4, 4) for _ in range(5)])
                self.ln_f = nn.LayerNorm(4)
                self.lm_head = nn.Linear(4, 8)

        model = GPT2ish()
        cb = LisaCallback(LisaPolicy(num_layers=2, interval_steps=5, seed=0))
        cb.on_train_begin(None, _State(0), None, model=model)
        # h.N.* layers detected; wte/wpe/ln_f/lm_head stay on
        assert len(self._trainable_layer_indices(model)) == 2
        for sub in ("wte", "wpe", "ln_f", "lm_head"):
            assert self._flag(model, sub)

    def test_model_none_is_noop(self):
        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        cb = LisaCallback(LisaPolicy(num_layers=1, interval_steps=5))
        # no model kwarg -> returns control, no crash
        assert cb.on_train_begin(None, _State(0), None) is None
        assert cb.on_step_end(None, _State(5), None) is None

    def test_clear_optimizer_state_tolerates_stateless_opt(self):
        from soup_cli.utils.lisa import LisaCallback

        # optimizer object with no .state attr -> best-effort no-op, no raise
        LisaCallback._clear_optimizer_state(object(), [])

    def test_is_real_trainer_callback_subclass(self):
        # CRITICAL: HF dispatches every event via getattr(cb, event) with no
        # hasattr guard, so LisaCallback must inherit TrainerCallback's no-op
        # stubs or training crashes on on_epoch_begin.
        from transformers import TrainerCallback

        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        cb = LisaCallback(LisaPolicy(num_layers=1, interval_steps=5))
        assert isinstance(cb, TrainerCallback)
        # a non-overridden event exists and is callable (inherited no-op)
        assert callable(cb.on_epoch_begin)

    def test_no_decoder_layers_raises(self):
        import torch.nn as nn

        from soup_cli.utils.lisa import LisaCallback, LisaPolicy

        class NoLayers(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(4, 4)
                self.lm_head = nn.Linear(4, 4)

        cb = LisaCallback(LisaPolicy(num_layers=1, interval_steps=5))
        with pytest.raises(RuntimeError, match="decoder layer"):
            cb.on_train_begin(None, _State(0), None, model=NoLayers())

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


# ---------------------------------------------------------------------------
# Task B3 — wiring
# ---------------------------------------------------------------------------
class _FakeTrainer:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, cb):
        self.callbacks.append(cb)


class _TCfg:
    lisa_enabled = True
    lisa_num_layers = 3
    lisa_interval_steps = 15
    lisa_reset_optimizer = True
    seed = 0


class TestAttachLisa:
    def test_attaches_when_enabled(self):
        from soup_cli.utils.lisa import LisaCallback
        from soup_cli.utils.peft_wiring import attach_lisa_callback

        tr = _FakeTrainer()
        assert attach_lisa_callback(tr, _TCfg()) is True
        cbs = [c for c in tr.callbacks if isinstance(c, LisaCallback)]
        assert len(cbs) == 1
        # policy fields threaded correctly (guards against a field-swap bug)
        assert cbs[0].policy.num_layers == 3
        assert cbs[0].policy.interval_steps == 15
        assert cbs[0].policy.reset_optimizer is True

    def test_noop_when_disabled(self):
        from soup_cli.utils.peft_wiring import attach_lisa_callback

        cfg = _TCfg()
        cfg.lisa_enabled = False
        tr = _FakeTrainer()
        assert attach_lisa_callback(tr, cfg) is False
        assert tr.callbacks == []


class TestSftRouting:
    def test_branch_and_attach_present(self):
        import soup_cli.trainer.sft as sft

        src = Path(sft.__file__).read_text(encoding="utf-8")
        assert "tcfg.lisa_enabled" in src
        assert "attach_lisa_callback(" in src


# ---------------------------------------------------------------------------
# Task B4 — #307: LISA for continued pre-training (``task: pretrain``)
#
# LISA is full-FT of a rotating set of decoder layers, which is exactly what
# continued pre-training does, so #307 widens the schema gate and wires the
# callback into ``trainer/pretrain.py``. The tests below are behavioural on
# purpose: a source-grep would pass on a branch that never runs.
# ---------------------------------------------------------------------------
_LISA_PRETRAIN = _LISA_BASE.replace("task: sft", "task: pretrain")

# Text rows for the continued-pre-training path (plain documents, no chat
# structure). Vocabulary restricted to the offline tiny tokenizer's words.
_PRETRAIN_ROWS = [
    {"text": "the cat sat on the mat"},
    {"text": "the dog ran fast"},
    {"text": "hello world one two"},
    {"text": "red blue green"},
]


def _plain(text: str) -> str:
    """Rich output as comparable text: ANSI stripped, runs of space collapsed.

    Raw ``result.output`` phrase assertions have reddened this suite before;
    route every rendered-output assertion through here even when it is green
    today.
    """
    return re.sub(r"\s+", " ", re.sub(r"\x1b\[[0-9;]*m", "", text))


def _requires_train_extra():
    for mod in ("torch", "transformers", "peft", "trl", "datasets"):
        pytest.importorskip(mod, reason=f"{mod} is only in the [train] extra")


def _pretrain_wrapper(tmp_path, monkeypatch, *, n_layers=4, **training_over):
    """A real ``PretrainTrainerWrapper`` over a real tiny offline checkpoint."""
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.pretrain import PretrainTrainerWrapper

    # Reuse #341's fixture rather than a second copy of it: a real (tiny) Llama
    # checkpoint plus an offline tokenizer, so nothing is downloaded and nothing
    # is randomly initialised at load time.
    from tests.test_issue341_seed_and_fullft import _tiny_llama_dir

    base = _tiny_llama_dir(tmp_path, n_layers=n_layers)
    monkeypatch.chdir(tmp_path)
    training = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "quantization": "none",
        "epochs": 1,
        "lr": 1e-3,
        "logging_steps": 100,
        "save_steps": 10_000,
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj"]},
    }
    training.update(training_over)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "pretrain",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 64},
                "training": training,
                "output": str(tmp_path / "out"),
            }
        )
    )
    return PretrainTrainerWrapper(cfg, device="cpu"), {"train": list(_PRETRAIN_ROWS)}


def _captured_console(monkeypatch):
    """Patch the pretrain trainer's console with a REAL wide Rich console.

    Real (not a recording double) because the summary line is Rich markup, and
    wide because a narrow console wraps the parameter counts across a newline.
    """
    from io import StringIO

    from rich.console import Console

    import soup_cli.trainer.pretrain as pretrain_mod

    buf = StringIO()
    monkeypatch.setattr(pretrain_mod, "console", Console(file=buf, width=200))
    return buf


def _lisa_callbacks(trainer):
    from soup_cli.utils.lisa import LisaCallback

    return [
        cb
        for cb in trainer.callback_handler.callbacks
        if isinstance(cb, LisaCallback)
    ]


class TestLisaPretrainSchema:
    def test_pretrain_task_accepted(self):
        cfg = _load(_LISA_PRETRAIN)
        assert cfg.task == "pretrain"
        assert cfg.training.lisa_enabled is True

    @pytest.mark.parametrize("task", ["dpo", "grpo", "embedding"])
    def test_other_tasks_refused_and_both_supported_tasks_named(self, task):
        """The widened gate must still refuse everything else — and say what it
        does accept. Asserting on the refused task alone would also pass for a
        gate that had been widened to accept every task but this one."""
        with pytest.raises(Exception) as excinfo:
            _load(_LISA_BASE.replace("task: sft", f"task: {task}"))
        message = str(excinfo.value)
        assert "'sft'" in message
        assert "'pretrain'" in message
        assert task in message

    def test_the_allow_list_is_exactly_sft_and_pretrain(self):
        """Pins the membership of ``_LISA_SUPPORTED_TASKS`` itself.

        The tests above prove the gate READS the tuple (emptying it, or
        dropping either entry, reddens them), but not what is IN it: a tuple
        widened to a task whose trainer never wires the callback — say
        ``orpo`` — would accept the config and then silently ignore LISA, and
        every other test here would still pass. This is the other half.
        """
        from soup_cli.config.schema import _LISA_SUPPORTED_TASKS

        assert set(_LISA_SUPPORTED_TASKS) == {"sft", "pretrain"}


class TestPretrainLisaWiring:
    @pytest.mark.parametrize("lisa_enabled", [True, False])
    def test_model_load_dtype_matches_whether_the_base_is_trainable(
        self, tmp_path, monkeypatch, lisa_enabled
    ):
        """LISA steps the base weights, so it needs fp32 master weights.

        The plain-LoRA control must keep the checkpoint-native dtype; otherwise
        a repair that simply hardcodes fp32 for every pretrain run survives.
        """
        _requires_train_extra()
        import torch
        from transformers import AutoModelForCausalLM

        _captured_console(monkeypatch)
        captured_dtypes = []
        real_from_pretrained = AutoModelForCausalLM.from_pretrained

        def _capture_load_dtype(*args, **kwargs):
            captured_dtypes.append(kwargs["torch_dtype"])
            return real_from_pretrained(*args, **kwargs)

        monkeypatch.setattr(
            AutoModelForCausalLM, "from_pretrained", _capture_load_dtype
        )
        overrides = {"lisa_enabled": True} if lisa_enabled else {}
        wrapper, dataset = _pretrain_wrapper(tmp_path, monkeypatch, **overrides)
        wrapper.setup(dataset)

        expected_dtype = torch.float32 if lisa_enabled else "auto"
        assert captured_dtypes == [expected_dtype]

    @pytest.mark.parametrize(
        "num_layers,interval_steps", [(3, 15), (2, 7)]
    )
    def test_callback_carries_the_configured_policy(
        self, tmp_path, monkeypatch, num_layers, interval_steps
    ):
        """Two distinct configurations: a callback built from hardcoded
        constants would satisfy one of them and fail the other."""
        _requires_train_extra()
        _captured_console(monkeypatch)
        wrapper, dataset = _pretrain_wrapper(
            tmp_path,
            monkeypatch,
            lisa_enabled=True,
            lisa_num_layers=num_layers,
            lisa_interval_steps=interval_steps,
        )
        wrapper.setup(dataset)

        callbacks = _lisa_callbacks(wrapper.trainer)
        assert len(callbacks) == 1
        assert callbacks[0].policy.num_layers == num_layers
        assert callbacks[0].policy.interval_steps == interval_steps

    def test_lisa_run_is_unwrapped_and_fully_trainable(self, tmp_path, monkeypatch):
        _requires_train_extra()
        from peft import PeftModel

        _captured_console(monkeypatch)
        wrapper, dataset = _pretrain_wrapper(
            tmp_path, monkeypatch, lisa_enabled=True, lisa_num_layers=2
        )
        wrapper.setup(dataset)

        assert not isinstance(wrapper.model, PeftModel)
        # Fully trainable at setup time is the correctness invariant: HF builds
        # the optimizer before on_train_begin, so a decoder parameter left out
        # of the param groups can never be re-activated by the callback.
        frozen = [
            name
            for name, param in wrapper.model.named_parameters()
            if not param.requires_grad
        ]
        assert frozen == []

    def test_lora_control_still_wraps_and_attaches_no_lisa_callback(
        self, tmp_path, monkeypatch
    ):
        """The control: a plain pretrain config must be untouched by #307."""
        _requires_train_extra()
        from peft import PeftModel

        _captured_console(monkeypatch)
        wrapper, dataset = _pretrain_wrapper(tmp_path, monkeypatch)
        wrapper.setup(dataset)

        assert isinstance(wrapper.model, PeftModel)
        assert _lisa_callbacks(wrapper.trainer) == []

    @pytest.mark.parametrize("n_layers", [2, 4])
    def test_summary_counts_every_parameter_and_says_lisa(
        self, tmp_path, monkeypatch, n_layers
    ):
        """``get_nb_trainable_parameters`` is a PeftModel method and a LISA run
        has no PeftModel, so the count comes off the parameters directly. Two
        model depths, so a summary printing a constant fails one of them."""
        _requires_train_extra()
        buf = _captured_console(monkeypatch)
        wrapper, dataset = _pretrain_wrapper(
            tmp_path, monkeypatch, n_layers=n_layers, lisa_enabled=True
        )
        wrapper.setup(dataset)

        expected = sum(param.numel() for param in wrapper.model.parameters())
        out = _plain(buf.getvalue())
        assert f"LISA: {expected:,} trainable / {expected:,} total (100.00%)" in out
        assert "LoRA applied" not in out

    def test_lora_control_summary_reports_the_adapter_count(
        self, tmp_path, monkeypatch
    ):
        """Negative control for the label AND the number: without LISA the
        summary must still name LoRA and report only the adapter as trainable,
        which is strictly fewer parameters than the whole model."""
        _requires_train_extra()
        buf = _captured_console(monkeypatch)
        wrapper, dataset = _pretrain_wrapper(tmp_path, monkeypatch)
        wrapper.setup(dataset)

        trainable, total = wrapper.model.get_nb_trainable_parameters()
        out = _plain(buf.getvalue())
        assert trainable < total
        assert f"LoRA applied: {trainable:,} trainable / {total:,} total" in out
        assert "LISA:" not in out


class TestSharedLisaSetup:
    """#307 — the SFT and pretrain trainers must not drift on what "LISA is on"
    means, so both route through ``peft_wiring.apply_lisa_setup``.

    Everything else about a LISA run (unwrapped model, every parameter
    trainable) is produced by the surrounding ``elif`` skipping the LoRA path,
    so it survives that call being deleted. The two behaviours the call itself
    owns are pinned here: the ``enable_input_require_grads`` hand-off and the
    fact that each trainer actually goes through the shared helper.
    """

    @staticmethod
    def _tcfg(lisa_enabled):
        from types import SimpleNamespace

        return SimpleNamespace(
            lisa_enabled=lisa_enabled, lisa_num_layers=2, lisa_interval_steps=20
        )

    @pytest.mark.parametrize("lisa_enabled,calls", [(True, 1), (False, 0)])
    def test_input_require_grads_is_enabled_only_when_lisa_is_on(
        self, lisa_enabled, calls
    ):
        """Without a LoRA adapter nothing else makes the embedding output
        require grad, so gradient checkpointing dies with "None of the inputs
        have requires_grad". Both directions, so a helper that unconditionally
        enables it (or unconditionally returns a fixed verdict) fails one."""

        class _Model:
            def __init__(self):
                self.enabled = 0

            def enable_input_require_grads(self):
                self.enabled += 1

        from soup_cli.utils.peft_wiring import apply_lisa_setup

        model = _Model()
        assert apply_lisa_setup(model, self._tcfg(lisa_enabled)) is lisa_enabled
        assert model.enabled == calls

    def test_a_model_without_the_hook_is_still_accepted(self):
        """The ``hasattr`` guard's own control: not every backend exposes
        ``enable_input_require_grads``, and LISA must still turn on there."""
        from soup_cli.utils.peft_wiring import apply_lisa_setup

        assert apply_lisa_setup(object(), self._tcfg(True)) is True

    @staticmethod
    def _announce(tcfg):
        """Drive the helper through a REAL wide Rich console.

        Real because the banner is Rich markup and a recording double would
        capture the object repr; wide because a narrow console wraps the
        figures across a newline and the assertion then silently misses them.
        """
        from io import StringIO

        from rich.console import Console

        from soup_cli.utils.peft_wiring import apply_lisa_setup

        buf = StringIO()
        apply_lisa_setup(object(), tcfg, Console(file=buf, width=200))
        return _plain(buf.getvalue())

    @pytest.mark.parametrize("num_layers,interval_steps", [(3, 15), (2, 7)])
    def test_the_announcement_carries_the_configured_policy(
        self, num_layers, interval_steps
    ):
        """LISA replaces the LoRA path silently otherwise: the banner is the
        only thing telling the user which policy is live. Two policies, with
        the other one's figures asserted ABSENT, so a hardcoded banner passes
        neither."""
        tcfg = self._tcfg(True)
        tcfg.lisa_num_layers = num_layers
        tcfg.lisa_interval_steps = interval_steps

        out = self._announce(tcfg)

        assert (
            f"LISA: layerwise importance sampling ({num_layers} layer(s) "
            f"every {interval_steps} steps, LoRA off)"
        ) in out
        assert f"{5 - num_layers} layer(s)" not in out
        assert f"every {22 - interval_steps} steps" not in out

    def test_nothing_is_announced_when_lisa_is_off(self):
        """The silent-case control: without it a helper that announces
        unconditionally would print "LoRA off" on every plain LoRA run."""
        assert self._announce(self._tcfg(False)) == ""

    @pytest.mark.parametrize("lisa_enabled,expected_calls", [(True, 1), (False, 0)])
    def test_the_sft_trainer_routes_lisa_through_the_shared_helper(
        self, tmp_path, monkeypatch, lisa_enabled, expected_calls
    ):
        """Pins the #307 refactor: the SFT trainer's LISA branch must keep
        delegating. Re-inlining it there (the drift this change exists to
        prevent) leaves no call to observe."""
        _requires_train_extra()
        import soup_cli.utils.peft_wiring as peft_wiring
        from tests.test_issue341_seed_and_fullft import _wrapper

        seen = []
        real = peft_wiring.apply_lisa_setup

        def _spy(model, tcfg, console=None):
            seen.append(model)
            return real(model, tcfg, console)

        monkeypatch.setattr(peft_wiring, "apply_lisa_setup", _spy)
        over = (
            {"lisa_enabled": True, "lisa_num_layers": 2} if lisa_enabled else {}
        )
        wrapper, dataset = _wrapper(tmp_path, monkeypatch, **over)
        wrapper.setup(dataset)

        assert len(seen) == expected_calls
        if expected_calls:
            assert seen[0] is wrapper.model

    @pytest.mark.parametrize("lisa_enabled", [True, False])
    def test_the_pretrain_trainer_routes_lisa_through_the_shared_helper(
        self, tmp_path, monkeypatch, lisa_enabled
    ):
        """The pretrain half of the same pin, and the one the mutation needs:
        this trainer asks the helper on EVERY run and lets its return value
        decide whether the LoRA path runs, so the call is expected with LISA
        both on and off. Rewriting the call site as ``if not
        tcfg.lisa_enabled`` keeps every other test in this module green while
        silently dropping ``enable_input_require_grads`` and the LISA
        announcement here — that mutant dies on this test alone.
        """
        _requires_train_extra()
        import soup_cli.utils.peft_wiring as peft_wiring

        _captured_console(monkeypatch)
        seen = []
        real = peft_wiring.apply_lisa_setup

        def _spy(model, tcfg, console=None):
            seen.append(model)
            return real(model, tcfg, console)

        monkeypatch.setattr(peft_wiring, "apply_lisa_setup", _spy)
        over = (
            {"lisa_enabled": True, "lisa_num_layers": 2} if lisa_enabled else {}
        )
        wrapper, dataset = _pretrain_wrapper(tmp_path, monkeypatch, **over)
        wrapper.setup(dataset)

        assert len(seen) == 1
        # ...and it is handed the trainer's own model, not some other object.
        # With LISA on the model is left unwrapped, so it is still
        # ``wrapper.model``; with LISA off ``get_peft_model`` wraps it right
        # after, so it is that wrapper's base.
        if lisa_enabled:
            assert seen[0] is wrapper.model
        else:
            assert wrapper.model.get_base_model() is seen[0]


class TestPretrainLisaEndToEnd:
    def test_a_pretrain_lisa_run_moves_only_the_sampled_decoder_layers(
        self, tmp_path, monkeypatch
    ):
        """The acceptance criterion, end to end: a real continued-pre-training
        run with LISA moves the sampled decoder layer and no OTHER decoder
        layer. Scoped to the decoder on purpose — the embeddings, the LM head
        and the final norm stay trainable throughout by design (#267), so they
        are expected to move and are not part of this claim.

        A wiring test alone would pass on a callback that is attached and never
        fires; this fails unless the sampling actually reaches the optimizer.
        """
        _requires_train_extra()
        import torch

        _captured_console(monkeypatch)
        wrapper, dataset = _pretrain_wrapper(
            tmp_path,
            monkeypatch,
            n_layers=4,
            lisa_enabled=True,
            lisa_num_layers=1,
            # One sample at on_train_begin, none afterwards, so the active set
            # is constant for the whole run and "what changed" is unambiguous.
            lisa_interval_steps=1_000,
        )
        wrapper.setup(dataset)
        before = {
            name: param.detach().clone()
            for name, param in wrapper.model.named_parameters()
        }

        wrapper.train()

        from soup_cli.utils.lisa import _LAYER_RE

        def _layers(names):
            return {
                int(_LAYER_RE.search(name).group(1))
                for name in names
                if _LAYER_RE.search(name)
            }

        active = _layers(
            name
            for name, param in wrapper.model.named_parameters()
            if param.requires_grad
        )
        moved = _layers(
            name
            for name, param in wrapper.model.named_parameters()
            if not torch.equal(param.detach().cpu(), before[name].cpu())
        )
        assert len(active) == 1, "one layer configured, one layer sampled"
        assert moved == active
