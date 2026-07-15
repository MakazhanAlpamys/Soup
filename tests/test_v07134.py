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
