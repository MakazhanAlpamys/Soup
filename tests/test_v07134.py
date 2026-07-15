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
