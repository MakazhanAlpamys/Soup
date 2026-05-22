"""v0.66.0 Part D — Catastrophic interference matrix (TDD).

When you have multiple adapters in the Registry, run all pairwise on a
probe set. Score interference (loss on adapter A's domain when adapter B
is also loaded). Surfaces which adapter pairs you cannot deploy together
via the v0.22 multi-adapter serve.

Public surface:

- ``InterferenceCell`` frozen dataclass — one (A, B) result
- ``InterferenceMatrix`` frozen dataclass — full N×N matrix
- ``compute_interference(loss_a_alone, loss_a_with_b)`` — math kernel
- ``classify_interference(score)`` — OK/MINOR/MAJOR taxonomy
- ``build_interference_matrix(adapters, probe_losses)`` — pure orchestrator
- ``render_matrix_json`` / ``render_matrix_markdown``
"""
from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_module_imports():
    from soup_cli.utils import interference

    for name in (
        "InterferenceCell",
        "InterferenceMatrix",
        "compute_interference",
        "classify_interference",
        "build_interference_matrix",
        "render_matrix_json",
        "render_matrix_markdown",
        "INTERFERENCE_VERDICTS",
    ):
        assert hasattr(interference, name), name


def test_verdicts_closed():
    from soup_cli.utils.interference import INTERFERENCE_VERDICTS

    assert INTERFERENCE_VERDICTS == frozenset({"OK", "MINOR", "MAJOR"})


# ---------------------------------------------------------------------------
# compute_interference — math kernel
# ---------------------------------------------------------------------------


def test_compute_interference_zero_when_identical_loss():
    from soup_cli.utils.interference import compute_interference

    # No change: score = 0
    assert compute_interference(2.0, 2.0) == pytest.approx(0.0)


def test_compute_interference_positive_when_loss_increases():
    """When loss_a_with_b > loss_a_alone, interference score is positive."""
    from soup_cli.utils.interference import compute_interference

    score = compute_interference(2.0, 3.0)
    # Relative increase: (3 - 2) / 2 = 0.5
    assert score == pytest.approx(0.5)


def test_compute_interference_negative_when_loss_decreases():
    """When loss decreases (beneficial interaction), score is negative."""
    from soup_cli.utils.interference import compute_interference

    score = compute_interference(2.0, 1.5)
    # (1.5 - 2) / 2 = -0.25
    assert score == pytest.approx(-0.25)


def test_compute_interference_rejects_bool():
    from soup_cli.utils.interference import compute_interference

    with pytest.raises(TypeError):
        compute_interference(True, 1.0)


def test_compute_interference_rejects_non_finite():
    from soup_cli.utils.interference import compute_interference

    with pytest.raises(ValueError):
        compute_interference(float("nan"), 1.0)
    with pytest.raises(ValueError):
        compute_interference(1.0, float("inf"))


def test_compute_interference_rejects_non_positive_baseline():
    """Cannot divide by zero or negative baseline loss."""
    from soup_cli.utils.interference import compute_interference

    with pytest.raises(ValueError):
        compute_interference(0.0, 1.0)
    with pytest.raises(ValueError):
        compute_interference(-1.0, 1.0)


def test_compute_interference_rejects_negative_combined_loss():
    from soup_cli.utils.interference import compute_interference

    with pytest.raises(ValueError):
        compute_interference(1.0, -0.5)


# ---------------------------------------------------------------------------
# classify_interference
# ---------------------------------------------------------------------------


def test_classify_ok_when_small_change():
    from soup_cli.utils.interference import classify_interference

    # < 5% increase
    assert classify_interference(0.02) == "OK"
    assert classify_interference(-0.02) == "OK"


def test_classify_minor_in_5_to_20_pct():
    from soup_cli.utils.interference import classify_interference

    assert classify_interference(0.10) == "MINOR"


def test_classify_major_above_20_pct():
    from soup_cli.utils.interference import classify_interference

    assert classify_interference(0.30) == "MAJOR"


def test_classify_exact_boundaries():
    from soup_cli.utils.interference import classify_interference

    # 0.05 → MINOR (boundary lands in more severe)
    assert classify_interference(0.05) == "MINOR"
    # 0.20 → MAJOR
    assert classify_interference(0.20) == "MAJOR"


def test_classify_uses_absolute_value():
    """Symmetric — large negative (beneficial) is still 'MINOR' worth noting."""
    from soup_cli.utils.interference import classify_interference

    assert classify_interference(-0.10) == "MINOR"


def test_classify_rejects_bool():
    from soup_cli.utils.interference import classify_interference

    with pytest.raises(TypeError):
        classify_interference(True)


def test_classify_rejects_non_finite():
    from soup_cli.utils.interference import classify_interference

    with pytest.raises(ValueError):
        classify_interference(float("nan"))


# ---------------------------------------------------------------------------
# InterferenceCell
# ---------------------------------------------------------------------------


def test_cell_frozen():
    from soup_cli.utils.interference import InterferenceCell

    cell = InterferenceCell(
        adapter_a="a", adapter_b="b", score=0.0, verdict="OK"
    )
    with pytest.raises((AttributeError, Exception)):
        cell.score = 1.0  # type: ignore[misc]


def test_cell_rejects_invalid_verdict():
    from soup_cli.utils.interference import InterferenceCell

    with pytest.raises(ValueError):
        InterferenceCell(adapter_a="a", adapter_b="b", score=0.0, verdict="X")


def test_cell_rejects_empty_adapter_name():
    from soup_cli.utils.interference import InterferenceCell

    with pytest.raises(ValueError):
        InterferenceCell(adapter_a="", adapter_b="b", score=0.0, verdict="OK")


def test_cell_rejects_bool_score():
    from soup_cli.utils.interference import InterferenceCell

    with pytest.raises(TypeError):
        InterferenceCell(adapter_a="a", adapter_b="b", score=True, verdict="OK")


def test_cell_rejects_non_finite_score():
    from soup_cli.utils.interference import InterferenceCell

    with pytest.raises(ValueError):
        InterferenceCell(
            adapter_a="a", adapter_b="b", score=float("inf"), verdict="OK"
        )


def test_cell_rejects_null_byte_adapter_name():
    from soup_cli.utils.interference import InterferenceCell

    with pytest.raises(ValueError):
        InterferenceCell(adapter_a="a\x00b", adapter_b="b", score=0.0, verdict="OK")


# ---------------------------------------------------------------------------
# InterferenceMatrix
# ---------------------------------------------------------------------------


def test_matrix_frozen():
    from soup_cli.utils.interference import InterferenceMatrix

    m = InterferenceMatrix(
        adapters=("a", "b"),
        cells=tuple(),
        worst_pair=None,
        worst_score=0.0,
    )
    with pytest.raises((AttributeError, Exception)):
        m.worst_score = 1.0  # type: ignore[misc]


def test_matrix_rejects_non_tuple_adapters():
    from soup_cli.utils.interference import InterferenceMatrix

    with pytest.raises(TypeError):
        InterferenceMatrix(
            adapters=["a", "b"],  # list, not tuple
            cells=tuple(),
            worst_pair=None,
            worst_score=0.0,
        )


def test_matrix_rejects_duplicate_adapter_names():
    from soup_cli.utils.interference import InterferenceMatrix

    with pytest.raises(ValueError, match="duplicate"):
        InterferenceMatrix(
            adapters=("a", "a"),
            cells=tuple(),
            worst_pair=None,
            worst_score=0.0,
        )


def test_matrix_rejects_too_few_adapters():
    from soup_cli.utils.interference import InterferenceMatrix

    with pytest.raises(ValueError, match="at least 2"):
        InterferenceMatrix(
            adapters=("a",),
            cells=tuple(),
            worst_pair=None,
            worst_score=0.0,
        )


# ---------------------------------------------------------------------------
# build_interference_matrix
# ---------------------------------------------------------------------------


def test_build_matrix_2x2():
    from soup_cli.utils.interference import build_interference_matrix

    adapters = ("a", "b")
    # Probe losses: (target, with) loss pairs
    #   a on a alone = 2.0
    #   a on b alone = 1.5
    #   b on a alone = 1.0
    #   b on b alone = 2.5
    #   a + b loaded: a target = 2.2, b target = 2.8
    losses = {
        ("a", "a"): 2.0,
        ("b", "b"): 2.5,
        ("a", "b"): 2.2,  # loss on a's domain when b is loaded
        ("b", "a"): 2.8,  # loss on b's domain when a is loaded
    }
    matrix = build_interference_matrix(adapters, losses)
    # Diagonal cells: A vs A — score 0
    diagonal = [c for c in matrix.cells if c.adapter_a == c.adapter_b]
    assert all(c.score == 0.0 for c in diagonal)
    # Off-diagonal: 2 cells (a→b, b→a)
    off_diag = [c for c in matrix.cells if c.adapter_a != c.adapter_b]
    assert len(off_diag) == 2


def test_build_matrix_worst_pair():
    from soup_cli.utils.interference import build_interference_matrix

    adapters = ("a", "b", "c")
    losses = {
        ("a", "a"): 1.0, ("b", "b"): 1.0, ("c", "c"): 1.0,
        ("a", "b"): 1.1,  # 10% increase
        ("a", "c"): 1.5,  # 50% increase (MAJOR)
        ("b", "a"): 1.05, ("b", "c"): 1.2,
        ("c", "a"): 1.3, ("c", "b"): 1.0,
    }
    matrix = build_interference_matrix(adapters, losses)
    # Worst pair = (a, c) with score 0.5
    assert matrix.worst_pair == ("a", "c")
    assert matrix.worst_score == pytest.approx(0.5)


def test_build_matrix_rejects_unknown_adapter_in_losses():
    from soup_cli.utils.interference import build_interference_matrix

    adapters = ("a", "b")
    losses = {("a", "a"): 1.0, ("b", "b"): 1.0, ("z", "x"): 0.5}
    with pytest.raises(ValueError, match="unknown"):
        build_interference_matrix(adapters, losses)


def test_build_matrix_rejects_missing_diagonal():
    from soup_cli.utils.interference import build_interference_matrix

    adapters = ("a", "b")
    # Missing ("b", "b")
    losses = {("a", "a"): 1.0}
    with pytest.raises(ValueError, match="diagonal"):
        build_interference_matrix(adapters, losses)


def test_build_matrix_rejects_bool_loss():
    from soup_cli.utils.interference import build_interference_matrix

    adapters = ("a", "b")
    losses = {("a", "a"): True, ("b", "b"): 1.0}
    with pytest.raises(TypeError):
        build_interference_matrix(adapters, losses)


def test_build_matrix_rejects_non_finite_loss():
    from soup_cli.utils.interference import build_interference_matrix

    adapters = ("a", "b")
    losses = {("a", "a"): float("nan"), ("b", "b"): 1.0}
    with pytest.raises(ValueError):
        build_interference_matrix(adapters, losses)


def test_build_matrix_rejects_too_few_adapters():
    from soup_cli.utils.interference import build_interference_matrix

    with pytest.raises(ValueError):
        build_interference_matrix(("a",), {("a", "a"): 1.0})


def test_build_matrix_rejects_too_many_adapters():
    from soup_cli.utils.interference import build_interference_matrix

    # 30 adapters = 900 pairs, but matrix cap is 16 by default
    adapters = tuple(f"a{i}" for i in range(30))
    losses = {(a, a): 1.0 for a in adapters}
    with pytest.raises(ValueError, match="too many"):
        build_interference_matrix(adapters, losses)


def test_build_matrix_handles_empty_pairwise_losses():
    """Missing off-diagonal entries default to baseline (score 0)."""
    from soup_cli.utils.interference import build_interference_matrix

    adapters = ("a", "b")
    losses = {("a", "a"): 1.0, ("b", "b"): 1.0}  # no off-diagonal
    matrix = build_interference_matrix(adapters, losses)
    # All cells have score 0 (missing pair = no measurement = assume no interference)
    for cell in matrix.cells:
        assert cell.score == 0.0


def test_build_matrix_rejects_non_pair_loss_keys():
    from soup_cli.utils.interference import build_interference_matrix

    adapters = ("a", "b")
    losses = {"not a tuple": 1.0, ("a", "a"): 1.0, ("b", "b"): 1.0}
    with pytest.raises(TypeError, match="tuple"):
        build_interference_matrix(adapters, losses)


def test_build_matrix_rejects_bool_adapter_name():
    from soup_cli.utils.interference import build_interference_matrix

    with pytest.raises(TypeError):
        build_interference_matrix((True, "b"), {})  # type: ignore[arg-type]


def test_build_matrix_rejects_null_byte_adapter_name():
    from soup_cli.utils.interference import build_interference_matrix

    with pytest.raises(ValueError):
        build_interference_matrix(("a\x00", "b"), {})


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_render_matrix_json_roundtrip():
    from soup_cli.utils.interference import (
        InterferenceCell,
        InterferenceMatrix,
        render_matrix_json,
    )

    m = InterferenceMatrix(
        adapters=("a", "b"),
        cells=(
            InterferenceCell(adapter_a="a", adapter_b="b", score=0.10, verdict="MINOR"),
        ),
        worst_pair=("a", "b"),
        worst_score=0.10,
    )
    text = render_matrix_json(m)
    payload = json.loads(text)
    assert payload["adapters"] == ["a", "b"]
    assert payload["worst_score"] == pytest.approx(0.10)


def test_render_matrix_json_rejects_non_matrix():
    from soup_cli.utils.interference import render_matrix_json

    with pytest.raises(TypeError):
        render_matrix_json("nope")


def test_render_matrix_markdown_has_table():
    from soup_cli.utils.interference import (
        InterferenceCell,
        InterferenceMatrix,
        render_matrix_markdown,
    )

    m = InterferenceMatrix(
        adapters=("a", "b"),
        cells=(
            InterferenceCell(adapter_a="a", adapter_b="b", score=0.30, verdict="MAJOR"),
        ),
        worst_pair=("a", "b"),
        worst_score=0.30,
    )
    text = render_matrix_markdown(m)
    assert "MAJOR" in text
    assert "a" in text


def test_render_matrix_markdown_empty_cells():
    from soup_cli.utils.interference import InterferenceMatrix, render_matrix_markdown

    m = InterferenceMatrix(
        adapters=("a", "b"),
        cells=tuple(),
        worst_pair=None,
        worst_score=0.0,
    )
    text = render_matrix_markdown(m)
    assert "no" in text.lower() or "empty" in text.lower()


def test_render_matrix_markdown_rejects_non_matrix():
    from soup_cli.utils.interference import render_matrix_markdown

    with pytest.raises(TypeError):
        render_matrix_markdown(None)


# ---------------------------------------------------------------------------
# Source-grep
# ---------------------------------------------------------------------------


def test_no_heavy_top_level_imports():
    import inspect

    from soup_cli.utils import interference

    source = inspect.getsource(interference)
    top_level_imports = [
        line for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    for line in top_level_imports:
        for bad in ("torch", "transformers", "peft", "safetensors"):
            assert bad not in line, f"top-level {bad} import: {line!r}"
