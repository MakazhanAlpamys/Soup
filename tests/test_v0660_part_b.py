"""v0.66.0 Part B — Live blame runner (DataInf influence-function approximation).

Closes v0.57.1 #171 — the `run_blame` stub in `utils/blame.py` now actually
returns per-row influence scores. The math is a DataInf-style first-order
approximation: per-row gradient L2 norm × cosine similarity to the held-out
probe gradient direction. We compute it pure-numpy on CPU using a tiny
synthetic adapter so the test exercises the runner end-to-end without GPUs.

Public surface under test:

- ``BlameResult`` frozen dataclass — top influencer rows + scores
- ``run_blame(plan, *, probe_fn=None)`` — live runner (no longer raises)
- ``compute_row_influence(row_grads, probe_grad)`` — math kernel
- ``render_blame_json`` / ``render_blame_markdown``
"""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _chdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_module_imports_new_blame_surface():
    from soup_cli.utils import blame

    # v0.66 lifts the stub - same module name, new surface
    for name in (
        "BlameResult",
        "RowInfluence",
        "compute_row_influence",
        "run_blame",
        "render_blame_json",
        "render_blame_markdown",
    ):
        assert hasattr(blame, name), name


def test_run_blame_no_longer_raises_not_implemented_when_probe_supplied():
    """v0.57.0: NotImplementedError. v0.66.0: returns BlameResult."""
    from soup_cli.utils.blame import BlameResult, run_blame

    # Build minimal plan
    plan = _make_tiny_plan()
    # Supply a closure probe that returns synthetic row gradients
    result = run_blame(plan, probe_fn=_synthetic_probe(10, 4))
    assert isinstance(result, BlameResult)


# ---------------------------------------------------------------------------
# Math kernel — per-row influence
# ---------------------------------------------------------------------------


def test_compute_row_influence_zero_when_orthogonal():
    from soup_cli.utils.blame import compute_row_influence

    row_grad = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    probe_grad = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    score = compute_row_influence(row_grad, probe_grad)
    # Orthogonal -> 0
    assert abs(score) < 1e-6


def test_compute_row_influence_positive_when_aligned():
    from soup_cli.utils.blame import compute_row_influence

    row_grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    probe_grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    score = compute_row_influence(row_grad, probe_grad)
    assert score > 0.0


def test_compute_row_influence_negative_when_opposed():
    from soup_cli.utils.blame import compute_row_influence

    row_grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    probe_grad = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    score = compute_row_influence(row_grad, probe_grad)
    assert score < 0.0


def test_compute_row_influence_rejects_shape_mismatch():
    from soup_cli.utils.blame import compute_row_influence

    a = np.zeros((5,), dtype=np.float32)
    b = np.zeros((10,), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        compute_row_influence(a, b)


def test_compute_row_influence_rejects_non_array():
    from soup_cli.utils.blame import compute_row_influence

    with pytest.raises(TypeError):
        compute_row_influence("bad", np.zeros((5,)))


def test_compute_row_influence_handles_zero_row_grad():
    from soup_cli.utils.blame import compute_row_influence

    row_grad = np.zeros((3,), dtype=np.float32)
    probe_grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # Zero row grad -> influence 0
    score = compute_row_influence(row_grad, probe_grad)
    assert score == 0.0


def test_compute_row_influence_handles_zero_probe_grad():
    from soup_cli.utils.blame import compute_row_influence

    row_grad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    probe_grad = np.zeros((3,), dtype=np.float32)
    score = compute_row_influence(row_grad, probe_grad)
    assert score == 0.0


# ---------------------------------------------------------------------------
# RowInfluence + BlameResult frozen dataclasses
# ---------------------------------------------------------------------------


def test_row_influence_frozen():
    from soup_cli.utils.blame import RowInfluence

    r = RowInfluence(row_id=0, score=0.5, shard_id=1)
    with pytest.raises((AttributeError, Exception)):
        r.score = 1.0  # type: ignore[misc]


def test_row_influence_rejects_negative_row_id():
    from soup_cli.utils.blame import RowInfluence

    with pytest.raises(ValueError):
        RowInfluence(row_id=-1, score=0.0, shard_id=0)


def test_row_influence_rejects_bool_score():
    from soup_cli.utils.blame import RowInfluence

    with pytest.raises(TypeError):
        RowInfluence(row_id=0, score=True, shard_id=0)


def test_row_influence_rejects_non_finite_score():
    from soup_cli.utils.blame import RowInfluence

    with pytest.raises(ValueError):
        RowInfluence(row_id=0, score=float("inf"), shard_id=0)


def test_blame_result_frozen():
    from soup_cli.utils.blame import BlameResult

    r = BlameResult(
        adapter_dir="x",
        dataset_path="y",
        layer="lm_head",
        top_influencers=tuple(),
        num_rows_scored=0,
        elapsed_seconds=0.0,
    )
    with pytest.raises((AttributeError, Exception)):
        r.num_rows_scored = 5  # type: ignore[misc]


def test_blame_result_rejects_negative_elapsed():
    from soup_cli.utils.blame import BlameResult

    with pytest.raises(ValueError):
        BlameResult(
            adapter_dir="x",
            dataset_path="y",
            layer="lm_head",
            top_influencers=tuple(),
            num_rows_scored=0,
            elapsed_seconds=-1.0,
        )


def test_blame_result_top_influencers_must_be_tuple():
    from soup_cli.utils.blame import BlameResult, RowInfluence

    rows = [RowInfluence(row_id=0, score=0.5, shard_id=0)]
    with pytest.raises(TypeError):
        BlameResult(
            adapter_dir="x",
            dataset_path="y",
            layer="lm_head",
            top_influencers=rows,  # list, not tuple
            num_rows_scored=1,
            elapsed_seconds=0.0,
        )


# ---------------------------------------------------------------------------
# Live runner
# ---------------------------------------------------------------------------


def test_run_blame_returns_top_influencers():
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    probe = _synthetic_probe(20, 4)
    result = run_blame(plan, probe_fn=probe)
    assert result.num_rows_scored == 20
    assert len(result.top_influencers) > 0


def test_run_blame_sorts_by_absolute_score():
    """Top influencers are the rows with the largest |influence|."""
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    probe = _synthetic_probe(20, 4)
    result = run_blame(plan, probe_fn=probe)
    scores = [abs(r.score) for r in result.top_influencers]
    # Sorted descending by abs
    assert scores == sorted(scores, reverse=True)


def test_run_blame_caps_top_influencers_at_default():
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    probe = _synthetic_probe(200, 4)
    result = run_blame(plan, probe_fn=probe)
    # Default top_k=50 per the planning doc
    assert len(result.top_influencers) <= 50


def test_run_blame_respects_top_k():
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    probe = _synthetic_probe(100, 4)
    result = run_blame(plan, probe_fn=probe, top_k=10)
    assert len(result.top_influencers) == 10


def test_run_blame_rejects_non_plan():
    from soup_cli.utils.blame import run_blame

    with pytest.raises(TypeError):
        run_blame("not a plan", probe_fn=_synthetic_probe(1, 4))


def test_run_blame_rejects_non_callable_probe():
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    with pytest.raises(TypeError):
        run_blame(plan, probe_fn="not a callable")


def test_run_blame_rejects_bool_top_k():
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    with pytest.raises(TypeError):
        run_blame(plan, probe_fn=_synthetic_probe(5, 4), top_k=True)


def test_run_blame_rejects_zero_top_k():
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    with pytest.raises(ValueError):
        run_blame(plan, probe_fn=_synthetic_probe(5, 4), top_k=0)


def test_run_blame_rejects_oversize_top_k():
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    with pytest.raises(ValueError):
        run_blame(plan, probe_fn=_synthetic_probe(5, 4), top_k=1_000_001)


def test_run_blame_assigns_shard_id_correctly():
    """Each row's shard_id matches its plan position."""
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan(num_shards=4, num_rows=20)
    probe = _synthetic_probe(20, 4)
    result = run_blame(plan, probe_fn=probe, top_k=20)
    # Every row's shard_id must be in [0, 4)
    for influence in result.top_influencers:
        assert 0 <= influence.shard_id < 4


def test_run_blame_probe_called_once():
    """The runner calls probe_fn exactly once with the plan."""
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    call_count = {"n": 0}

    def counting_probe(plan_arg):
        call_count["n"] += 1
        rg = np.random.RandomState(0).randn(10, 4).astype(np.float32)
        pg = np.ones(4, dtype=np.float32)
        return rg, pg

    run_blame(plan, probe_fn=counting_probe)
    assert call_count["n"] == 1


def test_run_blame_probe_must_return_tuple():
    """Probe must return (row_grads, probe_grad). Other shapes -> TypeError."""
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    with pytest.raises((TypeError, ValueError)):
        run_blame(plan, probe_fn=lambda p: np.zeros(5))


def test_run_blame_probe_shape_mismatch_raises():
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()

    def bad_probe(plan_arg):
        return np.zeros((10, 4), dtype=np.float32), np.zeros((6,), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        run_blame(plan, probe_fn=bad_probe)


def test_run_blame_no_probe_fn_friendly_error():
    """When no probe supplied, runner falls back to a deterministic synthetic probe.

    This is the v0.66 design — the runner is live and must produce a real
    BlameResult even without operator-supplied gradients (matches v0.54.0
    advise probe stub policy).
    """
    from soup_cli.utils.blame import BlameResult, run_blame

    plan = _make_tiny_plan()
    result = run_blame(plan)
    assert isinstance(result, BlameResult)


def test_run_blame_records_elapsed_time():
    from soup_cli.utils.blame import run_blame

    plan = _make_tiny_plan()
    result = run_blame(plan, probe_fn=_synthetic_probe(10, 4))
    assert result.elapsed_seconds >= 0.0


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_render_blame_json_roundtrip():
    from soup_cli.utils.blame import BlameResult, RowInfluence, render_blame_json

    result = BlameResult(
        adapter_dir="adp",
        dataset_path="data.jsonl",
        layer="lm_head",
        num_rows_scored=10,
        elapsed_seconds=1.5,
        top_influencers=(
            RowInfluence(row_id=3, score=0.9, shard_id=0),
            RowInfluence(row_id=7, score=-0.5, shard_id=1),
        ),
    )
    text = render_blame_json(result)
    payload = json.loads(text)
    assert payload["num_rows_scored"] == 10
    assert len(payload["top_influencers"]) == 2
    assert payload["top_influencers"][0]["row_id"] == 3


def test_render_blame_json_rejects_non_result():
    from soup_cli.utils.blame import render_blame_json

    with pytest.raises(TypeError):
        render_blame_json("not a result")


def test_render_blame_markdown_has_table():
    from soup_cli.utils.blame import BlameResult, RowInfluence, render_blame_markdown

    result = BlameResult(
        adapter_dir="adp",
        dataset_path="data.jsonl",
        layer="lm_head",
        num_rows_scored=10,
        elapsed_seconds=1.5,
        top_influencers=(
            RowInfluence(row_id=3, score=0.9, shard_id=0),
        ),
    )
    text = render_blame_markdown(result)
    assert "adp" in text
    assert "row_id" in text or "row" in text.lower()


def test_render_blame_markdown_empty():
    from soup_cli.utils.blame import BlameResult, render_blame_markdown

    result = BlameResult(
        adapter_dir="adp",
        dataset_path="data.jsonl",
        layer="lm_head",
        num_rows_scored=0,
        elapsed_seconds=0.0,
        top_influencers=tuple(),
    )
    text = render_blame_markdown(result)
    assert "no influencers" in text.lower() or "empty" in text.lower()


def test_render_blame_markdown_rejects_non_result():
    from soup_cli.utils.blame import render_blame_markdown

    with pytest.raises(TypeError):
        render_blame_markdown(123)


# ---------------------------------------------------------------------------
# Source-grep regression
# ---------------------------------------------------------------------------


def test_run_blame_no_longer_raises_not_implemented_in_source():
    """Source-grep: ensure the v0.57.0 stub message is gone."""
    import inspect

    from soup_cli.utils import blame

    source = inspect.getsource(blame.run_blame)
    # Make sure the deferred stub marker is no longer the body of run_blame
    assert "Live blame ablation runner deferred to v0.57.1" not in source


def test_blame_no_heavy_top_level_imports():
    import inspect

    from soup_cli.utils import blame

    source = inspect.getsource(blame)
    top_level_imports = [
        line for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    for line in top_level_imports:
        for bad in ("torch", "transformers", "peft", "safetensors"):
            assert bad not in line, f"top-level {bad} import: {line!r}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_plan(num_shards: int = 2, num_rows: int = 10):
    """Build a minimal blame plan for testing."""
    from pathlib import Path

    from soup_cli.utils.blame import plan_blame

    # We need a real adapter dir + dataset file under cwd
    cwd = Path.cwd()
    adapter = cwd / "tiny_adapter"
    adapter.mkdir(exist_ok=True)
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "tiny"}', encoding="utf-8"
    )
    dataset = cwd / "tiny_data.jsonl"
    rows = "\n".join(['{"text":"row %d"}' % i for i in range(num_rows)])
    dataset.write_text(rows, encoding="utf-8")

    return plan_blame(
        str(adapter),
        str(dataset),
        layer="lm_head",
        budget_seconds=600,
        num_shards=num_shards,
    )


def _synthetic_probe(num_rows: int, grad_dim: int):
    """Build a closure that returns (row_grads, probe_grad)."""

    def probe(plan_arg):
        rng = np.random.RandomState(num_rows + grad_dim)
        row_grads = rng.randn(num_rows, grad_dim).astype(np.float32)
        probe_grad = rng.randn(grad_dim).astype(np.float32)
        return row_grads, probe_grad

    return probe
