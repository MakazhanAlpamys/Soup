"""v0.63.0 Part C — Active-learning sampler tests."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_module_imports():
    from soup_cli.utils import active_sampler

    assert hasattr(active_sampler, "ActiveLearningPlan")
    assert hasattr(active_sampler, "score_uncertainty")
    assert hasattr(active_sampler, "pick_top_uncertain")
    assert hasattr(active_sampler, "validate_budget")
    assert hasattr(active_sampler, "sample_uncertain_rows")


# ---------------------------------------------------------------------------
# validate_budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 10, 100, 10_000])
def test_validate_budget_happy(value):
    from soup_cli.utils.active_sampler import validate_budget

    assert validate_budget(value) == value


@pytest.mark.parametrize("bad", [True, False, None, "10", -5, 0, 100_001, 1.5])
def test_validate_budget_rejects(bad):
    from soup_cli.utils.active_sampler import validate_budget

    with pytest.raises((TypeError, ValueError)):
        validate_budget(bad)


# ---------------------------------------------------------------------------
# score_uncertainty
# ---------------------------------------------------------------------------


def test_score_uncertainty_max_entropy():
    """Single-RM: uncertainty = 1 - |2*score - 1|. Score 0.5 -> max entropy."""
    from soup_cli.utils.active_sampler import score_uncertainty

    # Single RM with score 0.5 should yield maximum uncertainty
    s = score_uncertainty(scores=[0.5])
    assert math.isclose(s, 1.0, abs_tol=1e-6)

    s = score_uncertainty(scores=[0.0])
    assert math.isclose(s, 0.0, abs_tol=1e-6)

    s = score_uncertainty(scores=[1.0])
    assert math.isclose(s, 0.0, abs_tol=1e-6)


def test_score_uncertainty_two_rms_disagreement():
    """Two RMs: uncertainty = |s1 - s2|. Big gap -> max disagreement."""
    from soup_cli.utils.active_sampler import score_uncertainty

    s = score_uncertainty(scores=[0.1, 0.9])
    assert math.isclose(s, 0.8, abs_tol=1e-6)

    s = score_uncertainty(scores=[0.5, 0.5])
    assert math.isclose(s, 0.0, abs_tol=1e-6)


def test_score_uncertainty_empty():
    from soup_cli.utils.active_sampler import score_uncertainty

    assert score_uncertainty(scores=[]) == 0.0


def test_score_uncertainty_rejects_non_finite():
    from soup_cli.utils.active_sampler import score_uncertainty

    with pytest.raises(ValueError):
        score_uncertainty(scores=[float("nan")])
    with pytest.raises(ValueError):
        score_uncertainty(scores=[float("inf")])


def test_score_uncertainty_rejects_bool():
    from soup_cli.utils.active_sampler import score_uncertainty

    with pytest.raises(TypeError):
        score_uncertainty(scores=[True, False])


def test_score_uncertainty_rejects_out_of_range():
    from soup_cli.utils.active_sampler import score_uncertainty

    with pytest.raises(ValueError):
        score_uncertainty(scores=[1.5])
    with pytest.raises(ValueError):
        score_uncertainty(scores=[-0.1])


def test_score_uncertainty_rejects_too_many_rms():
    from soup_cli.utils.active_sampler import score_uncertainty

    # >2 RMs not yet supported (avg pairwise std would be the v0.63.1+ shape)
    with pytest.raises(ValueError):
        score_uncertainty(scores=[0.5, 0.6, 0.7])


# ---------------------------------------------------------------------------
# pick_top_uncertain
# ---------------------------------------------------------------------------


def test_pick_top_uncertain_orders_descending():
    from soup_cli.utils.active_sampler import pick_top_uncertain

    rows = [
        {"id": "a", "uncertainty": 0.2},
        {"id": "b", "uncertainty": 0.9},
        {"id": "c", "uncertainty": 0.5},
    ]
    top = pick_top_uncertain(rows, budget=2)
    assert [r["id"] for r in top] == ["b", "c"]


def test_pick_top_uncertain_budget_caps_output():
    from soup_cli.utils.active_sampler import pick_top_uncertain

    rows = [{"id": str(i), "uncertainty": i / 100} for i in range(50)]
    top = pick_top_uncertain(rows, budget=5)
    assert len(top) == 5


def test_pick_top_uncertain_handles_missing_uncertainty():
    from soup_cli.utils.active_sampler import pick_top_uncertain

    rows = [{"id": "a"}, {"id": "b", "uncertainty": 0.7}]
    top = pick_top_uncertain(rows, budget=2)
    assert top[0]["id"] == "b"  # 'a' treated as 0 uncertainty


def test_pick_top_uncertain_empty():
    from soup_cli.utils.active_sampler import pick_top_uncertain

    assert pick_top_uncertain([], budget=5) == []


def test_pick_top_uncertain_invalid_budget():
    from soup_cli.utils.active_sampler import pick_top_uncertain

    rows = [{"id": "a", "uncertainty": 0.5}]
    with pytest.raises((TypeError, ValueError)):
        pick_top_uncertain(rows, budget=True)
    with pytest.raises((TypeError, ValueError)):
        pick_top_uncertain(rows, budget=0)


def test_pick_top_uncertain_rejects_non_mapping():
    from soup_cli.utils.active_sampler import pick_top_uncertain

    with pytest.raises(TypeError):
        pick_top_uncertain([1, 2, 3], budget=2)


# ---------------------------------------------------------------------------
# ActiveLearningPlan
# ---------------------------------------------------------------------------


def test_active_learning_plan_frozen():
    from soup_cli.utils.active_sampler import ActiveLearningPlan

    plan = ActiveLearningPlan(
        rows_in=100,
        rows_selected=10,
        budget=10,
        mean_uncertainty=0.85,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.rows_in = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# sample_uncertain_rows
# ---------------------------------------------------------------------------


def test_sample_uncertain_rows_happy(tmp_path, monkeypatch):
    from soup_cli.utils.active_sampler import sample_uncertain_rows

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        {"id": str(i), "prompt": f"q{i}", "output": f"a{i}", "rm_score": s}
        for i, s in enumerate([0.1, 0.5, 0.9, 0.6, 0.05])
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    plan = sample_uncertain_rows(str(inp), output_path=str(out), budget=2)
    assert plan.rows_in == 5
    assert plan.rows_selected == 2
    out_rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert len(out_rows) == 2
    # Top uncertainty rows should be the 0.5 and 0.6 ones (closest to 0.5)
    ids = [r["id"] for r in out_rows]
    assert "1" in ids  # rm=0.5 -> uncertainty=1.0
    assert "3" in ids  # rm=0.6 -> uncertainty=0.8


def test_sample_uncertain_rows_dual_rm(tmp_path, monkeypatch):
    from soup_cli.utils.active_sampler import sample_uncertain_rows

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        {"id": "a", "rm_scores": [0.1, 0.9]},   # disagreement 0.8
        {"id": "b", "rm_scores": [0.5, 0.5]},   # disagreement 0.0
        {"id": "c", "rm_scores": [0.3, 0.7]},   # disagreement 0.4
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    plan = sample_uncertain_rows(str(inp), output_path=str(out), budget=2)
    assert plan.rows_selected == 2
    out_rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    ids = [r["id"] for r in out_rows]
    assert ids == ["a", "c"]


def test_sample_uncertain_rows_rejects_outside_cwd(tmp_path, monkeypatch):
    from soup_cli.utils.active_sampler import sample_uncertain_rows

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "stray.jsonl"
    outside.write_text('{"id":"x","rm_score":0.5}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    try:
        with pytest.raises(ValueError, match="outside"):
            sample_uncertain_rows(str(outside), output_path=str(out), budget=1)
    finally:
        if outside.exists():
            outside.unlink()


def test_sample_uncertain_rows_missing_input(tmp_path, monkeypatch):
    from soup_cli.utils.active_sampler import sample_uncertain_rows

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        sample_uncertain_rows(
            str(tmp_path / "missing.jsonl"),
            output_path=str(tmp_path / "o.jsonl"),
            budget=1,
        )


def test_sample_uncertain_rows_budget_bigger_than_input(tmp_path, monkeypatch):
    """Selecting more than input has — output capped to input size."""
    from soup_cli.utils.active_sampler import sample_uncertain_rows

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        {"id": "1", "rm_score": 0.5},
        {"id": "2", "rm_score": 0.7},
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    plan = sample_uncertain_rows(str(inp), output_path=str(out), budget=100)
    assert plan.rows_in == 2
    assert plan.rows_selected == 2


def test_sample_uncertain_rows_rejects_null_byte():
    from soup_cli.utils.active_sampler import sample_uncertain_rows

    with pytest.raises(ValueError):
        sample_uncertain_rows("bad\x00path.jsonl", output_path="o.jsonl", budget=1)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_active_sample_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["data", "active-sample", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "budget" in result.output.lower()


def test_cli_active_sample_happy(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [{"id": str(i), "rm_score": s} for i, s in enumerate([0.5, 0.95, 0.1])]
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = runner.invoke(
        app,
        ["data", "active-sample", "--input", str(inp), "--output", str(out), "--budget", "1"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert out.exists()
    out_rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    assert len(out_rows) == 1
    assert out_rows[0]["id"] == "0"  # rm=0.5 has highest uncertainty


def test_cli_active_sample_invalid_budget(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"id":"x","rm_score":0.5}\n', encoding="utf-8")
    result = runner.invoke(
        app,
        ["data", "active-sample", "--input", str(inp), "--budget", "0"],
    )
    assert result.exit_code != 0
