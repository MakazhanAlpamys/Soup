"""v0.57.0 Part C — adapters blame: plan emitter + budget check."""

from __future__ import annotations

import os
import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app as soup_app
from soup_cli.utils.blame import (
    BlamePlan,
    BlameShardWork,
    parse_budget,
    plan_blame,
    run_blame,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")

runner = CliRunner()


# ---------- parse_budget ----------


@pytest.mark.parametrize("spec,expected", [
    ("60", 60),
    ("60s", 60),
    ("5m", 300),
    ("2h", 7200),
    ("1h", 3600),
    ("90s", 90),
])
def test_parse_budget_happy(spec, expected):
    assert parse_budget(spec) == expected


def test_parse_budget_below_min():
    with pytest.raises(ValueError, match="below floor"):
        parse_budget("30")


def test_parse_budget_above_cap():
    with pytest.raises(ValueError, match="above cap"):
        parse_budget("25h")


def test_parse_budget_invalid_format():
    with pytest.raises(ValueError, match="invalid"):
        parse_budget("abc")


def test_parse_budget_empty():
    with pytest.raises(ValueError):
        parse_budget("")


def test_parse_budget_bool():
    with pytest.raises(TypeError):
        parse_budget(True)  # type: ignore[arg-type]


def test_parse_budget_null_byte():
    with pytest.raises(ValueError, match="null"):
        parse_budget("60\x00s")


def test_parse_budget_non_string():
    with pytest.raises(TypeError):
        parse_budget(60)  # type: ignore[arg-type]


# ---------- plan_blame ----------


def _setup_blame(tmp_path, monkeypatch, n_rows: int = 100):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adapter").mkdir()
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "\n".join(f'{{"text": "row{i}"}}' for i in range(n_rows)),
        encoding="utf-8",
    )
    return "adapter", "data.jsonl"


def test_plan_blame_happy(tmp_path, monkeypatch):
    adapter, dataset = _setup_blame(tmp_path, monkeypatch, n_rows=100)
    plan = plan_blame(
        adapter, dataset,
        layer="q_proj.7", budget_seconds=3600, num_shards=10,
    )
    assert isinstance(plan, BlamePlan)
    assert len(plan.shards) == 10
    assert plan.layer == "q_proj.7"
    assert plan.feasible is True
    # Each shard covers ~10 rows
    assert plan.shards[0].holdout_size == 10


def test_plan_blame_infeasible_budget(tmp_path, monkeypatch):
    adapter, dataset = _setup_blame(tmp_path, monkeypatch)
    plan = plan_blame(
        adapter, dataset,
        layer="q_proj.7", budget_seconds=60, num_shards=100,
    )
    # 60s / 100 shards = 0s/shard → infeasible
    assert plan.feasible is False
    assert "need" in plan.reason


def test_plan_blame_shard_offsets(tmp_path, monkeypatch):
    adapter, dataset = _setup_blame(tmp_path, monkeypatch, n_rows=50)
    plan = plan_blame(
        adapter, dataset,
        layer="layer", budget_seconds=3600, num_shards=5,
    )
    offsets = [s.holdout_offset for s in plan.shards]
    assert offsets == [0, 10, 20, 30, 40]


def test_plan_blame_uneven_split(tmp_path, monkeypatch):
    adapter, dataset = _setup_blame(tmp_path, monkeypatch, n_rows=23)
    plan = plan_blame(
        adapter, dataset,
        layer="layer", budget_seconds=3600, num_shards=5,
    )
    # ceil(23 / 5) = 5 → last shard gets 3 rows
    sizes = [s.holdout_size for s in plan.shards]
    assert sum(sizes) == 23


def test_plan_blame_empty_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adapter").mkdir()
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        plan_blame(
            "adapter", "empty.jsonl",
            layer="x", budget_seconds=3600, num_shards=5,
        )


def test_plan_blame_outside_cwd_adapter(tmp_path):
    with pytest.raises(ValueError):
        plan_blame(
            str(tmp_path), "data.jsonl",
            layer="x", budget_seconds=3600, num_shards=5,
        )


def test_plan_blame_outside_cwd_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adapter").mkdir()
    outside = os.path.join(os.path.dirname(str(tmp_path)), "data.jsonl")
    with pytest.raises(ValueError):
        plan_blame(
            "adapter", outside,
            layer="x", budget_seconds=3600, num_shards=5,
        )


def test_plan_blame_invalid_layer(tmp_path, monkeypatch):
    adapter, dataset = _setup_blame(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="non-empty"):
        plan_blame(
            adapter, dataset,
            layer="", budget_seconds=3600, num_shards=5,
        )
    with pytest.raises(ValueError, match="null"):
        plan_blame(
            adapter, dataset,
            layer="x\x00y", budget_seconds=3600, num_shards=5,
        )


def test_plan_blame_invalid_shards(tmp_path, monkeypatch):
    adapter, dataset = _setup_blame(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="num_shards"):
        plan_blame(
            adapter, dataset,
            layer="x", budget_seconds=3600, num_shards=1,
        )
    with pytest.raises(ValueError, match="num_shards"):
        plan_blame(
            adapter, dataset,
            layer="x", budget_seconds=3600, num_shards=200,
        )


@pytest.mark.parametrize("bool_value", [True, False])
def test_plan_blame_bool_shards(tmp_path, monkeypatch, bool_value):
    """bool subclasses int; both True (→1) and False (→0) must be rejected."""
    adapter, dataset = _setup_blame(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        plan_blame(
            adapter, dataset,
            layer="x", budget_seconds=3600,
            num_shards=bool_value,  # type: ignore[arg-type]
        )


def test_plan_blame_bool_budget_rejected(tmp_path, monkeypatch):
    """budget_seconds=True must not silently coerce to 1 (project bool policy)."""
    adapter, dataset = _setup_blame(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        plan_blame(
            adapter, dataset,
            layer="x",
            budget_seconds=True,  # type: ignore[arg-type]
            num_shards=5,
        )


def test_plan_blame_missing_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adapter").mkdir()
    # File doesn't exist BUT also doesn't exist in cwd containment check
    # The containment helper rejects non-existent first. Try with a real
    # outside-cwd file to confirm flow.
    (tmp_path / "missing.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        # Empty file → empty dataset rejection
        plan_blame(
            "adapter", "missing.jsonl",
            layer="x", budget_seconds=3600, num_shards=2,
        )


# ---------- run_blame stub ----------


def test_run_blame_stub_marker():
    plan = BlamePlan(
        adapter_dir="a", dataset_path="d", layer="x",
        budget_seconds=3600, num_shards=2, per_shard_seconds=600,
        shards=(BlameShardWork(0, 0, 1, 600), BlameShardWork(1, 1, 1, 600)),
        feasible=True, reason="ok",
    )
    with pytest.raises(NotImplementedError, match="v0.57.1"):
        run_blame(plan)


def test_run_blame_rejects_non_plan():
    with pytest.raises(TypeError):
        run_blame("not a plan")  # type: ignore[arg-type]


# ---------- Frozen invariants ----------


def test_blame_plan_frozen():
    import dataclasses
    plan = BlamePlan(
        adapter_dir="a", dataset_path="d", layer="x",
        budget_seconds=3600, num_shards=1, per_shard_seconds=3600,
        shards=(), feasible=False, reason="r",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.layer = "y"  # type: ignore[misc]


def test_blame_shard_work_frozen():
    import dataclasses
    shard = BlameShardWork(0, 0, 10, 60)
    with pytest.raises(dataclasses.FrozenInstanceError):
        shard.shard_id = 1  # type: ignore[misc]


# ---------- CLI ----------


def test_adapters_blame_help():
    result = runner.invoke(soup_app, ["adapters", "blame", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "--budget" in _strip_ansi(result.output)
    assert "--layer" in _strip_ansi(result.output)
    assert "--shards" in _strip_ansi(result.output)


def test_adapters_blame_plan_only(tmp_path, monkeypatch):
    adapter, dataset = _setup_blame(tmp_path, monkeypatch)
    result = runner.invoke(soup_app, [
        "adapters", "blame", adapter,
        "--dataset", dataset,
        "--layer", "q_proj.7",
        "--budget", "1h",
        "--shards", "5",
        "--plan-only",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "Blame plan" in _strip_ansi(result.output)


def test_adapters_blame_invalid_budget(tmp_path, monkeypatch):
    adapter, dataset = _setup_blame(tmp_path, monkeypatch)
    result = runner.invoke(soup_app, [
        "adapters", "blame", adapter,
        "--dataset", dataset,
        "--layer", "x",
        "--budget", "abc",
        "--shards", "5",
    ])
    assert result.exit_code == 2
    assert "Invalid --budget" in _strip_ansi(result.output)


def test_adapters_blame_live_runner_advisory(tmp_path, monkeypatch):
    adapter, dataset = _setup_blame(tmp_path, monkeypatch)
    result = runner.invoke(soup_app, [
        "adapters", "blame", adapter,
        "--dataset", dataset,
        "--layer", "x",
        "--budget", "1h",
        "--shards", "5",
    ])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "v0.57.1" in _strip_ansi(result.output)
