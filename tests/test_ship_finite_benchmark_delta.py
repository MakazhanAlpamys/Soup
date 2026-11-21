"""Tests that finite benchmark scores cannot produce non-finite deltas in ship verdict (#2804)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.commands import ship as ship_cmd
from soup_cli.utils.ship_verdict import (
    BenchmarkDelta,
    build_task_win,
    compute_benchmark_deltas,
    decide_ship,
    verdict_from_evidence,
    verdict_to_dict,
)
from tests.cli_runner import CliRunner

runner = CliRunner()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-compliant float token: {value}")


def test_benchmark_delta_dataclass_rejects_non_finite_delta():
    with pytest.raises(ValueError, match="benchmark delta for 'sample' must be finite"):
        BenchmarkDelta("sample", base=0.0, tuned=1.0, delta=float("inf"), regressed=False)
    with pytest.raises(ValueError, match="benchmark delta for 'sample' must be finite"):
        BenchmarkDelta("sample", base=0.0, tuned=1.0, delta=float("-inf"), regressed=True)
    with pytest.raises(ValueError, match="benchmark delta for 'sample' must be finite"):
        BenchmarkDelta("sample", base=0.0, tuned=1.0, delta=float("nan"), regressed=False)


def test_compute_benchmark_deltas_raises_on_overflow():
    with pytest.raises(ValueError, match="benchmark delta for 'sample' must be finite"):
        compute_benchmark_deltas({"sample": -1e308}, {"sample": 1e308})

    with pytest.raises(ValueError, match="benchmark delta for 'sample' must be finite"):
        compute_benchmark_deltas({"sample": 1e308}, {"sample": -1e308})


def test_decide_ship_raises_on_recomputed_overflow():
    task_win = build_task_win("metric", 0.5, 0.75)
    bd = BenchmarkDelta("sample", base=-1e308, tuned=1e308, delta=0.0, regressed=False)
    with pytest.raises(ValueError, match="benchmark delta for 'sample' must be finite"):
        decide_ship(task_win, [bd])


def test_verdict_from_evidence_raises_on_overflow():
    ev_pos = {
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.75},
        "benchmarks": {"sample": {"base": -1e308, "tuned": 1e308}},
    }
    match_msg = "invalid evidence.benchmarks: benchmark delta for 'sample' must be finite"
    with pytest.raises(ValueError, match=match_msg):
        verdict_from_evidence(ev_pos)

    ev_neg = {
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.75},
        "benchmarks": {"sample": {"base": 1e308, "tuned": -1e308}},
    }
    with pytest.raises(ValueError, match=match_msg):
        verdict_from_evidence(ev_neg)


def test_finite_controls_remain_valid():
    # Large finite delta that does not overflow
    ev_large = {
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.75},
        "benchmarks": {"sample": {"base": 0.0, "tuned": 1e308}},
    }
    v_large = verdict_from_evidence(ev_large)
    assert v_large.decision == "SHIP"
    assert v_large.benchmark_deltas[0].delta == 1e308
    strict_json_large = json.dumps(verdict_to_dict(v_large), allow_nan=False)
    assert "Infinity" not in strict_json_large

    # Ordinary delta
    ev_ord = {
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.75},
        "benchmarks": {"sample": {"base": 0.9, "tuned": 0.7}},
    }
    v_ord = verdict_from_evidence(ev_ord)
    assert v_ord.decision == "DON'T SHIP"
    assert v_ord.benchmark_deltas[0].delta == -0.2
    strict_json_ord = json.dumps(verdict_to_dict(v_ord), allow_nan=False)
    assert "Infinity" not in strict_json_ord


def test_cli_evidence_overflow_refused_exit_usage():
    with runner.isolated_filesystem():
        ev_file = Path("evidence.json")
        out_file = Path("verdict.json")
        ev_file.write_text(
            json.dumps({
                "task": {"mode": "metric", "base": 0.5, "tuned": 0.75},
                "benchmarks": {"sample": {"base": -1e308, "tuned": 1e308}},
            }),
            encoding="utf-8",
        )
        cmd = ["--evidence", "evidence.json", "--output", "verdict.json"]
        res = runner.invoke(ship_cmd.app, cmd)
        assert res.exit_code == 3
        assert "benchmark delta for 'sample' must be finite" in res.output
        assert not out_file.exists()


def test_cli_evidence_negative_overflow_refused_exit_usage():
    with runner.isolated_filesystem():
        ev_file = Path("evidence.json")
        out_file = Path("verdict.json")
        ev_file.write_text(
            json.dumps({
                "task": {"mode": "metric", "base": 0.5, "tuned": 0.75},
                "benchmarks": {"sample": {"base": 1e308, "tuned": -1e308}},
            }),
            encoding="utf-8",
        )
        cmd = ["--evidence", "evidence.json", "--output", "verdict.json"]
        res = runner.invoke(ship_cmd.app, cmd)
        assert res.exit_code == 3
        assert "benchmark delta for 'sample' must be finite" in res.output
        assert not out_file.exists()


def test_cli_controls_produce_strictly_valid_json():
    with runner.isolated_filesystem():
        ev_file = Path("evidence.json")
        out_file = Path("verdict.json")

        # Control 1: 0.0 vs 1e308 (SHIP)
        ev_file.write_text(
            json.dumps({
                "task": {"mode": "metric", "base": 0.5, "tuned": 0.75},
                "benchmarks": {"sample": {"base": 0.0, "tuned": 1e308}},
            }),
            encoding="utf-8",
        )
        cmd = ["--evidence", "evidence.json", "--output", "verdict.json"]
        res1 = runner.invoke(ship_cmd.app, cmd)
        assert res1.exit_code == 0
        content1 = out_file.read_text(encoding="utf-8")
        parsed1 = json.loads(content1, parse_constant=reject_constant)
        assert parsed1["decision"] == "SHIP"
        assert parsed1["benchmark_deltas"][0]["delta"] == 1e308

        # Control 2: 0.9 vs 0.7 (DON'T SHIP, regression)
        ev_file.write_text(
            json.dumps({
                "task": {"mode": "metric", "base": 0.5, "tuned": 0.75},
                "benchmarks": {"sample": {"base": 0.9, "tuned": 0.7}},
            }),
            encoding="utf-8",
        )
        res2 = runner.invoke(ship_cmd.app, cmd)
        assert res2.exit_code == 2
        content2 = out_file.read_text(encoding="utf-8")
        parsed2 = json.loads(content2, parse_constant=reject_constant)
        assert parsed2["decision"] == "DON'T SHIP"
        assert parsed2["benchmark_deltas"][0]["delta"] == -0.2
