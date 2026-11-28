"""A benchmark delta outside the float range is refused, not written as Infinity (#2804).

Both scores of a benchmark pass ``_validate_score`` (each is finite), but their difference
can still overflow: ``1e308 - (-1e308)`` is ``inf``. Before the fix ``soup ship --evidence``
printed SHIP (or DON'T SHIP for the other sign) and wrote a bare ``Infinity`` into the
verdict file, which a strict JSON decoder rejects. ``compute_benchmark_deltas`` and
``decide_ship`` each compute the difference, so each refuses it with a ``ValueError``
naming the benchmark; the evidence reader turns that into exit 3 (usage) and the MCP tool
into ``McpToolError``. Representable differences, rounding and the threshold are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soup_cli.commands import ship as ship_cmd
from soup_cli.mcp_server.registry import McpToolError, tool_ship_evidence
from soup_cli.utils.ship_verdict import (
    BenchmarkDelta,
    build_task_win,
    compute_benchmark_deltas,
    decide_ship,
)
from tests.cli_runner import CliRunner
from tests.conftest import strip_ansi

runner = CliRunner()

MESSAGE = "benchmark delta for 'sample' must be finite"
OVERFLOWS = [(-1e308, 1e308), (1e308, -1e308)]  # +inf, then -inf


def _evidence(base: float, tuned: float) -> dict:
    return {
        "task": {"mode": "metric", "base": 0.5, "tuned": 0.75},
        "benchmarks": {"sample": {"base": base, "tuned": tuned}},
    }


def _strict_loads(text: str) -> dict:
    def refuse(constant: str) -> None:
        raise ValueError(f"non-standard JSON constant {constant}")

    return json.loads(text, parse_constant=refuse)


def _cli(payload: dict):
    """Replay ``payload``; returns (result, the written verdict text or None)."""
    with runner.isolated_filesystem():
        Path("ev.json").write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(ship_cmd.app, ["--evidence", "ev.json", "--output", "out.json"])
        out = Path("out.json")
        written = out.read_text(encoding="utf-8") if out.exists() else None
    return result, written


@pytest.mark.parametrize("base, tuned", OVERFLOWS)
def test_the_cli_refuses_an_overflowing_delta_and_writes_nothing(base, tuned):
    result, written = _cli(_evidence(base, tuned))
    assert result.exit_code == 3, result.output
    assert written is None
    assert MESSAGE in " ".join(strip_ansi(result.output).split())


@pytest.mark.parametrize("base, tuned", OVERFLOWS)
def test_compute_benchmark_deltas_refuses_an_overflowing_delta(base, tuned):
    with pytest.raises(ValueError, match=MESSAGE):
        compute_benchmark_deltas({"sample": base}, {"sample": tuned})


@pytest.mark.parametrize("base, tuned", OVERFLOWS)
def test_decide_ship_refuses_an_overflowing_delta_given_directly(base, tuned):
    task_win = build_task_win("metric", 0.5, 0.75)
    item = BenchmarkDelta(name="sample", base=base, tuned=tuned, delta=0.0, regressed=False)
    with pytest.raises(ValueError, match=MESSAGE):
        decide_ship(task_win, [item])


@pytest.mark.parametrize("base, tuned", OVERFLOWS)
def test_the_mcp_tool_refuses_an_overflowing_delta(base, tuned):
    with runner.isolated_filesystem():
        Path("ev.json").write_text(json.dumps(_evidence(base, tuned)), encoding="utf-8")
        with pytest.raises(McpToolError) as exc_info:
            tool_ship_evidence({"evidence": "ev.json"})
    assert "must be finite" in str(exc_info.value)


@pytest.mark.parametrize(
    "base, tuned, exit_code, decision, delta",
    [(0.0, 1e308, 0, "SHIP", 1e308), (0.9, 0.7, 2, "DON'T SHIP", -0.2)],
)
def test_the_report_controls_still_write_strict_json(base, tuned, exit_code, decision, delta):
    result, written = _cli(_evidence(base, tuned))
    assert result.exit_code == exit_code, result.output
    verdict = _strict_loads(written)
    assert verdict["decision"] == decision
    assert verdict["benchmark_deltas"][0]["delta"] == pytest.approx(delta)
