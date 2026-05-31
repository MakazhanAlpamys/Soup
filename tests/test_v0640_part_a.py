"""v0.64.0 Part A — `soup tunability` probe across candidate bases.

Tests cover:
- CandidateBase frozen dataclass + validation
- TunabilityResult frozen dataclass + validation
- TunabilityReport frozen dataclass + immutable candidates tuple
- validate_probe_steps bounds + bool reject
- validate_holdout_size bounds + bool reject
- score_candidate happy + delta math (lower-better loss)
- pareto_frontier identifies non-dominated points
- run_tunability orchestrator (with mocked probe callable)
- write_report + load_report atomic + cwd containment + symlink rejection
- CLI smoke (--help / outside-cwd reject / unknown candidate)
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import pytest
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------


def test_module_imports():
    from soup_cli.utils import tunability

    assert hasattr(tunability, "CandidateBase")
    assert hasattr(tunability, "TunabilityResult")
    assert hasattr(tunability, "TunabilityReport")
    assert hasattr(tunability, "validate_probe_steps")
    assert hasattr(tunability, "validate_holdout_size")
    assert hasattr(tunability, "score_candidate")
    assert hasattr(tunability, "pareto_frontier")
    assert hasattr(tunability, "run_tunability")
    assert hasattr(tunability, "write_report")
    assert hasattr(tunability, "load_report")
    assert hasattr(tunability, "DEFAULT_CANDIDATES")


# ---------------------------------------------------------------------------
# DEFAULT_CANDIDATES catalog
# ---------------------------------------------------------------------------


def test_default_candidates_nonempty():
    from soup_cli.utils.tunability import DEFAULT_CANDIDATES

    assert len(DEFAULT_CANDIDATES) >= 6
    # Each entry must be CandidateBase
    from soup_cli.utils.tunability import CandidateBase
    for c in DEFAULT_CANDIDATES:
        assert isinstance(c, CandidateBase)


def test_default_candidates_immutable():
    from soup_cli.utils.tunability import DEFAULT_CANDIDATES

    # Tuple, not list
    assert isinstance(DEFAULT_CANDIDATES, tuple)


# ---------------------------------------------------------------------------
# CandidateBase
# ---------------------------------------------------------------------------


def test_candidate_base_frozen():
    from soup_cli.utils.tunability import CandidateBase

    c = CandidateBase(
        name="qwen3-0.6b",
        repo_id="Qwen/Qwen3-0.6B",
        params_b=0.6,
        license_id="apache-2.0",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.name = "other"  # type: ignore[misc]


def test_candidate_base_rejects_empty_name():
    from soup_cli.utils.tunability import CandidateBase

    with pytest.raises(ValueError, match="name"):
        CandidateBase(name="", repo_id="x/y", params_b=1.0, license_id="apache-2.0")


def test_candidate_base_rejects_null_byte():
    from soup_cli.utils.tunability import CandidateBase

    with pytest.raises(ValueError, match="null"):
        CandidateBase(name="bad\x00", repo_id="x/y", params_b=1.0, license_id="apache-2.0")


def test_candidate_base_rejects_negative_params():
    from soup_cli.utils.tunability import CandidateBase

    with pytest.raises(ValueError, match="params_b"):
        CandidateBase(name="x", repo_id="x/y", params_b=-1.0, license_id="apache-2.0")


def test_candidate_base_rejects_bool_params():
    from soup_cli.utils.tunability import CandidateBase

    with pytest.raises(TypeError, match="bool"):
        CandidateBase(name="x", repo_id="x/y", params_b=True, license_id="apache-2.0")  # type: ignore[arg-type]


def test_candidate_base_rejects_non_finite_params():
    from soup_cli.utils.tunability import CandidateBase

    with pytest.raises(ValueError, match="finite"):
        CandidateBase(name="x", repo_id="x/y", params_b=float("nan"), license_id="apache-2.0")


def test_candidate_base_rejects_oversize_name():
    from soup_cli.utils.tunability import CandidateBase

    with pytest.raises(ValueError, match="too long"):
        CandidateBase(name="x" * 513, repo_id="x/y", params_b=1.0, license_id="apache-2.0")


# ---------------------------------------------------------------------------
# validate_probe_steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [10, 100, 1000])
def test_validate_probe_steps_happy(value):
    from soup_cli.utils.tunability import validate_probe_steps

    assert validate_probe_steps(value) == value


def test_validate_probe_steps_boundary_min():
    from soup_cli.utils.tunability import validate_probe_steps

    assert validate_probe_steps(10) == 10
    with pytest.raises(ValueError):
        validate_probe_steps(9)


def test_validate_probe_steps_boundary_max():
    from soup_cli.utils.tunability import validate_probe_steps

    assert validate_probe_steps(10_000) == 10_000
    with pytest.raises(ValueError):
        validate_probe_steps(10_001)


@pytest.mark.parametrize("bad", [True, False, None, "100", -1, 0, 9, 10_001, 1.5])
def test_validate_probe_steps_rejects(bad):
    from soup_cli.utils.tunability import validate_probe_steps

    with pytest.raises((TypeError, ValueError)):
        validate_probe_steps(bad)


# ---------------------------------------------------------------------------
# validate_holdout_size
# ---------------------------------------------------------------------------


def test_validate_holdout_size_happy():
    from soup_cli.utils.tunability import validate_holdout_size

    assert validate_holdout_size(100) == 100


def test_validate_holdout_size_boundary():
    from soup_cli.utils.tunability import validate_holdout_size

    assert validate_holdout_size(10) == 10
    with pytest.raises(ValueError):
        validate_holdout_size(9)
    assert validate_holdout_size(100_000) == 100_000
    with pytest.raises(ValueError):
        validate_holdout_size(100_001)


@pytest.mark.parametrize("bad", [True, False, "100", -1, 0])
def test_validate_holdout_size_rejects(bad):
    from soup_cli.utils.tunability import validate_holdout_size

    with pytest.raises((TypeError, ValueError)):
        validate_holdout_size(bad)


# ---------------------------------------------------------------------------
# TunabilityResult
# ---------------------------------------------------------------------------


def test_tunability_result_happy():
    from soup_cli.utils.tunability import CandidateBase, TunabilityResult

    cand = CandidateBase(
        name="qwen3-0.6b", repo_id="x/y", params_b=0.6, license_id="apache-2.0"
    )
    r = TunabilityResult(
        candidate=cand,
        base_loss=2.5,
        probe_loss=2.0,
        delta=0.5,
        wall_clock_seconds=120.0,
        estimated_cost_usd=0.05,
    )
    assert r.delta == 0.5


def test_tunability_result_frozen():
    from soup_cli.utils.tunability import CandidateBase, TunabilityResult

    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    r = TunabilityResult(
        candidate=cand,
        base_loss=2.5,
        probe_loss=2.0,
        delta=0.5,
        wall_clock_seconds=120.0,
        estimated_cost_usd=0.05,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.delta = 0.0  # type: ignore[misc]


def test_tunability_result_rejects_non_finite():
    from soup_cli.utils.tunability import CandidateBase, TunabilityResult

    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    with pytest.raises(ValueError, match="finite"):
        TunabilityResult(
            candidate=cand,
            base_loss=float("nan"),
            probe_loss=2.0,
            delta=0.5,
            wall_clock_seconds=120.0,
            estimated_cost_usd=0.05,
        )


def test_tunability_result_rejects_negative_wall_clock():
    from soup_cli.utils.tunability import CandidateBase, TunabilityResult

    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    with pytest.raises(ValueError, match="wall_clock"):
        TunabilityResult(
            candidate=cand,
            base_loss=2.5,
            probe_loss=2.0,
            delta=0.5,
            wall_clock_seconds=-1.0,
            estimated_cost_usd=0.05,
        )


def test_tunability_result_rejects_negative_cost():
    from soup_cli.utils.tunability import CandidateBase, TunabilityResult

    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    with pytest.raises(ValueError, match="cost"):
        TunabilityResult(
            candidate=cand,
            base_loss=2.5,
            probe_loss=2.0,
            delta=0.5,
            wall_clock_seconds=120.0,
            estimated_cost_usd=-0.01,
        )


def test_tunability_result_rejects_bool_loss():
    from soup_cli.utils.tunability import CandidateBase, TunabilityResult

    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    with pytest.raises(TypeError, match="bool"):
        TunabilityResult(
            candidate=cand,
            base_loss=True,  # type: ignore[arg-type]
            probe_loss=2.0,
            delta=0.5,
            wall_clock_seconds=120.0,
            estimated_cost_usd=0.05,
        )


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------


def test_score_candidate_delta_math():
    """delta = base_loss - probe_loss (positive = improvement)."""
    from soup_cli.utils.tunability import score_candidate

    base_loss, probe_loss = 2.5, 2.0
    delta = score_candidate(base_loss=base_loss, probe_loss=probe_loss)
    assert delta == pytest.approx(0.5)


def test_score_candidate_zero_when_no_change():
    from soup_cli.utils.tunability import score_candidate

    assert score_candidate(base_loss=2.0, probe_loss=2.0) == 0.0


def test_score_candidate_negative_when_worse():
    from soup_cli.utils.tunability import score_candidate

    assert score_candidate(base_loss=2.0, probe_loss=2.5) == pytest.approx(-0.5)


def test_score_candidate_rejects_non_finite():
    from soup_cli.utils.tunability import score_candidate

    with pytest.raises(ValueError):
        score_candidate(base_loss=float("nan"), probe_loss=2.0)
    with pytest.raises(ValueError):
        score_candidate(base_loss=2.0, probe_loss=float("inf"))


def test_score_candidate_rejects_bool():
    from soup_cli.utils.tunability import score_candidate

    with pytest.raises(TypeError):
        score_candidate(base_loss=True, probe_loss=2.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# pareto_frontier
# ---------------------------------------------------------------------------


def test_pareto_frontier_simple():
    """Maximise delta, minimise cost. Strictly dominated entries get dropped."""
    from soup_cli.utils.tunability import CandidateBase, TunabilityResult, pareto_frontier

    def _mk(name: str, delta: float, cost: float) -> TunabilityResult:
        cand = CandidateBase(name=name, repo_id="x/y", params_b=1.0, license_id="apache-2.0")
        return TunabilityResult(
            candidate=cand,
            base_loss=2.5,
            probe_loss=2.5 - delta,
            delta=delta,
            wall_clock_seconds=60.0,
            estimated_cost_usd=cost,
        )

    # B dominates A: B has higher delta AND lower cost.
    # C is on the frontier: lower delta but lower cost than B.
    a = _mk("a", delta=0.1, cost=0.20)
    b = _mk("b", delta=0.5, cost=0.10)
    c = _mk("c", delta=0.05, cost=0.05)
    frontier = pareto_frontier([a, b, c])
    names = {r.candidate.name for r in frontier}
    # A is strictly dominated by B (lower delta, higher cost)
    assert "a" not in names
    assert "b" in names
    assert "c" in names


def test_pareto_frontier_empty():
    from soup_cli.utils.tunability import pareto_frontier

    assert pareto_frontier([]) == ()


def test_pareto_frontier_single():
    from soup_cli.utils.tunability import CandidateBase, TunabilityResult, pareto_frontier

    cand = CandidateBase(name="a", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    r = TunabilityResult(
        candidate=cand,
        base_loss=2.5,
        probe_loss=2.0,
        delta=0.5,
        wall_clock_seconds=60.0,
        estimated_cost_usd=0.05,
    )
    frontier = pareto_frontier([r])
    assert frontier == (r,)


def test_pareto_frontier_returns_tuple():
    from soup_cli.utils.tunability import pareto_frontier

    assert isinstance(pareto_frontier([]), tuple)


def test_pareto_frontier_rejects_non_sequence():
    from soup_cli.utils.tunability import pareto_frontier

    with pytest.raises(TypeError):
        pareto_frontier("not a list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TunabilityReport
# ---------------------------------------------------------------------------


def test_tunability_report_frozen():
    from soup_cli.utils.tunability import CandidateBase, TunabilityReport, TunabilityResult

    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    r = TunabilityResult(
        candidate=cand, base_loss=2.5, probe_loss=2.0, delta=0.5,
        wall_clock_seconds=60.0, estimated_cost_usd=0.05,
    )
    report = TunabilityReport(results=(r,), frontier=(r,), probe_steps=100, holdout_size=64)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.probe_steps = 0  # type: ignore[misc]


def test_tunability_report_results_tuple():
    from soup_cli.utils.tunability import CandidateBase, TunabilityReport, TunabilityResult

    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    r = TunabilityResult(
        candidate=cand, base_loss=2.5, probe_loss=2.0, delta=0.5,
        wall_clock_seconds=60.0, estimated_cost_usd=0.05,
    )
    # Lists rejected — frozen=True doesn't make lists immutable
    with pytest.raises(TypeError, match="tuple"):
        TunabilityReport(results=[r], frontier=(r,), probe_steps=100, holdout_size=64)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_tunability
# ---------------------------------------------------------------------------


def test_run_tunability_with_mocked_probe(tmp_path):
    """Inject a deterministic probe to exercise the orchestrator."""
    from soup_cli.utils.tunability import (
        CandidateBase,
        TunabilityResult,
        run_tunability,
    )

    candidates = (
        CandidateBase(name="cand-a", repo_id="x/y", params_b=0.5, license_id="apache-2.0"),
        CandidateBase(name="cand-b", repo_id="x/z", params_b=1.0, license_id="mit"),
    )

    def fake_probe(cand: CandidateBase, dataset_path: str, *, probe_steps: int,
                   holdout_size: int) -> TunabilityResult:
        # Synthetic: larger param count → bigger delta, longer wall-clock
        return TunabilityResult(
            candidate=cand,
            base_loss=2.5,
            probe_loss=2.5 - cand.params_b * 0.3,
            delta=cand.params_b * 0.3,
            wall_clock_seconds=cand.params_b * 60.0,
            estimated_cost_usd=cand.params_b * 0.05,
        )

    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "x", "completion": "y"}\n')

    report = run_tunability(
        candidates=candidates,
        dataset_path=str(dataset.relative_to(tmp_path)) if False else str(dataset),
        probe_steps=50,
        holdout_size=16,
        probe_fn=fake_probe,
    )
    assert len(report.results) == 2
    assert len(report.frontier) >= 1
    assert report.probe_steps == 50


def test_run_tunability_rejects_empty_candidates(tmp_path):
    from soup_cli.utils.tunability import run_tunability

    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    with pytest.raises(ValueError, match="candidates"):
        run_tunability(
            candidates=(),
            dataset_path=str(dataset),
            probe_steps=50,
            holdout_size=16,
        )


def test_run_tunability_rejects_invalid_probe_steps(tmp_path):
    from soup_cli.utils.tunability import CandidateBase, run_tunability

    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n")
    cands = (CandidateBase(name="a", repo_id="x/y", params_b=1.0, license_id="apache-2.0"),)
    with pytest.raises(ValueError):
        run_tunability(
            candidates=cands,
            dataset_path=str(dataset),
            probe_steps=9,  # below min
            holdout_size=16,
        )


# ---------------------------------------------------------------------------
# write_report / load_report
# ---------------------------------------------------------------------------


def test_write_report_atomic_roundtrip(tmp_path, monkeypatch):
    from soup_cli.utils.tunability import (
        CandidateBase,
        TunabilityReport,
        TunabilityResult,
        load_report,
        write_report,
    )

    monkeypatch.chdir(tmp_path)
    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    r = TunabilityResult(
        candidate=cand, base_loss=2.5, probe_loss=2.0, delta=0.5,
        wall_clock_seconds=60.0, estimated_cost_usd=0.05,
    )
    report = TunabilityReport(results=(r,), frontier=(r,), probe_steps=100, holdout_size=64)
    out = tmp_path / "tunability.json"
    write_report(report, str(out))

    loaded = load_report(str(out))
    assert loaded.probe_steps == report.probe_steps
    assert len(loaded.results) == 1
    assert loaded.results[0].candidate.name == "x"


def test_write_report_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.tunability import (
        CandidateBase,
        TunabilityReport,
        TunabilityResult,
        write_report,
    )

    monkeypatch.chdir(tmp_path)
    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    r = TunabilityResult(
        candidate=cand, base_loss=2.5, probe_loss=2.0, delta=0.5,
        wall_clock_seconds=60.0, estimated_cost_usd=0.05,
    )
    report = TunabilityReport(results=(r,), frontier=(r,), probe_steps=100, holdout_size=64)
    outside = tmp_path.parent / "evil.json"
    with pytest.raises(ValueError, match="cwd"):
        write_report(report, str(outside))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_write_report_symlink_rejected(tmp_path, monkeypatch):
    from soup_cli.utils.tunability import (
        CandidateBase,
        TunabilityReport,
        TunabilityResult,
        write_report,
    )

    monkeypatch.chdir(tmp_path)
    cand = CandidateBase(name="x", repo_id="x/y", params_b=1.0, license_id="apache-2.0")
    r = TunabilityResult(
        candidate=cand, base_loss=2.5, probe_loss=2.0, delta=0.5,
        wall_clock_seconds=60.0, estimated_cost_usd=0.05,
    )
    report = TunabilityReport(results=(r,), frontier=(r,), probe_steps=100, holdout_size=64)

    target = tmp_path / "real.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="symlink"):
        write_report(report, str(link))


def test_write_report_non_report_rejected(tmp_path):
    from soup_cli.utils.tunability import write_report

    with pytest.raises(TypeError):
        write_report("not a report", str(tmp_path / "out.json"))  # type: ignore[arg-type]


def test_load_report_missing_file(tmp_path, monkeypatch):
    from soup_cli.utils.tunability import load_report

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_report(str(tmp_path / "nope.json"))


def test_load_report_invalid_json(tmp_path):
    from soup_cli.utils.tunability import load_report

    p = tmp_path / "bad.json"
    p.write_text("not json")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_report(str(p))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_tunability_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["tunability", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "tunability" in result.output.lower()


def test_cli_tunability_list_default_candidates():
    from soup_cli.cli import app

    result = runner.invoke(app, ["tunability", "--list"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    # Should list at least one default candidate name.
    assert "qwen" in result.output.lower() or "llama" in result.output.lower() or \
        "phi" in result.output.lower() or "gemma" in result.output.lower() or \
        "smol" in result.output.lower()


def test_cli_tunability_requires_dataset(tmp_path, monkeypatch):
    """Without --list and without --dataset, exits with usage error."""
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["tunability"])
    assert result.exit_code != 0


def test_cli_tunability_outside_cwd(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "data.jsonl"
    outside.write_text("{}\n")
    result = runner.invoke(app, ["tunability", "--dataset", str(outside)])
    assert result.exit_code != 0


def test_cli_tunability_plan_only(tmp_path, monkeypatch):
    """--plan-only enumerates candidates without running probes."""
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt": "x"}\n')
    result = runner.invoke(app, ["tunability", "--dataset", str(dataset), "--plan-only"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


# ---------------------------------------------------------------------------
# Source-wiring regression guards
# ---------------------------------------------------------------------------


def test_cli_registers_tunability():
    """cli.py registers the tunability command."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "cli.py"
    text = src.read_text(encoding="utf-8")
    assert "tunability" in text


def test_version_bumped_to_0640():
    import soup_cli

    # Floor check so future minor releases don't regress this test (matches
    # v0.51.0 / v0.54.0 / v0.57.0 / v0.60.0 floor-check idiom). Exact-equality
    # at "0.64.0" broke on the v0.65.0 bump — the test name preserves the
    # intent (≥0.64.0 means v0.64.0 shipped).
    parts = tuple(int(p) for p in soup_cli.__version__.split(".")[:3])
    assert parts >= (0, 64, 0), soup_cli.__version__


def test_no_top_level_heavy_imports():
    """tunability module should not import torch/transformers/peft at top-level."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "soup_cli" / "utils" / "tunability.py"
    text = src.read_text(encoding="utf-8")
    # Heavy deps must be lazy-imported inside functions
    for bad in ["^import torch", "^from torch", "^import transformers", "^from transformers"]:
        # Strict line-start match
        import re
        assert not re.search(bad, text, re.MULTILINE), f"top-level {bad} found"
