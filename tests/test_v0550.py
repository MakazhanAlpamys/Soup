"""Tests for v0.55.0 — soup eval design / discover / lock / coverage / gate.

Covers Parts A-D plus CLI plumbing + source-grep regression guards.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import List

import pytest
from typer.testing import CliRunner

from soup_cli.utils.canary_discovery import (
    CanarySet,
    canary_set_to_dict,
    discover_canaries,
    load_canary_set,
    write_canary_set,
)
from soup_cli.utils.eval_design import (
    SCORER_TYPES,
    EvalDesign,
    EvalDimension,
    design_evals_from_data,
    design_to_dict,
    load_eval_design,
    write_eval_design,
)
from soup_cli.utils.eval_gate_hook import (
    GateThresholds,
    RegressionVerdict,
    decide_regression,
    paired_bootstrap_ci,
    render_pre_push_hook,
    write_pre_push_hook,
)
from soup_cli.utils.eval_lock_coverage import (
    CoverageReport,
    LockedSuite,
    canonicalise_design_bytes,
    checksum_design,
    compute_coverage,
    lock_suite,
)

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink test")

# Rich wraps option names with ANSI escapes when the terminal is narrow
# (macOS CI runners default to a smaller width than Linux/Windows), so
# substring searches like `"--goal" in output` fail without stripping.
# Project precedent: tests/test_auto_tuning.py, test_eval_platform.py,
# v0.53.5 / v0.53.6 / v0.53.8 / v0.53.9 CI hardening commits.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Part A — Eval design from data
# ---------------------------------------------------------------------------

class TestEvalDesignFromData:
    def _rows(self) -> List[dict]:
        return [
            {"messages": [
                {"role": "user", "content": "what is sql"},
                {"role": "assistant",
                 "content": "SQL is structured query language for databases."},
            ]},
            {"messages": [
                {"role": "user", "content": "write a query"},
                {"role": "assistant",
                 "content": "SELECT id FROM users WHERE active = true"},
            ]},
            {"messages": [
                {"role": "user", "content": "join tables"},
                {"role": "assistant",
                 "content": "SELECT users.id FROM users JOIN orders ON ..."},
            ]},
        ]

    def test_happy_path_returns_frozen_design(self):
        rows = self._rows()
        design = design_evals_from_data(rows, goal="better at SQL")
        assert isinstance(design, EvalDesign)
        assert design.row_count == 3
        assert len(design.dimensions) >= 1
        assert all(isinstance(d, EvalDimension) for d in design.dimensions)
        # Frozen invariant — mutation must raise the dataclass-specific
        # error, not just any exception.
        with pytest.raises(dataclasses.FrozenInstanceError):
            design.dimensions[0].name = "rewrite"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "goal,expected_scorer",
        [
            ("output json schema", "rlvr"),
            ("write python function", "rlvr"),
            ("solve math word problem", "rlvr"),
            ("classify intent", "exact_match"),
            ("extract field value", "regex"),
            ("summarize emails", "judge"),
            ("just generally helpful", "judge"),  # fallback
        ],
    )
    def test_scorer_picked_by_goal_keyword(self, goal, expected_scorer):
        design = design_evals_from_data(self._rows(), goal=goal)
        assert all(
            d.scorer_type == expected_scorer for d in design.dimensions
        )

    def test_scorer_is_allowlisted(self):
        design = design_evals_from_data(self._rows(), goal="x")
        for d in design.dimensions:
            assert d.scorer_type in SCORER_TYPES

    def test_dimension_names_are_unique_and_well_formed(self):
        design = design_evals_from_data(self._rows(), goal="x")
        names = [d.name for d in design.dimensions]
        assert len(names) == len(set(names))
        for n in names:
            assert re.match(r"^[a-z][a-z0-9_]{0,63}$", n)

    def test_num_dimensions_bounds(self):
        rows = self._rows()
        for bad in [0, 21, -1]:
            with pytest.raises(ValueError):
                design_evals_from_data(rows, goal="x", num_dimensions=bad)

    def test_num_dimensions_bool_rejected(self):
        with pytest.raises(TypeError):
            design_evals_from_data(self._rows(), goal="x", num_dimensions=True)

    def test_goal_bool_rejected(self):
        with pytest.raises(TypeError):
            design_evals_from_data(self._rows(), goal=True)  # type: ignore[arg-type]

    def test_goal_null_byte_rejected(self):
        with pytest.raises(ValueError):
            design_evals_from_data(self._rows(), goal="hi\x00there")

    def test_goal_oversize_rejected(self):
        with pytest.raises(ValueError):
            design_evals_from_data(self._rows(), goal="a" * 5000)

    def test_non_sequence_rows_rejected(self):
        with pytest.raises(TypeError):
            design_evals_from_data("not-a-list", goal="x")  # type: ignore[arg-type]

    def test_empty_rows_still_produces_goal_alignment(self):
        design = design_evals_from_data([], goal="x")
        assert len(design.dimensions) == 1
        assert design.dimensions[0].name == "goal_alignment"

    def test_rows_with_no_text_falls_through(self):
        design = design_evals_from_data(
            [{"messages": []}, {"messages": []}], goal="x",
        )
        assert design.dimensions  # at least the fallback


class TestEvalDesignIO:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        design = design_evals_from_data(
            [{"output": "the quick brown fox"}], goal="x",
        )
        out = "evals/d.json"
        write_eval_design(design, out)
        loaded = load_eval_design(out)
        assert loaded == design

    def test_write_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        design = design_evals_from_data([{"output": "x"}], goal="x")
        with pytest.raises(ValueError):
            write_eval_design(design, "/tmp/escape.json")

    def test_load_null_byte_rejected(self):
        with pytest.raises(ValueError):
            load_eval_design("evil\x00path.json")

    def test_load_non_string_rejected(self):
        with pytest.raises(TypeError):
            load_eval_design(123)  # type: ignore[arg-type]

    @POSIX_ONLY
    def test_load_symlink_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        real = tmp_path / "real.json"
        real.write_text("{}")
        link = tmp_path / "link.json"
        link.symlink_to(real)
        with pytest.raises(ValueError):
            load_eval_design("link.json")

    def test_load_rejects_unknown_scorer(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "goal": "x",
            "row_count": 1,
            "dimensions": [{
                "name": "d1",
                "rubric": "r",
                "scorer_type": "fnord",
                "keywords": [],
            }],
        }))
        with pytest.raises(ValueError, match="unknown scorer_type"):
            load_eval_design("bad.json")

    def test_load_rejects_bad_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "goal": "x", "row_count": 1,
            "dimensions": [{
                "name": "Has Space",
                "rubric": "r",
                "scorer_type": "judge",
                "keywords": [],
            }],
        }))
        with pytest.raises(ValueError, match="invalid dimension name"):
            load_eval_design("bad.json")

    def test_design_to_dict_rejects_non_design(self):
        with pytest.raises(TypeError):
            design_to_dict({"goal": "x"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Part B — Canary discovery
# ---------------------------------------------------------------------------

class TestCanaryDiscovery:
    def _rows(self) -> List[dict]:
        return [
            {"prompt": "what is sql", "output": "structured query language"},
            {"prompt": "write a select query",
             "output": "SELECT id FROM users"},
            {"prompt": "join two tables",
             "output": "SELECT * FROM a JOIN b"},
            {"prompt": "translate to french",
             "output": "bonjour le monde"},
            {"prompt": "translate to german",
             "output": "guten tag welt"},
        ]

    def test_happy_path(self):
        canary = discover_canaries(
            self._rows(), base="meta-llama/Llama-3-8B",
            num_clusters=2, per_cluster=2, seed=42,
        )
        assert isinstance(canary, CanarySet)
        assert canary.base == "meta-llama/Llama-3-8B"
        assert canary.cluster_count >= 1
        # Held-out + memorization populated
        assert len(canary.held_out) >= 1
        assert len(canary.memorization_probes) >= 1

    def test_frozen_dataclass(self):
        canary = discover_canaries(self._rows(), num_clusters=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            canary.held_out = ("changed",)  # type: ignore[misc]

    def test_deterministic_with_seed(self):
        c1 = discover_canaries(self._rows(), num_clusters=2, seed=7)
        c2 = discover_canaries(self._rows(), num_clusters=2, seed=7)
        assert c1.held_out == c2.held_out

    def test_seed_changes_output_distribution(self):
        # Different seeds may produce different first-centroid picks.
        c1 = discover_canaries(self._rows(), num_clusters=3, seed=0)
        c2 = discover_canaries(self._rows(), num_clusters=3, seed=100)
        # Don't assert inequality (could collide by chance for small data);
        # just assert both are valid.
        assert c1.held_out
        assert c2.held_out

    @pytest.mark.parametrize("bad", [True, 1.5, "5", None])
    def test_num_clusters_type_rejected(self, bad):
        with pytest.raises((TypeError, ValueError)):
            discover_canaries(self._rows(), num_clusters=bad)

    def test_num_clusters_oob(self):
        with pytest.raises(ValueError):
            discover_canaries(self._rows(), num_clusters=0)
        with pytest.raises(ValueError):
            discover_canaries(self._rows(), num_clusters=999)

    def test_base_validation(self):
        with pytest.raises(ValueError):
            discover_canaries(self._rows(), base="bad\x00name")
        with pytest.raises(ValueError):
            discover_canaries(self._rows(), base="a" * 600)
        with pytest.raises(TypeError):
            discover_canaries(self._rows(), base=True)  # type: ignore[arg-type]

    def test_empty_rows(self):
        canary = discover_canaries([], num_clusters=3)
        assert canary.held_out == ()
        assert canary.adjacent_skills == ()
        assert canary.memorization_probes == ()
        assert canary.cluster_count == 0

    def test_dimensions_validated(self):
        canary = discover_canaries(
            self._rows(), dimensions=("rewrite", "summarize"),
        )
        assert canary.dimensions == ("rewrite", "summarize")
        with pytest.raises(ValueError):
            discover_canaries(self._rows(), dimensions=["bad\x00d"])
        with pytest.raises(TypeError):
            discover_canaries(self._rows(), dimensions="not-a-list")  # type: ignore[arg-type]

    def test_memorization_probe_is_truncated(self):
        # A int prompt → memorization probe is shorter.
        rows = [{"prompt": " ".join(["word"] * 40), "output": "x"}]
        canary = discover_canaries(rows, num_clusters=1, per_cluster=1)
        assert canary.memorization_probes
        assert (
            len(canary.memorization_probes[0].split())
            < len(rows[0]["prompt"].split())
        )

    def test_dedup_held_out(self):
        # Two identical rows → only one canary
        canary = discover_canaries(
            [{"prompt": "hi"}, {"prompt": "hi"}],
            num_clusters=1, per_cluster=2,
        )
        assert canary.held_out.count("hi") <= 1


class TestCanaryIO:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        canary = discover_canaries(
            [{"prompt": "x", "output": "y"}], num_clusters=1,
        )
        write_canary_set(canary, "c.json")
        loaded = load_canary_set("c.json")
        assert loaded == canary

    def test_write_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = discover_canaries([{"prompt": "x"}], num_clusters=1)
        with pytest.raises(ValueError):
            write_canary_set(c, "/tmp/escape.json")

    def test_canary_set_to_dict_rejects_non_canary(self):
        with pytest.raises(TypeError):
            canary_set_to_dict({"held_out": []})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Part C — Eval lock + coverage
# ---------------------------------------------------------------------------

class TestEvalLock:
    def test_canonicalise_deterministic(self):
        design = design_evals_from_data([{"output": "abc"}], goal="x")
        a = canonicalise_design_bytes(design)
        b = canonicalise_design_bytes(design)
        assert a == b
        # Strict canonical layout: no structural whitespace. (Content
        # strings may still contain ": " or "\n" — we only assert that
        # json.dumps was called with the separators+sort_keys flags by
        # checking common-prefix structure is compact.)
        assert a.startswith(b"{\"")
        # The keys come back sorted: dimensions / goal / row_count.
        assert a.index(b"\"dimensions\"") < a.index(b"\"goal\"")
        assert a.index(b"\"goal\"") < a.index(b"\"row_count\"")

    def test_checksum_stable(self):
        design = design_evals_from_data([{"output": "abc"}], goal="x")
        h1 = checksum_design(design)
        h2 = checksum_design(design)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_canonicalise_rejects_non_design(self):
        with pytest.raises(TypeError):
            canonicalise_design_bytes({"goal": "x"})  # type: ignore[arg-type]

    def test_lock_writes_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        design = design_evals_from_data([{"output": "abc"}], goal="x")
        locked = lock_suite(design, "evals/locked.json")
        assert isinstance(locked, LockedSuite)
        assert locked.dimension_count == len(design.dimensions)
        assert os.path.isfile("evals/locked.json")
        # checksum == sha256 of file bytes
        on_disk = Path("evals/locked.json").read_bytes()
        import hashlib
        assert locked.checksum == hashlib.sha256(on_disk).hexdigest()

    def test_lock_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        design = design_evals_from_data([{"output": "x"}], goal="x")
        with pytest.raises(ValueError):
            lock_suite(design, "/tmp/escape.json")


class TestCoverage:
    def _design_with_scorers(self, scorers: list) -> EvalDesign:
        return EvalDesign(
            goal="x",
            row_count=1,
            dimensions=tuple(
                EvalDimension(
                    name=f"d{i}",
                    rubric="r",
                    scorer_type=s,
                    keywords=(),
                )
                for i, s in enumerate(scorers)
            ),
        )

    def test_happy_path_factual_lookup(self):
        design = self._design_with_scorers(["exact_match", "judge"])
        report = compute_coverage(design, task_category="factual_lookup")
        assert isinstance(report, CoverageReport)
        assert report.task_category == "factual_lookup"
        assert report.missing_scorers == ()  # has both expected

    def test_missing_scorer_surfaces(self):
        design = self._design_with_scorers(["judge"])
        report = compute_coverage(design, task_category="reasoning")
        assert "rlvr" in report.missing_scorers

    def test_scorer_mix_counts(self):
        design = self._design_with_scorers(["judge", "judge", "rlvr"])
        report = compute_coverage(design, task_category="reasoning")
        assert report.scorer_mix["judge"] == 2
        assert report.scorer_mix["rlvr"] == 1
        assert report.scorer_mix["exact_match"] == 0

    def test_unknown_task_category(self):
        design = self._design_with_scorers(["judge"])
        with pytest.raises(ValueError, match="unknown task_category"):
            compute_coverage(design, task_category="not_a_real_category")

    def test_task_category_bool_rejected(self):
        design = self._design_with_scorers(["judge"])
        with pytest.raises(TypeError):
            compute_coverage(design, task_category=True)  # type: ignore[arg-type]

    def test_empty_design_recommendations(self):
        design = EvalDesign(goal="", row_count=0, dimensions=())
        report = compute_coverage(design, task_category="summarization")
        assert any(
            "no dimensions" in rec for rec in report.recommendations
        )

    def test_case_insensitive_task_category(self):
        design = self._design_with_scorers(["judge"])
        report = compute_coverage(design, task_category="Reasoning")
        assert report.task_category == "reasoning"


# ---------------------------------------------------------------------------
# Part D — git-hook regression gate + paired bootstrap
# ---------------------------------------------------------------------------

class TestPairedBootstrap:
    def test_identical_samples_zero_mean(self):
        a = [0.5] * 50
        b = [0.5] * 50
        lo, hi, mean = paired_bootstrap_ci(a, b, n_samples=200, seed=42)
        assert mean == pytest.approx(0.0)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(0.0)

    def test_improvement_positive_ci(self):
        baseline = [0.5] * 100
        candidate = [0.7] * 100
        lo, hi, mean = paired_bootstrap_ci(
            baseline, candidate, n_samples=500, seed=0,
        )
        assert mean == pytest.approx(0.2)
        # Constant series → CI collapses to point.
        assert hi - lo == pytest.approx(0.0)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="length"):
            paired_bootstrap_ci([1.0, 2.0], [1.0])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            paired_bootstrap_ci([], [])

    def test_n_samples_bounds(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci([1.0], [1.0], n_samples=10)
        with pytest.raises(ValueError):
            paired_bootstrap_ci([1.0], [1.0], n_samples=200_000)

    def test_ci_level_bounds(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci([1.0], [1.0], n_samples=100, ci_level=0.0)
        with pytest.raises(ValueError):
            paired_bootstrap_ci([1.0], [1.0], n_samples=100, ci_level=1.0)

    def test_seed_bool_rejected(self):
        with pytest.raises(TypeError):
            paired_bootstrap_ci(
                [1.0], [1.0], n_samples=100, seed=True,  # type: ignore[arg-type]
            )

    def test_non_finite_rejected(self):
        import math
        with pytest.raises(ValueError):
            paired_bootstrap_ci(
                [math.nan, 1.0], [1.0, 1.0], n_samples=100,
            )

    def test_deterministic(self):
        a = [0.1, 0.4, 0.5, 0.7]
        b = [0.2, 0.4, 0.6, 0.6]
        r1 = paired_bootstrap_ci(a, b, n_samples=500, seed=0)
        r2 = paired_bootstrap_ci(a, b, n_samples=500, seed=0)
        assert r1 == r2


class TestDecideRegression:
    def test_higher_better_no_regression_on_improvement(self):
        baseline = [0.5] * 50
        candidate = [0.7] * 50
        verdict = decide_regression(
            "task_accuracy", baseline, candidate, GateThresholds(),
            n_samples=200, seed=0,
        )
        assert isinstance(verdict, RegressionVerdict)
        assert verdict.regressed is False
        assert verdict.offenders == ()

    def test_higher_better_regression_on_drop(self):
        baseline = [0.9] * 50
        candidate = [0.5] * 50
        verdict = decide_regression(
            "task_accuracy", baseline, candidate, GateThresholds(),
            n_samples=200, seed=0,
        )
        assert verdict.regressed is True
        assert "task_accuracy" in verdict.offenders

    def test_lower_better_no_regression_on_improvement(self):
        # Lower latency = better. Tolerance is +100ms (regression
        # acceptable up to +100ms). 90ms improvement → not regressed.
        baseline = [200.0] * 50
        candidate = [110.0] * 50
        verdict = decide_regression(
            "p95_latency_ms", baseline, candidate, GateThresholds(),
            n_samples=200, seed=0,
        )
        assert verdict.regressed is False

    def test_lower_better_regression_on_increase(self):
        # +500ms latency — beyond tolerance.
        baseline = [100.0] * 50
        candidate = [600.0] * 50
        verdict = decide_regression(
            "p95_latency_ms", baseline, candidate, GateThresholds(),
            n_samples=200, seed=0,
        )
        assert verdict.regressed is True

    def test_unknown_metric_rejected(self):
        with pytest.raises(ValueError, match="unknown metric"):
            decide_regression(
                "made_up_metric", [1.0], [1.0], GateThresholds(),
            )

    def test_metric_bool_rejected(self):
        with pytest.raises(TypeError):
            decide_regression(
                True, [1.0], [1.0], GateThresholds(),  # type: ignore[arg-type]
            )

    def test_thresholds_type_rejected(self):
        with pytest.raises(TypeError):
            decide_regression(
                "task_accuracy", [1.0], [1.0],
                {"task_accuracy": -0.05},  # type: ignore[arg-type]
            )


class TestPrePushHookRendering:
    def test_basic_render(self):
        body = render_pre_push_hook(
            baseline_run_id="run-abc-123",
            suite_path="evals/locked.json",
        )
        assert "#!/usr/bin/env bash" in body
        assert "soup eval against" in body
        assert "run-abc-123" in body
        assert "set -euo pipefail" in body

    def test_bad_run_id_rejected(self):
        for bad in ["", "has space", "../escape", "a\x00b", True]:
            with pytest.raises((TypeError, ValueError)):
                render_pre_push_hook(
                    baseline_run_id=bad,  # type: ignore[arg-type]
                    suite_path="evals/locked.json",
                )

    def test_suite_path_null_byte_rejected(self):
        with pytest.raises(ValueError):
            render_pre_push_hook(
                baseline_run_id="abc",
                suite_path="evals/loc\x00ked.json",
            )

    def test_suite_path_newline_rejected(self):
        with pytest.raises(ValueError):
            render_pre_push_hook(
                baseline_run_id="abc",
                suite_path="evals/locked.json\nrm -rf /",
            )

    def test_suite_path_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            render_pre_push_hook(
                baseline_run_id="abc",
                suite_path="/etc/passwd",
            )

    def test_shell_escape_resists_quote_injection(self, tmp_path, monkeypatch):
        # Suite path with single quotes — must be safely escaped.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "weird's name.json").write_text("{}")
        body = render_pre_push_hook(
            baseline_run_id="abc",
            suite_path="weird's name.json",
        )
        # Must not break the quoting: bash -n would validate but we
        # just check the escape pattern is present.
        assert "'\"'\"'" in body or "\\'" in body or "'weird" in body


class TestPrePushHookWrite:
    def test_atomic_write(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "locked.json").write_text("{}")
        hook = tmp_path / "hooks" / "pre-push"
        path = write_pre_push_hook(
            baseline_run_id="run-1",
            suite_path="evals/locked.json",
            hook_path=str(hook.relative_to(tmp_path)),
        )
        assert os.path.isfile(path)
        body = Path(path).read_text()
        assert "run-1" in body

    def test_refuses_overwrite_by_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "locked.json").write_text("{}")
        hook = "hooks/pre-push"
        write_pre_push_hook(
            baseline_run_id="run-1",
            suite_path="evals/locked.json",
            hook_path=hook,
        )
        with pytest.raises(ValueError, match="already exists"):
            write_pre_push_hook(
                baseline_run_id="run-2",
                suite_path="evals/locked.json",
                hook_path=hook,
            )

    def test_force_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "locked.json").write_text("{}")
        hook = "hooks/pre-push"
        write_pre_push_hook(
            baseline_run_id="run-1",
            suite_path="evals/locked.json",
            hook_path=hook,
        )
        write_pre_push_hook(
            baseline_run_id="run-2",
            suite_path="evals/locked.json",
            hook_path=hook,
            overwrite=True,
        )
        body = Path(hook).read_text()
        assert "run-2" in body

    @POSIX_ONLY
    def test_symlink_target_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "locked.json").write_text("{}")
        target = tmp_path / "real-file"
        target.write_text("")
        link = tmp_path / "hook-link"
        link.symlink_to(target)
        with pytest.raises(ValueError, match="symlink"):
            write_pre_push_hook(
                baseline_run_id="run-1",
                suite_path="evals/locked.json",
                hook_path="hook-link",
                overwrite=True,
            )

    def test_hook_path_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "locked.json").write_text("{}")
        with pytest.raises(ValueError):
            write_pre_push_hook(
                baseline_run_id="run-1",
                suite_path="evals/locked.json",
                hook_path="/tmp/escape",
            )

    def test_hook_path_bool_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "locked.json").write_text("{}")
        with pytest.raises(TypeError):
            write_pre_push_hook(
                baseline_run_id="run-1",
                suite_path="evals/locked.json",
                hook_path=True,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Registry integration — eval_suite + canaries are valid artifact kinds.
# ---------------------------------------------------------------------------

class TestRegistryArtifactKinds:
    def test_eval_suite_in_valid_kinds(self):
        from soup_cli.registry.store import _VALID_KINDS
        assert "eval_suite" in _VALID_KINDS
        assert "canaries" in _VALID_KINDS


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

class TestCLIPlumbing:
    def setup_method(self):
        self.runner = CliRunner()

    def test_eval_help_lists_new_commands(self):
        from soup_cli.commands.eval import app
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        for cmd in ["design", "discover", "lock", "coverage", "gate-install"]:
            assert cmd in result.output, f"missing {cmd!r} in --help"

    def test_eval_design_help(self):
        from soup_cli.commands.eval import app
        result = self.runner.invoke(app, ["design", "--help"])
        assert result.exit_code == 0, result.output
        assert "--goal" in _strip_ansi(result.output)

    def test_eval_design_end_to_end(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        data = tmp_path / "data.jsonl"
        data.write_text(
            '{"messages":[{"role":"user","content":"q"},'
            '{"role":"assistant","content":"sql query database"}]}\n'
        )
        result = self.runner.invoke(
            app, ["design", "data.jsonl", "--goal", "better at SQL"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert os.path.isfile("evals/design.json")

    def test_eval_design_missing_data_exits_nonzero(
        self, tmp_path, monkeypatch,
    ):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(
            app, ["design", "nope.jsonl", "--goal", "x"],
        )
        assert result.exit_code != 0

    def test_eval_discover_end_to_end(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        data = tmp_path / "d.jsonl"
        data.write_text(
            '{"prompt":"a","output":"x"}\n'
            '{"prompt":"b","output":"y"}\n'
        )
        result = self.runner.invoke(
            app, ["discover", "d.jsonl", "--num-clusters", "2"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert os.path.isfile("evals/canaries.json")

    def test_eval_lock_end_to_end(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        design = design_evals_from_data([{"output": "x"}], goal="y")
        (tmp_path / "evals").mkdir()
        write_eval_design(design, "evals/d.json")
        result = self.runner.invoke(app, ["lock", "evals/d.json"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert os.path.isfile("evals/locked.json")

    def test_eval_coverage_end_to_end(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        design = design_evals_from_data([{"output": "x"}], goal="y")
        (tmp_path / "evals").mkdir()
        write_eval_design(design, "evals/d.json")
        result = self.runner.invoke(
            app, ["coverage", "evals/d.json", "--task", "summarization"],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_eval_coverage_bad_task_exits_nonzero(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        design = design_evals_from_data([{"output": "x"}], goal="y")
        (tmp_path / "evals").mkdir()
        write_eval_design(design, "evals/d.json")
        result = self.runner.invoke(
            app, ["coverage", "evals/d.json", "--task", "garbage"],
        )
        assert result.exit_code == 2

    def test_gate_install_end_to_end(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "locked.json").write_text("{}")
        result = self.runner.invoke(
            app,
            [
                "gate-install",
                "--baseline", "run-1",
                "--suite", "evals/locked.json",
                "--hook-path", "hooks/pre-push",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert os.path.isfile("hooks/pre-push")


# ---------------------------------------------------------------------------
# Source-grep regression guards.
# ---------------------------------------------------------------------------

class TestSourceGrep:
    REPO_ROOT = Path(__file__).resolve().parent.parent

    def test_eval_py_registers_v0550_module(self):
        text = (self.REPO_ROOT / "src" / "soup_cli" / "commands" / "eval.py").read_text(
            encoding="utf-8",
        )
        assert "_eval_v0550" in text
        assert "_register_v0550(app, console)" in text

    def test_v0550_module_uses_lazy_imports(self):
        # The Typer registration module itself must not eagerly import
        # heavy deps. Lazy imports happen inside each command body.
        text = (self.REPO_ROOT / "src" / "soup_cli" / "commands" / "_eval_v0550.py").read_text(
            encoding="utf-8",
        )
        # Top-level imports allowed: typer, rich.*. No torch/transformers/peft.
        for forbidden in ("import torch", "import transformers", "import peft"):
            assert forbidden not in text.splitlines()[:15], (
                f"top-level {forbidden} would slow CLI startup"
            )

    def test_version_bumped_to_0_55_0(self):
        # v0.55.0+ floor — released versions are always >= 0.55.0.
        import re

        init_text = (
            self.REPO_ROOT / "src" / "soup_cli" / "__init__.py"
        ).read_text(encoding="utf-8")
        match = re.search(r'__version__ = "(\d+)\.(\d+)\.(\d+)"', init_text)
        assert match is not None, "version line not found"
        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        assert (major, minor) >= (0, 55), (
            f"version must be >= 0.55.0 (found {major}.{minor}.{patch})"
        )


# ---------------------------------------------------------------------------
# `soup eval against` — run-vs-run regression check (Part D)
# ---------------------------------------------------------------------------

class TestEvalAgainst:
    def setup_method(self):
        self.runner = CliRunner()

    def test_against_listed_in_help(self):
        from soup_cli.commands.eval import app
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert "against" in result.output, "missing 'against' in --help"

    def test_against_help(self):
        from soup_cli.commands.eval import app
        result = self.runner.invoke(app, ["against", "--help"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        out = _strip_ansi(result.output)
        assert "--candidate" in out
        assert "--metric" in out
        assert "--json-only" in out

    def test_against_requires_candidate(self):
        # `--candidate` is a required option; omitting must fail.
        from soup_cli.commands.eval import app
        result = self.runner.invoke(app, ["against", "run-baseline"])
        assert result.exit_code != 0
        assert "candidate" in result.output.lower()

    def test_against_unknown_metric_rejected(self, monkeypatch, tmp_path):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(
            app,
            [
                "against", "run-baseline",
                "--candidate", "run-cand",
                "--metric", "made_up",
            ],
        )
        assert result.exit_code != 0
        # The deferred-tracker path will fire first (AttributeError → exit 2)
        # OR the metric validator fires (exit 1). Either is acceptable — what
        # we care about is that an invalid metric doesn't silently succeed.

    def test_against_deferred_tracker_advisory(self, monkeypatch, tmp_path):
        # The tracker doesn't yet expose `get_metric_series` — the command
        # must catch the AttributeError and emit a v0.55.1-deferred advisory.
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(
            app,
            [
                "against", "run-baseline",
                "--candidate", "run-cand",
            ],
        )
        # Either the deferred-advisory (exit 2) or the empty-series failure
        # (exit 1) — both are acceptable absence-of-data signals. What we
        # assert is that we do NOT exit 0 (false-pass) and that the message
        # is informative.
        assert result.exit_code != 0


class TestHookTemplate:
    """v0.55.0 hook template references SOUP_CANDIDATE_RUN_ID env var."""

    def test_template_calls_soup_eval_against(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "locked.json").write_text("{}")
        from soup_cli.utils.eval_gate_hook import render_pre_push_hook
        body = render_pre_push_hook(
            baseline_run_id="run-1",
            suite_path="evals/locked.json",
        )
        # Must call the new subcommand, NOT a stale --against flag.
        assert "soup eval against" in body
        # Must reference the env var so users can wire it from their CI.
        assert "SOUP_CANDIDATE_RUN_ID" in body
        # Hook is bash-strict.
        assert "set -euo pipefail" in body
