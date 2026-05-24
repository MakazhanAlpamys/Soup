"""v0.67.0 Part A — Evolutionary merge (CMA-ES) over LoRA adapter weights.

Tests for ``soup_cli/utils/cmaes_merge.py``:

- Closed allowlist + frozen dataclasses + parse_budget reuse from blame.py
- ``run_cmaes_merge`` orchestrator with operator-supplied ``eval_fn`` injection
- ``soup adapters merge --strategy cmaes --eval <suite> --budget 1h`` CLI plumbing
- Validation matrix (bool/null-byte/non-finite/oversize rejection)
- Budget bounds reused from v0.57 ``blame.parse_budget`` (60s..24h)
- Source-grep regression guard for the new strategy alias in adapter_merge
"""

from __future__ import annotations

import dataclasses
import math

import pytest

# -----------------------------------------------------------------------------
# Module surface — imports + constants
# -----------------------------------------------------------------------------


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import cmaes_merge

        assert hasattr(cmaes_merge, "CmaesPlan")
        assert hasattr(cmaes_merge, "CmaesResult")
        assert hasattr(cmaes_merge, "run_cmaes_merge")
        assert hasattr(cmaes_merge, "validate_population_size")
        assert hasattr(cmaes_merge, "validate_generations")

    def test_constants_immutable(self) -> None:
        from soup_cli.utils import cmaes_merge

        assert cmaes_merge.MIN_POPULATION >= 2
        assert cmaes_merge.MAX_POPULATION <= 256
        assert cmaes_merge.MIN_GENERATIONS >= 1
        assert cmaes_merge.MAX_GENERATIONS <= 10000


# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------


class TestValidatePopulationSize:
    def test_happy(self) -> None:
        from soup_cli.utils.cmaes_merge import validate_population_size

        assert validate_population_size(8) == 8

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.cmaes_merge import validate_population_size

        with pytest.raises(TypeError):
            validate_population_size(True)  # type: ignore[arg-type]

    def test_below_floor(self) -> None:
        from soup_cli.utils.cmaes_merge import validate_population_size

        with pytest.raises(ValueError):
            validate_population_size(1)

    def test_above_cap(self) -> None:
        from soup_cli.utils.cmaes_merge import MAX_POPULATION, validate_population_size

        with pytest.raises(ValueError):
            validate_population_size(MAX_POPULATION + 1)

    def test_non_int_rejected(self) -> None:
        from soup_cli.utils.cmaes_merge import validate_population_size

        with pytest.raises(TypeError):
            validate_population_size("8")  # type: ignore[arg-type]


class TestValidateGenerations:
    def test_happy(self) -> None:
        from soup_cli.utils.cmaes_merge import validate_generations

        assert validate_generations(20) == 20

    def test_bool_rejected(self) -> None:
        from soup_cli.utils.cmaes_merge import validate_generations

        with pytest.raises(TypeError):
            validate_generations(False)  # type: ignore[arg-type]

    def test_below_floor(self) -> None:
        from soup_cli.utils.cmaes_merge import validate_generations

        with pytest.raises(ValueError):
            validate_generations(0)

    def test_above_cap(self) -> None:
        from soup_cli.utils.cmaes_merge import MAX_GENERATIONS, validate_generations

        with pytest.raises(ValueError):
            validate_generations(MAX_GENERATIONS + 1)


# -----------------------------------------------------------------------------
# Frozen dataclasses
# -----------------------------------------------------------------------------


class TestCmaesPlan:
    def test_construct(self) -> None:
        from soup_cli.utils.cmaes_merge import CmaesPlan

        plan = CmaesPlan(
            adapters=("a", "b"),
            eval_suite="suite.yaml",
            budget_seconds=3600,
            population_size=8,
            max_generations=20,
            seed=42,
        )
        assert plan.population_size == 8
        assert plan.eval_suite == "suite.yaml"

    def test_frozen(self) -> None:
        from soup_cli.utils.cmaes_merge import CmaesPlan

        plan = CmaesPlan(
            adapters=("a", "b"),
            eval_suite="s.yaml",
            budget_seconds=600,
            population_size=4,
            max_generations=5,
            seed=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.population_size = 99  # type: ignore[misc]


class TestCmaesResult:
    def test_construct(self) -> None:
        from soup_cli.utils.cmaes_merge import CmaesResult

        result = CmaesResult(
            best_weights=(0.5, 0.5),
            best_score=0.93,
            generations_run=10,
            evaluations=80,
            wall_clock_seconds=120.5,
            converged=True,
            history=(0.85, 0.87, 0.9, 0.93),
        )
        assert result.best_score == 0.93
        assert result.generations_run == 10
        assert result.converged is True

    def test_frozen(self) -> None:
        from soup_cli.utils.cmaes_merge import CmaesResult

        result = CmaesResult(
            best_weights=(0.5, 0.5),
            best_score=0.5,
            generations_run=1,
            evaluations=4,
            wall_clock_seconds=1.0,
            converged=False,
            history=(0.5,),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.best_score = 9.99  # type: ignore[misc]

    def test_weights_must_be_simplex(self) -> None:
        from soup_cli.utils.cmaes_merge import CmaesResult

        with pytest.raises(ValueError):
            CmaesResult(
                best_weights=(0.3, 0.3),  # sums to 0.6, not 1.0
                best_score=0.5,
                generations_run=1,
                evaluations=4,
                wall_clock_seconds=1.0,
                converged=False,
                history=(0.5,),
            )

    def test_bool_score_rejected(self) -> None:
        from soup_cli.utils.cmaes_merge import CmaesResult

        with pytest.raises(TypeError):
            CmaesResult(
                best_weights=(0.5, 0.5),
                best_score=True,  # type: ignore[arg-type]
                generations_run=1,
                evaluations=4,
                wall_clock_seconds=1.0,
                converged=False,
                history=(0.5,),
            )

    def test_non_finite_score_rejected(self) -> None:
        from soup_cli.utils.cmaes_merge import CmaesResult

        with pytest.raises(ValueError):
            CmaesResult(
                best_weights=(0.5, 0.5),
                best_score=math.nan,
                generations_run=1,
                evaluations=4,
                wall_clock_seconds=1.0,
                converged=False,
                history=(0.5,),
            )


# -----------------------------------------------------------------------------
# build_cmaes_plan
# -----------------------------------------------------------------------------


class TestBuildCmaesPlan:
    def test_happy(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.cmaes_merge import build_cmaes_plan

        monkeypatch.chdir(tmp_path)
        adapter_a = tmp_path / "a"
        adapter_a.mkdir()
        adapter_b = tmp_path / "b"
        adapter_b.mkdir()
        suite = tmp_path / "suite.yaml"
        suite.write_text("dummy: 1\n", encoding="utf-8")

        plan = build_cmaes_plan(
            adapters=[str(adapter_a), str(adapter_b)],
            eval_suite=str(suite),
            budget_spec="10m",
            population_size=6,
            max_generations=15,
            seed=42,
        )
        assert plan.population_size == 6
        assert plan.budget_seconds == 600
        assert plan.max_generations == 15

    def test_invalid_budget_string(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.cmaes_merge import build_cmaes_plan

        monkeypatch.chdir(tmp_path)
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
        suite = tmp_path / "s.yaml"
        suite.write_text("x: 1\n", encoding="utf-8")

        with pytest.raises(ValueError):
            build_cmaes_plan(
                adapters=[str(tmp_path / "a"), str(tmp_path / "b")],
                eval_suite=str(suite),
                budget_spec="not-a-budget",
                population_size=4,
                max_generations=5,
                seed=0,
            )

    def test_below_min_adapters(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.cmaes_merge import build_cmaes_plan

        monkeypatch.chdir(tmp_path)
        (tmp_path / "a").mkdir()
        suite = tmp_path / "s.yaml"
        suite.write_text("x: 1\n", encoding="utf-8")

        with pytest.raises(ValueError):
            build_cmaes_plan(
                adapters=[str(tmp_path / "a")],
                eval_suite=str(suite),
                budget_spec="60s",
                population_size=4,
                max_generations=5,
                seed=0,
            )

    def test_eval_suite_outside_cwd(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.cmaes_merge import build_cmaes_plan

        cwd = tmp_path / "work"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        (cwd / "a").mkdir()
        (cwd / "b").mkdir()
        outside = tmp_path / "outside.yaml"
        outside.write_text("x: 1\n", encoding="utf-8")

        with pytest.raises(ValueError):
            build_cmaes_plan(
                adapters=[str(cwd / "a"), str(cwd / "b")],
                eval_suite=str(outside),
                budget_spec="60s",
                population_size=4,
                max_generations=5,
                seed=0,
            )


# -----------------------------------------------------------------------------
# run_cmaes_merge orchestrator — operator-supplied eval_fn
# -----------------------------------------------------------------------------


class TestRunCmaesMerge:
    def test_returns_result(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.cmaes_merge import build_cmaes_plan, run_cmaes_merge

        monkeypatch.chdir(tmp_path)
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
        suite = tmp_path / "s.yaml"
        suite.write_text("x: 1\n", encoding="utf-8")

        plan = build_cmaes_plan(
            adapters=[str(tmp_path / "a"), str(tmp_path / "b")],
            eval_suite=str(suite),
            budget_spec="60s",
            population_size=4,
            max_generations=3,
            seed=42,
        )

        # eval_fn returns a deterministic score with maximum near weights=(0.5, 0.5)
        def eval_fn(weights):
            target = (0.5, 0.5)
            diff = sum((w - t) ** 2 for w, t in zip(weights, target))
            return 1.0 - diff

        result = run_cmaes_merge(plan, eval_fn=eval_fn)
        assert result.generations_run >= 1
        assert result.evaluations >= 1
        assert 0.0 <= result.best_score <= 1.0
        # weights normalised to simplex
        assert math.isclose(sum(result.best_weights), 1.0, abs_tol=1e-6)

    def test_eval_fn_required(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.cmaes_merge import build_cmaes_plan, run_cmaes_merge

        monkeypatch.chdir(tmp_path)
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
        suite = tmp_path / "s.yaml"
        suite.write_text("x: 1\n", encoding="utf-8")

        plan = build_cmaes_plan(
            adapters=[str(tmp_path / "a"), str(tmp_path / "b")],
            eval_suite=str(suite),
            budget_spec="60s",
            population_size=4,
            max_generations=2,
            seed=0,
        )
        with pytest.raises(TypeError):
            run_cmaes_merge(plan, eval_fn=None)  # type: ignore[arg-type]

    def test_eval_fn_exceptions_logged(self, tmp_path, monkeypatch) -> None:
        from soup_cli.utils.cmaes_merge import build_cmaes_plan, run_cmaes_merge

        monkeypatch.chdir(tmp_path)
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
        suite = tmp_path / "s.yaml"
        suite.write_text("x: 1\n", encoding="utf-8")

        plan = build_cmaes_plan(
            adapters=[str(tmp_path / "a"), str(tmp_path / "b")],
            eval_suite=str(suite),
            budget_spec="60s",
            population_size=4,
            max_generations=2,
            seed=0,
        )

        def eval_fn(weights):
            if weights[0] > 0.8:
                raise RuntimeError("simulated eval failure")
            return 0.5

        # Should not crash; failed evals get sentinel score
        result = run_cmaes_merge(plan, eval_fn=eval_fn)
        assert result.evaluations >= 1

    def test_non_plan_rejected(self) -> None:
        from soup_cli.utils.cmaes_merge import run_cmaes_merge

        with pytest.raises(TypeError):
            run_cmaes_merge("not-a-plan", eval_fn=lambda w: 0.5)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# adapter_merge integration: cmaes is a known strategy
# -----------------------------------------------------------------------------


class TestAdapterMergeIntegration:
    def test_cmaes_in_supported_strategies(self) -> None:
        from soup_cli.utils.adapter_merge import SUPPORTED_STRATEGIES

        assert "cmaes" in SUPPORTED_STRATEGIES


# -----------------------------------------------------------------------------
# CLI smoke
# -----------------------------------------------------------------------------


class TestCliSmoke:
    def test_merge_help_lists_cmaes(self) -> None:
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        runner = CliRunner()
        result = runner.invoke(app, ["merge", "--help"])
        assert result.exit_code == 0
        assert "cmaes" in result.output.lower()

    def test_merge_cmaes_requires_eval(self, tmp_path, monkeypatch) -> None:
        """cmaes without --eval should exit 2 with friendly message."""
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        # We don't bother making valid adapters; we just want to see the strategy
        # validation reject missing --eval before file-handling kicks in.
        result = runner.invoke(
            app,
            [
                "merge",
                str(tmp_path / "a"),
                str(tmp_path / "b"),
                "--strategy",
                "cmaes",
                "--output",
                str(tmp_path / "out"),
            ],
        )
        # --strategy cmaes WITHOUT --eval must fail
        assert result.exit_code == 2
        assert "eval" in result.output.lower()


# -----------------------------------------------------------------------------
# Source-grep regression guards
# -----------------------------------------------------------------------------


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (root / "soup_cli" / "utils" / "cmaes_merge.py").read_text(
            encoding="utf-8"
        )
        # Only first 30 non-comment lines (matches v0.66 review-grep policy)
        head_lines = [
            line
            for line in src.splitlines()[:50]
            if line.strip() and not line.strip().startswith("#")
        ]
        head = "\n".join(head_lines)
        for forbidden in ("import torch", "import transformers", "import peft"):
            assert forbidden not in head, f"top-level {forbidden!r} in cmaes_merge"

    def test_adapter_merge_lists_cmaes(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (root / "soup_cli" / "utils" / "adapter_merge.py").read_text(
            encoding="utf-8"
        )
        assert "cmaes" in src
