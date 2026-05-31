"""v0.67.0 review-fix follow-ups (from tdd-guide + security-review waves).

HIGH-priority gaps surfaced by the TDD review:

- Plateau-convergence path in CMA-ES
- POSIX symlink rejection at write boundaries (vector_bank.write_bank,
  cmaes_merge has no disk surface, only the load-side has it today)
- Vector-bank forward-compat: unknown JSON fields silently ignored
- Adapter PR empty-sections rendering (no None leakage)
- Bisect non-monotonic eval_fn does not crash
- Exact-boundary tests at MIN/MAX values
- validate_user_id bool rejection
- soup.lock version-drift advisory check
"""

from __future__ import annotations

import json
import os

import pytest

# -----------------------------------------------------------------------------
# Part A — Plateau convergence
# -----------------------------------------------------------------------------


class TestCmaesPlateauConvergence:
    def test_converges_on_flat_landscape(self, tmp_path, monkeypatch) -> None:
        """When eval_fn returns the same score for 3+ generations,
        ``converged=True`` and the loop short-circuits early."""
        from soup_cli.utils.cmaes_merge import build_cmaes_plan, run_cmaes_merge

        monkeypatch.chdir(tmp_path)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        suite = tmp_path / "s.yaml"
        suite.write_text("x: 1\n", encoding="utf-8")

        plan = build_cmaes_plan(
            adapters=[str(tmp_path / "a"), str(tmp_path / "b")],
            eval_suite=str(suite),
            budget_spec="60s",
            population_size=4,
            max_generations=20,
            seed=42,
        )

        # Flat constant landscape: every weight returns 0.5
        def eval_fn(weights):
            return 0.5

        result = run_cmaes_merge(plan, eval_fn=eval_fn)
        assert result.converged is True
        # Plateau detection breaks out before max_generations
        assert result.generations_run < plan.max_generations


# -----------------------------------------------------------------------------
# Part A — Exact boundary tests
# -----------------------------------------------------------------------------


class TestCmaesValidatorBoundaries:
    def test_population_min_accepted(self) -> None:
        from soup_cli.utils.cmaes_merge import MIN_POPULATION, validate_population_size

        assert validate_population_size(MIN_POPULATION) == MIN_POPULATION

    def test_population_min_minus_one_rejected(self) -> None:
        from soup_cli.utils.cmaes_merge import MIN_POPULATION, validate_population_size

        with pytest.raises(ValueError):
            validate_population_size(MIN_POPULATION - 1)

    def test_population_max_accepted(self) -> None:
        from soup_cli.utils.cmaes_merge import MAX_POPULATION, validate_population_size

        assert validate_population_size(MAX_POPULATION) == MAX_POPULATION

    def test_generations_min_accepted(self) -> None:
        from soup_cli.utils.cmaes_merge import MIN_GENERATIONS, validate_generations

        assert validate_generations(MIN_GENERATIONS) == MIN_GENERATIONS

    def test_generations_max_accepted(self) -> None:
        from soup_cli.utils.cmaes_merge import MAX_GENERATIONS, validate_generations

        assert validate_generations(MAX_GENERATIONS) == MAX_GENERATIONS


# -----------------------------------------------------------------------------
# Part B — Forward-compat: unknown fields silently ignored
# -----------------------------------------------------------------------------


class TestVectorBankForwardCompat:
    def test_unknown_fields_ignored(self, tmp_path, monkeypatch) -> None:
        """A bank JSON with extra unknown fields should round-trip without
        error — guards against breakage when v0.67.1+ adds new fields."""
        from soup_cli.utils.vector_bank import load_bank

        monkeypatch.chdir(tmp_path)
        path = tmp_path / "bank.json"
        path.write_text(
            json.dumps(
                {
                    "name": "fc",
                    "base_model": "m",
                    "projection_seed": 0,
                    "vector_dim": 2,
                    "entries": [
                        {"user_id": "u", "scaling": [0.1, 0.2]},
                    ],
                    # Forward-compat: unknown field — must be tolerated
                    "_future_field": 99,
                    "unknown_key": "unknown_value",
                }
            ),
            encoding="utf-8",
        )
        loaded = load_bank(str(path))
        assert loaded.name == "fc"
        assert loaded.vector_dim == 2

    @pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink test")
    def test_write_bank_symlink_rejected(self, tmp_path, monkeypatch) -> None:
        """`write_bank` must reject a pre-placed symlink at the target path
        (TOCTOU defence — mirrors v0.55.0 / v0.56.0 policy)."""
        from soup_cli.utils.vector_bank import VectorBank, write_bank

        monkeypatch.chdir(tmp_path)
        target = tmp_path / "out.json"
        real_target = tmp_path / "real.json"
        real_target.write_text("{}", encoding="utf-8")
        os.symlink(real_target, target)

        bank = VectorBank(
            name="b",
            base_model="m",
            projection_seed=0,
            vector_dim=1,
            entries=(),
        )
        with pytest.raises(ValueError):
            write_bank(bank, str(target))


# -----------------------------------------------------------------------------
# Part B — validate_user_id bool rejection
# -----------------------------------------------------------------------------


class TestValidateUserIdBool:
    def test_bool_rejected(self) -> None:
        from soup_cli.utils.vector_bank import validate_user_id

        with pytest.raises(TypeError):
            validate_user_id(True)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Part B — Exact boundaries
# -----------------------------------------------------------------------------


class TestVectorBankBoundaries:
    def test_vector_dim_at_max_accepted(self) -> None:
        from soup_cli.utils.vector_bank import MAX_VECTOR_DIM, VectorBank

        # MAX value should be accepted (use it with empty entries)
        VectorBank(
            name="b",
            base_model="m",
            projection_seed=0,
            vector_dim=MAX_VECTOR_DIM,
            entries=(),
        )

    def test_vector_dim_above_max_rejected(self) -> None:
        from soup_cli.utils.vector_bank import MAX_VECTOR_DIM, VectorBank

        with pytest.raises(ValueError):
            VectorBank(
                name="b",
                base_model="m",
                projection_seed=0,
                vector_dim=MAX_VECTOR_DIM + 1,
                entries=(),
            )


# -----------------------------------------------------------------------------
# Part D — Empty-sections PR renders without None leakage
# -----------------------------------------------------------------------------


class TestPRRenderEmptySections:
    def test_all_empty_sections(self) -> None:
        """A PR with no deltas / no samples / no dataset_diff should
        render valid Markdown without any `None` literals leaking."""
        from soup_cli.utils.adapter_pr import AdapterPR, render_pr_markdown

        pr = AdapterPR(
            title="empty-pr",
            base_sha="a" * 64,
            adapter_path="adapter/",
            dataset_diff="",
            deltas=(),
            samples=(),
        )
        md = render_pr_markdown(pr)
        assert "None" not in md
        assert "empty-pr" in md

    def test_json_handles_empty_sections(self) -> None:
        from soup_cli.utils.adapter_pr import AdapterPR, render_pr_json

        pr = AdapterPR(
            title="t",
            base_sha="a" * 64,
            adapter_path="adapter/",
            dataset_diff="",
            deltas=(),
            samples=(),
        )
        data = json.loads(render_pr_json(pr))
        assert data["deltas"] == []
        assert data["samples"] == []


# -----------------------------------------------------------------------------
# Part E — soup_version drift is advisory-only (not a drift signal)
# -----------------------------------------------------------------------------


class TestSoupLockVersionDriftAdvisory:
    def test_version_change_not_drift(self) -> None:
        """``soup_version`` differing between expected and actual locks
        should NOT count as drift — operators upgrade Soup legitimately."""
        from soup_cli.utils.soup_lock import SoupLock, check_lock_drift

        base = dict(
            base_model="m",
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
            closure_sha="d" * 64,
            created_at="2026-05-24",
        )
        expected = SoupLock(soup_version="0.67.0", **base)
        actual = SoupLock(soup_version="0.68.0", **base)

        drift = check_lock_drift(expected, actual)
        assert drift.ok is True

    def test_created_at_change_not_drift(self) -> None:
        from soup_cli.utils.soup_lock import SoupLock, check_lock_drift

        base = dict(
            soup_version="0.67.0",
            base_model="m",
            base_model_sha="a" * 64,
            dataset_sha="b" * 64,
            env_hash="c" * 64,
            closure_sha="d" * 64,
        )
        expected = SoupLock(created_at="2026-05-24", **base)
        actual = SoupLock(created_at="2026-06-01", **base)

        drift = check_lock_drift(expected, actual)
        assert drift.ok is True


# -----------------------------------------------------------------------------
# Part F — Non-monotonic eval_fn does not crash
# -----------------------------------------------------------------------------


class TestBisectNonMonotonic:
    def test_non_monotonic_returns_valid_result(self) -> None:
        """If the eval_fn flips back and forth (non-monotonic regression),
        the bisect must still return a valid ``BisectResult`` rather than
        crash. The boundary it finds is undefined but must be in-range."""
        from soup_cli.utils.adapter_bisect import (
            BisectPlan,
            BisectResult,
            run_bisect,
        )

        plan = BisectPlan(history=("c0", "c1", "c2", "c3", "c4", "c5"))

        # Non-monotonic: c0 ok, c1 fail, c2 ok, c3 fail, ...
        def eval_fn(checkpoint: str) -> bool:
            return int(checkpoint[1:]) % 2 == 0

        # Must not raise; result is a valid BisectResult
        result = run_bisect(plan, eval_fn=eval_fn)
        assert isinstance(result, BisectResult)
        # Boundary is some valid checkpoint id
        if result.first_broken is not None:
            assert result.first_broken in plan.history


# -----------------------------------------------------------------------------
# Source-grep regression guards on policies the agents asked us to verify
# -----------------------------------------------------------------------------


class TestSourceWiringRegressions:
    """Project-wide policies verified by source-grep across v0.67.0 modules."""

    def test_all_modules_have_future_annotations(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        for module in (
            "src/soup_cli/utils/cmaes_merge.py",
            "src/soup_cli/utils/vector_bank.py",
            "src/soup_cli/utils/mole_routing.py",
            "src/soup_cli/utils/adapter_pr.py",
            "src/soup_cli/utils/soup_lock.py",
            "src/soup_cli/utils/adapter_bisect.py",
            "src/soup_cli/commands/lock.py",
        ):
            src = (root / module).read_text(encoding="utf-8")
            assert "from __future__ import annotations" in src, (
                f"{module}: missing 'from __future__ import annotations'"
            )

    def test_atomic_write_used_in_disk_surfaces(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        for module in (
            "src/soup_cli/utils/vector_bank.py",
            "src/soup_cli/utils/adapter_pr.py",
            "src/soup_cli/utils/soup_lock.py",
        ):
            src = (root / module).read_text(encoding="utf-8")
            assert "atomic_write_text" in src, (
                f"{module}: must use atomic_write_text for disk writes"
            )

    def test_no_top_level_torch_imports(self) -> None:
        """v0.67.0 modules MUST stay torch-free at import time."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        for module in (
            "src/soup_cli/utils/cmaes_merge.py",
            "src/soup_cli/utils/vector_bank.py",
            "src/soup_cli/utils/mole_routing.py",
            "src/soup_cli/utils/adapter_pr.py",
            "src/soup_cli/utils/soup_lock.py",
            "src/soup_cli/utils/adapter_bisect.py",
            "src/soup_cli/commands/lock.py",
        ):
            src = (root / module).read_text(encoding="utf-8")
            head = "\n".join(
                line for line in src.splitlines()[:50]
                if line.strip() and not line.strip().startswith("#")
            )
            for forbidden in (
                "import torch", "import transformers",
                "import peft", "import safetensors",
            ):
                assert forbidden not in head, (
                    f"{module}: top-level {forbidden!r} forbidden"
                )

    def test_supported_strategies_has_cmaes(self) -> None:
        from soup_cli.utils.adapter_merge import SUPPORTED_STRATEGIES

        assert "cmaes" in SUPPORTED_STRATEGIES

    def test_subprocess_call_uses_argv_list(self) -> None:
        """`soup adapters bisect` must use argv list mode (no shell=True)."""
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (root / "src" / "soup_cli" / "commands" / "adapters.py").read_text(
            encoding="utf-8"
        )
        # The bisect subprocess call site
        # Must NOT use shell=True; must use shlex.split + shlex.quote
        assert "shell=True" not in src
        # Must include shlex.quote pattern for the {ckpt} substitution
        assert "shlex.quote" in src
