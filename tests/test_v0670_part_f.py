"""v0.67.0 Part F — ``soup adapters bisect``.

Binary-search over a training-step history (or dataset-commit history)
to find the step that broke an eval.

Tests for ``soup_cli/utils/adapter_bisect.py``:

- Frozen ``BisectPlan`` / ``BisectStep`` / ``BisectResult`` dataclasses
- ``build_bisect_plan(history)`` validates input + computes initial mid
- ``run_bisect(plan, eval_fn)`` runs the binary search
- ``bisect_next_step(state)`` pure step kernel (one iteration)
- CLI smoke
"""

from __future__ import annotations

import dataclasses

import pytest


class TestPublicSurface:
    def test_module_importable(self) -> None:
        from soup_cli.utils import adapter_bisect

        assert hasattr(adapter_bisect, "BisectPlan")
        assert hasattr(adapter_bisect, "BisectStep")
        assert hasattr(adapter_bisect, "BisectResult")
        assert hasattr(adapter_bisect, "build_bisect_plan")
        assert hasattr(adapter_bisect, "run_bisect")
        assert hasattr(adapter_bisect, "bisect_next_step")


class TestBisectPlan:
    def test_construct(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan

        plan = BisectPlan(history=("ckpt-100", "ckpt-200", "ckpt-300"))
        assert len(plan.history) == 3

    def test_frozen(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan

        plan = BisectPlan(history=("a", "b"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.history = ()  # type: ignore[misc]

    def test_too_short_rejected(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan

        with pytest.raises(ValueError):
            BisectPlan(history=("only-one",))

    def test_must_be_tuple(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan

        with pytest.raises(TypeError):
            BisectPlan(history=["a", "b"])  # type: ignore[arg-type]

    def test_entries_must_be_str(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan

        with pytest.raises(TypeError):
            BisectPlan(history=("a", 2))  # type: ignore[arg-type]

    def test_empty_entry_rejected(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan

        with pytest.raises(ValueError):
            BisectPlan(history=("a", ""))

    def test_null_byte_rejected(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan

        with pytest.raises(ValueError):
            BisectPlan(history=("a", "b\x00"))

    def test_duplicate_rejected(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan

        with pytest.raises(ValueError):
            BisectPlan(history=("a", "b", "a"))

    def test_oversize_history_rejected(self) -> None:
        from soup_cli.utils.adapter_bisect import MAX_HISTORY, BisectPlan

        too_many = tuple(f"ckpt-{i}" for i in range(MAX_HISTORY + 1))
        with pytest.raises(ValueError):
            BisectPlan(history=too_many)


class TestBisectStep:
    def test_construct(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectStep

        step = BisectStep(checkpoint="ckpt-100", ok=True)
        assert step.checkpoint == "ckpt-100"

    def test_frozen(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectStep

        step = BisectStep(checkpoint="ckpt-1", ok=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.ok = False  # type: ignore[misc]

    def test_ok_must_be_bool(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectStep

        with pytest.raises(TypeError):
            BisectStep(checkpoint="ckpt", ok=1)  # type: ignore[arg-type]


class TestBuildBisectPlan:
    def test_happy(self) -> None:
        from soup_cli.utils.adapter_bisect import build_bisect_plan

        plan = build_bisect_plan(["a", "b", "c", "d"])
        assert len(plan.history) == 4


class TestBisectNextStep:
    def test_picks_midpoint(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan, bisect_next_step

        plan = BisectPlan(history=("a", "b", "c", "d", "e"))
        nxt = bisect_next_step(plan, lo=0, hi=4)
        assert nxt == 2

    def test_invalid_range(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan, bisect_next_step

        plan = BisectPlan(history=("a", "b"))
        with pytest.raises(ValueError):
            bisect_next_step(plan, lo=5, hi=0)


class TestRunBisect:
    def test_finds_first_broken(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan, run_bisect

        # Eval says: first three checkpoints pass, last three fail.
        # The bisect should find the boundary at index 3 (first failing).
        plan = BisectPlan(
            history=("c0", "c1", "c2", "c3", "c4", "c5")
        )

        def eval_fn(checkpoint: str) -> bool:
            return checkpoint in ("c0", "c1", "c2")

        result = run_bisect(plan, eval_fn=eval_fn)
        assert result.first_broken == "c3"

    def test_all_ok(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan, run_bisect

        plan = BisectPlan(history=("c0", "c1", "c2"))
        result = run_bisect(plan, eval_fn=lambda _: True)
        # All pass: first_broken is None
        assert result.first_broken is None
        assert result.verdict == "ALL_OK"

    def test_all_broken(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan, run_bisect

        plan = BisectPlan(history=("c0", "c1", "c2"))
        result = run_bisect(plan, eval_fn=lambda _: False)
        # All fail: first_broken is c0
        assert result.first_broken == "c0"

    def test_eval_fn_must_be_callable(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan, run_bisect

        plan = BisectPlan(history=("a", "b"))
        with pytest.raises(TypeError):
            run_bisect(plan, eval_fn=None)  # type: ignore[arg-type]

    def test_non_plan_rejected(self) -> None:
        from soup_cli.utils.adapter_bisect import run_bisect

        with pytest.raises(TypeError):
            run_bisect("not-a-plan", eval_fn=lambda _: True)  # type: ignore[arg-type]

    def test_logs_history_of_probes(self) -> None:
        from soup_cli.utils.adapter_bisect import BisectPlan, run_bisect

        plan = BisectPlan(
            history=("c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7")
        )

        probed: list[str] = []

        def eval_fn(checkpoint: str) -> bool:
            probed.append(checkpoint)
            # Boundary at c4
            return int(checkpoint[1:]) < 4

        result = run_bisect(plan, eval_fn=eval_fn)
        assert result.first_broken == "c4"
        # Probes are logged in `result.steps`
        assert len(result.steps) == len(probed)
        # ~log2(8)=3 midpoint probes plus 2 endpoint probes = 5 total
        assert len(result.steps) <= 6


class TestCliSmoke:
    def test_bisect_help(self) -> None:
        from typer.testing import CliRunner

        from soup_cli.commands.adapters import app

        runner = CliRunner()
        result = runner.invoke(app, ["bisect", "--help"])
        assert result.exit_code == 0
        assert "bisect" in result.output.lower()


class TestSourceWiring:
    def test_no_top_level_heavy_imports(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        src = (root / "src" / "soup_cli" / "utils" / "adapter_bisect.py").read_text(
            encoding="utf-8"
        )
        head_lines = [
            line
            for line in src.splitlines()[:50]
            if line.strip() and not line.strip().startswith("#")
        ]
        head = "\n".join(head_lines)
        for forbidden in ("import torch", "import transformers", "import peft"):
            assert forbidden not in head, f"top-level {forbidden!r}"
