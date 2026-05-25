"""v0.70.0 Part E — Iterative DPO loop driver.

Sample → RM-score → re-pair → retrain over N rounds. Schema + CLI live;
the actual round-orchestrator (which would invoke `soup train --task dpo`
between rounds) is deferred to v0.70.1 (mirrors v0.68.0 local-rl
nightly-train pattern).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from typer.testing import CliRunner


class TestIterativeDPOPublicSurface:
    def test_module_imports(self):
        from soup_cli.utils import iterative_dpo

        assert hasattr(iterative_dpo, "IterativeDPOPlan")
        assert hasattr(iterative_dpo, "IterativeDPORound")
        assert hasattr(iterative_dpo, "validate_rounds")
        assert hasattr(iterative_dpo, "validate_pairs_per_round")
        assert hasattr(iterative_dpo, "build_iterative_dpo_plan")
        assert hasattr(iterative_dpo, "run_iterative_dpo")


class TestValidateRounds:
    def test_happy(self):
        from soup_cli.utils.iterative_dpo import validate_rounds

        assert validate_rounds(5) == 5

    def test_min_boundary(self):
        from soup_cli.utils.iterative_dpo import validate_rounds

        assert validate_rounds(1) == 1

    def test_max_boundary(self):
        from soup_cli.utils.iterative_dpo import validate_rounds

        # 100 rounds is plenty.
        assert validate_rounds(100) == 100

    def test_zero_rejected(self):
        from soup_cli.utils.iterative_dpo import validate_rounds

        with pytest.raises(ValueError, match=">= 1"):
            validate_rounds(0)

    def test_above_cap_rejected(self):
        from soup_cli.utils.iterative_dpo import validate_rounds

        with pytest.raises(ValueError, match="100"):
            validate_rounds(101)

    def test_bool_rejected(self):
        from soup_cli.utils.iterative_dpo import validate_rounds

        with pytest.raises(ValueError, match="bool"):
            validate_rounds(True)

    def test_non_int_rejected(self):
        from soup_cli.utils.iterative_dpo import validate_rounds

        with pytest.raises(ValueError, match="int"):
            validate_rounds(5.5)


class TestValidatePairsPerRound:
    def test_happy(self):
        from soup_cli.utils.iterative_dpo import validate_pairs_per_round

        assert validate_pairs_per_round(500) == 500

    def test_min_boundary(self):
        from soup_cli.utils.iterative_dpo import validate_pairs_per_round

        assert validate_pairs_per_round(10) == 10

    def test_max_boundary(self):
        from soup_cli.utils.iterative_dpo import validate_pairs_per_round

        assert validate_pairs_per_round(1_000_000) == 1_000_000

    def test_below_min_rejected(self):
        from soup_cli.utils.iterative_dpo import validate_pairs_per_round

        with pytest.raises(ValueError, match=">= 10"):
            validate_pairs_per_round(9)

    def test_above_cap_rejected(self):
        from soup_cli.utils.iterative_dpo import validate_pairs_per_round

        with pytest.raises(ValueError, match="1000000"):
            validate_pairs_per_round(1_000_001)

    def test_bool_rejected(self):
        from soup_cli.utils.iterative_dpo import validate_pairs_per_round

        with pytest.raises(ValueError, match="bool"):
            validate_pairs_per_round(True)


class TestIterativeDPORound:
    def test_basic(self):
        from soup_cli.utils.iterative_dpo import IterativeDPORound

        rnd = IterativeDPORound(
            round_index=1,
            prompts_path="./data/prompts.jsonl",
            pairs_path="./data/round1_pairs.jsonl",
            adapter_path="./output/round1",
            pairs_count=512,
        )
        assert rnd.round_index == 1
        assert rnd.pairs_count == 512

    def test_frozen(self):
        from soup_cli.utils.iterative_dpo import IterativeDPORound

        rnd = IterativeDPORound(
            round_index=1,
            prompts_path="./p.jsonl",
            pairs_path="./pairs.jsonl",
            adapter_path="./out",
            pairs_count=100,
        )
        with pytest.raises(FrozenInstanceError):
            rnd.round_index = 2  # type: ignore[misc]

    def test_negative_round_rejected(self):
        from soup_cli.utils.iterative_dpo import IterativeDPORound

        with pytest.raises(ValueError, match="round_index"):
            IterativeDPORound(
                round_index=-1,
                prompts_path="./p.jsonl",
                pairs_path="./pairs.jsonl",
                adapter_path="./out",
                pairs_count=100,
            )

    def test_bool_round_rejected(self):
        from soup_cli.utils.iterative_dpo import IterativeDPORound

        with pytest.raises(ValueError, match="bool"):
            IterativeDPORound(
                round_index=True,
                prompts_path="./p.jsonl",
                pairs_path="./pairs.jsonl",
                adapter_path="./out",
                pairs_count=100,
            )

    def test_null_byte_path_rejected(self):
        from soup_cli.utils.iterative_dpo import IterativeDPORound

        with pytest.raises(ValueError, match="null byte"):
            IterativeDPORound(
                round_index=0,
                prompts_path="./bad\x00",
                pairs_path="./pairs.jsonl",
                adapter_path="./out",
                pairs_count=100,
            )

    def test_negative_pairs_rejected(self):
        from soup_cli.utils.iterative_dpo import IterativeDPORound

        with pytest.raises(ValueError, match="pairs_count"):
            IterativeDPORound(
                round_index=0,
                prompts_path="./p.jsonl",
                pairs_path="./pairs.jsonl",
                adapter_path="./out",
                pairs_count=-1,
            )


class TestIterativeDPOPlan:
    def test_basic(self):
        from soup_cli.utils.iterative_dpo import (
            IterativeDPOPlan,
            IterativeDPORound,
        )

        plan = IterativeDPOPlan(
            base_model="meta-llama/Llama-3.1-8B",
            reward_model="./output_rm",
            rounds=(
                IterativeDPORound(
                    round_index=0,
                    prompts_path="./p.jsonl",
                    pairs_path="./r0.jsonl",
                    adapter_path="./out/r0",
                    pairs_count=512,
                ),
                IterativeDPORound(
                    round_index=1,
                    prompts_path="./p.jsonl",
                    pairs_path="./r1.jsonl",
                    adapter_path="./out/r1",
                    pairs_count=512,
                ),
            ),
        )
        assert len(plan.rounds) == 2

    def test_frozen(self):
        from soup_cli.utils.iterative_dpo import (
            IterativeDPOPlan,
            IterativeDPORound,
        )

        plan = IterativeDPOPlan(
            base_model="m",
            reward_model="./rm",
            rounds=(
                IterativeDPORound(
                    round_index=0,
                    prompts_path="./p.jsonl",
                    pairs_path="./r0.jsonl",
                    adapter_path="./out/r0",
                    pairs_count=10,
                ),
            ),
        )
        with pytest.raises(FrozenInstanceError):
            plan.base_model = "evil"  # type: ignore[misc]

    def test_rounds_must_be_tuple(self):
        from soup_cli.utils.iterative_dpo import (
            IterativeDPOPlan,
            IterativeDPORound,
        )

        rounds_list = [
            IterativeDPORound(
                round_index=0,
                prompts_path="./p.jsonl",
                pairs_path="./r0.jsonl",
                adapter_path="./out/r0",
                pairs_count=10,
            ),
        ]
        with pytest.raises(TypeError, match="tuple"):
            IterativeDPOPlan(
                base_model="m",
                reward_model="./rm",
                rounds=rounds_list,  # type: ignore[arg-type]
            )

    def test_zero_rounds_rejected(self):
        from soup_cli.utils.iterative_dpo import IterativeDPOPlan

        with pytest.raises(ValueError, match="rounds"):
            IterativeDPOPlan(
                base_model="m",
                reward_model="./rm",
                rounds=(),
            )

    def test_non_consecutive_round_indices_rejected(self):
        from soup_cli.utils.iterative_dpo import (
            IterativeDPOPlan,
            IterativeDPORound,
        )

        with pytest.raises(ValueError, match="consecutive"):
            IterativeDPOPlan(
                base_model="m",
                reward_model="./rm",
                rounds=(
                    IterativeDPORound(
                        round_index=0,
                        prompts_path="./p.jsonl",
                        pairs_path="./r0.jsonl",
                        adapter_path="./out/r0",
                        pairs_count=10,
                    ),
                    IterativeDPORound(
                        round_index=5,  # gap
                        prompts_path="./p.jsonl",
                        pairs_path="./r5.jsonl",
                        adapter_path="./out/r5",
                        pairs_count=10,
                    ),
                ),
            )

    def test_null_byte_base_rejected(self):
        from soup_cli.utils.iterative_dpo import (
            IterativeDPOPlan,
            IterativeDPORound,
        )

        with pytest.raises(ValueError, match="null byte"):
            IterativeDPOPlan(
                base_model="m\x00",
                reward_model="./rm",
                rounds=(
                    IterativeDPORound(
                        round_index=0,
                        prompts_path="./p.jsonl",
                        pairs_path="./r0.jsonl",
                        adapter_path="./out/r0",
                        pairs_count=10,
                    ),
                ),
            )


class TestBuildIterativeDPOPlan:
    def test_happy(self, tmp_path, monkeypatch):
        from soup_cli.utils.iterative_dpo import build_iterative_dpo_plan

        monkeypatch.chdir(tmp_path)
        (tmp_path / "prompts.jsonl").write_text(
            '{"prompt": "hello"}\n', encoding="utf-8"
        )
        plan = build_iterative_dpo_plan(
            base_model="meta-llama/Llama-3.1-8B",
            reward_model="./rm",
            prompts_path="./prompts.jsonl",
            output_dir="./out",
            rounds=3,
            pairs_per_round=100,
        )
        assert len(plan.rounds) == 3
        assert plan.rounds[0].round_index == 0
        assert plan.rounds[2].round_index == 2
        assert plan.rounds[0].adapter_path != plan.rounds[1].adapter_path

    def test_rounds_validation(self, tmp_path, monkeypatch):
        from soup_cli.utils.iterative_dpo import build_iterative_dpo_plan

        monkeypatch.chdir(tmp_path)
        (tmp_path / "prompts.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="rounds"):
            build_iterative_dpo_plan(
                base_model="m",
                reward_model="./rm",
                prompts_path="./prompts.jsonl",
                output_dir="./out",
                rounds=0,
                pairs_per_round=100,
            )


class TestRunIterativeDPODeferred:
    def test_non_plan_rejected(self):
        from soup_cli.utils.iterative_dpo import run_iterative_dpo

        with pytest.raises(TypeError, match="IterativeDPOPlan"):
            run_iterative_dpo({"rounds": 1})  # type: ignore[arg-type]

    def test_deferred(self):
        from soup_cli.utils.iterative_dpo import (
            IterativeDPOPlan,
            IterativeDPORound,
            run_iterative_dpo,
        )

        plan = IterativeDPOPlan(
            base_model="m",
            reward_model="./rm",
            rounds=(
                IterativeDPORound(
                    round_index=0,
                    prompts_path="./p.jsonl",
                    pairs_path="./r0.jsonl",
                    adapter_path="./out/r0",
                    pairs_count=10,
                ),
            ),
        )
        with pytest.raises(NotImplementedError, match="v0.70.1"):
            run_iterative_dpo(plan)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestIterativeDPOCli:
    def test_help(self):
        from soup_cli.commands.iterative_dpo import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "iterative" in result.output.lower() or "dpo" in result.output.lower()

    def test_plan_only_happy(self, tmp_path, monkeypatch):
        from soup_cli.commands.iterative_dpo import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "prompts.jsonl").write_text("{}\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--base-model",
                "meta-llama/Llama-3.1-8B",
                "--reward-model",
                "./rm",
                "--prompts",
                "./prompts.jsonl",
                "--output-dir",
                "./out",
                "--rounds",
                "3",
                "--pairs-per-round",
                "100",
                "--plan-only",
            ],
        )
        assert result.exit_code == 0, (result.output, repr(result.exception))

    def test_invalid_rounds_exits_2(self, tmp_path, monkeypatch):
        from soup_cli.commands.iterative_dpo import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "prompts.jsonl").write_text("{}\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--base-model",
                "m",
                "--reward-model",
                "./rm",
                "--prompts",
                "./prompts.jsonl",
                "--output-dir",
                "./out",
                "--rounds",
                "0",
                "--pairs-per-round",
                "100",
                "--plan-only",
            ],
        )
        assert result.exit_code == 2

    def test_live_deferred_exits_3(self, tmp_path, monkeypatch):
        """Without --plan-only, the deferred live runner exits 3."""
        from soup_cli.commands.iterative_dpo import app

        monkeypatch.chdir(tmp_path)
        (tmp_path / "prompts.jsonl").write_text("{}\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--base-model",
                "m",
                "--reward-model",
                "./rm",
                "--prompts",
                "./prompts.jsonl",
                "--output-dir",
                "./out",
                "--rounds",
                "2",
                "--pairs-per-round",
                "100",
            ],
        )
        assert result.exit_code == 3, (result.output, repr(result.exception))


class TestSourceWiring:
    def test_module_no_top_level_torch(self):
        from pathlib import Path

        src = (
            Path(__file__).resolve().parent.parent
            / "soup_cli"
            / "utils"
            / "iterative_dpo.py"
        )
        body = src.read_text(encoding="utf-8")
        assert "\nimport torch" not in body
        assert "\nfrom torch" not in body

    def test_cli_registered(self):
        """soup iterative-dpo command registered on the top-level Typer app."""
        from pathlib import Path

        cli_src = (
            Path(__file__).resolve().parent.parent
            / "soup_cli"
            / "cli.py"
        )
        body = cli_src.read_text(encoding="utf-8")
        # Either app.command or app.add_typer wiring.
        assert "iterative_dpo" in body or "iterative-dpo" in body
