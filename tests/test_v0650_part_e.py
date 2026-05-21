"""v0.65.0 Part E — IRT eval-cost optimizer tests.

1PL Rasch model fit on per-item correctness signals + subset selection
that preserves ranking power. Profile ``full | small | tiny`` selects how
many items to keep.
"""
from __future__ import annotations

import json
import os
import platform

import pytest
from typer.testing import CliRunner

from soup_cli.utils.irt import (
    IRT_PROFILES,
    IrtSubsetPlan,
    ItemDifficulty,
    fit_difficulty,
    load_response_rows,
    pick_irt_subset,
)


class TestProfiles:
    def test_profiles_immutable(self):
        with pytest.raises(TypeError):
            IRT_PROFILES["evil"] = 1.0  # type: ignore[index]

    def test_known_profiles(self):
        assert "full" in IRT_PROFILES
        assert "small" in IRT_PROFILES
        assert "tiny" in IRT_PROFILES

    def test_full_equals_one(self):
        assert IRT_PROFILES["full"] == 1.0

    def test_tiny_smaller_than_small(self):
        assert IRT_PROFILES["tiny"] < IRT_PROFILES["small"]


class TestItemDifficulty:
    def test_frozen(self):
        d = ItemDifficulty(item_id="q1", difficulty=0.0, info=1.0)
        with pytest.raises(Exception):
            d.item_id = "x"  # type: ignore[misc]

    def test_invalid_id(self):
        with pytest.raises(ValueError, match="item_id"):
            ItemDifficulty(item_id="", difficulty=0.0, info=1.0)

    def test_null_byte_id(self):
        with pytest.raises(ValueError, match="null"):
            ItemDifficulty(item_id="q\x00", difficulty=0.0, info=1.0)

    def test_invalid_difficulty(self):
        with pytest.raises(ValueError, match="difficulty"):
            ItemDifficulty(item_id="q", difficulty=float("nan"), info=1.0)

    def test_invalid_info(self):
        with pytest.raises(ValueError, match="info"):
            ItemDifficulty(item_id="q", difficulty=0.0, info=-1.0)

    def test_bool_difficulty(self):
        with pytest.raises(ValueError, match="difficulty"):
            ItemDifficulty(
                item_id="q", difficulty=True, info=1.0,  # type: ignore[arg-type]
            )


class TestFitDifficulty:
    def test_basic(self):
        # 3 items, 5 respondents
        # item1: 5/5 correct → easy → difficulty negative
        # item2: 0/5 correct → hard → difficulty positive
        # item3: mixed
        rows = []
        for i in range(5):
            rows.extend([
                {"item_id": "easy", "correct": True},
                {"item_id": "hard", "correct": False},
                {"item_id": "mixed", "correct": i % 2 == 0},
            ])
        result = fit_difficulty(rows)
        by_id = {d.item_id: d for d in result}
        assert "easy" in by_id
        assert "hard" in by_id
        assert "mixed" in by_id
        # Easy items have NEGATIVE difficulty under Rasch.
        assert by_id["easy"].difficulty < by_id["mixed"].difficulty
        assert by_id["mixed"].difficulty < by_id["hard"].difficulty

    def test_returns_tuple(self):
        rows = [
            {"item_id": "q1", "correct": True},
            {"item_id": "q1", "correct": False},
        ]
        result = fit_difficulty(rows)
        assert isinstance(result, tuple)

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            fit_difficulty([])

    def test_non_list(self):
        with pytest.raises(TypeError):
            fit_difficulty("not a list")  # type: ignore[arg-type]

    def test_missing_item_id(self):
        with pytest.raises(ValueError, match="item_id"):
            fit_difficulty([{"correct": True}])

    def test_missing_correct(self):
        with pytest.raises(ValueError, match="correct"):
            fit_difficulty([{"item_id": "q1"}])

    def test_non_bool_correct(self):
        with pytest.raises(ValueError, match="correct"):
            fit_difficulty([{"item_id": "q1", "correct": "yes"}])

    def test_too_many_rows(self):
        rows = [
            {"item_id": f"q{i % 100}", "correct": i % 2 == 0}
            for i in range(1_000_001)
        ]
        with pytest.raises(ValueError, match="cap"):
            fit_difficulty(rows)

    def test_oversize_item_id(self):
        with pytest.raises(ValueError, match="item_id"):
            fit_difficulty([{"item_id": "a" * 257, "correct": True}])


class TestPickIrtSubset:
    def test_full_keeps_all(self):
        difficulty = tuple(
            ItemDifficulty(item_id=f"q{i}", difficulty=0.0, info=1.0)
            for i in range(10)
        )
        plan = pick_irt_subset(difficulty, size="full")
        assert len(plan.item_ids) == 10
        assert plan.size == "full"

    def test_small_reduces(self):
        difficulty = tuple(
            ItemDifficulty(item_id=f"q{i}", difficulty=float(i), info=1.0)
            for i in range(100)
        )
        plan = pick_irt_subset(difficulty, size="small")
        assert len(plan.item_ids) < 100
        assert plan.total_items == 100

    def test_tiny_smaller(self):
        difficulty = tuple(
            ItemDifficulty(item_id=f"q{i}", difficulty=float(i), info=1.0)
            for i in range(100)
        )
        small = pick_irt_subset(difficulty, size="small")
        tiny = pick_irt_subset(difficulty, size="tiny")
        assert len(tiny.item_ids) <= len(small.item_ids)

    def test_invalid_size(self):
        difficulty = (ItemDifficulty(item_id="q", difficulty=0.0, info=1.0),)
        with pytest.raises(ValueError, match="size"):
            pick_irt_subset(difficulty, size="evil")

    def test_empty_difficulty(self):
        with pytest.raises(ValueError, match="empty"):
            pick_irt_subset((), size="full")

    def test_non_tuple_difficulty(self):
        with pytest.raises(TypeError):
            pick_irt_subset([], size="full")  # type: ignore[arg-type]

    def test_subset_returns_plan(self):
        difficulty = tuple(
            ItemDifficulty(item_id=f"q{i}", difficulty=0.0, info=float(i + 1))
            for i in range(10)
        )
        plan = pick_irt_subset(difficulty, size="small")
        assert isinstance(plan, IrtSubsetPlan)
        assert plan.cost_ratio <= 1.0
        assert plan.cost_ratio > 0.0

    def test_picks_high_info_items(self):
        # High-info items should be preferred.
        difficulty = (
            ItemDifficulty(item_id="boring", difficulty=0.0, info=0.001),
            ItemDifficulty(item_id="useful", difficulty=0.0, info=10.0),
        )
        plan = pick_irt_subset(difficulty, size="tiny")
        # If we only pick one, it should be the useful one.
        assert "useful" in plan.item_ids


class TestIrtSubsetPlan:
    def test_frozen(self):
        p = IrtSubsetPlan(
            size="small", item_ids=("q1",), total_items=10, cost_ratio=0.1,
        )
        with pytest.raises(Exception):
            p.size = "tiny"  # type: ignore[misc]

    def test_to_dict(self):
        p = IrtSubsetPlan(
            size="small", item_ids=("q1", "q2"),
            total_items=10, cost_ratio=0.2,
        )
        d = p.to_dict()
        assert d["size"] == "small"
        assert d["item_ids"] == ["q1", "q2"]
        assert d["total_items"] == 10

    def test_invalid_size(self):
        with pytest.raises(ValueError, match="size"):
            IrtSubsetPlan(
                size="evil", item_ids=("q",),
                total_items=1, cost_ratio=1.0,
            )

    def test_invalid_cost(self):
        with pytest.raises(ValueError, match="cost_ratio"):
            IrtSubsetPlan(
                size="full", item_ids=("q",),
                total_items=1, cost_ratio=2.0,
            )

    def test_invalid_total(self):
        with pytest.raises(ValueError, match="total_items"):
            IrtSubsetPlan(
                size="full", item_ids=("q",),
                total_items=-1, cost_ratio=1.0,
            )

    def test_item_ids_must_be_tuple(self):
        with pytest.raises(ValueError, match="item_ids"):
            IrtSubsetPlan(
                size="full", item_ids=["q"],  # type: ignore[arg-type]
                total_items=1, cost_ratio=1.0,
            )

    def test_subset_exceeds_total(self):
        with pytest.raises(ValueError, match="total"):
            IrtSubsetPlan(
                size="full", item_ids=("q1", "q2"),
                total_items=1, cost_ratio=1.0,
            )


class TestLoadResponseRows:
    def test_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "responses.jsonl"
        p.write_text(
            '{"item_id": "q1", "correct": true}\n'
            '{"item_id": "q1", "correct": false}\n'
        )
        rows = load_response_rows(str(p))
        assert len(rows) == 2

    def test_skips_malformed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "responses.jsonl"
        p.write_text(
            '{"item_id": "q1", "correct": true}\n'
            'not valid json\n'
            '{"item_id": "q2", "correct": false}\n'
        )
        rows = load_response_rows(str(p))
        assert len(rows) == 2

    def test_outside_cwd(self, tmp_path, monkeypatch):
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        outside = tmp_path / "ev.jsonl"
        outside.write_text("{}")
        with pytest.raises(ValueError):
            load_response_rows(str(outside))

    def test_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_response_rows(str(tmp_path / "nope.jsonl"))

    def test_null_byte_path(self):
        with pytest.raises(ValueError, match="null"):
            load_response_rows("path\x00.jsonl")

    def test_non_string_path(self):
        with pytest.raises(TypeError):
            load_response_rows(42)  # type: ignore[arg-type]

    def test_oversize_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "huge.jsonl"
        # 2 GiB sparse-like — write a small file then check cap rejects bigger.
        # We can't easily make a giant file; instead simulate by setting cap
        # smaller. Skip if not testable.
        # Instead test row cap:
        lines = "\n".join(
            json.dumps({"item_id": f"q{i}", "correct": True})
            for i in range(1_000_001)
        )
        p.write_text(lines)
        with pytest.raises(ValueError, match="cap"):
            load_response_rows(str(p))

    @pytest.mark.skipif(platform.system() == "Windows", reason="POSIX symlink")
    def test_symlink_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "real.jsonl"
        target.write_text('{"item_id": "q1", "correct": true}')
        link = tmp_path / "link.jsonl"
        os.symlink(target, link)
        with pytest.raises(ValueError, match="symlink"):
            load_response_rows(str(link))


class TestIrtCli:
    def test_help_listed(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "irt-subset" in result.output.lower()

    def test_irt_help(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, ["irt-subset", "--help"])
        assert result.exit_code == 0

    def test_irt_runs(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "responses.jsonl"
        p.write_text("\n".join(
            json.dumps({"item_id": f"q{i % 10}", "correct": i % 2 == 0})
            for i in range(100)
        ))
        out = tmp_path / "plan.json"
        runner = CliRunner()
        result = runner.invoke(app, [
            "irt-subset", str(p), "--size", "small", "--output", str(out),
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        assert out.exists()

    def test_irt_unknown_size(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        p = tmp_path / "responses.jsonl"
        p.write_text('{"item_id": "q", "correct": true}')
        runner = CliRunner()
        result = runner.invoke(app, [
            "irt-subset", str(p), "--size", "evil",
        ])
        assert result.exit_code != 0


class TestSourceWiring:
    def test_no_heavy_imports(self):
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "soup_cli" / "utils" / "irt.py"
        text = src.read_text(encoding="utf-8")
        forbidden_imports = (
            "import torch\n",
            "import transformers\n",
            "import scipy\n",
            "from torch",
            "from transformers",
            "from scipy",
        )
        for forbidden in forbidden_imports:
            assert forbidden not in text, f"Found heavy top-level: {forbidden!r}"
