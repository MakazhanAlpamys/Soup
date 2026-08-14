"""Tests for training intelligence — forgetting detection + checkpoint intel (Part G)."""

import pytest

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestForgettingConfig:
    def test_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.forgetting_detection is False
        assert cfg.forgetting_eval_steps == 100
        assert cfg.forgetting_threshold == 0.10
        assert cfg.forgetting_benchmark == "mini_mmlu"
        assert cfg.forgetting_stop is False

    def test_threshold_bounded(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(forgetting_threshold=0.8)  # > 0.50 max
        with pytest.raises(ValidationError):
            TrainingConfig(forgetting_threshold=0.001)  # < 0.01 min

    def test_eval_steps_bounded(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(forgetting_eval_steps=5)

    def test_benchmark_literal(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(forgetting_benchmark="evil-benchmark")


class TestCheckpointIntelConfig:
    def test_defaults(self):
        from soup_cli.config.schema import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.checkpoint_intelligence is False
        assert cfg.checkpoint_eval_steps == 200
        assert cfg.checkpoint_eval_metric == "composite"
        assert cfg.checkpoint_keep_top == 3
        assert cfg.early_stop_on_regression is False
        assert cfg.early_stop_patience == 2

    def test_keep_top_bounded(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(checkpoint_keep_top=0)
        with pytest.raises(ValidationError):
            TrainingConfig(checkpoint_keep_top=21)

    def test_patience_bounded(self):
        from pydantic import ValidationError

        from soup_cli.config.schema import TrainingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(early_stop_patience=0)
        with pytest.raises(ValidationError):
            TrainingConfig(early_stop_patience=11)


# ---------------------------------------------------------------------------
# ForgettingDetector
# ---------------------------------------------------------------------------

class TestForgettingDetector:
    def test_built_in_benchmarks_exist(self):
        from soup_cli.eval.forgetting import MINI_BENCHMARKS

        assert "mini_mmlu" in MINI_BENCHMARKS
        assert "mini_common_sense" in MINI_BENCHMARKS
        assert "mini_instruction" in MINI_BENCHMARKS

        # v0.71.38 expands the v0.25.0 5-item starter sets so a single-item flip
        # (1/N < 0.05) actually trips the ship gate. Every suite is now large
        # enough that its quantum is finer than the 0.05 forgetting threshold.
        assert "mini_arithmetic" in MINI_BENCHMARKS
        for name, bench in MINI_BENCHMARKS.items():
            assert len(bench) > 20, f"{name} expected >20 items, got {len(bench)}"
            assert 1.0 / len(bench) < 0.05, f"{name} quantum too coarse"
            for item in bench:
                assert "question" in item
                assert "answer" in item
                assert isinstance(item["question"], str)
                assert isinstance(item["answer"], str)

    def test_baseline_mocked(self):
        from soup_cli.eval.forgetting import ForgettingDetector

        def fake_gen(prompt: str) -> str:
            # Always returns the correct answer for mini_mmlu first item
            return "A"

        detector = ForgettingDetector(
            generate_fn=fake_gen, benchmark="mini_mmlu",
        )
        baseline = detector.run_baseline()
        assert 0.0 <= baseline <= 1.0

    def test_check_forgetting_level_green(self):
        from soup_cli.eval.forgetting import ForgettingDetector

        detector = ForgettingDetector(
            generate_fn=lambda p: "dummy", benchmark="mini_mmlu",
            threshold=0.10,
        )
        # Manually set baseline and current accuracy
        detector._baseline_accuracy = 0.9
        result = detector._build_result(step=100, accuracy=0.88)
        assert result.warning_level == "green"

    def test_check_forgetting_level_yellow(self):
        from soup_cli.eval.forgetting import ForgettingDetector

        detector = ForgettingDetector(
            generate_fn=lambda p: "dummy", threshold=0.10,
        )
        detector._baseline_accuracy = 0.9
        result = detector._build_result(step=100, accuracy=0.75)
        assert result.warning_level == "yellow"

    def test_check_forgetting_level_red(self):
        from soup_cli.eval.forgetting import ForgettingDetector

        detector = ForgettingDetector(
            generate_fn=lambda p: "dummy", threshold=0.10,
        )
        detector._baseline_accuracy = 0.9
        result = detector._build_result(step=100, accuracy=0.60)
        assert result.warning_level == "red"

    def test_check_forgetting_integration(self):
        """Full check_forgetting() integration: baseline then eval."""
        from soup_cli.eval.forgetting import ForgettingDetector

        # Model that returns the right answer for the first 3 questions
        calls = {"n": 0}

        def gen(prompt: str) -> str:
            calls["n"] += 1
            return "A" if calls["n"] <= 3 else "Z"

        detector = ForgettingDetector(
            generate_fn=gen, benchmark="mini_mmlu", threshold=0.10,
        )
        # First call implicitly computes baseline then re-evaluates
        result = detector.check_forgetting(step=100)
        assert result.step == 100
        assert result.warning_level in ("green", "yellow", "red")
        assert detector._baseline_accuracy is not None

    def test_unknown_benchmark_rejected(self):
        from soup_cli.eval.forgetting import ForgettingDetector

        with pytest.raises(ValueError):
            ForgettingDetector(
                generate_fn=lambda p: "x",
                benchmark="evil_benchmark",
            )

    def test_forgetting_stop_schema(self):
        """forgetting_stop is a proper TrainingConfig bool with False default."""
        from soup_cli.config.schema import TrainingConfig

        default_cfg = TrainingConfig()
        assert default_cfg.forgetting_stop is False

        enabled = TrainingConfig(forgetting_detection=True, forgetting_stop=True)
        assert enabled.forgetting_stop is True


# ---------------------------------------------------------------------------
# CheckpointTracker
# ---------------------------------------------------------------------------

class TestCheckpointTracker:
    def test_initial_best_none(self):
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        tracker = CheckpointTracker(metric="composite")
        assert tracker.best is None

    def test_record_becomes_best(self):
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        tracker = CheckpointTracker(metric="composite")
        tracker.record(step=100, score=0.8)
        assert tracker.best is not None
        assert tracker.best.score == 0.8
        assert tracker.best.step == 100

    def test_record_better_replaces_best(self):
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        tracker = CheckpointTracker(metric="composite")
        tracker.record(step=100, score=0.6)
        tracker.record(step=200, score=0.9)
        assert tracker.best.score == 0.9
        assert tracker.best.step == 200

    def test_record_worse_keeps_best(self):
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        tracker = CheckpointTracker(metric="composite")
        tracker.record(step=100, score=0.9)
        tracker.record(step=200, score=0.5)
        assert tracker.best.score == 0.9
        assert tracker.best.step == 100

    def test_should_early_stop_no_regression(self):
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        tracker = CheckpointTracker(metric="composite", patience=2)
        tracker.record(step=100, score=0.7)
        tracker.record(step=200, score=0.8)
        tracker.record(step=300, score=0.9)
        assert tracker.should_early_stop() is False

    def test_should_early_stop_on_patience(self):
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        tracker = CheckpointTracker(metric="composite", patience=2)
        tracker.record(step=100, score=0.9)
        tracker.record(step=200, score=0.8)
        tracker.record(step=300, score=0.7)
        assert tracker.should_early_stop() is True

    def test_prune_keeps_top_n(self, tmp_path):
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        tracker = CheckpointTracker(metric="composite", keep_top=2)

        # Create fake checkpoint dirs
        for step, score in [(100, 0.6), (200, 0.9), (300, 0.75), (400, 0.5)]:
            ckpt = tmp_path / f"checkpoint-{step}"
            ckpt.mkdir()
            (ckpt / "dummy.txt").write_text("x", encoding="utf-8")
            tracker.record(step=step, score=score)

        removed = tracker.prune_checkpoints(tmp_path)

        surviving = sorted(
            p.name for p in tmp_path.iterdir() if p.name.startswith("checkpoint-")
        )
        # Top 2 by score: 200 (0.9) and 300 (0.75)
        assert surviving == ["checkpoint-200", "checkpoint-300"]
        # Deleted the other two and reported them
        assert sorted(removed) == [100, 400]
        assert not (tmp_path / "checkpoint-100").exists()
        assert not (tmp_path / "checkpoint-400").exists()

    def test_prune_refuses_non_checkpoint_dirs(self, tmp_path):
        from soup_cli.eval.checkpoint_intelligence import CheckpointTracker

        tracker = CheckpointTracker(metric="composite", keep_top=1)
        tracker.record(step=100, score=0.9)

        # A sibling dir that should never be touched
        sibling = tmp_path / "user_data"
        sibling.mkdir()
        (sibling / "file.txt").write_text("keep", encoding="utf-8")

        ckpt = tmp_path / "checkpoint-100"
        ckpt.mkdir()

        tracker.prune_checkpoints(tmp_path)

        assert sibling.exists()
        assert (sibling / "file.txt").exists()

    def test_composite_metric_weights(self):
        from soup_cli.eval.checkpoint_intelligence import compute_composite

        composite = compute_composite(judge=0.8, mmlu=0.6, custom=0.9)
        assert 0.6 < composite < 0.9

    def test_composite_all_zero(self):
        from soup_cli.eval.checkpoint_intelligence import compute_composite

        assert compute_composite(judge=0.0, mmlu=0.0, custom=0.0) == 0.0

    def test_composite_all_ones(self):
        from soup_cli.eval.checkpoint_intelligence import compute_composite

        assert compute_composite(judge=1.0, mmlu=1.0, custom=1.0) == 1.0

    def test_composite_missing_metrics(self):
        """None-valued metrics drop out of the weighted average."""
        from soup_cli.eval.checkpoint_intelligence import compute_composite

        # Only judge supplied — composite equals judge score exactly.
        assert compute_composite(judge=0.7, mmlu=None, custom=None) == 0.7

    def test_composite_all_none_returns_zero(self):
        from soup_cli.eval.checkpoint_intelligence import compute_composite

        assert compute_composite() == 0.0


# ---------------------------------------------------------------------------
# SQLite tracker extension
# ---------------------------------------------------------------------------

class TestTrackerSchema:
    def test_checkpoint_quality_table_created(self, tmp_path):
        from soup_cli.experiment.tracker import ExperimentTracker

        db_path = tmp_path / "experiments.db"
        tracker = ExperimentTracker(db_path=db_path)
        tracker.init_db()

        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='checkpoint_quality'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_forgetting_eval_table_created(self, tmp_path):
        from soup_cli.experiment.tracker import ExperimentTracker

        db_path = tmp_path / "experiments.db"
        tracker = ExperimentTracker(db_path=db_path)
        tracker.init_db()

        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='forgetting_eval'"
        )
        assert cursor.fetchone() is not None
        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
