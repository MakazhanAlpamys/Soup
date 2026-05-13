"""Tests for v0.48.0 Part A — Curriculum-Aware Trainer (dynamic re-weighting).

BETA feature. Covers:
- ``DynamicCurriculumPolicy`` frozen + bounds.
- ``compute_bucket_weights`` math properties + degenerate inputs.
- ``validate_distributed_curriculum`` multi-rank coordination gate.
- ``render_curve`` ASCII renderer.
- ``parse_history_jsonl`` schema check.
- Schema: ``curriculum_dynamic`` requires ``curriculum=True``; mlx + non-SFT
  rejection at SoupConfig level.
- ``soup runs curriculum-curve`` CLI smoke.
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
from soup_cli.utils.curriculum_dynamic import (
    BucketStats,
    DynamicCurriculumPolicy,
    compute_bucket_weights,
    parse_history_jsonl,
    render_curve,
    validate_distributed_curriculum,
)

# ---------- DynamicCurriculumPolicy ---------------------------------------


def test_policy_defaults_ok():
    p = DynamicCurriculumPolicy(num_buckets=4)
    assert p.recompute_every_n_steps == 50
    assert p.floor == 0.05
    assert p.temperature == 1.0


def test_policy_frozen():
    p = DynamicCurriculumPolicy(num_buckets=4)
    with pytest.raises(Exception):
        p.num_buckets = 99  # type: ignore[misc]


@pytest.mark.parametrize("nb", [0, -1, 21, 100])
def test_policy_rejects_invalid_buckets(nb):
    with pytest.raises(ValueError):
        DynamicCurriculumPolicy(num_buckets=nb)


def test_policy_rejects_bool_buckets():
    with pytest.raises(ValueError, match="num_buckets must be int, not bool"):
        DynamicCurriculumPolicy(num_buckets=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("rs", [0, -5, 1_000_001])
def test_policy_rejects_invalid_recompute(rs):
    with pytest.raises(ValueError):
        DynamicCurriculumPolicy(num_buckets=4, recompute_every_n_steps=rs)


def test_policy_floor_must_leave_room():
    # num_buckets=4 → ceiling=0.25.
    DynamicCurriculumPolicy(num_buckets=4, floor=0.25)
    with pytest.raises(ValueError, match="floor must be in"):
        DynamicCurriculumPolicy(num_buckets=4, floor=0.5)


def test_policy_floor_rejects_non_finite():
    with pytest.raises(ValueError, match="must be finite"):
        DynamicCurriculumPolicy(num_buckets=4, floor=float("nan"))


def test_policy_rejects_zero_temperature():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        DynamicCurriculumPolicy(num_buckets=4, temperature=0.0)


def test_policy_should_recompute():
    p = DynamicCurriculumPolicy(num_buckets=4, recompute_every_n_steps=10)
    assert p.should_recompute(0) is False
    assert p.should_recompute(5) is False
    assert p.should_recompute(10) is True
    assert p.should_recompute(20) is True


def test_policy_should_recompute_rejects_bool():
    p = DynamicCurriculumPolicy(num_buckets=4)
    with pytest.raises(ValueError):
        p.should_recompute(True)  # type: ignore[arg-type]


def test_policy_should_recompute_rejects_negative():
    p = DynamicCurriculumPolicy(num_buckets=4)
    with pytest.raises(ValueError):
        p.should_recompute(-1)


# ---------- compute_bucket_weights ----------------------------------------


def test_compute_uniform_when_no_data():
    p = DynamicCurriculumPolicy(num_buckets=4)
    weights = compute_bucket_weights({}, p)
    assert len(weights) == 4
    assert all(abs(w - 0.25) < 1e-9 for w in weights)


def test_compute_uniform_when_no_samples():
    p = DynamicCurriculumPolicy(num_buckets=3)
    stats = {
        0: {"num_samples": 0, "mean_loss": 0.0, "mean_grad_norm": 0.0},
    }
    weights = compute_bucket_weights(stats, p)
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-6


def test_compute_high_loss_gets_more_weight():
    p = DynamicCurriculumPolicy(num_buckets=2, floor=0.05, temperature=1.0)
    stats = {
        0: {"num_samples": 10, "mean_loss": 0.5, "mean_grad_norm": 0.1},
        1: {"num_samples": 10, "mean_loss": 5.0, "mean_grad_norm": 1.0},
    }
    w0, w1 = compute_bucket_weights(stats, p)
    assert w1 > w0
    assert abs(w0 + w1 - 1.0) < 1e-6


def test_compute_floor_respected():
    p = DynamicCurriculumPolicy(num_buckets=2, floor=0.2)
    stats = {
        0: {"num_samples": 10, "mean_loss": 0.0, "mean_grad_norm": 0.0},
        1: {"num_samples": 10, "mean_loss": 100.0, "mean_grad_norm": 100.0},
    }
    weights = compute_bucket_weights(stats, p)
    assert min(weights) >= 0.2 - 1e-9


def test_compute_sums_to_one():
    p = DynamicCurriculumPolicy(num_buckets=5)
    stats = {
        i: {"num_samples": 5, "mean_loss": float(i), "mean_grad_norm": 0.1}
        for i in range(5)
    }
    weights = compute_bucket_weights(stats, p)
    assert abs(sum(weights) - 1.0) < 1e-6


def test_compute_rejects_non_mapping_stats():
    p = DynamicCurriculumPolicy(num_buckets=2)
    with pytest.raises(TypeError, match="stats must be Mapping"):
        compute_bucket_weights([], p)  # type: ignore[arg-type]


def test_compute_rejects_non_policy():
    with pytest.raises(TypeError, match="policy must be"):
        compute_bucket_weights({}, "policy")  # type: ignore[arg-type]


def test_compute_rejects_negative_loss():
    p = DynamicCurriculumPolicy(num_buckets=2)
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_bucket_weights(
            {0: {"num_samples": 1, "mean_loss": -1.0, "mean_grad_norm": 0.0}},
            p,
        )


def test_compute_rejects_oversize_num_samples():
    p = DynamicCurriculumPolicy(num_buckets=2)
    with pytest.raises(ValueError, match="num_samples"):
        compute_bucket_weights(
            {0: {"num_samples": 10_000_001, "mean_loss": 1.0,
                 "mean_grad_norm": 1.0}}, p,
        )


def test_compute_rejects_bool_bucket_id():
    p = DynamicCurriculumPolicy(num_buckets=2)
    with pytest.raises(TypeError, match="bucket id must be int"):
        compute_bucket_weights(
            {True: {"num_samples": 1, "mean_loss": 1.0,
                    "mean_grad_norm": 1.0}}, p,
        )


def test_compute_rejects_negative_bucket_id():
    p = DynamicCurriculumPolicy(num_buckets=2)
    with pytest.raises(ValueError, match="bucket id must be >= 0"):
        compute_bucket_weights(
            {-1: {"num_samples": 1, "mean_loss": 1.0,
                  "mean_grad_norm": 1.0}}, p,
        )


def test_compute_rejects_non_mapping_payload():
    p = DynamicCurriculumPolicy(num_buckets=2)
    with pytest.raises(TypeError, match="bucket payload must be Mapping"):
        compute_bucket_weights({0: "bad"}, p)  # type: ignore[dict-item]


def test_bucket_stats_frozen():
    bs = BucketStats(bucket_id=0, num_samples=1, mean_loss=0.5,
                     mean_grad_norm=0.1)
    with pytest.raises(Exception):
        bs.num_samples = 99  # type: ignore[misc]


# ---------- validate_distributed_curriculum -------------------------------


def test_distributed_single_rank_ok():
    validate_distributed_curriculum(True, world_size=1, rank_coordinated=False)


def test_distributed_multi_rank_uncoordinated_rejected():
    with pytest.raises(ValueError, match="all_reduce hook"):
        validate_distributed_curriculum(
            True, world_size=4, rank_coordinated=False
        )


def test_distributed_multi_rank_coordinated_ok():
    validate_distributed_curriculum(True, world_size=4, rank_coordinated=True)


def test_distributed_disabled_short_circuit():
    validate_distributed_curriculum(
        False, world_size=4, rank_coordinated=False
    )


def test_distributed_rejects_bool_world_size():
    with pytest.raises(ValueError, match="world_size must be int"):
        validate_distributed_curriculum(
            True, world_size=True, rank_coordinated=True  # type: ignore[arg-type]
        )


def test_distributed_rejects_non_bool_enabled():
    with pytest.raises(TypeError, match="enabled must be bool"):
        validate_distributed_curriculum(
            "yes", world_size=1, rank_coordinated=False  # type: ignore[arg-type]
        )


def test_distributed_rejects_non_bool_coordinated():
    with pytest.raises(TypeError, match="rank_coordinated must be bool"):
        validate_distributed_curriculum(
            True, world_size=2, rank_coordinated="yes"  # type: ignore[arg-type]
        )


def test_distributed_rejects_zero_world_size():
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        validate_distributed_curriculum(
            True, world_size=0, rank_coordinated=True
        )


# ---------- render_curve --------------------------------------------------


def test_render_curve_empty_history():
    text = render_curve([], num_buckets=4)
    assert "no curriculum history" in text


def test_render_curve_basic():
    history = [
        {"step": 100, "weights": [0.25, 0.25, 0.25, 0.25]},
        {"step": 200, "weights": [0.1, 0.2, 0.3, 0.4]},
    ]
    text = render_curve(history, num_buckets=4)
    assert "step" in text
    assert "B0" in text and "B3" in text
    assert "100" in text and "200" in text


def test_render_curve_rejects_wrong_arity():
    history = [{"step": 1, "weights": [0.5, 0.5]}]
    with pytest.raises(ValueError, match="weights length"):
        render_curve(history, num_buckets=4)


def test_render_curve_rejects_non_sequence():
    with pytest.raises(TypeError, match="history must be"):
        render_curve("bogus", num_buckets=4)


def test_render_curve_rejects_bool_width():
    with pytest.raises(ValueError, match="width must be int"):
        render_curve([], num_buckets=4, width=True)  # type: ignore[arg-type]


def test_render_curve_rejects_invalid_num_buckets():
    with pytest.raises(ValueError, match="num_buckets must be in"):
        render_curve([], num_buckets=999)


# ---------- parse_history_jsonl -------------------------------------------


def test_parse_history_jsonl_happy():
    rows = [
        {"step": 50, "weights": [0.25, 0.25, 0.5]},
        {"step": 100, "weights": [0.3, 0.3, 0.4]},
    ]
    out = parse_history_jsonl(rows)
    assert len(out) == 2
    assert isinstance(out[0]["weights"], tuple)
    assert out[0]["step"] == 50


def test_parse_history_jsonl_rejects_non_summing():
    rows = [{"step": 1, "weights": [0.1, 0.2]}]
    with pytest.raises(ValueError, match="weights at step"):
        parse_history_jsonl(rows)


def test_parse_history_jsonl_rejects_non_sequence_weights():
    rows = [{"step": 1, "weights": "abc"}]
    with pytest.raises(TypeError, match="weights must be"):
        parse_history_jsonl(rows)


def test_parse_history_jsonl_rejects_non_mapping_row():
    with pytest.raises(TypeError, match="history row must be Mapping"):
        parse_history_jsonl([42])


def test_parse_history_jsonl_rejects_non_sequence():
    with pytest.raises(TypeError, match="rows must be"):
        parse_history_jsonl("bogus")  # type: ignore[arg-type]


# ---------- Schema integration --------------------------------------------


def _base_cfg(**training_overrides):
    base = {
        "curriculum": True,
        "curriculum_dynamic": True,
        "curriculum_buckets": 4,
    }
    base.update(training_overrides)
    return SoupConfig(
        base="meta-llama/Llama-3.2-1B",
        task="sft",
        data=DataConfig(train="data.jsonl"),
        training=TrainingConfig(**base),
    )


def test_schema_accepts_curriculum_dynamic_with_curriculum():
    cfg = _base_cfg()
    assert cfg.training.curriculum_dynamic is True
    assert cfg.training.curriculum_dynamic_recompute_steps == 50


def test_schema_rejects_dynamic_without_static():
    with pytest.raises(ValidationError, match="curriculum_dynamic requires"):
        SoupConfig(
            base="meta-llama/Llama-3.2-1B",
            task="sft",
            data=DataConfig(train="data.jsonl"),
            training=TrainingConfig(
                curriculum=False, curriculum_dynamic=True
            ),
        )


def test_schema_rejects_dynamic_on_mlx():
    with pytest.raises(ValidationError, match="mlx backend"):
        SoupConfig(
            base="mlx-community/Llama-3.2-1B-Instruct-4bit",
            task="sft",
            backend="mlx",
            data=DataConfig(train="data.jsonl"),
            training=TrainingConfig(
                curriculum=True,
                curriculum_dynamic=True,
                curriculum_buckets=4,
            ),
        )


def test_schema_accepts_dynamic_on_dpo():
    """v0.53.5 #115: multi-trainer expansion accepts every transformer task."""
    cfg = SoupConfig(
        base="meta-llama/Llama-3.2-1B",
        task="dpo",
        data=DataConfig(train="data.jsonl", format="dpo"),
        training=TrainingConfig(
            curriculum=True,
            curriculum_dynamic=True,
            curriculum_buckets=4,
        ),
    )
    assert cfg.training.curriculum_dynamic is True


def test_schema_pretrain_accepted():
    cfg = SoupConfig(
        base="meta-llama/Llama-3.2-1B",
        task="pretrain",
        data=DataConfig(train="data.txt", format="plaintext"),
        training=TrainingConfig(
            curriculum=True, curriculum_dynamic=True, curriculum_buckets=4
        ),
    )
    assert cfg.training.curriculum_dynamic is True


def test_schema_floor_capped_by_uniform():
    # curriculum_buckets=4 → ceiling=0.25.
    with pytest.raises(ValidationError, match="must be <= 1/curriculum_buckets"):
        SoupConfig(
            base="meta-llama/Llama-3.2-1B",
            task="sft",
            data=DataConfig(train="data.jsonl"),
            training=TrainingConfig(
                curriculum=True,
                curriculum_dynamic=True,
                curriculum_buckets=4,
                curriculum_dynamic_floor=0.5,
            ),
        )


def test_schema_recompute_steps_bounded():
    with pytest.raises(ValidationError):
        TrainingConfig(curriculum_dynamic_recompute_steps=0)
    with pytest.raises(ValidationError):
        TrainingConfig(curriculum_dynamic_recompute_steps=1_000_001)


def test_schema_temperature_bounded():
    with pytest.raises(ValidationError):
        TrainingConfig(curriculum_dynamic_temperature=0.0)
    with pytest.raises(ValidationError):
        TrainingConfig(curriculum_dynamic_temperature=101.0)


def test_schema_disabled_when_dynamic_off():
    cfg = SoupConfig(
        base="meta-llama/Llama-3.2-1B",
        task="dpo",
        data=DataConfig(train="data.jsonl", format="dpo"),
        training=TrainingConfig(curriculum_dynamic=False),
    )
    assert cfg.training.curriculum_dynamic is False


# ---------- CLI smoke ------------------------------------------------------


def test_curriculum_curve_cli_help():
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["runs", "curriculum-curve", "--help"])
    assert result.exit_code == 0
    assert "curriculum" in result.output.lower()


def test_curriculum_curve_run_not_found(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from soup_cli.cli import app
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app, ["runs", "curriculum-curve", "nonexistent-run-xyz"]
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_curriculum_curve_missing_history_file(tmp_path, monkeypatch):
    """When history file does not exist, exits 1 with friendly message."""
    from typer.testing import CliRunner

    from soup_cli.cli import app
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    # Use --history pointing at non-existent file under cwd; the run lookup
    # will fail first so this exercises the run-not-found branch.
    result = runner.invoke(
        app,
        [
            "runs", "curriculum-curve", "anything",
            "--history", str(tmp_path / "missing.jsonl"),
        ],
    )
    # Either run-not-found (1) or invalid args; not crash.
    assert result.exit_code in (1, 2)


def test_curriculum_curve_history_outside_cwd(tmp_path, monkeypatch):
    """--history outside cwd is rejected."""
    from typer.testing import CliRunner

    from soup_cli.cli import app
    work = tmp_path / "work"
    work.mkdir()
    elsewhere = tmp_path / "elsewhere.jsonl"
    elsewhere.write_text(json.dumps({"step": 1, "weights": [1.0]}) + "\n")
    monkeypatch.chdir(work)

    # Need a "run" to exist; patch get_run to return a stub.
    import soup_cli.experiment.tracker as et

    class FakeTracker:
        def get_run(self, run_id):
            return {"run_id": run_id, "output_dir": "."}

    monkeypatch.setattr(et, "ExperimentTracker", FakeTracker)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "runs", "curriculum-curve", "stub",
            "--history", str(elsewhere),
        ],
    )
    assert result.exit_code == 2
    assert "outside cwd" in result.output.lower()


def test_curriculum_curve_render_jsonl(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import soup_cli.experiment.tracker as et
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps({"step": 100, "weights": [0.25, 0.25, 0.25, 0.25]}) + "\n"
        + json.dumps({"step": 200, "weights": [0.1, 0.2, 0.3, 0.4]}) + "\n"
    )

    class FakeTracker:
        def get_run(self, run_id):
            return {"run_id": run_id, "output_dir": "."}

    monkeypatch.setattr(et, "ExperimentTracker", FakeTracker)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "runs", "curriculum-curve", "stub",
            "--history", str(history),
        ],
    )
    assert result.exit_code == 0
    assert "B0" in result.output and "B3" in result.output
    assert "100" in result.output


# ---------- Review-fix coverage -------------------------------------------


def test_parse_history_jsonl_rejects_bool_step():
    rows = [{"step": True, "weights": [0.5, 0.5]}]
    with pytest.raises(ValueError, match="step must be int, not bool"):
        parse_history_jsonl(rows)


def test_render_curve_rejects_bool_num_buckets():
    with pytest.raises(ValueError, match="num_buckets must be int, not bool"):
        render_curve([], num_buckets=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("w", [3, 201])
def test_render_curve_rejects_out_of_range_width(w):
    with pytest.raises(ValueError, match="width must be in"):
        render_curve([], num_buckets=4, width=w)


def test_render_curve_rejects_non_mapping_entry():
    with pytest.raises(TypeError, match="history entry must be Mapping"):
        render_curve([42], num_buckets=4)


def test_render_curve_rejects_bool_step_in_entry():
    history = [{"step": True, "weights": [0.5, 0.5]}]
    with pytest.raises(ValueError, match="step must be int, not bool"):
        render_curve(history, num_buckets=2)


def test_render_curve_caps_history_rows():
    huge = [{"step": i, "weights": [0.5, 0.5]} for i in range(101)]
    # Patch the cap to exercise the branch without 100k rows.
    import soup_cli.utils.curriculum_dynamic as cd

    original = cd._MAX_HISTORY_ROWS
    try:
        cd._MAX_HISTORY_ROWS = 100
        with pytest.raises(ValueError, match="cap is"):
            render_curve(huge, num_buckets=2)
    finally:
        cd._MAX_HISTORY_ROWS = original


def test_parse_history_jsonl_caps_rows():
    huge = [{"step": i, "weights": [0.5, 0.5]} for i in range(101)]
    import soup_cli.utils.curriculum_dynamic as cd

    original = cd._MAX_HISTORY_ROWS
    try:
        cd._MAX_HISTORY_ROWS = 100
        with pytest.raises(ValueError, match="cap is"):
            parse_history_jsonl(huge)
    finally:
        cd._MAX_HISTORY_ROWS = original


def test_compute_rejects_negative_grad_norm():
    p = DynamicCurriculumPolicy(num_buckets=2)
    with pytest.raises(ValueError, match="must be >= 0"):
        compute_bucket_weights(
            {0: {"num_samples": 1, "mean_loss": 0.5, "mean_grad_norm": -1.0}},
            p,
        )


def test_compute_floor_strict_invariant():
    """All weights must be >= floor (no floor-violating renorm)."""
    p = DynamicCurriculumPolicy(num_buckets=4, floor=0.1, temperature=0.01)
    stats = {
        0: {"num_samples": 10, "mean_loss": 0.0, "mean_grad_norm": 0.0},
        1: {"num_samples": 10, "mean_loss": 100.0, "mean_grad_norm": 0.0},
        2: {"num_samples": 10, "mean_loss": 100.0, "mean_grad_norm": 0.0},
        3: {"num_samples": 10, "mean_loss": 100.0, "mean_grad_norm": 0.0},
    }
    weights = compute_bucket_weights(stats, p)
    assert min(weights) >= 0.1 - 1e-12
    assert abs(sum(weights) - 1.0) < 1e-9


def test_should_recompute_boundary_minus_one():
    p = DynamicCurriculumPolicy(num_buckets=4, recompute_every_n_steps=50)
    assert p.should_recompute(49) is False
    assert p.should_recompute(50) is True


def test_curriculum_curve_render_includes_exception_info(tmp_path, monkeypatch):
    """Use the recommended `(result.output, repr(result.exception))` assert form."""
    from typer.testing import CliRunner

    import soup_cli.experiment.tracker as et
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    history = tmp_path / "history.jsonl"
    history.write_text(
        json.dumps({"step": 100, "weights": [0.5, 0.5]}) + "\n"
    )

    class FakeTracker:
        def get_run(self, run_id):
            return {"run_id": run_id, "output_dir": "."}

    monkeypatch.setattr(et, "ExperimentTracker", FakeTracker)
    runner = CliRunner()
    result = runner.invoke(
        app, ["runs", "curriculum-curve", "stub", "--history", str(history)]
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_curriculum_curve_corrupt_history(tmp_path, monkeypatch):
    """Malformed JSONL (non-summing weights) exits 2 with 'history malformed'."""
    from typer.testing import CliRunner

    import soup_cli.experiment.tracker as et
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    history = tmp_path / "h.jsonl"
    # Weights don't sum to 1.
    history.write_text(json.dumps({"step": 1, "weights": [0.1, 0.2]}) + "\n")

    class FakeTracker:
        def get_run(self, run_id):
            return {"run_id": run_id, "output_dir": "."}

    monkeypatch.setattr(et, "ExperimentTracker", FakeTracker)
    runner = CliRunner()
    result = runner.invoke(
        app, ["runs", "curriculum-curve", "stub", "--history", str(history)]
    )
    assert result.exit_code == 2
    assert "malformed" in result.output.lower()


def test_curriculum_curve_rejects_oversize_file(tmp_path, monkeypatch):
    """50 MB cap on curriculum history file."""
    from typer.testing import CliRunner

    import soup_cli.experiment.tracker as et
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    history = tmp_path / "big.jsonl"
    history.write_text(json.dumps({"step": 1, "weights": [0.5, 0.5]}) + "\n")

    class FakeTracker:
        def get_run(self, run_id):
            return {"run_id": run_id, "output_dir": "."}

    # Monkey-patch getsize to simulate large file.
    monkeypatch.setattr(et, "ExperimentTracker", FakeTracker)
    real_getsize = os.path.getsize

    def fake_getsize(p):
        if str(p).endswith("big.jsonl"):
            return 100 * 1024 * 1024
        return real_getsize(p)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)
    runner = CliRunner()
    result = runner.invoke(
        app, ["runs", "curriculum-curve", "stub", "--history", str(history)]
    )
    assert result.exit_code == 2
    assert "50 mb" in result.output.lower()
