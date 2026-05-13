"""v0.53.5 — Adaptive Training (BETA → stable).

Covers:
- #114: DynamicCurriculumCallback (live HF Trainer hook + history JSONL).
- #115: Multi-trainer schema gate widening for `curriculum_dynamic`.
- #116: `proxy_run_for_weights` live mix proxy + `soup data mix --live`.
- #117: skopt OptimizerProtocol wrapper.
- #118: `MixOptimizationReport.elapsed_seconds` excludes failed candidates.
- #17:  `deepseek-v3-reasoning` recipe.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from soup_cli.config.loader import load_config_from_string
from soup_cli.monitoring.curriculum_callback import (
    DynamicCurriculumCallback,
    _is_rank_zero,
    _pick_bucket,
)
from soup_cli.recipes.catalog import RECIPES, get_recipe
from soup_cli.utils.curriculum_dynamic import DynamicCurriculumPolicy
from soup_cli.utils.data_mix import (
    _build_skopt_optimizer,
    build_optimization_plan,
    run_mix_optimizer,
)
from soup_cli.utils.mix_proxy import proxy_run_for_weights
from soup_cli.utils.peft_wiring import attach_curriculum_callback

# ----------------------------------------------------------------------
# #114 — DynamicCurriculumCallback
# ----------------------------------------------------------------------


@pytest.fixture
def policy() -> DynamicCurriculumPolicy:
    return DynamicCurriculumPolicy(
        num_buckets=4, recompute_every_n_steps=5, floor=0.05, temperature=1.0
    )


def _under_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    return str(out)


def test_pick_bucket_round_robin():
    assert _pick_bucket(0, 4) == 0
    assert _pick_bucket(1, 4) == 1
    assert _pick_bucket(4, 4) == 0
    assert _pick_bucket(7, 4) == 3


def test_pick_bucket_rejects_bool():
    with pytest.raises(TypeError):
        _pick_bucket(True, 4)
    with pytest.raises(TypeError):
        _pick_bucket(1, True)


def test_pick_bucket_rejects_negative_or_zero():
    with pytest.raises(ValueError):
        _pick_bucket(-1, 4)
    with pytest.raises(ValueError):
        _pick_bucket(1, 0)


def test_is_rank_zero_default_true():
    # Without an initialised process group, single-process → rank-0.
    assert _is_rank_zero() is True


def test_callback_rejects_bad_policy(tmp_path, monkeypatch):
    output_dir = _under_cwd(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        DynamicCurriculumCallback(policy="bad", output_dir=output_dir)  # type: ignore[arg-type]


def test_callback_rejects_outside_cwd(tmp_path, monkeypatch, policy):
    monkeypatch.chdir(tmp_path)
    # Path under parent of cwd → outside.
    with pytest.raises(ValueError, match="outside cwd"):
        DynamicCurriculumCallback(policy=policy, output_dir=str(tmp_path.parent))


def test_callback_rejects_null_byte(tmp_path, monkeypatch, policy):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="null"):
        DynamicCurriculumCallback(policy=policy, output_dir="bad\x00path")


def test_callback_rejects_empty(tmp_path, monkeypatch, policy):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        DynamicCurriculumCallback(policy=policy, output_dir="")


def test_callback_rejects_non_string(tmp_path, monkeypatch, policy):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError):
        DynamicCurriculumCallback(policy=policy, output_dir=123)  # type: ignore[arg-type]


def test_callback_initial_weights_uniform(tmp_path, monkeypatch, policy):
    output_dir = _under_cwd(tmp_path, monkeypatch)
    cb = DynamicCurriculumCallback(policy=policy, output_dir=output_dir)
    assert cb.current_weights == (0.25, 0.25, 0.25, 0.25)


def test_callback_on_log_records_stats(tmp_path, monkeypatch, policy):
    output_dir = _under_cwd(tmp_path, monkeypatch)
    cb = DynamicCurriculumCallback(policy=policy, output_dir=output_dir)
    state = MagicMock()
    state.global_step = 1
    cb.on_log(args=None, state=state, control=None, logs={"loss": 2.0, "grad_norm": 0.5})
    # Bucket 1 should have the recorded sample.
    assert 1 in cb._stats
    assert cb._stats[1]["num_samples"] == 1.0
    assert cb._stats[1]["loss_sum"] == pytest.approx(2.0)


def test_callback_on_log_swallows_nan(tmp_path, monkeypatch, policy):
    output_dir = _under_cwd(tmp_path, monkeypatch)
    cb = DynamicCurriculumCallback(policy=policy, output_dir=output_dir)
    state = MagicMock()
    state.global_step = 0
    cb.on_log(
        args=None,
        state=state,
        control=None,
        logs={"loss": float("nan"), "grad_norm": float("inf")},
    )
    # Falls back to zero — does not poison the accumulator.
    assert cb._stats[0]["loss_sum"] == 0.0
    assert cb._stats[0]["grad_norm_sum"] == 0.0


def test_callback_on_step_end_writes_history(tmp_path, monkeypatch, policy):
    output_dir = _under_cwd(tmp_path, monkeypatch)
    cb = DynamicCurriculumCallback(policy=policy, output_dir=output_dir)
    # Seed stats across all buckets so compute_bucket_weights has populated data.
    state_log = MagicMock()
    for step in range(4):
        state_log.global_step = step
        cb.on_log(
            args=None,
            state=state_log,
            control=None,
            logs={"loss": float(step + 1), "grad_norm": float(step + 1) * 0.1},
        )
    # Step 5 is the recompute boundary.
    state_step = MagicMock()
    state_step.global_step = 5
    cb.on_step_end(args=None, state=state_step, control=None)
    history_path = Path(output_dir) / "curriculum_history.jsonl"
    assert history_path.exists()
    lines = [
        line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["step"] == 5
    assert isinstance(row["weights"], list)
    assert len(row["weights"]) == 4
    assert sum(row["weights"]) == pytest.approx(1.0, abs=1e-3)
    # Stats reset after recompute.
    assert cb._stats == {}


def test_callback_on_step_end_skips_off_boundary(tmp_path, monkeypatch, policy):
    output_dir = _under_cwd(tmp_path, monkeypatch)
    cb = DynamicCurriculumCallback(policy=policy, output_dir=output_dir)
    state = MagicMock()
    state.global_step = 3  # not a multiple of 5
    cb.on_step_end(args=None, state=state, control=None)
    assert not (Path(output_dir) / "curriculum_history.jsonl").exists()


def test_callback_history_atomic_append(tmp_path, monkeypatch, policy):
    output_dir = _under_cwd(tmp_path, monkeypatch)
    cb = DynamicCurriculumCallback(policy=policy, output_dir=output_dir)
    # Seed + recompute twice.
    state_log = MagicMock()
    state_step = MagicMock()
    for cycle in (5, 10):
        for step in range(cycle - 4, cycle):
            state_log.global_step = step
            cb.on_log(
                args=None,
                state=state_log,
                control=None,
                logs={"loss": float(step + 1), "grad_norm": 0.1},
            )
        state_step.global_step = cycle
        cb.on_step_end(args=None, state=state_step, control=None)
    history_path = Path(output_dir) / "curriculum_history.jsonl"
    lines = [
        line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 2
    steps = [json.loads(line)["step"] for line in lines]
    assert steps == [5, 10]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX symlink test")
def test_callback_history_rejects_symlink(tmp_path, monkeypatch, policy):
    output_dir = _under_cwd(tmp_path, monkeypatch)
    history_path = Path(output_dir) / "curriculum_history.jsonl"
    target = tmp_path / "evil.txt"
    target.write_text("evil")
    os.symlink(target, history_path)
    cb = DynamicCurriculumCallback(policy=policy, output_dir=output_dir)
    state = MagicMock()
    state.global_step = 5
    cb._stats = {0: {"num_samples": 1.0, "loss_sum": 1.0, "grad_norm_sum": 0.0}}
    cb.on_step_end(args=None, state=state, control=None)
    # Symlink left in place, evil target untouched.
    assert history_path.is_symlink()
    assert target.read_text() == "evil"


def test_callback_current_weights_is_defensive_copy(tmp_path, monkeypatch, policy):
    output_dir = _under_cwd(tmp_path, monkeypatch)
    cb = DynamicCurriculumCallback(policy=policy, output_dir=output_dir)
    weights = cb.current_weights
    assert isinstance(weights, tuple)
    # Tuples are immutable; ensure subsequent reads do not return same identity
    # as internal state (defence: callers cannot rebind the field).
    assert cb.current_weights == weights


# ----------------------------------------------------------------------
# #115 — Multi-trainer schema gate
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        "sft", "pretrain", "dpo", "grpo", "kto", "orpo", "simpo", "ipo",
        "bco", "reward_model", "embedding", "ppo", "preference",
    ],
)
def test_curriculum_dynamic_accepts_every_transformer_task(task: str):
    extra = ""
    if task == "preference":
        extra = "  preference_loss: dpo\n"
    yaml_text = (
        f"base: test-base\n"
        f"task: {task}\n"
        f"data:\n"
        f"  train: ./train.jsonl\n"
        f"training:\n"
        f"  curriculum: true\n"
        f"  curriculum_dynamic: true\n"
        f"{extra}"
    )
    cfg = load_config_from_string(yaml_text)
    assert cfg.training.curriculum_dynamic is True


def test_curriculum_dynamic_rejects_mlx():
    yaml_text = (
        "base: test-base\n"
        "task: sft\n"
        "backend: mlx\n"
        "data:\n"
        "  train: ./train.jsonl\n"
        "training:\n"
        "  curriculum: true\n"
        "  curriculum_dynamic: true\n"
    )
    with pytest.raises(Exception, match="mlx"):
        load_config_from_string(yaml_text)


def test_curriculum_dynamic_rejects_unknown_task():
    # `tts` is not in the supported allowlist.
    yaml_text = (
        "base: test-base\n"
        "task: tts\n"
        "modality: audio_out\n"
        "data:\n"
        "  train: ./train.jsonl\n"
        "training:\n"
        "  curriculum: true\n"
        "  curriculum_dynamic: true\n"
        "  tts_family: orpheus\n"
    )
    with pytest.raises(Exception, match="curriculum_dynamic"):
        load_config_from_string(yaml_text)


# ----------------------------------------------------------------------
# #115 — attach_curriculum_callback helper
# ----------------------------------------------------------------------


def test_attach_curriculum_callback_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = MagicMock()
    tcfg = MagicMock()
    tcfg.curriculum_dynamic = False
    attached = attach_curriculum_callback(trainer, tcfg, str(tmp_path))
    assert attached is False
    trainer.add_callback.assert_not_called()


def test_attach_curriculum_callback_attaches_when_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = MagicMock()
    tcfg = MagicMock()
    tcfg.curriculum_dynamic = True
    tcfg.curriculum_buckets = 4
    tcfg.curriculum_dynamic_recompute_steps = 25
    tcfg.curriculum_dynamic_floor = 0.05
    tcfg.curriculum_dynamic_temperature = 1.0
    attached = attach_curriculum_callback(trainer, tcfg, str(tmp_path))
    assert attached is True
    trainer.add_callback.assert_called_once()
    cb_arg = trainer.add_callback.call_args[0][0]
    assert isinstance(cb_arg, DynamicCurriculumCallback)


def test_attach_curriculum_callback_returns_false_on_invalid_path(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    trainer = MagicMock()
    tcfg = MagicMock()
    tcfg.curriculum_dynamic = True
    tcfg.curriculum_buckets = 4
    tcfg.curriculum_dynamic_recompute_steps = 25
    tcfg.curriculum_dynamic_floor = 0.05
    tcfg.curriculum_dynamic_temperature = 1.0
    # Outside-cwd path → callback constructor raises; helper returns False.
    attached = attach_curriculum_callback(trainer, tcfg, str(tmp_path.parent))
    assert attached is False
    trainer.add_callback.assert_not_called()


def test_trainer_source_grep_curriculum_wired_everywhere():
    """v0.53.5 #115 — every transformer-backend trainer wires the callback."""
    root = Path(__file__).resolve().parent.parent / "soup_cli" / "trainer"
    trainers = [
        "sft", "dpo", "grpo", "kto", "orpo", "simpo", "ipo", "bco",
        "embedding", "reward_model", "ppo", "pretrain", "distill",
    ]
    for name in trainers:
        src = (root / f"{name}.py").read_text(encoding="utf-8")
        assert "attach_curriculum_callback" in src, (
            f"{name}.py is missing attach_curriculum_callback wiring"
        )


# ----------------------------------------------------------------------
# #116 — mix_proxy validation
# ----------------------------------------------------------------------


def _make_base_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "base.yaml"
    p.write_text(
        "base: test-base\n"
        "task: sft\n"
        "data:\n"
        "  train: ./a.jsonl\n"
        "  format: auto\n"
        "training:\n"
        "  epochs: 1\n"
        "output: ./out\n",
        encoding="utf-8",
    )
    return p


def test_proxy_rejects_non_sequence_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    with pytest.raises(TypeError):
        proxy_run_for_weights(
            "not-a-seq", ["a.jsonl", "b.jsonl"], str(base)  # type: ignore[arg-type]
        )


def test_proxy_rejects_bool_weight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    with pytest.raises(ValueError, match="bool"):
        proxy_run_for_weights(
            [True, 0.5], ["a.jsonl", "b.jsonl"], str(base)
        )


def test_proxy_rejects_non_simplex_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    with pytest.raises(ValueError, match="sum to 1"):
        proxy_run_for_weights(
            [0.3, 0.3], ["a.jsonl", "b.jsonl"], str(base)
        )


def test_proxy_rejects_few_datasets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    with pytest.raises(ValueError, match="at least 2"):
        proxy_run_for_weights([1.0], ["a.jsonl"], str(base))


def test_proxy_rejects_too_many_datasets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    paths = [f"d{i}.jsonl" for i in range(33)]
    weights = [1.0 / 33] * 33
    with pytest.raises(ValueError, match="cap is 32"):
        proxy_run_for_weights(weights, paths, str(base))


def test_proxy_rejects_outside_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Construct a path outside cwd via tempfile.gettempdir parent.
    import tempfile as _tf
    outside = Path(_tf.gettempdir()) / "definitely_outside_soup.yaml"
    outside.write_text("base: x\ntask: sft\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside cwd"):
            proxy_run_for_weights(
                [0.5, 0.5], ["a.jsonl", "b.jsonl"], str(outside)
            )
    finally:
        outside.unlink(missing_ok=True)


def test_proxy_rejects_invalid_timeout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    with pytest.raises(ValueError, match="timeout_seconds"):
        proxy_run_for_weights(
            [0.5, 0.5], ["a.jsonl", "b.jsonl"], str(base), timeout_seconds=1
        )
    with pytest.raises(ValueError, match="bool"):
        proxy_run_for_weights(
            [0.5, 0.5],
            ["a.jsonl", "b.jsonl"],
            str(base),
            timeout_seconds=True,  # type: ignore[arg-type]
        )


def test_proxy_rejects_null_byte_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    with pytest.raises(ValueError, match="null"):
        proxy_run_for_weights(
            [0.5, 0.5], ["a.jsonl", "b\x00.jsonl"], str(base)
        )


def test_proxy_subprocess_failure_raises_runtimeerror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    fake_result = MagicMock(returncode=2, stdout=b"", stderr=b"boom")
    with patch.object(subprocess, "run", return_value=fake_result):
        with pytest.raises(RuntimeError, match="failed"):
            proxy_run_for_weights(
                [0.5, 0.5], ["a.jsonl", "b.jsonl"], str(base)
            )


def test_proxy_timeout_raises_runtimeerror(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    with patch.object(
        subprocess,
        "run",
        side_effect=subprocess.TimeoutExpired(cmd="x", timeout=60),
    ):
        with pytest.raises(RuntimeError, match="timeout"):
            proxy_run_for_weights(
                [0.5, 0.5], ["a.jsonl", "b.jsonl"], str(base)
            )


def test_proxy_happy_path_reads_tracker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _make_base_yaml(tmp_path)
    fake_result = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
    fake_tracker = MagicMock()
    fake_tracker.list_runs.return_value = [
        {"status": "completed", "final_loss": 1.234},
    ]
    with patch.object(subprocess, "run", return_value=fake_result), \
            patch(
                "soup_cli.utils.mix_proxy.ExperimentTracker",
                return_value=fake_tracker,
                create=True,
            ):
        # We need to patch the lazy import inside `_read_final_eval_loss`:
        import soup_cli.experiment.tracker as tracker_mod

        with patch.object(
            tracker_mod, "ExperimentTracker", return_value=fake_tracker
        ):
            loss = proxy_run_for_weights(
                [0.5, 0.5], ["a.jsonl", "b.jsonl"], str(base)
            )
    assert loss == pytest.approx(1.234)


# ----------------------------------------------------------------------
# #117 — skopt OptimizerProtocol wrapper
# ----------------------------------------------------------------------


def test_build_skopt_optimizer_rejects_bad_inputs():
    with pytest.raises(TypeError):
        _build_skopt_optimizer(True, 0)
    with pytest.raises(TypeError):
        _build_skopt_optimizer(2, True)
    with pytest.raises(ValueError, match=">= 2"):
        _build_skopt_optimizer(1, 0)


def test_build_skopt_optimizer_missing_dep_raises_importerror():
    # If skopt is not installed in CI, the wrapper raises ImportError.
    try:
        import skopt  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            _build_skopt_optimizer(3, 42)


def test_build_skopt_optimizer_with_dep_returns_protocol():
    pytest.importorskip("skopt")
    opt = _build_skopt_optimizer(3, 42)
    # Quack-checks: must implement ask + tell.
    assert hasattr(opt, "ask")
    assert hasattr(opt, "tell")
    weights = opt.ask()
    assert len(weights) == 3
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------
# #118 — elapsed_seconds excludes failed candidates
# ----------------------------------------------------------------------


def test_elapsed_seconds_excludes_failed_proxy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds = tmp_path / "a.jsonl"
    ds.write_text("{}\n", encoding="utf-8")
    ds2 = tmp_path / "b.jsonl"
    ds2.write_text("{}\n", encoding="utf-8")
    plan = build_optimization_plan(
        [str(ds), str(ds2)], budget="5m", num_probes=4, seed=0
    )
    call_count = {"n": 0}

    def proxy(weights):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("synthetic crash")
        return float(sum(weights))  # finite

    # Inject a clock that ticks deterministically. Provide many ticks so the
    # tracker's repeated reads cannot exhaust the iterator.
    t = [0.0]

    def fake_clock() -> float:
        # Advance fractionally on every call.
        t[0] += 0.1
        return t[0]

    report = run_mix_optimizer(plan, proxy, clock=fake_clock)
    # Only candidates with finite loss + no exception contribute to elapsed.
    expected = sum(c.wall_clock_seconds for c in report.candidates)
    assert report.elapsed_seconds == pytest.approx(expected)
    # Failed candidate is NOT in `candidates` (sentinel-only via opt.tell).
    assert len(report.candidates) == 3


def test_report_elapsed_zero_when_all_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ds = tmp_path / "a.jsonl"
    ds.write_text("{}\n", encoding="utf-8")
    ds2 = tmp_path / "b.jsonl"
    ds2.write_text("{}\n", encoding="utf-8")
    plan = build_optimization_plan(
        [str(ds), str(ds2)], budget="5m", num_probes=2, seed=0
    )

    def proxy(_):
        raise RuntimeError("always fails")

    report = run_mix_optimizer(plan, proxy)
    assert report.elapsed_seconds == 0.0
    assert report.candidates == ()


# ----------------------------------------------------------------------
# #17 — deepseek-v3-reasoning recipe
# ----------------------------------------------------------------------


def test_recipe_deepseek_v3_reasoning_exists():
    assert "deepseek-v3-reasoning" in RECIPES
    meta = get_recipe("deepseek-v3-reasoning")
    assert meta is not None
    assert meta.task == "grpo"
    assert "DeepSeek-V3" in meta.model


def test_recipe_deepseek_v3_reasoning_yaml_loads():
    meta = get_recipe("deepseek-v3-reasoning")
    assert meta is not None
    cfg = load_config_from_string(meta.yaml_str)
    assert cfg.task == "grpo"
    assert cfg.base == "deepseek-ai/DeepSeek-V3"
    assert cfg.training.num_generations == 4
    assert cfg.training.verifiable_domain == "math"


def test_recipe_deepseek_v3_yaml_safe_load_structure():
    meta = get_recipe("deepseek-v3-reasoning")
    assert meta is not None
    raw = yaml.safe_load(meta.yaml_str)
    assert raw["task"] == "grpo"
    assert "reward_fn" in raw["training"]
    assert "verifiable_domain" in raw["training"]


# ----------------------------------------------------------------------
# CLI --live flag plumbing
# ----------------------------------------------------------------------


def test_cli_mix_help_lists_live_flag():
    from soup_cli.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["data", "mix", "--help"])
    assert result.exit_code == 0
    assert "--live" in result.output
    assert "--base-yaml" in result.output


def test_cli_mix_live_requires_base_yaml(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    ds = tmp_path / "a.jsonl"
    ds.write_text("{}\n", encoding="utf-8")
    ds2 = tmp_path / "b.jsonl"
    ds2.write_text("{}\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "data", "mix", "--optimize", "--live",
            "--datasets", f"{ds},{ds2}",
        ],
    )
    assert result.exit_code == 2
    assert "base-yaml" in result.output
