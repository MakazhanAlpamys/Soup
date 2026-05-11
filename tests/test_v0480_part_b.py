"""Tests for v0.48.0 Part B — Data Mixing Optimizer.

BETA feature. Covers:
- ``parse_budget`` digits + suffix matrix.
- ``validate_datasets`` containment, dedup, symlink, bounds.
- ``build_optimization_plan`` happy path + rejection.
- ``BudgetTracker`` start/elapsed/exceeded semantics.
- ``MixCandidate`` schema, simplex constraint, finite checks.
- ``run_mix_optimizer`` happy / partial / NaN-loss skip / proxy crash.
- ``render_mix_recipe_yaml`` shape + injection defence.
- ``write_mix_recipe`` atomic + TOCTOU symlink reject.
- ``load_mix_recipe`` round-trip.
- ``soup data mix`` CLI smoke (--help, --optimize, --apply, error paths).
"""

from __future__ import annotations

import os

import pytest

from soup_cli.utils.data_mix import (
    BudgetTracker,
    MixCandidate,
    MixOptimizationReport,
    build_optimization_plan,
    load_mix_recipe,
    parse_budget,
    render_mix_recipe_yaml,
    run_mix_optimizer,
    validate_datasets,
    write_mix_recipe,
)

# ---------- parse_budget --------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("60", 60),
        ("60s", 60),
        ("5m", 300),
        ("1h", 3600),
        ("24h", 86400),
    ],
)
def test_parse_budget_happy(raw, expected):
    assert parse_budget(raw) == expected


@pytest.mark.parametrize("raw", ["", "59", "0", "25h", "abc", "10x", "-5m"])
def test_parse_budget_rejects(raw):
    with pytest.raises(ValueError):
        parse_budget(raw)


def test_parse_budget_rejects_null_byte():
    with pytest.raises(ValueError, match="null bytes"):
        parse_budget("1h\x00")


def test_parse_budget_rejects_non_str():
    with pytest.raises(TypeError):
        parse_budget(60)  # type: ignore[arg-type]


# ---------- validate_datasets --------------------------------------------


def _make_files(tmp_path, names):
    for n in names:
        (tmp_path / n).write_text("{}\n")


def test_validate_datasets_happy(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    out = validate_datasets(["a.jsonl", "b.jsonl"])
    assert len(out) == 2
    assert all(os.path.isabs(p) for p in out)


def test_validate_datasets_requires_two(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="at least 2"):
        validate_datasets(["a.jsonl"])


def test_validate_datasets_rejects_empty():
    with pytest.raises(ValueError, match="at least 2"):
        validate_datasets([])


def test_validate_datasets_rejects_non_sequence():
    with pytest.raises(TypeError, match="non-string Sequence"):
        validate_datasets("a.jsonl,b.jsonl")  # type: ignore[arg-type]


def test_validate_datasets_rejects_dedup(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        validate_datasets(["a.jsonl", "a.jsonl"])


def test_validate_datasets_rejects_oversize_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = [f"d{i}.jsonl" for i in range(33)]
    with pytest.raises(ValueError, match="cap is 32"):
        validate_datasets(raw)


def test_validate_datasets_rejects_outside_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    elsewhere = tmp_path / "outside.jsonl"
    elsewhere.write_text("{}")
    monkeypatch.chdir(work)
    with pytest.raises(ValueError, match="outside cwd"):
        validate_datasets([str(elsewhere), "a.jsonl"])


def test_validate_datasets_rejects_null_byte(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="null bytes"):
        validate_datasets(["a\x00.jsonl", "b.jsonl"])


def test_validate_datasets_rejects_non_string(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError, match="dataset must be str"):
        validate_datasets([42, "b.jsonl"])  # type: ignore[list-item]


def test_validate_datasets_rejects_empty_string(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        validate_datasets(["", "b.jsonl"])


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_validate_datasets_rejects_symlink(tmp_path, monkeypatch):
    real = tmp_path / "real.jsonl"
    real.write_text("{}")
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)
    other = tmp_path / "other.jsonl"
    other.write_text("{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        validate_datasets(["link.jsonl", "other.jsonl"])


# ---------- build_optimization_plan --------------------------------------


def test_build_plan_happy(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    plan = build_optimization_plan(["a.jsonl", "b.jsonl"], budget="60s")
    assert plan.num_probes == 8
    assert plan.budget_seconds == 60
    assert plan.seed == 42
    assert len(plan.datasets) == 2


@pytest.mark.parametrize("nb", [0, -1, 257])
def test_build_plan_rejects_invalid_num_probes(tmp_path, monkeypatch, nb):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="num_probes"):
        build_optimization_plan(
            ["a.jsonl", "b.jsonl"], budget="60s", num_probes=nb
        )


def test_build_plan_rejects_bool_num_probes(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="num_probes must be int, not bool"):
        build_optimization_plan(
            ["a.jsonl", "b.jsonl"], budget="60s", num_probes=True  # type: ignore[arg-type]
        )


def test_build_plan_rejects_invalid_seed(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        build_optimization_plan(
            ["a.jsonl", "b.jsonl"], budget="60s", seed=-1
        )


# ---------- BudgetTracker -------------------------------------------------


def test_budget_tracker_basic():
    fake_time = [0.0]

    def clock():
        return fake_time[0]

    tracker = BudgetTracker(60, clock=clock)
    tracker.start()
    assert tracker.elapsed == 0.0
    fake_time[0] = 30.0
    assert tracker.elapsed == 30.0
    assert not tracker.exceeded()
    fake_time[0] = 60.0
    assert tracker.exceeded()


def test_budget_tracker_double_start_raises():
    tracker = BudgetTracker(60)
    tracker.start()
    with pytest.raises(RuntimeError, match="start called twice"):
        tracker.start()


def test_budget_tracker_remaining():
    fake_time = [0.0]
    tracker = BudgetTracker(60, clock=lambda: fake_time[0])
    tracker.start()
    fake_time[0] = 20.0
    assert tracker.remaining == 40.0
    fake_time[0] = 100.0
    assert tracker.remaining == 0.0


def test_budget_tracker_rejects_invalid_budget():
    with pytest.raises(ValueError):
        BudgetTracker(0)
    with pytest.raises(ValueError):
        BudgetTracker(10**10)


def test_budget_tracker_rejects_bool_budget():
    with pytest.raises(ValueError, match="budget_seconds must be int, not bool"):
        BudgetTracker(True)  # type: ignore[arg-type]


def test_budget_tracker_elapsed_before_start_zero():
    tracker = BudgetTracker(60)
    assert tracker.elapsed == 0.0


# ---------- MixCandidate --------------------------------------------------


def test_mix_candidate_happy():
    c = MixCandidate(
        weights=(0.5, 0.5), eval_loss=1.0, wall_clock_seconds=5.0
    )
    assert c.eval_loss == 1.0


def test_mix_candidate_frozen():
    c = MixCandidate(
        weights=(0.5, 0.5), eval_loss=1.0, wall_clock_seconds=5.0
    )
    with pytest.raises(Exception):
        c.eval_loss = 2.0  # type: ignore[misc]


def test_mix_candidate_rejects_non_simplex():
    with pytest.raises(ValueError, match="sum to 1"):
        MixCandidate(
            weights=(0.5, 0.3), eval_loss=1.0, wall_clock_seconds=1.0
        )


def test_mix_candidate_rejects_negative_weight():
    with pytest.raises(ValueError, match="weight must be in"):
        MixCandidate(
            weights=(-0.1, 1.1), eval_loss=1.0, wall_clock_seconds=1.0
        )


def test_mix_candidate_rejects_bool_weight():
    with pytest.raises(ValueError, match="weight must be float, not bool"):
        MixCandidate(
            weights=(True, False), eval_loss=1.0, wall_clock_seconds=1.0  # type: ignore[arg-type]
        )


def test_mix_candidate_rejects_empty_weights():
    with pytest.raises(ValueError, match="weights must be non-empty"):
        MixCandidate(weights=(), eval_loss=1.0, wall_clock_seconds=1.0)


def test_mix_candidate_rejects_non_tuple_weights():
    with pytest.raises(TypeError, match="weights must be tuple"):
        MixCandidate(
            weights=[0.5, 0.5], eval_loss=1.0, wall_clock_seconds=1.0  # type: ignore[arg-type]
        )


def test_mix_candidate_rejects_nan_loss():
    with pytest.raises(ValueError, match="eval_loss must be finite"):
        MixCandidate(
            weights=(0.5, 0.5),
            eval_loss=float("nan"),
            wall_clock_seconds=1.0,
        )


def test_mix_candidate_rejects_negative_wall_clock():
    with pytest.raises(ValueError, match="wall_clock_seconds must be >= 0"):
        MixCandidate(
            weights=(0.5, 0.5), eval_loss=1.0, wall_clock_seconds=-1.0
        )


def test_mix_candidate_rejects_oversize_loss():
    with pytest.raises(ValueError, match="eval_loss exceeds"):
        MixCandidate(
            weights=(0.5, 0.5),
            eval_loss=1e7,
            wall_clock_seconds=1.0,
        )


# ---------- run_mix_optimizer ---------------------------------------------


def _make_plan(tmp_path, monkeypatch, num_probes=4, budget="60s"):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    return build_optimization_plan(
        ["a.jsonl", "b.jsonl"], budget=budget, num_probes=num_probes
    )


def test_run_mix_optimizer_happy(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch, num_probes=4)
    calls = []

    def proxy(weights):
        calls.append(weights)
        return float(weights[0])  # lower loss when first dataset dominant

    report = run_mix_optimizer(plan, proxy)
    assert isinstance(report, MixOptimizationReport)
    assert len(calls) == 4
    assert report.partial is False
    assert len(report.best_weights) == 2
    assert abs(sum(report.best_weights) - 1.0) < 1e-6


def test_run_mix_optimizer_budget_exceeded(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch, num_probes=20)
    fake_time = [0.0]

    def clock():
        # Each probe consumes 30s; budget is 60s → 2 probes max.
        fake_time[0] += 30.0
        return fake_time[0]

    def proxy(weights):
        return 0.5

    report = run_mix_optimizer(plan, proxy, clock=clock)
    assert report.partial is True
    assert len(report.candidates) <= 2


def test_run_mix_optimizer_proxy_crash_skipped(tmp_path, monkeypatch):
    """Proxy exceptions are isolated per-candidate (project pattern)."""
    plan = _make_plan(tmp_path, monkeypatch, num_probes=2)

    def proxy(weights):
        raise RuntimeError("boom")

    report = run_mix_optimizer(plan, proxy)
    # Every candidate raised → no valid observations.
    assert len(report.candidates) == 0
    # Uniform fallback used.
    assert abs(sum(report.best_weights) - 1.0) < 1e-6


def test_run_mix_optimizer_propagates_keyboard_interrupt(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch, num_probes=2)

    def proxy(weights):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_mix_optimizer(plan, proxy)


def test_run_mix_optimizer_skips_nan(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch, num_probes=3)
    seq = iter([float("nan"), 0.5, 0.3])

    def proxy(weights):
        return next(seq)

    report = run_mix_optimizer(plan, proxy)
    # NaN observation skipped → 2 valid candidates recorded.
    assert len(report.candidates) == 2
    assert report.best_eval_loss == 0.3


def test_run_mix_optimizer_rejects_non_float_loss(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch, num_probes=2)

    def proxy(weights):
        return "loss"

    with pytest.raises(TypeError, match="proxy_run must return float"):
        run_mix_optimizer(plan, proxy)


def test_run_mix_optimizer_rejects_bool_loss(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch, num_probes=2)

    def proxy(weights):
        return True

    with pytest.raises(TypeError, match="proxy_run must return float"):
        run_mix_optimizer(plan, proxy)


def test_run_mix_optimizer_rejects_non_plan():
    with pytest.raises(TypeError, match="plan must be MixOptimizationPlan"):
        run_mix_optimizer({}, lambda w: 0.0)  # type: ignore[arg-type]


def test_run_mix_optimizer_rejects_non_callable_proxy(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="proxy_run must be callable"):
        run_mix_optimizer(plan, "not-callable")  # type: ignore[arg-type]


def test_run_mix_optimizer_custom_optimizer(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch, num_probes=2)

    class StubOpt:
        def __init__(self):
            self.told = []

        def ask(self):
            return (0.7, 0.3)

        def tell(self, weights, loss):
            self.told.append((weights, loss))

    opt = StubOpt()

    def proxy(w):
        return 1.0

    report = run_mix_optimizer(plan, proxy, optimizer=opt)
    assert len(opt.told) == 2
    assert report.best_weights == (0.7, 0.3)


def test_run_mix_optimizer_rejects_bad_optimizer(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="OptimizerProtocol"):
        run_mix_optimizer(plan, lambda w: 0.0, optimizer="not-optimizer")  # type: ignore[arg-type]


def test_run_mix_optimizer_all_nan_uniform_fallback(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch, num_probes=2)

    def proxy(w):
        return float("inf")

    report = run_mix_optimizer(plan, proxy)
    assert len(report.best_weights) == 2
    assert abs(sum(report.best_weights) - 1.0) < 1e-6


# ---------- render_mix_recipe_yaml ---------------------------------------


def _report(tmp_path):
    return MixOptimizationReport(
        datasets=(str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")),
        candidates=(),
        best_weights=(0.6, 0.4),
        best_eval_loss=0.123,
        partial=False,
        elapsed_seconds=10.0,
    )


def test_render_recipe_shape(tmp_path):
    text = render_mix_recipe_yaml(_report(tmp_path))
    assert "data:" in text
    assert "interleave:" in text
    assert "strategy: probs" in text
    assert "0.600000" in text
    assert "0.400000" in text
    assert "a.jsonl" in text


def test_render_recipe_rejects_non_report():
    with pytest.raises(TypeError, match="MixOptimizationReport"):
        render_mix_recipe_yaml({})  # type: ignore[arg-type]


def test_render_recipe_rejects_newline_in_path():
    bad = MixOptimizationReport(
        datasets=("a\n.jsonl", "b.jsonl"),
        candidates=(),
        best_weights=(0.5, 0.5),
        best_eval_loss=0.0,
        partial=False,
        elapsed_seconds=0.0,
    )
    with pytest.raises(ValueError, match="control characters"):
        render_mix_recipe_yaml(bad)


def test_render_recipe_rejects_null_byte_path():
    bad = MixOptimizationReport(
        datasets=("a\x00.jsonl", "b.jsonl"),
        candidates=(),
        best_weights=(0.5, 0.5),
        best_eval_loss=0.0,
        partial=False,
        elapsed_seconds=0.0,
    )
    with pytest.raises(ValueError, match="control characters"):
        render_mix_recipe_yaml(bad)


def test_render_recipe_partial_note():
    rep = MixOptimizationReport(
        datasets=("a.jsonl", "b.jsonl"),
        candidates=(),
        best_weights=(0.5, 0.5),
        best_eval_loss=0.5,
        partial=True,
        elapsed_seconds=10.0,
    )
    text = render_mix_recipe_yaml(rep)
    assert "Budget exceeded" in text


def test_render_recipe_handles_inf_loss():
    rep = MixOptimizationReport(
        datasets=("a.jsonl", "b.jsonl"),
        candidates=(),
        best_weights=(0.5, 0.5),
        best_eval_loss=float("inf"),
        partial=False,
        elapsed_seconds=10.0,
    )
    text = render_mix_recipe_yaml(rep)
    assert "no valid observation" in text


# ---------- write_mix_recipe + load_mix_recipe ---------------------------


def test_write_recipe_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = _report(tmp_path)
    out = write_mix_recipe(report, "recipe.yaml")
    assert os.path.isfile(out)
    loaded = load_mix_recipe("recipe.yaml")
    assert "interleave" in loaded
    assert "train" in loaded


def test_write_recipe_outside_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    report = _report(tmp_path)
    with pytest.raises(ValueError, match="outside cwd"):
        write_mix_recipe(report, str(tmp_path / "out.yaml"))


def test_write_recipe_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "exists.yaml"
    target.write_text("# pre")
    with pytest.raises(ValueError, match="already exists"):
        write_mix_recipe(_report(tmp_path), "exists.yaml")


def test_write_recipe_overwrite_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "exists.yaml"
    target.write_text("# pre")
    out = write_mix_recipe(_report(tmp_path), "exists.yaml", overwrite=True)
    text = open(out).read()
    assert "data:" in text


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_write_recipe_symlink_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.yaml"
    real.write_text("# real")
    link = tmp_path / "link.yaml"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        write_mix_recipe(_report(tmp_path), "link.yaml", overwrite=True)


def test_write_recipe_null_byte_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="null bytes"):
        write_mix_recipe(_report(tmp_path), "out\x00.yaml")


def test_write_recipe_non_str(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TypeError):
        write_mix_recipe(_report(tmp_path), 42)  # type: ignore[arg-type]


def test_load_recipe_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_mix_recipe("missing.yaml")


def test_load_recipe_outside_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "out.yaml"
    outside.write_text("data: {interleave: {strategy: probs}}")
    monkeypatch.chdir(work)
    with pytest.raises(ValueError, match="outside cwd"):
        load_mix_recipe(str(outside))


def test_load_recipe_not_yaml_mapping(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "bad.yaml"
    p.write_text("- just a list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_mix_recipe("bad.yaml")


def test_load_recipe_missing_data_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "missing.yaml"
    p.write_text("other: value\n")
    with pytest.raises(ValueError, match="missing required"):
        load_mix_recipe("missing.yaml")


# ---------- CLI smoke ----------------------------------------------------


def test_mix_cli_help():
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["data", "mix", "--help"])
    assert result.exit_code == 0
    assert "optimiz" in result.output.lower() or "optimize" in result.output.lower()


def test_mix_cli_requires_mode():
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["data", "mix"])
    assert result.exit_code == 2


def test_mix_cli_mutual_exclusion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rec.yaml").write_text("data: {}\n")
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["data", "mix", "--optimize", "--apply", "rec.yaml"],
    )
    assert result.exit_code == 2


def test_mix_cli_optimize_requires_datasets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    result = runner.invoke(app, ["data", "mix", "--optimize"])
    assert result.exit_code == 2


def test_mix_cli_optimize_happy(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "data", "mix",
            "--optimize",
            "--datasets", "a.jsonl,b.jsonl",
            "--budget", "60s",
            "--num-probes", "2",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert os.path.isfile(tmp_path / "mix_recipe.yaml")


def test_mix_cli_optimize_invalid_datasets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "data", "mix", "--optimize",
            "--datasets", "missing-and-only-one.jsonl",
            "--budget", "60s",
        ],
    )
    assert result.exit_code == 2
    assert "validation failed" in result.output.lower()


def test_mix_cli_apply_happy(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    # First generate a recipe.
    result = runner.invoke(
        app,
        [
            "data", "mix",
            "--optimize",
            "--datasets", "a.jsonl,b.jsonl",
            "--budget", "60s",
            "--num-probes", "2",
            "--output", "rec.yaml",
        ],
    )
    assert result.exit_code == 0
    # Now apply it.
    result = runner.invoke(app, ["data", "mix", "--apply", "rec.yaml"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "interleave" in result.output


def test_mix_cli_apply_outside_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "rec.yaml"
    outside.write_text("data: {interleave: {strategy: probs}}\n")
    monkeypatch.chdir(work)
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    result = runner.invoke(
        app, ["data", "mix", "--apply", str(outside)]
    )
    assert result.exit_code == 2


def test_optimization_plan_frozen(tmp_path, monkeypatch):
    plan = _make_plan(tmp_path, monkeypatch)
    with pytest.raises(Exception):
        plan.num_probes = 99  # type: ignore[misc]


# ---------- Review-fix coverage -------------------------------------------


def test_optimization_report_frozen(tmp_path):
    rep = MixOptimizationReport(
        datasets=("a.jsonl", "b.jsonl"),
        candidates=(),
        best_weights=(0.5, 0.5),
        best_eval_loss=0.0,
        partial=False,
        elapsed_seconds=0.0,
    )
    with pytest.raises(Exception):
        rep.partial = True  # type: ignore[misc]


def test_build_plan_rejects_seed_at_upper_boundary(tmp_path, monkeypatch):
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="seed must be in"):
        build_optimization_plan(
            ["a.jsonl", "b.jsonl"], budget="60s", seed=2**31
        )


@pytest.mark.parametrize("raw", ["59s", "0s", "0m", "0h"])
def test_parse_budget_rejects_below_min_with_suffix(raw):
    with pytest.raises(ValueError):
        parse_budget(raw)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_load_mix_recipe_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.yaml"
    real.write_text("data: {interleave: {strategy: probs}}\n")
    link = tmp_path / "link.yaml"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        load_mix_recipe("link.yaml")


def test_load_mix_recipe_rejects_oversize(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    big = tmp_path / "big.yaml"
    big.write_bytes(b"data: {x: y}\n" + b"# pad\n" * 200_000)
    with pytest.raises(ValueError, match="bytes cap"):
        load_mix_recipe("big.yaml")


def test_mix_cli_optimize_recipe_content(tmp_path, monkeypatch):
    """Recipe file is non-empty and contains the canonical interleave block."""
    _make_files(tmp_path, ["a.jsonl", "b.jsonl"])
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner

    from soup_cli.cli import app
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "data", "mix", "--optimize",
            "--datasets", "a.jsonl,b.jsonl",
            "--budget", "60s",
            "--num-probes", "2",
            "--output", "rec.yaml",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    text = (tmp_path / "rec.yaml").read_text()
    assert "interleave:" in text
    assert "strategy: probs" in text
    assert "a.jsonl" in text


def test_validate_datasets_single_entry_message(tmp_path, monkeypatch):
    """Single-entry input should fail with 'at least 2' immediately."""
    _make_files(tmp_path, ["a.jsonl"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="at least 2"):
        validate_datasets(["a.jsonl"])
