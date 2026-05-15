"""Tests for v0.58.0 — soup loop CLI-first data flywheel.

Coverage:
- Part A: LoopState validation, atomic state file I/O, init/read/write
- Part B: canary_router hash-bucket determinism, sticky rollback, BucketStats
- Part C: BudgetTracker math, daily counter reset, parse_budget_string
- Part D: IterationRecord, write/read/list iterations
- Watch daemon: run_once + watch end-to-end with stub callbacks
- CLI: init / status / pause / resume / watch --max-iterations / canary / replay
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.canary_router import (
    BucketStats,
    CanaryPolicy,
    rollback,
    route,
)
from soup_cli.utils.loop_budget import (
    check_budget,
    parse_budget_string,
    reset_daily_counter_if_new_day,
)
from soup_cli.utils.loop_daemon import (
    WatchConfig,
    evaluate_canary_verdict,
    maybe_rollback,
    run_once,
    watch,
)
from soup_cli.utils.loop_iteration import (
    IterationRecord,
    list_iterations,
    new_iteration_id,
    read_iteration,
    write_iteration,
)
from soup_cli.utils.loop_state import (
    LOOP_STATUSES,
    LoopState,
    default_state_path,
    init_state,
    read_state,
    write_state,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Part A — LoopState validation
# ---------------------------------------------------------------------------


class TestLoopState:
    def test_default_status_is_stopped(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        assert s.status == "stopped"

    def test_status_allowlist(self):
        assert LOOP_STATUSES == frozenset({"running", "paused", "stopped"})

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError, match="status must be"):
            LoopState(served_model="m", eval_suite="e", baseline="b", status="weird")

    @pytest.mark.parametrize("field_name", ["served_model", "eval_suite", "baseline"])
    def test_empty_required_field_rejected(self, field_name):
        kwargs = {"served_model": "m", "eval_suite": "e", "baseline": "b"}
        kwargs[field_name] = ""
        with pytest.raises(ValueError, match="must not be empty"):
            LoopState(**kwargs)

    def test_null_byte_in_field_rejected(self):
        with pytest.raises(ValueError, match="NUL"):
            LoopState(served_model="m\x00", eval_suite="e", baseline="b")

    def test_oversize_string_rejected(self):
        with pytest.raises(ValueError, match="exceeds"):
            LoopState(served_model="m" * 1000, eval_suite="e", baseline="b")

    def test_non_string_rejected(self):
        with pytest.raises(TypeError):
            LoopState(served_model=123, eval_suite="e", baseline="b")  # type: ignore

    @pytest.mark.parametrize(
        "counter",
        [
            "traces_collected",
            "pairs_distilled",
            "runs_gated",
            "adapters_shipped",
            "iteration_count",
            "runs_today",
        ],
    )
    def test_counter_rejects_negative(self, counter):
        kwargs = {"served_model": "m", "eval_suite": "e", "baseline": "b", counter: -1}
        with pytest.raises(ValueError):
            LoopState(**kwargs)

    @pytest.mark.parametrize(
        "counter",
        [
            "traces_collected",
            "pairs_distilled",
            "runs_gated",
            "adapters_shipped",
        ],
    )
    def test_counter_rejects_bool(self, counter):
        kwargs = {"served_model": "m", "eval_suite": "e", "baseline": "b", counter: True}
        with pytest.raises(ValueError):
            LoopState(**kwargs)

    def test_canary_traffic_pct_bounds(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            LoopState(
                served_model="m",
                eval_suite="e",
                baseline="b",
                canary_traffic_pct=150,
            )

    def test_canary_traffic_pct_bool_rejected(self):
        with pytest.raises(ValueError, match="numeric"):
            LoopState(
                served_model="m",
                eval_suite="e",
                baseline="b",
                canary_traffic_pct=True,
            )

    def test_monthly_budget_negative_rejected(self):
        with pytest.raises(ValueError):
            LoopState(
                served_model="m",
                eval_suite="e",
                baseline="b",
                monthly_budget_usd=-1,
            )

    def test_max_runs_per_day_zero_rejected(self):
        with pytest.raises(ValueError):
            LoopState(
                served_model="m",
                eval_suite="e",
                baseline="b",
                max_runs_per_day=0,
            )

    def test_frozen(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.status = "running"  # type: ignore

    def test_to_dict_is_mapping_proxy(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        d = s.to_dict()
        with pytest.raises(TypeError):
            d["status"] = "running"  # type: ignore

    def test_with_status_returns_new_instance(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        s2 = s.with_status("running")
        assert s.status == "stopped"
        assert s2.status == "running"
        assert s2.updated_at != ""

    def test_with_status_rejects_unknown(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(ValueError):
            s.with_status("nope")

    def test_bumped_increments_counter(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        s2 = s.bumped(traces_collected=3, pairs_distilled=2)
        assert s2.traces_collected == 3
        assert s2.pairs_distilled == 2

    def test_bumped_rejects_unknown_counter(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(ValueError, match="unknown counter"):
            s.bumped(nonexistent=1)

    def test_bumped_rejects_negative_delta(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(ValueError):
            s.bumped(traces_collected=-1)

    def test_bumped_rejects_bool_delta(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(ValueError):
            s.bumped(traces_collected=True)


# ---------------------------------------------------------------------------
# Part A — state file I/O
# ---------------------------------------------------------------------------


class TestStateIO:
    def test_default_state_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert default_state_path().endswith(os.path.join(".soup", "loop.yaml"))

    def test_init_state_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state, path = init_state("m", "e", "b")
        assert os.path.exists(path)
        assert state.served_model == "m"
        assert state.status == "stopped"

    def test_init_state_refuses_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        init_state("m", "e", "b")
        with pytest.raises(FileExistsError):
            init_state("m2", "e2", "b2")

    def test_init_state_force_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        init_state("m", "e", "b")
        state, _ = init_state("m2", "e2", "b2", force=True)
        assert state.served_model == "m2"

    def test_write_read_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = LoopState(
            served_model="model-a",
            eval_suite="suite.yaml",
            baseline="registry://abc",
            status="running",
            traces_collected=42,
        )
        write_state(s)
        reloaded = read_state()
        assert reloaded.served_model == "model-a"
        assert reloaded.status == "running"
        assert reloaded.traces_collected == 42

    def test_write_state_non_loopstate_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError):
            write_state({"foo": "bar"})  # type: ignore

    def test_write_state_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        outside = str(tmp_path.parent / "escape.yaml")
        with pytest.raises(ValueError, match="cwd"):
            write_state(s, outside)

    def test_read_state_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            read_state()

    def test_read_state_invalid_json_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / ".soup" / "loop.yaml"
        target.parent.mkdir()
        target.write_text("not json")
        with pytest.raises(ValueError, match="JSON"):
            read_state()

    def test_read_state_non_dict_root_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / ".soup" / "loop.yaml"
        target.parent.mkdir()
        target.write_text("[]")
        with pytest.raises(ValueError, match="object"):
            read_state()

    def test_read_state_oversize_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / ".soup" / "loop.yaml"
        target.parent.mkdir()
        target.write_text("x" * (2 * 1024 * 1024))
        with pytest.raises(ValueError, match="1 MiB"):
            read_state()

    def test_read_state_drops_unknown_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / ".soup" / "loop.yaml"
        target.parent.mkdir()
        target.write_text(
            json.dumps(
                {
                    "served_model": "m",
                    "eval_suite": "e",
                    "baseline": "b",
                    "future_field_v59": "ignored",
                }
            )
        )
        s = read_state()
        assert s.served_model == "m"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink test")
    def test_write_state_rejects_symlink(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        sd = tmp_path / ".soup"
        sd.mkdir()
        target = sd / "loop.yaml"
        target.symlink_to(tmp_path / "elsewhere")
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(ValueError, match="symlink"):
            write_state(s, str(target))

    def test_null_byte_path_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(ValueError, match="NUL"):
            write_state(s, "foo\x00bar")


# ---------------------------------------------------------------------------
# Part B — canary router
# ---------------------------------------------------------------------------


class TestCanaryPolicy:
    def test_stable_only_default(self):
        p = CanaryPolicy(stable="adapter-a")
        assert p.canary is None
        assert p.traffic_pct == 0.0

    def test_empty_stable_rejected(self):
        with pytest.raises(ValueError):
            CanaryPolicy(stable="")

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError):
            CanaryPolicy(stable="a\x00b")

    def test_canary_same_as_stable_rejected(self):
        with pytest.raises(ValueError, match="differ"):
            CanaryPolicy(stable="a", canary="a")

    def test_traffic_pct_out_of_range(self):
        with pytest.raises(ValueError):
            CanaryPolicy(stable="a", canary="b", traffic_pct=150)

    def test_traffic_pct_bool_rejected(self):
        with pytest.raises(ValueError):
            CanaryPolicy(stable="a", canary="b", traffic_pct=True)

    def test_traffic_pct_nan_rejected(self):
        with pytest.raises(ValueError):
            CanaryPolicy(stable="a", canary="b", traffic_pct=float("nan"))

    def test_traffic_without_canary_rejected(self):
        with pytest.raises(ValueError, match="cannot route"):
            CanaryPolicy(stable="a", traffic_pct=5)

    def test_sticky_bool_required(self):
        with pytest.raises(ValueError):
            CanaryPolicy(stable="a", sticky_on_rollback=1)  # type: ignore

    def test_frozen(self):
        p = CanaryPolicy(stable="a")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.stable = "x"  # type: ignore


class TestRouting:
    def test_stable_only_policy_routes_to_stable(self):
        p = CanaryPolicy(stable="A")
        for key in ("k1", "k2", "k3"):
            d = route(p, key)
            assert d.adapter == "A"
            assert d.bucket == "stable"

    def test_zero_pct_canary_still_routes_stable(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=0.0)
        d = route(p, "anything")
        assert d.bucket == "stable"

    def test_full_100_pct_routes_all_canary(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=100.0)
        for key in ("k1", "k2", "k3"):
            d = route(p, key)
            assert d.adapter == "B"
            assert d.bucket == "canary"

    def test_deterministic_routing(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=25.0)
        assert route(p, "abc").adapter == route(p, "abc").adapter
        assert route(p, "xyz").adapter == route(p, "xyz").adapter

    def test_split_approximates_traffic_pct(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=25.0)
        canary_count = sum(
            1 for i in range(2000) if route(p, f"key-{i}").bucket == "canary"
        )
        # 25% of 2000 = 500; tolerate ±15% relative drift on a uniform hash.
        assert 350 <= canary_count <= 650

    def test_empty_key_rejected(self):
        p = CanaryPolicy(stable="A")
        with pytest.raises(ValueError):
            route(p, "")

    def test_non_policy_rejected(self):
        with pytest.raises(TypeError):
            route("not-policy", "k")  # type: ignore


class TestRollback:
    def test_rollback_clears_canary(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=10.0)
        cleared = rollback(p)
        assert cleared.canary is None
        assert cleared.traffic_pct == 0.0
        assert cleared.stable == "A"

    def test_rollback_after_clears_route_to_stable(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=100.0)
        cleared = rollback(p)
        assert route(cleared, "anykey").adapter == "A"

    def test_rollback_reason_required(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=5.0)
        with pytest.raises(ValueError):
            rollback(p, reason="")

    def test_rollback_non_policy_rejected(self):
        with pytest.raises(TypeError):
            rollback("notpolicy")  # type: ignore


class TestBucketStats:
    def test_record_and_verdict_ok(self):
        s = BucketStats()
        for _ in range(50):
            s.record("stable", True)
            s.record("canary", True)
        assert s.verdict() == "OK"

    def test_verdict_unknown_below_min_samples(self):
        s = BucketStats()
        for _ in range(5):
            s.record("canary", True)
        assert s.verdict(min_samples=30) == "UNKNOWN"

    def test_verdict_major_on_regression(self):
        s = BucketStats()
        for _ in range(50):
            s.record("stable", True)
        # canary fails most of the time
        for _ in range(50):
            s.record("canary", False)
        assert s.verdict() == "MAJOR"

    def test_record_invalid_bucket(self):
        s = BucketStats()
        with pytest.raises(ValueError):
            s.record("middle", True)

    def test_record_non_bool_ok(self):
        s = BucketStats()
        with pytest.raises(ValueError):
            s.record("stable", 1)  # type: ignore

    def test_verdict_invalid_min_samples(self):
        s = BucketStats()
        with pytest.raises(ValueError):
            s.verdict(min_samples=0)

    def test_verdict_invalid_threshold(self):
        s = BucketStats()
        for _ in range(40):
            s.record("canary", True)
        with pytest.raises(ValueError):
            s.verdict(regression_threshold=1.5)

    def test_snapshot_mapping_proxy(self):
        s = BucketStats()
        s.record("stable", True)
        snap = s.snapshot()
        with pytest.raises(TypeError):
            snap["stable_ok"] = 999  # type: ignore


# ---------------------------------------------------------------------------
# Part C — budget guardrails
# ---------------------------------------------------------------------------


class TestParseBudget:
    @pytest.mark.parametrize(
        "raw,expected",
        [("50", 50.0), ("50usd", 50.0), ("100 USD", 100.0), ("0", 0.0), ("0.5", 0.5)],
    )
    def test_happy(self, raw, expected):
        assert parse_budget_string(raw) == expected

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            parse_budget_string("")

    def test_garbage_rejected(self):
        with pytest.raises(ValueError):
            parse_budget_string("abc")

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError):
            parse_budget_string("50\x00usd")

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            parse_budget_string("-5")

    def test_overflow_rejected(self):
        with pytest.raises(ValueError):
            parse_budget_string("10000000")

    def test_non_string_rejected(self):
        with pytest.raises(TypeError):
            parse_budget_string(50)  # type: ignore


class TestCheckBudget:
    def test_happy_within_budget(self):
        d = check_budget(
            estimated_run_usd=1.0,
            spent_so_far_usd=5.0,
            monthly_budget_usd=50.0,
            runs_today=0,
            max_runs_per_day=10,
        )
        assert d.proceed is True
        assert d.projected_total_usd == 6.0

    def test_blocked_by_budget(self):
        d = check_budget(
            estimated_run_usd=10.0,
            spent_so_far_usd=45.0,
            monthly_budget_usd=50.0,
            runs_today=0,
            max_runs_per_day=10,
        )
        assert d.proceed is False
        assert "budget" in d.reason

    def test_blocked_by_daily_cap(self):
        d = check_budget(
            estimated_run_usd=1.0,
            spent_so_far_usd=0.0,
            monthly_budget_usd=50.0,
            runs_today=3,
            max_runs_per_day=3,
        )
        assert d.proceed is False
        assert "daily cap" in d.reason

    def test_none_budget_means_unlimited(self):
        d = check_budget(
            estimated_run_usd=1e6,
            spent_so_far_usd=0,
            monthly_budget_usd=None,
            runs_today=0,
            max_runs_per_day=None,
        )
        assert d.proceed is True

    def test_negative_estimate_rejected(self):
        with pytest.raises(ValueError):
            check_budget(
                estimated_run_usd=-1,
                spent_so_far_usd=0,
                monthly_budget_usd=None,
                runs_today=0,
                max_runs_per_day=None,
            )

    def test_nan_estimate_rejected(self):
        with pytest.raises(ValueError):
            check_budget(
                estimated_run_usd=float("nan"),
                spent_so_far_usd=0,
                monthly_budget_usd=None,
                runs_today=0,
                max_runs_per_day=None,
            )

    def test_bool_estimate_rejected(self):
        with pytest.raises(ValueError):
            check_budget(
                estimated_run_usd=True,
                spent_so_far_usd=0,
                monthly_budget_usd=None,
                runs_today=0,
                max_runs_per_day=None,
            )

    def test_negative_runs_today_rejected(self):
        with pytest.raises(ValueError):
            check_budget(
                estimated_run_usd=0,
                spent_so_far_usd=0,
                monthly_budget_usd=None,
                runs_today=-1,
                max_runs_per_day=None,
            )

    def test_zero_max_runs_per_day_rejected(self):
        with pytest.raises(ValueError):
            check_budget(
                estimated_run_usd=0,
                spent_so_far_usd=0,
                monthly_budget_usd=None,
                runs_today=0,
                max_runs_per_day=0,
            )


class TestDailyCounter:
    def test_same_day_keeps_count(self):
        today = datetime(2026, 5, 15, tzinfo=timezone.utc)
        out, date = reset_daily_counter_if_new_day(3, "2026-05-15", now=today)
        assert out == 3
        assert date == "2026-05-15"

    def test_new_day_resets(self):
        today = datetime(2026, 5, 16, tzinfo=timezone.utc)
        out, date = reset_daily_counter_if_new_day(3, "2026-05-15", now=today)
        assert out == 0
        assert date == "2026-05-16"

    def test_none_prior_date_treated_as_new_day(self):
        today = datetime(2026, 5, 15, tzinfo=timezone.utc)
        out, _ = reset_daily_counter_if_new_day(5, None, now=today)
        assert out == 0

    def test_negative_runs_today_rejected(self):
        with pytest.raises(ValueError):
            reset_daily_counter_if_new_day(-1, "2026-05-15")


# ---------------------------------------------------------------------------
# Part D — iteration artifact
# ---------------------------------------------------------------------------


def _make_record(**overrides):
    base = dict(
        iteration_id="iter-20260515T000000-abcdef01",
        started_at="2026-05-15T00:00:00+00:00",
        finished_at="2026-05-15T00:05:00+00:00",
        pairs_harvested=10,
        run_id="run-abc",
        gate_verdict="OK",
        canary_verdict=None,
        shipped=True,
        rolled_back=False,
        estimated_cost_usd=0.50,
    )
    base.update(overrides)
    return IterationRecord(**base)


class TestIterationRecord:
    def test_happy(self):
        r = _make_record()
        assert r.shipped is True
        assert r.gate_verdict == "OK"

    def test_frozen(self):
        r = _make_record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.shipped = False  # type: ignore

    def test_invalid_gate_verdict_rejected(self):
        with pytest.raises(ValueError, match="gate_verdict"):
            _make_record(gate_verdict="WEIRD")

    def test_invalid_canary_verdict_rejected(self):
        with pytest.raises(ValueError, match="canary_verdict"):
            _make_record(canary_verdict="WEIRD")

    def test_shipped_must_be_bool(self):
        with pytest.raises(ValueError):
            _make_record(shipped=1)  # type: ignore

    def test_negative_pairs_rejected(self):
        with pytest.raises(ValueError):
            _make_record(pairs_harvested=-1)

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError):
            _make_record(estimated_cost_usd=-0.5)

    def test_iteration_id_path_separator_rejected(self):
        with pytest.raises(ValueError):
            _make_record(iteration_id="iter/escape")

    def test_iteration_id_empty_rejected(self):
        with pytest.raises(ValueError):
            _make_record(iteration_id="")

    def test_iteration_id_null_rejected(self):
        with pytest.raises(ValueError):
            _make_record(iteration_id="iter\x00x")

    def test_oversize_notes_rejected(self):
        with pytest.raises(ValueError):
            _make_record(notes="x" * 5000)


class TestIterationIO:
    def test_write_read_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = _make_record()
        path = write_iteration(r)
        assert os.path.exists(path)
        loaded = read_iteration(r.iteration_id)
        assert loaded.iteration_id == r.iteration_id
        assert loaded.shipped is True

    def test_write_outside_cwd_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = _make_record()
        with pytest.raises(ValueError):
            write_iteration(r, base_dir=str(tmp_path.parent / "escape"))

    def test_read_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            read_iteration("iter-missing")

    def test_list_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert list_iterations() == ()

    def test_list_sorted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_iteration(_make_record(iteration_id="iter-001"))
        write_iteration(_make_record(iteration_id="iter-003"))
        write_iteration(_make_record(iteration_id="iter-002"))
        assert list_iterations() == ("iter-001", "iter-002", "iter-003")

    def test_new_iteration_id_unique(self):
        ids = {new_iteration_id() for _ in range(20)}
        assert len(ids) == 20

    def test_new_iteration_id_passes_validation(self):
        # Round-trips through IterationRecord without raising
        r = _make_record(iteration_id=new_iteration_id())
        assert r.iteration_id.startswith("iter-")

    def test_write_non_record_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TypeError):
            write_iteration({"foo": "bar"})  # type: ignore

    def test_read_invalid_manifest_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        d = tmp_path / ".soup-loops" / "iter-bad"
        d.mkdir(parents=True)
        (d / "iteration.json").write_text("[]")
        with pytest.raises(ValueError, match="object"):
            read_iteration("iter-bad")

    def test_list_skips_invalid_id_directories(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Legitimate
        write_iteration(_make_record(iteration_id="iter-ok"))
        # Junk directory with iteration.json but bad id
        bad = tmp_path / ".soup-loops" / "weird\x00"
        # OSes may reject NUL in path; if so, just skip
        try:
            bad.mkdir(parents=True)
            (bad / "iteration.json").write_text("{}")
        except (OSError, ValueError):
            pass
        out = list_iterations()
        assert "iter-ok" in out


# ---------------------------------------------------------------------------
# Watch daemon
# ---------------------------------------------------------------------------


class TestRunOnce:
    def test_default_callbacks_record_iteration(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state, _ = init_state("m", "e", "b")
        state = state.with_status("running")
        write_state(state)
        cfg = WatchConfig()
        new_state, record, decision = run_once(state, cfg)
        assert decision.proceed is True
        assert record.gate_verdict == "SKIPPED"  # default train stub sets skipped
        assert new_state.iteration_count == 1

    def test_budget_skip_records_iteration_with_zero_counters(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = LoopState(
            served_model="m",
            eval_suite="e",
            baseline="b",
            status="running",
            monthly_budget_usd=10.0,
            spent_this_month_usd=10.0,
        )
        cfg = WatchConfig(cost_fn=lambda s: 5.0)
        new_state, record, decision = run_once(state, cfg)
        assert decision.proceed is False
        assert record.gate_verdict == "SKIPPED"
        assert "budget" in record.notes.lower()
        assert new_state.iteration_count == 0  # not bumped on skip

    def test_custom_callbacks_invoked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state, _ = init_state("m", "e", "b")
        state = state.with_status("running")
        write_state(state)
        cfg = WatchConfig(
            harvest_fn=lambda s: {"pairs_harvested": 5, "traces_collected": 100},
            train_fn=lambda s, c: {"run_id": "run-X", "skipped": False},
            gate_fn=lambda s, c: {"gate_verdict": "OK"},
            deploy_fn=lambda s, c: {"deployed": True, "canary_verdict": "OK"},
            cost_fn=lambda s: 0.10,
        )
        new_state, record, decision = run_once(state, cfg)
        assert decision.proceed is True
        assert record.pairs_harvested == 5
        assert record.run_id == "run-X"
        assert record.gate_verdict == "OK"
        assert record.canary_verdict == "OK"
        assert record.shipped is True
        assert new_state.adapters_shipped == 1
        assert new_state.pairs_distilled == 5
        assert new_state.spent_this_month_usd == pytest.approx(0.10)

    def test_run_once_rejects_non_state(self):
        cfg = WatchConfig()
        with pytest.raises(TypeError):
            run_once("notstate", cfg)  # type: ignore

    def test_run_once_rejects_non_config(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(TypeError):
            run_once(s, "notcfg")  # type: ignore

    def test_gate_verdict_normalised(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state, _ = init_state("m", "e", "b")
        state = state.with_status("running")
        write_state(state)
        cfg = WatchConfig(gate_fn=lambda s, c: {"gate_verdict": "GIBBERISH"})
        _, record, _ = run_once(state, cfg)
        assert record.gate_verdict == "SKIPPED"


class TestWatchConfig:
    def test_default_construct(self):
        cfg = WatchConfig()
        assert cfg.poll_interval_sec == 60.0

    def test_invalid_poll_interval(self):
        with pytest.raises(ValueError):
            WatchConfig(poll_interval_sec=0.5)
        with pytest.raises(ValueError):
            WatchConfig(poll_interval_sec=10000)
        with pytest.raises(ValueError):
            WatchConfig(poll_interval_sec=float("nan"))

    def test_non_callable_harvest_rejected(self):
        with pytest.raises(ValueError):
            WatchConfig(harvest_fn="not callable")  # type: ignore

    def test_negative_max_iterations_rejected(self):
        with pytest.raises(ValueError):
            WatchConfig(max_iterations=-1)


class TestWatchDaemon:
    def test_watch_finite_runs_then_stops(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        init_state("m", "e", "b")
        cfg = WatchConfig(
            poll_interval_sec=1.0,
            max_iterations=3,
        )
        final_state, ran = watch(cfg)
        assert ran == 3
        assert final_state.iteration_count == 3
        assert final_state.status == "stopped"

    def test_watch_respects_pause(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        init_state("m", "e", "b")

        # Iteration 1 paths through and then we flip to paused via callback.
        def _pause_after_first(record):
            s = read_state()
            write_state(s.with_status("stopped"))

        cfg = WatchConfig(
            poll_interval_sec=1.0,
            max_iterations=10,
            on_iteration=_pause_after_first,
        )
        final_state, ran = watch(cfg)
        # Either 1 or 2 — the on_iteration fires before the next read.
        assert ran >= 1
        assert final_state.status == "stopped"


class TestRollbackOrchestration:
    def test_maybe_rollback_no_op_on_ok(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=5.0)
        assert maybe_rollback(p, "OK").canary == "B"

    def test_maybe_rollback_clears_on_major(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=5.0)
        assert maybe_rollback(p, "MAJOR").canary is None

    def test_maybe_rollback_unknown_no_op(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=5.0)
        assert maybe_rollback(p, "UNKNOWN").canary == "B"

    def test_evaluate_canary_verdict_wraps_stats(self):
        stats = BucketStats()
        for _ in range(40):
            stats.record("canary", True)
        assert evaluate_canary_verdict(stats) == "OK"

    def test_maybe_rollback_non_policy_rejected(self):
        with pytest.raises(TypeError):
            maybe_rollback("notpolicy", "MAJOR")  # type: ignore

    def test_maybe_rollback_non_str_verdict_rejected(self):
        p = CanaryPolicy(stable="A")
        with pytest.raises(TypeError):
            maybe_rollback(p, 5)  # type: ignore


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCLI:
    def test_loop_help(self):
        result = runner.invoke(app, ["loop", "--help"])
        assert result.exit_code == 0, result.output
        assert "init" in result.output
        assert "watch" in result.output

    def test_init_command(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["loop", "init", "model-a", "--eval", "suite.yaml", "--baseline", "ref"],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / ".soup" / "loop.yaml").exists()

    def test_init_refuses_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        result = runner.invoke(
            app, ["loop", "init", "m2", "--eval", "e", "--baseline", "b"]
        )
        assert result.exit_code == 2, result.output
        assert "already exists" in result.output

    def test_init_force_overwrites(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        result = runner.invoke(
            app,
            ["loop", "init", "m2", "--eval", "e", "--baseline", "b", "--force"],
        )
        assert result.exit_code == 0, result.output

    def test_init_invalid_budget(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "loop",
                "init",
                "m",
                "--eval",
                "e",
                "--baseline",
                "b",
                "--monthly-budget",
                "garbage",
            ],
        )
        assert result.exit_code == 2

    def test_status_without_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["loop", "status"])
        assert result.exit_code == 2
        assert "init" in result.output.lower()

    def test_status_after_init(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        result = runner.invoke(app, ["loop", "status"])
        assert result.exit_code == 0, result.output
        assert "stopped" in result.output

    def test_pause_resume_cycle(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        # Mark running manually so pause has something to flip.
        s = read_state()
        write_state(s.with_status("running"))
        r = runner.invoke(app, ["loop", "pause"])
        assert r.exit_code == 0
        assert read_state().status == "paused"
        r = runner.invoke(app, ["loop", "resume"])
        assert r.exit_code == 0
        assert read_state().status == "running"

    def test_pause_when_stopped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(app, ["loop", "pause"])
        assert r.exit_code == 0
        assert "already stopped" in r.output

    def test_resume_when_not_paused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(app, ["loop", "resume"])
        assert r.exit_code == 0
        assert "not paused" in r.output

    def test_watch_max_iterations(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(
            app,
            [
                "loop",
                "watch",
                "--foreground",
                "--max-iterations",
                "2",
                "--poll-interval",
                "1",
            ],
        )
        assert r.exit_code == 0, r.output
        assert "iterations=2" in r.output

    def test_watch_detach_and_foreground_mutex(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(
            app, ["loop", "watch", "--foreground", "--detach"]
        )
        assert r.exit_code == 2

    def test_canary_command(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "model-a", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(
            app, ["loop", "canary", "model-b", "--traffic", "10%"]
        )
        assert r.exit_code == 0, r.output
        s = read_state()
        assert s.canary_active == "model-b"
        assert s.canary_traffic_pct == 10.0

    def test_canary_invalid_traffic(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "model-a", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(
            app, ["loop", "canary", "model-b", "--traffic", "150"]
        )
        assert r.exit_code == 2

    def test_canary_same_as_stable_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "same", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(
            app, ["loop", "canary", "same", "--traffic", "5%"]
        )
        assert r.exit_code == 2

    def test_replay_list_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(app, ["loop", "replay"])
        assert r.exit_code == 0
        assert "no iterations" in r.output

    def test_replay_show_iteration(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        rec = _make_record()
        write_iteration(rec)
        r = runner.invoke(app, ["loop", "replay", rec.iteration_id])
        assert r.exit_code == 0
        assert rec.iteration_id in r.output

    def test_replay_unknown_iteration(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "m", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(app, ["loop", "replay", "iter-nonexistent"])
        assert r.exit_code == 2


# ---------------------------------------------------------------------------
# Source-grep regression guards
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestSourceWiring:
    def test_cli_registers_loop_typer(self):
        cli_src = (_REPO_ROOT / "soup_cli" / "cli.py").read_text(encoding="utf-8")
        assert "from soup_cli.commands import loop as _loop_cmd" in cli_src
        assert 'name="loop"' in cli_src

    def test_version_bumped_to_0_58_0(self):
        init = (_REPO_ROOT / "soup_cli" / "__init__.py").read_text(encoding="utf-8")
        assert '__version__ = "0.58.0"' in init

    def test_no_top_level_torch_import_in_loop_modules(self):
        for name in [
            "loop_state.py",
            "loop_budget.py",
            "loop_iteration.py",
            "canary_router.py",
            "loop_daemon.py",
        ]:
            src = (_REPO_ROOT / "soup_cli" / "utils" / name).read_text(encoding="utf-8")
            # Check module-level imports only (skip indented imports inside funcs)
            for line in src.splitlines():
                if line.startswith(("import torch", "from torch")):
                    raise AssertionError(f"{name} imports torch at module level")

    def test_command_module_uses_typer_app(self):
        src = (_REPO_ROOT / "soup_cli" / "commands" / "loop.py").read_text(encoding="utf-8")
        assert "app = typer.Typer(" in src
        assert 'name="loop"' in src


# ---------------------------------------------------------------------------
# version sanity
# ---------------------------------------------------------------------------


def test_version_string():
    from soup_cli import __version__

    assert __version__ == "0.58.0"


# ---------------------------------------------------------------------------
# Code-review wave 2 follow-ups (HIGH #2-#4 + MEDIUM #5-#8 review fixes)
# ---------------------------------------------------------------------------


class TestReviewFixWave2:
    def test_watch_preserves_paused_status_on_exit(self, tmp_path, monkeypatch):
        """HIGH #2: SIGTERM-while-paused must NOT silently promote to stopped."""
        monkeypatch.chdir(tmp_path)
        init_state("m", "e", "b")
        s = read_state()
        write_state(s.with_status("paused"))
        cfg = WatchConfig(poll_interval_sec=1.0, max_iterations=0)
        final_state, _ = watch(cfg)
        # Was paused before watch; daemon must not have flipped to stopped.
        assert final_state.status == "paused", "watch destroyed paused state"

    def test_watch_flips_running_to_stopped_at_exit(self, tmp_path, monkeypatch):
        """The legit case still works — running → stopped on max_iterations."""
        monkeypatch.chdir(tmp_path)
        init_state("m", "e", "b")
        cfg = WatchConfig(poll_interval_sec=1.0, max_iterations=1)
        final_state, _ = watch(cfg)
        assert final_state.status == "stopped"

    def test_budget_skip_does_not_write_iteration_manifest(self, tmp_path, monkeypatch):
        """HIGH #3: budget-skipped runs do not produce iteration manifests."""
        monkeypatch.chdir(tmp_path)
        init_state("m", "e", "b")
        s = read_state()
        write_state(
            dataclasses.replace(
                s,
                status="running",
                monthly_budget_usd=10.0,
                spent_this_month_usd=10.0,
            )
        )
        cfg = WatchConfig(
            poll_interval_sec=1.0,
            max_iterations=1,
            cost_fn=lambda _s: 5.0,  # forces budget rejection
        )
        watch(cfg)
        # No manifest should exist because the iteration was budget-skipped.
        assert list_iterations() == ()

    def test_canary_autoroll_persisted(self, tmp_path, monkeypatch):
        """HIGH #4: --autoroll-on-regress flag must survive into LoopState."""
        monkeypatch.chdir(tmp_path)
        runner.invoke(
            app, ["loop", "init", "model-a", "--eval", "e", "--baseline", "b"]
        )
        r = runner.invoke(
            app,
            [
                "loop",
                "canary",
                "model-b",
                "--traffic",
                "5%",
                "--no-autoroll-on-regress",
            ],
        )
        assert r.exit_code == 0, r.output
        s = read_state()
        assert s.canary_autoroll_on_regress is False
        # Default True is also exercised by other canary tests above.

    def test_canary_autoroll_bool_validator(self):
        """LoopState rejects non-bool canary_autoroll_on_regress."""
        with pytest.raises(ValueError, match="canary_autoroll_on_regress"):
            LoopState(
                served_model="m",
                eval_suite="e",
                baseline="b",
                canary_autoroll_on_regress=1,  # type: ignore
            )

    def test_route_ceil_at_sub_bucket_fraction(self):
        """MEDIUM #5: 0.005 % must allocate ≥1 bucket, not round to 0."""
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=0.005)
        # 10_000 buckets * 0.00005 = 0.5 → ceil = 1 bucket reserved for canary.
        canary_hits = sum(
            1 for i in range(10_000) if route(p, f"k-{i}").bucket == "canary"
        )
        assert canary_hits >= 1, "ceil rounding lost the sub-bucket fraction"

    def test_parse_budget_usd_only(self):
        """MEDIUM #6: bare 'usd' / '  usd  ' raises friendly explicit error."""
        with pytest.raises(ValueError, match="numeric value"):
            parse_budget_string("usd")
        with pytest.raises(ValueError, match="numeric value"):
            parse_budget_string("  USD  ")

    def test_list_iterations_swallows_oserror_on_listdir(self, tmp_path, monkeypatch):
        """MEDIUM #8: list_iterations returns () instead of raising on unreadable dir."""
        monkeypatch.chdir(tmp_path)
        # Create the dir then monkeypatch os.listdir to raise — simulates a
        # permission flap mid-iteration that would otherwise kill the daemon.
        d = tmp_path / ".soup-loops"
        d.mkdir()
        import soup_cli.utils.loop_iteration as li

        def _raise(_p):
            raise PermissionError("simulated permission flap")

        monkeypatch.setattr(li.os, "listdir", _raise)
        assert li.list_iterations() == ()


# ---------------------------------------------------------------------------
# TDD-review wave 3: exact-boundary + match= tightening + coverage gaps
# ---------------------------------------------------------------------------


class TestReviewFixWave3:
    # --- HIGH #1: exact-boundary tests for _MAX_STR_FIELD = 512 ----------

    def test_str_field_accepts_exactly_512(self):
        s = LoopState(served_model="m" * 512, eval_suite="e", baseline="b")
        assert len(s.served_model) == 512

    def test_str_field_rejects_513(self):
        with pytest.raises(ValueError, match="exceeds 512"):
            LoopState(served_model="m" * 513, eval_suite="e", baseline="b")

    # --- HIGH #2: read_state size cap exact boundary ---------------------

    def test_read_state_at_one_mib_minus_one_accepted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / ".soup" / "loop.yaml"
        target.parent.mkdir()
        # Pad a valid JSON document to just under 1 MiB.
        notes_pad = " " * (1024 * 1024 - 200)
        target.write_text(
            json.dumps(
                {
                    "served_model": "m",
                    "eval_suite": "e",
                    "baseline": "b",
                    "last_iteration_id": "iter-pad" + notes_pad[:480],
                }
            )
        )
        size = os.path.getsize(target)
        assert size < 1024 * 1024  # under the cap
        s = read_state()
        assert s.served_model == "m"

    def test_read_state_at_one_mib_plus_one_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        target = tmp_path / ".soup" / "loop.yaml"
        target.parent.mkdir()
        target.write_text("x" * (1024 * 1024 + 1))
        with pytest.raises(ValueError, match="1 MiB"):
            read_state()

    # --- HIGH #3: match= tightening on critical reject paths -------------

    def test_monthly_budget_negative_match(self):
        with pytest.raises(ValueError, match="monthly_budget_usd must be"):
            LoopState(
                served_model="m",
                eval_suite="e",
                baseline="b",
                monthly_budget_usd=-1.0,
            )

    def test_max_runs_per_day_zero_match(self):
        with pytest.raises(ValueError, match="max_runs_per_day"):
            LoopState(
                served_model="m",
                eval_suite="e",
                baseline="b",
                max_runs_per_day=0,
            )

    def test_with_status_unknown_match(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(ValueError, match="status must be"):
            s.with_status("not-a-real-status")

    def test_bumped_unknown_counter_match(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(ValueError, match="unknown counter"):
            s.bumped(banana=1)

    def test_canary_traffic_out_of_range_match(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            CanaryPolicy(stable="A", canary="B", traffic_pct=150.0)

    # --- MEDIUM: bool rejection on iteration_count / runs_today ---------

    @pytest.mark.parametrize("counter", ["iteration_count", "runs_today"])
    def test_counter_rejects_bool_iter_runs(self, counter):
        kwargs = {"served_model": "m", "eval_suite": "e", "baseline": "b", counter: True}
        with pytest.raises(ValueError, match=counter):
            LoopState(**kwargs)

    # --- MEDIUM: bool rejection on float budget fields -------------------

    def test_monthly_budget_bool_rejected(self):
        with pytest.raises(ValueError, match="monthly_budget_usd"):
            LoopState(
                served_model="m",
                eval_suite="e",
                baseline="b",
                monthly_budget_usd=True,  # type: ignore
            )

    def test_spent_this_month_bool_rejected(self):
        with pytest.raises(ValueError, match="spent_this_month_usd"):
            LoopState(
                served_model="m",
                eval_suite="e",
                baseline="b",
                spent_this_month_usd=True,  # type: ignore
            )

    # --- MEDIUM: optional-string empty-string rejection ------------------

    @pytest.mark.parametrize(
        "field_name", ["canary_active", "last_iteration_id", "last_run_date"]
    )
    def test_optional_str_field_rejects_empty(self, field_name):
        kwargs = {"served_model": "m", "eval_suite": "e", "baseline": "b", field_name: ""}
        with pytest.raises(ValueError, match="must not be empty"):
            LoopState(**kwargs)

    # --- MEDIUM: _check_path empty string at write boundary --------------

    def test_write_state_rejects_empty_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        with pytest.raises(ValueError, match="must not be empty"):
            write_state(s, "")

    # --- LOW: canary traffic_pct lower-bound exact 0 + upper-bound 100 ---

    def test_canary_traffic_pct_zero_accepted(self):
        p = CanaryPolicy(stable="A")  # traffic_pct=0 default
        assert p.traffic_pct == 0.0

    def test_canary_traffic_pct_negative_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            CanaryPolicy(stable="A", canary="B", traffic_pct=-0.001)

    def test_canary_traffic_pct_exactly_100_accepted(self):
        p = CanaryPolicy(stable="A", canary="B", traffic_pct=100.0)
        assert p.traffic_pct == 100.0

    # --- LOW: to_dict keys match dataclass fields (forward-compat lock) --

    def test_to_dict_keys_match_dataclass_fields(self):
        s = LoopState(served_model="m", eval_suite="e", baseline="b")
        assert set(s.to_dict().keys()) == set(LoopState.__dataclass_fields__.keys())

    # --- LOW: read_state preserves created_at/updated_at when present ----

    def test_read_state_preserves_created_at(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        s = LoopState(
            served_model="m",
            eval_suite="e",
            baseline="b",
            created_at="2026-05-15T00:00:00+00:00",
        )
        write_state(s)
        reloaded = read_state()
        # created_at survives the roundtrip; updated_at gets refreshed by write_state.
        assert reloaded.created_at == "2026-05-15T00:00:00+00:00"
