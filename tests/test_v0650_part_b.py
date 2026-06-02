"""v0.65.0 Part B — Behaviour battery tests.

Closed allowlist over XSTest / HarmBench / JailbreakBench / ELEPHANT /
SycEval; pre/post diff report; ``soup eval behavior`` CLI surface.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.utils.behavior_battery import (
    SUPPORTED_BATTERIES,
    BatterySpec,
    BehaviorDiffReport,
    BehaviorScore,
    classify_behavior_score,
    compute_behavior_diff,
    get_battery_spec,
    list_batteries,
    load_battery_probes,
    validate_battery_name,
)

# ─── Allowlist + spec ───


class TestSupportedBatteries:
    def test_known_set(self):
        assert "xstest" in SUPPORTED_BATTERIES
        assert "harmbench" in SUPPORTED_BATTERIES
        assert "jailbreakbench" in SUPPORTED_BATTERIES
        assert "elephant" in SUPPORTED_BATTERIES
        assert "syceval" in SUPPORTED_BATTERIES

    def test_immutable_frozenset(self):
        assert isinstance(SUPPORTED_BATTERIES, frozenset)
        with pytest.raises(AttributeError):
            SUPPORTED_BATTERIES.add("evil")  # type: ignore[attr-defined]


class TestValidateBatteryName:
    def test_happy(self):
        assert validate_battery_name("xstest") == "xstest"

    def test_case_insensitive(self):
        assert validate_battery_name("XSTEST") == "xstest"

    def test_unknown(self):
        with pytest.raises(ValueError, match="unknown"):
            validate_battery_name("not-a-battery")

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            validate_battery_name("")

    def test_null_byte(self):
        with pytest.raises(ValueError, match="null"):
            validate_battery_name("xstest\x00")

    def test_oversize(self):
        with pytest.raises(ValueError, match="int"):
            validate_battery_name("a" * 33)

    def test_non_string(self):
        with pytest.raises(TypeError):
            validate_battery_name(42)  # type: ignore[arg-type]

    def test_bool(self):
        with pytest.raises(TypeError):
            validate_battery_name(True)  # type: ignore[arg-type]


class TestGetBatterySpec:
    def test_known_returns_spec(self):
        spec = get_battery_spec("xstest")
        assert isinstance(spec, BatterySpec)
        assert spec.name == "xstest"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_battery_spec("not-real")

    def test_frozen(self):
        spec = get_battery_spec("xstest")
        with pytest.raises(Exception):
            spec.name = "x"  # type: ignore[misc]

    def test_list_batteries_returns_sorted(self):
        names = list_batteries()
        assert list(names) == sorted(names)
        assert "xstest" in names


# ─── BehaviorScore ───


class TestBehaviorScore:
    def test_happy(self):
        s = BehaviorScore(
            battery="xstest", value=0.85, verdict="OK", num_probes=100,
        )
        assert s.value == 0.85

    def test_frozen(self):
        s = BehaviorScore(
            battery="xstest", value=0.85, verdict="OK", num_probes=100,
        )
        with pytest.raises(Exception):
            s.value = 0.5  # type: ignore[misc]

    def test_invalid_battery(self):
        with pytest.raises(ValueError, match="battery"):
            BehaviorScore(
                battery="not-real", value=0.5, verdict="OK", num_probes=10,
            )

    def test_invalid_value(self):
        with pytest.raises(ValueError, match="value"):
            BehaviorScore(
                battery="xstest", value=1.5, verdict="OK", num_probes=10,
            )

    def test_nan_value(self):
        with pytest.raises(ValueError, match="finite"):
            BehaviorScore(
                battery="xstest", value=float("nan"), verdict="OK", num_probes=10,
            )

    def test_invalid_verdict(self):
        with pytest.raises(ValueError, match="verdict"):
            BehaviorScore(
                battery="xstest", value=0.5, verdict="BAD", num_probes=10,
            )

    def test_invalid_num_probes(self):
        with pytest.raises(ValueError, match="num_probes"):
            BehaviorScore(
                battery="xstest", value=0.5, verdict="OK", num_probes=-1,
            )

    def test_bool_num_probes(self):
        with pytest.raises(ValueError, match="num_probes"):
            BehaviorScore(
                battery="xstest", value=0.5, verdict="OK",
                num_probes=True,  # type: ignore[arg-type]
            )


# ─── classify_behavior_score ───


class TestClassifyBehaviorScore:
    def test_ok(self):
        assert classify_behavior_score(0.9) == "OK"
        assert classify_behavior_score(0.85) == "OK"

    def test_minor(self):
        assert classify_behavior_score(0.7) == "MINOR"
        assert classify_behavior_score(0.6) == "MINOR"

    def test_major(self):
        assert classify_behavior_score(0.3) == "MAJOR"
        assert classify_behavior_score(0.0) == "MAJOR"

    def test_boundary_ok(self):
        assert classify_behavior_score(0.85) == "OK"

    def test_boundary_minor(self):
        assert classify_behavior_score(0.60) == "MINOR"

    def test_invalid(self):
        with pytest.raises(ValueError, match="value"):
            classify_behavior_score(1.5)

    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            classify_behavior_score(float("nan"))

    def test_bool(self):
        with pytest.raises(ValueError, match="value"):
            classify_behavior_score(True)  # type: ignore[arg-type]


# ─── BehaviorDiffReport ───


class TestBehaviorDiffReport:
    def test_happy(self):
        pre = BehaviorScore(
            battery="xstest", value=0.9, verdict="OK", num_probes=10,
        )
        post = BehaviorScore(
            battery="xstest", value=0.5, verdict="MAJOR", num_probes=10,
        )
        r = BehaviorDiffReport(
            run_id="r1",
            battery="xstest",
            pre=pre,
            post=post,
            delta=-0.4,
            overall="MAJOR",
        )
        assert r.delta == -0.4

    def test_frozen(self):
        pre = BehaviorScore(
            battery="xstest", value=0.9, verdict="OK", num_probes=10,
        )
        post = BehaviorScore(
            battery="xstest", value=0.8, verdict="OK", num_probes=10,
        )
        r = BehaviorDiffReport(
            run_id="r", battery="xstest", pre=pre, post=post,
            delta=-0.1, overall="OK",
        )
        with pytest.raises(Exception):
            r.delta = 0.0  # type: ignore[misc]

    def test_battery_mismatch(self):
        pre = BehaviorScore(
            battery="xstest", value=0.9, verdict="OK", num_probes=10,
        )
        post = BehaviorScore(
            battery="harmbench", value=0.9, verdict="OK", num_probes=10,
        )
        with pytest.raises(ValueError, match="battery"):
            BehaviorDiffReport(
                run_id="r", battery="xstest", pre=pre, post=post,
                delta=0.0, overall="OK",
            )

    def test_invalid_overall(self):
        pre = BehaviorScore(
            battery="xstest", value=0.9, verdict="OK", num_probes=10,
        )
        post = BehaviorScore(
            battery="xstest", value=0.8, verdict="OK", num_probes=10,
        )
        with pytest.raises(ValueError, match="overall"):
            BehaviorDiffReport(
                run_id="r", battery="xstest", pre=pre, post=post,
                delta=-0.1, overall="X",
            )


# ─── compute_behavior_diff ───


class TestComputeBehaviorDiff:
    def test_basic_ok(self):
        pre_responses = ["safe answer"] * 10
        post_responses = ["safe answer"] * 10
        oracle = ["safe"] * 10
        r = compute_behavior_diff(
            run_id="r1", battery="xstest",
            pre_responses=pre_responses,
            post_responses=post_responses,
            oracle=oracle,
        )
        assert r.battery == "xstest"
        # Both pre/post match oracle perfectly -> verdict should be OK.
        assert r.overall == "OK"

    def test_regression_detected(self):
        # Pre: all responses are "safe", post: all responses are "unsafe".
        pre_responses = ["safe"] * 10
        post_responses = ["unsafe"] * 10
        oracle = ["safe"] * 10
        r = compute_behavior_diff(
            run_id="r1", battery="xstest",
            pre_responses=pre_responses,
            post_responses=post_responses,
            oracle=oracle,
        )
        assert r.overall == "MAJOR"
        assert r.delta < 0

    def test_length_mismatch_pre(self):
        with pytest.raises(ValueError, match="length"):
            compute_behavior_diff(
                run_id="r", battery="xstest",
                pre_responses=["a"], post_responses=["a", "b"],
                oracle=["a", "b"],
            )

    def test_length_mismatch_oracle(self):
        with pytest.raises(ValueError, match="length"):
            compute_behavior_diff(
                run_id="r", battery="xstest",
                pre_responses=["a"], post_responses=["a"],
                oracle=["a", "b"],
            )

    def test_unknown_battery(self):
        with pytest.raises(ValueError):
            compute_behavior_diff(
                run_id="r", battery="not-real",
                pre_responses=["x"], post_responses=["x"],
                oracle=["x"],
            )

    def test_empty(self):
        with pytest.raises(ValueError, match="empty"):
            compute_behavior_diff(
                run_id="r", battery="xstest",
                pre_responses=[], post_responses=[], oracle=[],
            )

    def test_bool_responses_list(self):
        with pytest.raises(ValueError):
            compute_behavior_diff(
                run_id="r", battery="xstest",
                pre_responses=[True],  # type: ignore[list-item]
                post_responses=["a"], oracle=["a"],
            )


# ─── load_battery_probes (bundled fixture) ───


class TestLoadBatteryProbes:
    def test_xstest_bundled(self):
        probes = load_battery_probes("xstest")
        assert isinstance(probes, tuple)
        assert len(probes) > 0
        for p in probes:
            assert isinstance(p, dict)
            assert "prompt" in p

    def test_unknown(self):
        with pytest.raises(ValueError):
            load_battery_probes("not-real")


# ─── CLI smoke ───


class TestBehaviorCli:
    def test_help_listed(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "behavior" in result.output.lower()

    def test_behavior_help(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, ["behavior", "--help"])
        assert result.exit_code == 0
        assert "xstest" in result.output.lower() or "battery" in result.output.lower()

    def test_behavior_unknown_battery(self):
        from soup_cli.commands.eval import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "behavior", "test_run",
            "--battery", "evilcorp",
        ])
        assert result.exit_code != 0

    def test_behavior_with_evidence(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        monkeypatch.chdir(tmp_path)
        ev = tmp_path / "ev.json"
        ev.write_text(json.dumps({
            "pre_responses": ["safe"] * 5,
            "post_responses": ["safe"] * 5,
            "oracle": ["safe"] * 5,
        }))
        out = tmp_path / "out.json"
        runner = CliRunner()
        result = runner.invoke(app, [
            "behavior", "test_run",
            "--battery", "xstest",
            "--evidence", str(ev),
            "--output", str(out),
        ])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        data = json.loads(out.read_text())
        assert data["battery"] == "xstest"

    def test_behavior_outside_cwd_evidence(self, tmp_path, monkeypatch):
        from soup_cli.commands.eval import app
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        outside = tmp_path / "ev.json"
        outside.write_text("{}")
        runner = CliRunner()
        result = runner.invoke(app, [
            "behavior", "test_run",
            "--battery", "xstest",
            "--evidence", str(outside),
        ])
        assert result.exit_code != 0
