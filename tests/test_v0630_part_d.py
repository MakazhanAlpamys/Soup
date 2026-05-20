"""v0.63.0 Part D — soup ab mSPRT A/B harness tests."""

from __future__ import annotations

import dataclasses
import json

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_module_imports():
    from soup_cli.utils import ab_test

    assert hasattr(ab_test, "MsprtConfig")
    assert hasattr(ab_test, "MsprtVerdict")
    assert hasattr(ab_test, "msprt_step")
    assert hasattr(ab_test, "run_msprt")
    assert hasattr(ab_test, "validate_metric_name")


# ---------------------------------------------------------------------------
# Metric allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", ["latency", "judge_score", "retry_rate"])
def test_validate_metric_name_happy(m):
    from soup_cli.utils.ab_test import validate_metric_name

    assert validate_metric_name(m) == m


def test_validate_metric_name_case_insensitive():
    from soup_cli.utils.ab_test import validate_metric_name

    assert validate_metric_name("LATENCY") == "latency"


@pytest.mark.parametrize("bad", [None, 1, True, "", "x" * 33, "ban\x00ana", "unknown"])
def test_validate_metric_name_rejects(bad):
    from soup_cli.utils.ab_test import validate_metric_name

    with pytest.raises((TypeError, ValueError)):
        validate_metric_name(bad)


# ---------------------------------------------------------------------------
# MsprtConfig
# ---------------------------------------------------------------------------


def test_msprt_config_defaults():
    from soup_cli.utils.ab_test import MsprtConfig

    cfg = MsprtConfig(metric="latency")
    assert cfg.alpha == 0.05
    assert cfg.beta == 0.20
    assert cfg.effect_size > 0


def test_msprt_config_frozen():
    from soup_cli.utils.ab_test import MsprtConfig

    cfg = MsprtConfig(metric="judge_score")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.alpha = 0.1  # type: ignore[misc]


@pytest.mark.parametrize("field,bad", [
    ("alpha", 0.0),
    ("alpha", 1.0),
    ("alpha", -0.1),
    ("alpha", float("nan")),
    ("alpha", True),
    ("beta", 0.0),
    ("beta", 1.0),
    ("beta", float("inf")),
    ("effect_size", 0.0),
    ("effect_size", -0.1),
    ("effect_size", float("nan")),
])
def test_msprt_config_rejects(field, bad):
    from soup_cli.utils.ab_test import MsprtConfig

    kwargs = {"metric": "latency", field: bad}
    with pytest.raises((TypeError, ValueError)):
        MsprtConfig(**kwargs)


# ---------------------------------------------------------------------------
# msprt_step + run_msprt
# ---------------------------------------------------------------------------


def test_msprt_step_returns_verdict():
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    cfg = MsprtConfig(metric="latency")
    verdict = msprt_step(cfg, control=[1.0, 1.1, 0.9], treatment=[2.0, 2.1, 1.9])
    assert verdict.decision in ("continue", "reject_h0", "accept_h0")
    assert verdict.n_control == 3
    assert verdict.n_treatment == 3


def test_msprt_step_rejects_h0_on_huge_effect():
    """When treatment is dramatically different (with realistic noise), reject H0.

    Both arms carry a tiny amount of variability — zero-variance arms hit
    the degenerate-pooled-var fall-through which always returns ``continue``
    (proper SPRT behaviour: with no observed noise the test cannot bound
    Type-I error honestly).
    """
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    cfg = MsprtConfig(metric="latency", effect_size=0.1)
    control = [1.0 + 0.01 * (i % 5) for i in range(50)]  # ~uniform spread
    treatment = [10.0 + 0.01 * (i % 5) for i in range(50)]
    verdict = msprt_step(cfg, control=control, treatment=treatment)
    assert verdict.decision == "reject_h0"


def test_msprt_step_continues_with_tiny_samples():
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    cfg = MsprtConfig(metric="latency")
    verdict = msprt_step(cfg, control=[1.0], treatment=[1.01])
    assert verdict.decision == "continue"


def test_msprt_step_empty_continues():
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    cfg = MsprtConfig(metric="latency")
    verdict = msprt_step(cfg, control=[], treatment=[])
    assert verdict.decision == "continue"


def test_msprt_step_rejects_non_list():
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    cfg = MsprtConfig(metric="latency")
    with pytest.raises(TypeError):
        msprt_step(cfg, control="not a list", treatment=[1.0])  # type: ignore[arg-type]


def test_msprt_step_rejects_non_finite_value():
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    cfg = MsprtConfig(metric="latency")
    with pytest.raises(ValueError):
        msprt_step(cfg, control=[1.0, float("nan")], treatment=[1.0])
    with pytest.raises(ValueError):
        msprt_step(cfg, control=[1.0, float("inf")], treatment=[1.0])


def test_msprt_step_rejects_bool_value():
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    cfg = MsprtConfig(metric="latency")
    with pytest.raises(TypeError):
        msprt_step(cfg, control=[True, False], treatment=[1.0])


def test_msprt_step_caps_samples():
    """Internal cap prevents OOM on tampered data."""
    from soup_cli.utils.ab_test import MsprtConfig, msprt_step

    cfg = MsprtConfig(metric="latency")
    # 1M samples each side - should not blow up
    huge = [1.0] * 50_000
    verdict = msprt_step(cfg, control=huge, treatment=huge)
    assert verdict.n_control > 0


# ---------------------------------------------------------------------------
# MsprtVerdict
# ---------------------------------------------------------------------------


def test_msprt_verdict_frozen():
    from soup_cli.utils.ab_test import MsprtVerdict

    v = MsprtVerdict(
        decision="continue",
        log_likelihood_ratio=0.5,
        n_control=10,
        n_treatment=10,
        mean_control=1.0,
        mean_treatment=1.1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.decision = "reject_h0"  # type: ignore[misc]


def test_msprt_verdict_validates_decision():
    from soup_cli.utils.ab_test import MsprtVerdict

    with pytest.raises(ValueError):
        MsprtVerdict(
            decision="bogus",
            log_likelihood_ratio=0.0,
            n_control=1,
            n_treatment=1,
            mean_control=0.0,
            mean_treatment=0.0,
        )


# ---------------------------------------------------------------------------
# run_msprt (file driver)
# ---------------------------------------------------------------------------


def test_run_msprt_happy(tmp_path, monkeypatch):
    from soup_cli.utils.ab_test import MsprtConfig, run_msprt

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "ab.jsonl"
    # control: 30 latency=1.0 samples, treatment: 30 latency=2.0 samples
    rows = []
    for _ in range(30):
        rows.append({"arm": "control", "latency": 1.0})
        rows.append({"arm": "treatment", "latency": 2.0})
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    cfg = MsprtConfig(metric="latency", effect_size=0.5)
    verdict = run_msprt(str(inp), config=cfg)
    assert verdict.decision in ("reject_h0", "continue", "accept_h0")
    assert verdict.n_control == 30
    assert verdict.n_treatment == 30


def test_run_msprt_rejects_outside_cwd(tmp_path, monkeypatch):
    from soup_cli.utils.ab_test import MsprtConfig, run_msprt

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "stray.jsonl"
    outside.write_text('{"arm":"control","latency":1.0}\n', encoding="utf-8")
    try:
        cfg = MsprtConfig(metric="latency")
        with pytest.raises(ValueError, match="outside"):
            run_msprt(str(outside), config=cfg)
    finally:
        if outside.exists():
            outside.unlink()


def test_run_msprt_missing_input(tmp_path, monkeypatch):
    from soup_cli.utils.ab_test import MsprtConfig, run_msprt

    monkeypatch.chdir(tmp_path)
    cfg = MsprtConfig(metric="latency")
    with pytest.raises(FileNotFoundError):
        run_msprt(str(tmp_path / "missing.jsonl"), config=cfg)


def test_run_msprt_rejects_null_byte():
    from soup_cli.utils.ab_test import MsprtConfig, run_msprt

    cfg = MsprtConfig(metric="latency")
    with pytest.raises(ValueError):
        run_msprt("bad\x00path.jsonl", config=cfg)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_ab_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["ab", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "metric" in result.output.lower()


def test_cli_ab_happy(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "ab.jsonl"
    rows = []
    for _ in range(30):
        rows.append({"arm": "control", "latency": 1.0})
        rows.append({"arm": "treatment", "latency": 2.0})
    inp.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = runner.invoke(
        app,
        ["ab", "--input", str(inp), "--metric", "latency"],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert (
        "reject_h0" in result.output.lower()
        or "continue" in result.output.lower()
        or "accept_h0" in result.output.lower()
    )


def test_cli_ab_unknown_metric(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    inp = tmp_path / "ab.jsonl"
    inp.write_text('{"arm":"control","latency":1.0}\n', encoding="utf-8")
    result = runner.invoke(
        app,
        ["ab", "--input", str(inp), "--metric", "bogus"],
    )
    assert result.exit_code != 0


def test_cli_ab_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "stray_ab.jsonl"
    outside.write_text('{"arm":"control","latency":1.0}\n', encoding="utf-8")
    try:
        result = runner.invoke(
            app,
            ["ab", "--input", str(outside), "--metric", "latency"],
        )
        assert result.exit_code != 0
        assert "outside" in result.output.lower()
    finally:
        if outside.exists():
            outside.unlink()
