"""v0.63.0 Part E — Online-eval drift alarm tests."""

from __future__ import annotations

import dataclasses
import json

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def test_module_imports():
    from soup_cli.utils import drift_alarm

    assert hasattr(drift_alarm, "DriftReport")
    assert hasattr(drift_alarm, "compute_token_distribution")
    assert hasattr(drift_alarm, "rolling_kl")
    assert hasattr(drift_alarm, "run_drift_check")
    assert hasattr(drift_alarm, "validate_webhook_url")
    assert hasattr(drift_alarm, "validate_threshold")


# ---------------------------------------------------------------------------
# validate_threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.01, 0.05, 0.5, 1.0, 10.0])
def test_validate_threshold_happy(value):
    from soup_cli.utils.drift_alarm import validate_threshold

    assert validate_threshold(value) == float(value)


@pytest.mark.parametrize(
    "bad", [True, False, None, "0.1", -0.1, 0.0, float("nan"), float("inf"), 100.1],
)
def test_validate_threshold_rejects(bad):
    from soup_cli.utils.drift_alarm import validate_threshold

    with pytest.raises((TypeError, ValueError)):
        validate_threshold(bad)


# ---------------------------------------------------------------------------
# validate_webhook_url — SSRF parity with v0.30.0
# ---------------------------------------------------------------------------


def test_validate_webhook_url_https_ok():
    from soup_cli.utils.drift_alarm import validate_webhook_url

    assert validate_webhook_url("https://hooks.slack.com/services/xxx") is not None


def test_validate_webhook_url_loopback_http_ok():
    from soup_cli.utils.drift_alarm import validate_webhook_url

    assert validate_webhook_url("http://127.0.0.1:9000/hook") is not None
    assert validate_webhook_url("http://localhost:8080/hook") is not None


@pytest.mark.parametrize(
    "bad",
    [
        None,
        True,
        "",
        "ftp://example.com",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://10.0.0.1/hook",  # RFC1918
        "http://169.254.169.254/latest",  # AWS metadata
        "http://0.0.0.0/hook",  # bind-any
        "http://192.168.1.1/hook",  # RFC1918
        "http://example.com\x00.com",  # null byte
        "x" * 4097,
    ],
)
def test_validate_webhook_url_rejects(bad):
    from soup_cli.utils.drift_alarm import validate_webhook_url

    with pytest.raises((TypeError, ValueError)):
        validate_webhook_url(bad)


# ---------------------------------------------------------------------------
# compute_token_distribution
# ---------------------------------------------------------------------------


def test_compute_token_distribution_basic():
    from soup_cli.utils.drift_alarm import compute_token_distribution

    rows = ["hello world", "hello there", "world peace"]
    dist = compute_token_distribution(rows)
    # All probabilities sum to 1
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    # "hello" appears twice, "world" twice — they share the top
    assert dist["hello"] > 0


def test_compute_token_distribution_empty():
    from soup_cli.utils.drift_alarm import compute_token_distribution

    dist = compute_token_distribution([])
    assert dist == {}


def test_compute_token_distribution_skips_non_string():
    from soup_cli.utils.drift_alarm import compute_token_distribution

    dist = compute_token_distribution(["hello", 42, None, "world"])
    assert "hello" in dist and "world" in dist


def test_compute_token_distribution_rejects_non_iterable():
    from soup_cli.utils.drift_alarm import compute_token_distribution

    with pytest.raises(TypeError):
        compute_token_distribution(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# rolling_kl
# ---------------------------------------------------------------------------


def test_rolling_kl_identical_distributions():
    from soup_cli.utils.drift_alarm import rolling_kl

    p = {"a": 0.5, "b": 0.5}
    q = {"a": 0.5, "b": 0.5}
    assert rolling_kl(p, q) == pytest.approx(0.0, abs=1e-9)


def test_rolling_kl_different_distributions():
    from soup_cli.utils.drift_alarm import rolling_kl

    p = {"a": 0.9, "b": 0.1}
    q = {"a": 0.1, "b": 0.9}
    kl = rolling_kl(p, q)
    assert kl > 0.5  # large divergence


def test_rolling_kl_handles_missing_keys():
    """Smoothing: token in p but not q should not crash."""
    from soup_cli.utils.drift_alarm import rolling_kl

    p = {"a": 0.5, "b": 0.5}
    q = {"a": 1.0}
    kl = rolling_kl(p, q)
    assert kl > 0.0


def test_rolling_kl_rejects_negative_prob():
    from soup_cli.utils.drift_alarm import rolling_kl

    with pytest.raises(ValueError):
        rolling_kl({"a": -0.1, "b": 1.1}, {"a": 0.5, "b": 0.5})


def test_rolling_kl_rejects_non_finite():
    from soup_cli.utils.drift_alarm import rolling_kl

    with pytest.raises(ValueError):
        rolling_kl({"a": float("nan")}, {"a": 1.0})


def test_rolling_kl_rejects_non_mapping():
    from soup_cli.utils.drift_alarm import rolling_kl

    with pytest.raises(TypeError):
        rolling_kl([0.5, 0.5], {"a": 1.0})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------


def test_drift_report_frozen():
    from soup_cli.utils.drift_alarm import DriftReport

    r = DriftReport(
        kl_divergence=0.5,
        threshold=0.2,
        drift_detected=True,
        n_reference=100,
        n_live=100,
        top_drift_tokens=(("hello", 0.3),),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.kl_divergence = 0.6  # type: ignore[misc]


def test_drift_report_validates():
    from soup_cli.utils.drift_alarm import DriftReport

    # KL must be >= 0
    with pytest.raises(ValueError):
        DriftReport(
            kl_divergence=-0.1,
            threshold=0.2,
            drift_detected=False,
            n_reference=1,
            n_live=1,
            top_drift_tokens=(),
        )
    # n_reference / n_live must be >= 0
    with pytest.raises(ValueError):
        DriftReport(
            kl_divergence=0.1,
            threshold=0.2,
            drift_detected=False,
            n_reference=-1,
            n_live=1,
            top_drift_tokens=(),
        )


# ---------------------------------------------------------------------------
# run_drift_check
# ---------------------------------------------------------------------------


def test_run_drift_check_happy(tmp_path, monkeypatch):
    from soup_cli.utils.drift_alarm import run_drift_check

    monkeypatch.chdir(tmp_path)
    ref = tmp_path / "ref.jsonl"
    live = tmp_path / "live.jsonl"
    ref.write_text(
        "\n".join(
            json.dumps({"output": text}) for text in ["hello world"] * 10
        ),
        encoding="utf-8",
    )
    live.write_text(
        "\n".join(
            json.dumps({"output": text}) for text in ["completely different content"] * 10
        ),
        encoding="utf-8",
    )

    report = run_drift_check(
        reference_path=str(ref),
        live_path=str(live),
        threshold=0.1,
    )
    assert report.drift_detected is True
    assert report.kl_divergence > 0.1


def test_run_drift_check_below_threshold(tmp_path, monkeypatch):
    from soup_cli.utils.drift_alarm import run_drift_check

    monkeypatch.chdir(tmp_path)
    ref = tmp_path / "ref.jsonl"
    live = tmp_path / "live.jsonl"
    payload = "\n".join(
        json.dumps({"output": text}) for text in ["hello world"] * 10
    )
    ref.write_text(payload, encoding="utf-8")
    live.write_text(payload, encoding="utf-8")

    report = run_drift_check(
        reference_path=str(ref),
        live_path=str(live),
        threshold=0.1,
    )
    assert report.drift_detected is False
    assert report.kl_divergence == pytest.approx(0.0, abs=1e-9)


def test_run_drift_check_rejects_outside_cwd(tmp_path, monkeypatch):
    from soup_cli.utils.drift_alarm import run_drift_check

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "stray.jsonl"
    outside.write_text('{"output":"x"}\n', encoding="utf-8")
    live = tmp_path / "live.jsonl"
    live.write_text('{"output":"y"}\n', encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside"):
            run_drift_check(
                reference_path=str(outside),
                live_path=str(live),
                threshold=0.1,
            )
    finally:
        if outside.exists():
            outside.unlink()


def test_run_drift_check_missing_files(tmp_path, monkeypatch):
    from soup_cli.utils.drift_alarm import run_drift_check

    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_drift_check(
            reference_path=str(tmp_path / "missing.jsonl"),
            live_path=str(tmp_path / "missing.jsonl"),
            threshold=0.1,
        )


def test_run_drift_check_rejects_null_byte():
    from soup_cli.utils.drift_alarm import run_drift_check

    with pytest.raises(ValueError):
        run_drift_check(
            reference_path="bad\x00path.jsonl",
            live_path="ok.jsonl",
            threshold=0.1,
        )


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_drift_alarm_help():
    from soup_cli.cli import app

    result = runner.invoke(app, ["drift-alarm", "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert "threshold" in result.output.lower()


def test_cli_drift_alarm_happy(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    ref = tmp_path / "ref.jsonl"
    live = tmp_path / "live.jsonl"
    ref.write_text(
        "\n".join(json.dumps({"output": "hello world"}) for _ in range(5)),
        encoding="utf-8",
    )
    live.write_text(
        "\n".join(json.dumps({"output": "hello world"}) for _ in range(5)),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "drift-alarm",
            "--reference",
            str(ref),
            "--live",
            str(live),
            "--threshold",
            "0.1",
        ],
    )
    assert result.exit_code == 0, (result.output, repr(result.exception))


def test_cli_drift_alarm_outside_cwd_rejected(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "stray.jsonl"
    outside.write_text('{"output":"x"}\n', encoding="utf-8")
    live = tmp_path / "live.jsonl"
    live.write_text('{"output":"y"}\n', encoding="utf-8")
    try:
        result = runner.invoke(
            app,
            [
                "drift-alarm",
                "--reference",
                str(outside),
                "--live",
                str(live),
                "--threshold",
                "0.1",
            ],
        )
        assert result.exit_code != 0
        assert "outside" in result.output.lower()
    finally:
        if outside.exists():
            outside.unlink()


def test_cli_drift_alarm_invalid_threshold(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    ref = tmp_path / "ref.jsonl"
    ref.write_text('{"output":"x"}\n', encoding="utf-8")
    live = tmp_path / "live.jsonl"
    live.write_text('{"output":"y"}\n', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "drift-alarm",
            "--reference",
            str(ref),
            "--live",
            str(live),
            "--threshold",
            "-0.1",
        ],
    )
    assert result.exit_code != 0


def test_cli_drift_alarm_webhook_validated(tmp_path, monkeypatch):
    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    ref = tmp_path / "ref.jsonl"
    ref.write_text('{"output":"x"}\n', encoding="utf-8")
    live = tmp_path / "live.jsonl"
    live.write_text('{"output":"y"}\n', encoding="utf-8")

    # Invalid webhook URL -> exit 2
    result = runner.invoke(
        app,
        [
            "drift-alarm",
            "--reference",
            str(ref),
            "--live",
            str(live),
            "--threshold",
            "0.1",
            "--slack-url",
            "http://10.0.0.1/hook",  # RFC1918 rejected
        ],
    )
    assert result.exit_code != 0
