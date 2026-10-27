"""Regression tests for stale v0.77.0 help text (#2221)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.cloud import lambda_labs, modal, runpod
from soup_cli.config.schema import TrainingConfig
from tests.conftest import strip_ansi

runner = CliRunner()


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


def _help(monkeypatch, *args: str) -> str:
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, [*args, "--help"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return _plain(result.output.replace("│", " "))


def test_mcp_serve_help_describes_the_execution_gate(monkeypatch):
    out = _help(monkeypatch, "mcp", "serve")

    assert "never execute in this version" not in out
    assert "future MCP tools" not in out
    assert "train_execute" in out
    assert "export_execute" in out
    assert "Refused with --transport sse or http" in out


def test_ab_help_and_root_help_use_bayes_factor_wording(monkeypatch):
    for args in ((), ("ab",)):
        out = _help(monkeypatch, *args)
        assert "mSPRT" not in out

    out = _help(monkeypatch, "ab")
    assert "Bayes factor with a confidence-sequence accept rule" in out
    assert "valid under re-running after every new row" in out


def test_ab_verdict_table_is_not_titled_msprt(monkeypatch, tmp_path: Path):
    path = tmp_path / "ab.jsonl"
    rows = [
        {"arm": "control", "latency": 1.0},
        {"arm": "treatment", "latency": 2.0},
    ] * 20
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["ab", "--input", path.name, "--metric", "latency"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    out = _plain(result.output)
    assert "mSPRT" not in out
    assert "A/B verdict — latency" in out


def test_train_gpu_help_lists_every_provider_gpu(monkeypatch):
    out = _help(monkeypatch, "train")

    assert "modal: t4, l4, a10g, a100, a100-80gb, l40s, h100" in out
    assert "lambda: a10, a100, a6000, h100" in out
    assert "runpod: a100, a40, h100, rtx3090, rtx4090" in out
    assert "Default a100" in out
    assert "A type another provider lists is refused with exit 2" in out

    for provider in (modal, lambda_labs, runpod):
        for gpu in provider.SUPPORTED_GPUS:
            assert gpu in out


def test_kv_cache_type_description_matches_actual_runtime():
    description = TrainingConfig.model_fields["kv_cache_type"].description or ""

    assert "--kv-cache-type" in description
    assert "Applied by soup serve" not in description
    assert "Recorded in soup.yaml only" in description
    assert "neither soup train nor soup serve reads this key" in description
