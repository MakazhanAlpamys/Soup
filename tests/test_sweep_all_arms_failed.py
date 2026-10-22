"""#1808 — ``soup sweep`` exits non-zero when every arm failed.

Each failing arm was recorded as ``failed`` and the loop went on; after the
summary the command returned normally, so a sweep in which no arm completed
exited 0 with a table of ``failed`` rows. A script chaining on the exit code
read that as success.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli.commands.sweep as sweep_mod
from soup_cli.cli import app

from .conftest import strip_ansi


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture()
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "soup.yaml"
    path.write_text("base: some-model\ndata:\n  train: ./d.jsonl\n", encoding="utf-8")
    return str(path)


def _sweep(config: str, params: list[str]):
    argv = ["sweep", "--config", config, "--yes"]
    for param in params:
        argv += ["--param", param]
    return CliRunner().invoke(app, argv)


def test_every_arm_failing_exits_non_zero_and_says_so(config, monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise FileNotFoundError("Data file not found: missing.jsonl")

    monkeypatch.setattr(sweep_mod, "_run_single", fail)
    result = _sweep(config, ["lr=1e-5,2e-5"])
    out = _plain(result.output)
    assert result.exit_code == 1, (result.exit_code, out)
    assert "no sweep run completed" in out.lower()
    assert "2 of 2" in out
    assert "missing.jsonl" in out  # the first error, so the cause is on screen


def test_a_single_failing_arm_exits_non_zero_too(config, monkeypatch) -> None:
    monkeypatch.setattr(sweep_mod, "_run_single", lambda *a, **k: 1 / 0)
    result = _sweep(config, ["lr=1e-5"])
    assert result.exit_code == 1, _plain(result.output)
    assert "1 of 1" in _plain(result.output)


def test_control_one_completed_arm_still_exits_zero(config, monkeypatch) -> None:
    calls = {"n": 0}

    def one_fails(cfg, combo, run_name, config_path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"final_loss": 0.5, "run_id": "r2", "duration": "1s"}

    monkeypatch.setattr(sweep_mod, "_run_single", one_fails)
    result = _sweep(config, ["lr=1e-5,2e-5"])
    out = _plain(result.output)
    assert result.exit_code == 0, (result.exit_code, out)
    assert "no sweep run completed" not in out.lower()


def test_control_all_arms_completing_exits_zero(config, monkeypatch) -> None:
    monkeypatch.setattr(
        sweep_mod, "_run_single",
        lambda cfg, combo, name, path: {"final_loss": 0.4, "run_id": name, "duration": "1s"},
    )
    result = _sweep(config, ["lr=1e-5,2e-5"])
    assert result.exit_code == 0, _plain(result.output)
