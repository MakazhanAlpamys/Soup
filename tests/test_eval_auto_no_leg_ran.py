"""#1808 — ``soup eval auto`` exits non-zero when no leg evaluated anything.

A benchmark leg that could not run was swallowed and "Auto-eval complete."
exited 0 when it was the only leg. The custom leg's ``except SystemExit`` could
not catch ``typer.Exit`` (a ``RuntimeError``), so a custom-only failure exited 1
with a traceback-style abort: the exit code depended on which leg failed. Now
both legs are caught the same way, "Auto-eval complete." is printed only when a
leg ran, and nothing having run exits 1 whichever leg it was.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import soup_cli.commands.eval as eval_mod
from soup_cli.cli import app

from .conftest import strip_ansi


def _plain(text: str) -> str:
    return " ".join(strip_ansi(text).split())


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "HuggingFaceTB/SmolLM2-135M"}', encoding="utf-8"
    )
    (tmp_path / "eval.yaml").write_text(
        "base: HuggingFaceTB/SmolLM2-135M\ntask: sft\ndata:\n  train: ./missing.jsonl\n"
        "output: out\n",
        encoding="utf-8",
    )
    (tmp_path / "tasks.jsonl").write_text('{"prompt": "p", "expected": "e"}\n', encoding="utf-8")
    return tmp_path


def _failing(*args, **kwargs):
    raise typer.Exit(1)


def _ran(*args, **kwargs):
    return None


def _auto(*extra: str):
    return CliRunner().invoke(app, ["eval", "auto", "-c", "eval.yaml", *extra])


def test_benchmark_only_leg_that_cannot_run_exits_non_zero(workdir, monkeypatch) -> None:
    monkeypatch.setattr(eval_mod, "benchmark", _failing)
    result = _auto("--benchmarks", "mmlu")
    out = _plain(result.output)
    assert result.exit_code == 1, (result.exit_code, out)
    assert "Auto-eval complete." not in out
    assert "nothing was evaluated" in out.lower()


def test_custom_only_leg_that_cannot_run_exits_the_same_code(workdir, monkeypatch) -> None:
    monkeypatch.setattr(eval_mod, "custom", _failing)
    result = _auto("--tasks", "tasks.jsonl")
    out = _plain(result.output)
    assert result.exit_code == 1, (result.exit_code, out)
    assert "Custom eval skipped" in out  # caught like the benchmark leg, not aborted
    assert "Auto-eval complete." not in out


def test_both_legs_failing_exits_non_zero(workdir, monkeypatch) -> None:
    monkeypatch.setattr(eval_mod, "benchmark", _failing)
    monkeypatch.setattr(eval_mod, "custom", _failing)
    result = _auto("--benchmarks", "mmlu", "--tasks", "tasks.jsonl")
    assert result.exit_code == 1, _plain(result.output)


def test_control_one_of_two_legs_running_still_completes(workdir, monkeypatch) -> None:
    monkeypatch.setattr(eval_mod, "benchmark", _failing)
    monkeypatch.setattr(eval_mod, "custom", _ran)
    result = _auto("--benchmarks", "mmlu", "--tasks", "tasks.jsonl")
    out = _plain(result.output)
    assert result.exit_code == 0, (result.exit_code, out)
    assert "Benchmark eval skipped" in out
    assert "Auto-eval complete." in out


def test_control_a_leg_that_runs_exits_zero(workdir, monkeypatch) -> None:
    monkeypatch.setattr(eval_mod, "custom", _ran)
    result = _auto("--tasks", "tasks.jsonl")
    out = _plain(result.output)
    assert result.exit_code == 0, (result.exit_code, out)
    assert "Auto-eval complete." in out


def test_control_a_benchmark_leg_that_runs_exits_zero(workdir, monkeypatch) -> None:
    # The benchmark leg's own `legs_ran += 1`: with no tasks file the benchmark leg
    # is the only one, so if it ran and is not counted the command would say
    # "nothing was evaluated" (#1905 review).
    monkeypatch.setattr(eval_mod, "benchmark", _ran)
    result = _auto("--benchmarks", "mmlu")
    out = _plain(result.output)
    assert result.exit_code == 0, (result.exit_code, out)
    assert "Auto-eval complete." in out
    assert "nothing was evaluated" not in out.lower()
