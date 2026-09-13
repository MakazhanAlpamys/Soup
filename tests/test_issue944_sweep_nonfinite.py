"""#944: diverged sweep arms must stay visible and rank after finite losses."""

from __future__ import annotations

from unittest.mock import patch

from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import sweep as sweep_mod

from .conftest import strip_ansi


def _result(name: str, loss: float) -> dict:
    return {
        "name": name,
        "params": {"lr": name},
        "run_id": name,
        "final_loss": loss,
        "duration": "1s",
        "status": "completed",
    }


def test_mixed_nonfinite_losses_sort_last_in_full_order():
    results = [
        _result("one", 1.0),
        _result("nan", float("nan")),
        _result("half", 0.5),
        _result("inf", float("inf")),
        _result("eight", 0.8),
    ]

    ordered = sorted(results, key=sweep_mod._loss_sort_key)

    assert [result["name"] for result in ordered] == ["half", "eight", "one", "nan", "inf"]


def test_failed_zero_loss_does_not_outrank_a_completed_run():
    failed = _result("failed", 0.0)
    failed["status"] = "failed"

    ordered = sorted([failed, _result("completed", 0.5)], key=sweep_mod._loss_sort_key)

    assert [result["name"] for result in ordered] == ["completed", "failed"]


def test_summary_labels_diverged_arms_and_never_selects_them_as_best(monkeypatch):
    console = Console(record=True, width=120)
    monkeypatch.setattr(sweep_mod, "console", console)
    results = [
        _result("nan-run", float("nan")),
        _result("finite-run", 0.5),
        _result("inf-run", float("inf")),
    ]

    sweep_mod._display_summary(results, {"lr": ["nan", "finite", "inf"]})
    output = console.export_text()

    assert output.count("diverged") == 2
    assert output.index("finite-run") < output.index("nan-run") < output.index("inf-run")
    assert "Best run: finite-run (loss: 0.5000)" in output


def test_all_diverged_summary_has_no_best_marker_or_best_run(monkeypatch):
    console = Console(record=True, width=120)
    monkeypatch.setattr(sweep_mod, "console", console)

    sweep_mod._display_summary(
        [_result("nan-run", float("nan")), _result("inf-run", float("inf"))],
        {"lr": ["nan", "inf"]},
    )
    output = console.export_text()

    assert output.count("diverged") == 2
    assert "*" not in output
    assert "Best run:" not in output


def test_early_stop_treats_a_diverged_recent_arm_as_worse(tmp_path, monkeypatch):
    config_file = tmp_path / "soup.yaml"
    config_file.write_text(
        "base: test-model\n"
        "data:\n"
        "  train: ./data.jsonl\n",
        encoding="utf-8",
    )
    run_results = [
        {"final_loss": 1.0, "run_id": "run-1", "duration": "1s"},
        {"final_loss": float("nan"), "run_id": "run-2", "duration": "1s"},
        {"final_loss": 0.5, "run_id": "run-3", "duration": "1s"},
        {"final_loss": float("inf"), "run_id": "run-4", "duration": "1s"},
        {"final_loss": 0.8, "run_id": "run-5", "duration": "1s"},
    ]
    runner = CliRunner()

    with patch.object(sweep_mod, "_run_single", side_effect=run_results) as run_single:
        result = runner.invoke(
            app,
            [
                "sweep",
                "--config",
                str(config_file),
                "--param",
                "lr=1,2,3,4,5",
                "--early-stop",
                "1.5",
                "--yes",
            ],
        )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert run_single.call_count == 2
    output = strip_ansi(result.output)
    assert output.lower().count("diverged") == 3
    assert "skipping 3 remaining run(s)" in output
