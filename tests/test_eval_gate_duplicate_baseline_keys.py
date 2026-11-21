"""Baseline JSON keys must be unambiguous before the gate loads a model."""

from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils.exit_codes import EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE_ERROR
from tests.conftest import strip_ansi


@pytest.mark.parametrize(
    ("baseline", "exit_code", "repeated_key"),
    [
        ('{"custom":0.9}', EXIT_GATE_FAILED, None),
        ('{"custom":0.1}', EXIT_OK, None),
        ('{"custom":0.9,"custom":0.1}', EXIT_USAGE_ERROR, "custom"),
        ('{"scores":{"custom":0.9},"provenance":{"scorer_revision":1}}', EXIT_GATE_FAILED, None),
        ('{"scores":{"custom":0.9,"custom":0.1}}', EXIT_USAGE_ERROR, "custom"),
        ('{"scores":{"custom":0.9},"scores":{"custom":0.1}}', EXIT_USAGE_ERROR, "scores"),
        ('{"[red]":0.9,"[red]":0.1}', EXIT_USAGE_ERROR, "[red]"),
    ],
    ids=[
        "legacy-regression",
        "legacy-pass",
        "legacy-duplicate",
        "stamped-regression",
        "stamped-duplicate-task",
        "stamped-duplicate-envelope",
        "duplicate-key-with-markup",
    ],
)
def test_gate_baseline_object_keys(tmp_path, monkeypatch, baseline, exit_code, repeated_key):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "suite.yaml").write_text(
        "suite: duplicate-baseline\ntasks:\n"
        "  - {name: custom, type: custom, tasks: tasks.jsonl, scorer: exact, threshold: 0.25}\n",
        encoding="utf-8",
    )
    (tmp_path / "tasks.jsonl").write_text(
        '{"prompt":"hit","expected":"gold"}\n{"prompt":"miss","expected":"gold"}\n',
        encoding="utf-8",
    )
    (tmp_path / "baseline.json").write_text(baseline, encoding="utf-8")
    factory = Mock(return_value=lambda prompt: "gold" if prompt == "hit" else "other")
    monkeypatch.setattr("soup_cli.eval.quant_check.make_model_generator", factory)

    result = CliRunner().invoke(
        app, ["eval", "gate", "--suite", "suite.yaml", "--model", "synthetic",
              "--baseline", "baseline.json"],
    )

    assert result.exit_code == exit_code, (result.output, repr(result.exception))
    if repeated_key is None:
        factory.assert_called_once_with("synthetic")
    else:
        factory.assert_not_called()
        assert f"duplicate JSON object key {repeated_key!r}" in " ".join(
            strip_ansi(result.output).split()
        )
